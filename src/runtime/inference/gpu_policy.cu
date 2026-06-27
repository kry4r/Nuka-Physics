// ---------------------------------------------------------------------------
// gpu_policy.cu -- the device-resident Go2 policy kernels + a general batched MLP
// forward (see gpu_policy.hpp). This TU owns the CUDA tokens; the host controller
// + app reach it only through the CUDA-free gpu_policy.hpp interface.
//
// The MLP forward reproduces MlpPolicy::Forward arithmetic (RunningMeanStd
// normalize + clamp, per-output double-accumulated dot, fp32 ELU) so the device
// action matches the host oracle to ~1e-5 (the CPU-oracle-validates-GPU pattern).
// One control step (Go2GpuPolicy::Step) runs obs-assembly -> MLP -> drive-write
// entirely on the world backend's main stream: the obs kernels read the SAME
// device fields DownloadField reads, so there is no host round-trip.
// ---------------------------------------------------------------------------

#include "runtime/inference/gpu_policy.hpp"

#include "runtime/inference/mlp_policy.hpp"
#include "scene/terrain/heightfield.hpp"
#include "scene/terrain/heightfield_sample.hpp"

#include "phi/backend_cuda/cuda_internal.cuh"  // CudaBackend, CudaBackendMainStream

#include "math/vec3.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace nuka::runtime::inference {

namespace {

constexpr uint32_t kBlock = 128u;

bool AllocDev(void** p, size_t bytes) {
    *p = nullptr;
    if (bytes == 0u) return true;
    if (cudaMalloc(p, bytes) != cudaSuccess) {
        *p = nullptr;
        return false;
    }
    return true;
}
void FreeDev(void* p) {
    if (p != nullptr) cudaFree(p);
}
bool UploadDev(void* dst, const void* src, size_t bytes) {
    return cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}

cudaStream_t MainStreamOf(void* backend) {
    if (backend == nullptr) return nullptr;
    return phi::CudaBackendMainStream(reinterpret_cast<phi::CudaBackend*>(backend));
}

__device__ inline float DevClampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// World gravity [0,0,-1] rotated into the body frame (R(q)^T @ [0,0,-1]); q is
// w-first. Bit-for-bit the host ProjectedGravity (go2_policy_controller.cpp).
__device__ inline void DevProjectedGravity(float w, float x, float y, float z,
                                           float* out) {
    const float vz = -1.0f;
    const float a = w * w - (x * x + y * y + z * z);
    const float cx = y * vz, cy = -x * vz, cz = 0.0f;
    const float qv = z * vz;
    out[0] = -2.0f * w * cx + 2.0f * x * qv;
    out[1] = -2.0f * w * cy + 2.0f * y * qv;
    out[2] = a * vz - 2.0f * w * cz + 2.0f * z * qv;
}

// ---------------------------------------------------------------------------
// General batched MLP kernels. Sizes are PARAMS (any obs/act dim, any batch); the
// per-output dot accumulates in double, matching the host. Strided in/out so the
// final head writes the caller's action buffer directly.
// ---------------------------------------------------------------------------
__global__ void NormalizeKernel(const float* obs, uint32_t obs_stride, float* x,
                                uint32_t x_stride, const float* mean,
                                const float* std_dev, float clip, uint32_t obs_dim,
                                uint32_t batch) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * obs_dim) return;
    const uint32_t s = idx / obs_dim;
    const uint32_t i = idx - s * obs_dim;
    float y = (obs[s * obs_stride + i] - mean[i]) / std_dev[i];
    y = y < -clip ? -clip : (y > clip ? clip : y);
    x[s * x_stride + i] = y;
}

__global__ void MlpLayerKernel(const float* x, uint32_t x_stride, float* y,
                               uint32_t y_stride, const float* W, const float* b,
                               uint32_t in_dim, uint32_t out_dim, uint32_t batch,
                               uint32_t act, float alpha) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * out_dim) return;
    const uint32_t s = idx / out_dim;
    const uint32_t o = idx - s * out_dim;
    const float* xr = x + static_cast<size_t>(s) * x_stride;
    const float* wr = W + static_cast<size_t>(o) * in_dim;
    double acc = static_cast<double>(b[o]);
    for (uint32_t i = 0; i < in_dim; ++i) {
        acc += static_cast<double>(wr[i]) * static_cast<double>(xr[i]);
    }
    float v = static_cast<float>(acc);
    if (act == 1u) v = v > 0.0f ? v : alpha * (expf(v) - 1.0f);
    y[static_cast<size_t>(s) * y_stride + o] = v;
}

// ---------------------------------------------------------------------------
// Go2 obs/drive kernels. Field pointers are env-base (env_index already folded in);
// a dog's root link is dog*links_per_dog, leg slot u lives at root+1+slot[u].
// ---------------------------------------------------------------------------
struct Go2DevParams {
    const float* link_pose;      // env-base, stride 7 [pos.xyz, quat.wxyz]
    const float* link_velocity;  // env-base, stride 6 [ang.xyz, lin.xyz]
    const float* q;              // env-base, stride 1
    const float* qdot;           // env-base, stride 1
    const float* base_pose;      // env-base, stride 7
    float*       drive_target;   // env-base, stride 1
    const float* action_in;      // num_dogs*act_dim
    float*       obs;            // num_dogs*obs_dim
    float*       last_action;    // num_dogs*njoints
    const float* default_angles;  // device [njoints]
    const uint32_t* slot;        // device [njoints]
    const float* terrain_values;  // device grid
    uint32_t ter_nrow, ter_ncol;
    float    ter_ox, ter_oy, ter_oz, ter_base_z, ter_scale_z, ter_cx, ter_cy;
    float    lin_vel_scale, ang_vel_scale, dof_pos_scale, dof_vel_scale;
    float    cmd_scale0, cmd_scale1, cmd_scale2;
    float    command0, command1, command2;
    float    action_scale, obs_clip, action_clip;
    float    scan_x_min, scan_y_min, scan_res, scan_scale, scan_clip, scan_z_offset;
    uint32_t njoints, proprio_dim, scan_nx, scan_ny, obs_dim, act_dim;
    uint32_t num_dogs, links_per_dog;
};

// Proprio block (one thread per dog): base lin/ang vel, projected gravity, command,
// joint pos/vel deviations, last action; then clamp [0,proprio_dim).
__global__ void Go2ProprioKernel(Go2DevParams p) {
    const uint32_t dog = blockIdx.x * blockDim.x + threadIdx.x;
    if (dog >= p.num_dogs) return;
    const uint32_t root = dog * p.links_per_dog;
    const float* v = p.link_velocity + static_cast<size_t>(root) * 6u;
    const float* pose = p.link_pose + static_cast<size_t>(root) * 7u;
    float pg[3];
    DevProjectedGravity(pose[3], pose[4], pose[5], pose[6], pg);
    float* obs = p.obs + static_cast<size_t>(dog) * p.obs_dim;
    uint32_t o = 0u;
    obs[o++] = v[3] * p.lin_vel_scale;
    obs[o++] = v[4] * p.lin_vel_scale;
    obs[o++] = v[5] * p.lin_vel_scale;
    obs[o++] = v[0] * p.ang_vel_scale;
    obs[o++] = v[1] * p.ang_vel_scale;
    obs[o++] = v[2] * p.ang_vel_scale;
    obs[o++] = pg[0];
    obs[o++] = pg[1];
    obs[o++] = pg[2];
    obs[o++] = p.command0 * p.cmd_scale0;
    obs[o++] = p.command1 * p.cmd_scale1;
    obs[o++] = p.command2 * p.cmd_scale2;
    for (uint32_t u = 0; u < p.njoints; ++u) {
        const float qv = p.q[root + 1u + p.slot[u]];
        obs[o++] = (qv - p.default_angles[u]) * p.dof_pos_scale;
    }
    for (uint32_t u = 0; u < p.njoints; ++u) {
        obs[o++] = p.qdot[root + 1u + p.slot[u]] * p.dof_vel_scale;
    }
    const float* la = p.last_action + static_cast<size_t>(dog) * p.njoints;
    for (uint32_t u = 0; u < p.njoints; ++u) obs[o++] = la[u];
    for (uint32_t i = 0; i < p.proprio_dim; ++i) {
        obs[i] = DevClampf(obs[i], -p.obs_clip, p.obs_clip);
    }
}

// Height scan (one thread per dog x scan cell): bilinear terrain sample under the
// yaw-rotated grid, relative to the base, scaled. Same layout obs[proprio+ix*ny+iy].
__global__ void Go2HeightScanKernel(Go2DevParams p) {
    const uint32_t scan_n = p.scan_nx * p.scan_ny;
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= p.num_dogs * scan_n) return;
    const uint32_t dog = idx / scan_n;
    const uint32_t cell = idx - dog * scan_n;
    const uint32_t ix = cell / p.scan_ny;
    const uint32_t iy = cell - ix * p.scan_ny;
    const float* b = p.base_pose + static_cast<size_t>(dog) * 7u;
    const float bx = b[0], by = b[1], bz = b[2];
    const float qw = b[3], qx = b[4], qy = b[5], qz = b[6];
    const float yaw = atan2f(2.0f * (qw * qz + qx * qy),
                             1.0f - 2.0f * (qy * qy + qz * qz));
    const float cy = cosf(yaw), sy = sinf(yaw);
    const float px = p.scan_x_min + static_cast<float>(ix) * p.scan_res;
    const float py = p.scan_y_min + static_cast<float>(iy) * p.scan_res;
    const float wx = bx + (px * cy - py * sy);
    const float wy = by + (px * sy + py * cy);
    const float surf = ::nuka::terrain::SampleHeightFieldZRaw(
        p.terrain_values, p.ter_nrow, p.ter_ncol,
        math::Vec3{p.ter_ox, p.ter_oy, p.ter_oz}, p.ter_base_z, p.ter_scale_z,
        p.ter_cx, p.ter_cy, wx, wy);
    const float val =
        DevClampf(bz - p.scan_z_offset - surf, -p.scan_clip, p.scan_clip) *
        p.scan_scale;
    p.obs[static_cast<size_t>(dog) * p.obs_dim + p.proprio_dim + ix * p.scan_ny + iy] =
        val;
}

// Drive write (one thread per dog x joint): clamp the action, persist it as the
// next-step last_action, and set the PD target default + scale*action at the slot.
__global__ void Go2DriveWriteKernel(Go2DevParams p) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= p.num_dogs * p.njoints) return;
    const uint32_t dog = idx / p.njoints;
    const uint32_t u = idx - dog * p.njoints;
    const uint32_t root = dog * p.links_per_dog;
    const float a = DevClampf(p.action_in[static_cast<size_t>(dog) * p.act_dim + u],
                              -p.action_clip, p.action_clip);
    p.last_action[static_cast<size_t>(dog) * p.njoints + u] = a;
    p.drive_target[root + 1u + p.slot[u]] = p.default_angles[u] + p.action_scale * a;
}

// PD gains + per-joint force limits onto every dog's leg slots (one thread per dog
// x joint). Non-leg slots are untouched.
__global__ void Go2GainsKernel(float* stiffness, float* damping, float* force_limit,
                               const uint32_t* slot, const float* fl, float kp,
                               float kd, uint32_t num_dogs, uint32_t njoints,
                               uint32_t links_per_dog) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_dogs * njoints) return;
    const uint32_t dog = idx / njoints;
    const uint32_t u = idx - dog * njoints;
    const uint32_t s = dog * links_per_dog + 1u + slot[u];
    stiffness[s] = kp;
    damping[s] = kd;
    force_limit[s] = fl[u];
}

uint32_t Grid(uint32_t n) { return (n + kBlock - 1u) / kBlock; }

}  // namespace

// ===========================================================================
// GpuMlp::Impl -- device weights + the on-stream forward.
// ===========================================================================
struct GpuMlp::Impl {
    struct DevLayer {
        uint32_t in = 0, out = 0, act = 0;
        float* w = nullptr;
        float* b = nullptr;
    };
    uint32_t obs_dim = 0, act_dim = 0, max_width = 0;
    float clip = 5.0f, alpha = 1.0f;
    float* mean = nullptr;
    float* std = nullptr;
    std::vector<DevLayer> layers;
    // ping-pong activations, sized batch_cap*max_width.
    float* buf_a = nullptr;
    float* buf_b = nullptr;
    uint32_t batch_cap = 0;
    // standalone-forward staging + dedicated stream.
    float* io_obs = nullptr;
    float* io_out = nullptr;
    uint32_t io_cap = 0;
    cudaStream_t stream = nullptr;
    bool ready = false;

    ~Impl() {
        for (DevLayer& L : layers) {
            FreeDev(L.w);
            FreeDev(L.b);
        }
        FreeDev(mean);
        FreeDev(std);
        FreeDev(buf_a);
        FreeDev(buf_b);
        FreeDev(io_obs);
        FreeDev(io_out);
        if (stream != nullptr) cudaStreamDestroy(stream);
    }

    bool EnsureActivation(uint32_t batch) {
        if (batch <= batch_cap && buf_a != nullptr) return true;
        FreeDev(buf_a);
        FreeDev(buf_b);
        buf_a = buf_b = nullptr;
        const size_t bytes = static_cast<size_t>(batch) * max_width * sizeof(float);
        if (!AllocDev(reinterpret_cast<void**>(&buf_a), bytes) ||
            !AllocDev(reinterpret_cast<void**>(&buf_b), bytes)) {
            return false;
        }
        batch_cap = batch;
        return true;
    }

    // d_obs has stride obs_dim; d_out has stride act_dim. All launches on `s`.
    bool ForwardDevice(const float* d_obs, float* d_out, uint32_t batch,
                       cudaStream_t s) {
        if (!ready || batch == 0u) return false;
        if (!EnsureActivation(batch)) return false;
        NormalizeKernel<<<Grid(batch * obs_dim), kBlock, 0, s>>>(
            d_obs, obs_dim, buf_a, max_width, mean, std, clip, obs_dim, batch);
        const float* in = buf_a;
        float* other = buf_b;
        for (size_t l = 0; l < layers.size(); ++l) {
            const DevLayer& L = layers[l];
            const bool last = (l + 1u == layers.size());
            float* out = last ? d_out : other;
            const uint32_t out_stride = last ? act_dim : max_width;
            MlpLayerKernel<<<Grid(batch * L.out), kBlock, 0, s>>>(
                in, max_width, out, out_stride, L.w, L.b, L.in, L.out, batch, L.act,
                alpha);
            if (!last) {
                float* prev = const_cast<float*>(in);
                in = other;
                other = prev;
            }
        }
        return true;
    }
};

GpuMlp::GpuMlp() : impl_(new Impl()) {}
GpuMlp::~GpuMlp() { delete impl_; }

bool GpuMlp::InitFromHost(const MlpPolicy& host) {
    if (!host.Ready()) return false;
    Impl& m = *impl_;
    m.obs_dim = host.ObsDim();
    m.act_dim = host.ActDim();
    m.clip = host.Clip();
    m.alpha = host.EluAlpha();
    m.max_width = m.obs_dim;

    const size_t nbytes = static_cast<size_t>(m.obs_dim) * sizeof(float);
    if (!AllocDev(reinterpret_cast<void**>(&m.mean), nbytes) ||
        !AllocDev(reinterpret_cast<void**>(&m.std), nbytes) ||
        !UploadDev(m.mean, host.Mean(), nbytes) ||
        !UploadDev(m.std, host.Std(), nbytes)) {
        std::fprintf(stderr, "[gpu_policy] normalizer upload failed\n");
        return false;
    }
    for (uint32_t l = 0; l < host.LayerCount(); ++l) {
        const MlpPolicy::LayerView v = host.GetLayer(l);
        Impl::DevLayer L;
        L.in = v.in;
        L.out = v.out;
        L.act = v.act;
        const size_t wbytes = static_cast<size_t>(v.in) * v.out * sizeof(float);
        const size_t bbytes = static_cast<size_t>(v.out) * sizeof(float);
        if (!AllocDev(reinterpret_cast<void**>(&L.w), wbytes) ||
            !AllocDev(reinterpret_cast<void**>(&L.b), bbytes) ||
            !UploadDev(L.w, v.w, wbytes) || !UploadDev(L.b, v.b, bbytes)) {
            std::fprintf(stderr, "[gpu_policy] layer %u upload failed\n", l);
            FreeDev(L.w);
            FreeDev(L.b);
            return false;
        }
        m.max_width = std::max(m.max_width, v.out);
        m.layers.push_back(L);
    }
    if (m.layers.empty()) return false;
    cudaStreamCreateWithFlags(&m.stream, cudaStreamNonBlocking);
    m.ready = true;
    return true;
}

bool GpuMlp::Ready() const { return impl_ != nullptr && impl_->ready; }
uint32_t GpuMlp::ObsDim() const { return impl_->obs_dim; }
uint32_t GpuMlp::ActDim() const { return impl_->act_dim; }

bool GpuMlp::Forward(const float* obs, float* out, uint32_t batch) {
    Impl& m = *impl_;
    if (!m.ready || batch == 0u) return false;
    if (batch > m.io_cap || m.io_obs == nullptr) {
        FreeDev(m.io_obs);
        FreeDev(m.io_out);
        m.io_obs = m.io_out = nullptr;
        if (!AllocDev(reinterpret_cast<void**>(&m.io_obs),
                      static_cast<size_t>(batch) * m.obs_dim * sizeof(float)) ||
            !AllocDev(reinterpret_cast<void**>(&m.io_out),
                      static_cast<size_t>(batch) * m.act_dim * sizeof(float))) {
            return false;
        }
        m.io_cap = batch;
    }
    if (cudaMemcpyAsync(m.io_obs, obs,
                        static_cast<size_t>(batch) * m.obs_dim * sizeof(float),
                        cudaMemcpyHostToDevice, m.stream) != cudaSuccess) {
        return false;
    }
    if (!m.ForwardDevice(m.io_obs, m.io_out, batch, m.stream)) return false;
    if (cudaMemcpyAsync(out, m.io_out,
                        static_cast<size_t>(batch) * m.act_dim * sizeof(float),
                        cudaMemcpyDeviceToHost, m.stream) != cudaSuccess) {
        return false;
    }
    return cudaStreamSynchronize(m.stream) == cudaSuccess;
}

// ===========================================================================
// Go2GpuPolicy.
// ===========================================================================
struct Go2GpuPolicy::GImpl {
    GpuMlp mlp;
    Go2GpuContract c{};
    uint32_t links_per_env = 0, num_dogs = 0, links_per_dog = 0;
    uint32_t obs_dim = 0, act_dim = 0, scan_n = 0;
    // device config + persistent state.
    float* d_default = nullptr;
    uint32_t* d_slot = nullptr;
    float* d_force_limit = nullptr;
    float* d_terrain = nullptr;
    float* d_obs = nullptr;
    float* d_act = nullptr;
    float* d_last_action = nullptr;
    // terrain meta.
    uint32_t ter_nrow = 0, ter_ncol = 0;
    float ter_ox = 0, ter_oy = 0, ter_oz = 0, ter_base_z = 0, ter_scale_z = 0;
    float ter_cx = 0, ter_cy = 0;
    bool ready = false;

    ~GImpl() {
        FreeDev(d_default);
        FreeDev(d_slot);
        FreeDev(d_force_limit);
        FreeDev(d_terrain);
        FreeDev(d_obs);
        FreeDev(d_act);
        FreeDev(d_last_action);
    }
};

Go2GpuPolicy::Go2GpuPolicy() : g_(new GImpl()) {}
Go2GpuPolicy::~Go2GpuPolicy() { delete g_; }

bool Go2GpuPolicy::Init(const MlpPolicy& host, const Go2GpuContract& contract,
                        const ::nuka::terrain::HeightField& terrain,
                        uint32_t links_per_env, uint32_t num_dogs,
                        uint32_t links_per_dog) {
    GImpl& g = *g_;
    g.ready = false;
    if (!g.mlp.InitFromHost(host)) return false;
    g.c = contract;
    g.links_per_env = links_per_env;
    g.num_dogs = num_dogs > 0u ? num_dogs : 1u;
    g.links_per_dog = links_per_dog;
    g.obs_dim = g.mlp.ObsDim();
    g.act_dim = g.mlp.ActDim();
    g.scan_n = contract.scan_nx * contract.scan_ny;
    if (contract.proprio_dim + g.scan_n != g.obs_dim) {
        std::fprintf(stderr, "[gpu_policy] obs width %u != policy %u\n",
                     contract.proprio_dim + g.scan_n, g.obs_dim);
        return false;
    }

    const uint32_t nj = contract.njoints;
    if (!AllocDev(reinterpret_cast<void**>(&g.d_default), nj * sizeof(float)) ||
        !AllocDev(reinterpret_cast<void**>(&g.d_slot), nj * sizeof(uint32_t)) ||
        !AllocDev(reinterpret_cast<void**>(&g.d_force_limit), nj * sizeof(float)) ||
        !UploadDev(g.d_default, contract.default_angles, nj * sizeof(float)) ||
        !UploadDev(g.d_slot, contract.slot_of_urdf, nj * sizeof(uint32_t)) ||
        !UploadDev(g.d_force_limit, contract.force_limit, nj * sizeof(float))) {
        std::fprintf(stderr, "[gpu_policy] contract upload failed\n");
        return false;
    }

    const size_t tbytes = terrain.values.size() * sizeof(float);
    if (!AllocDev(reinterpret_cast<void**>(&g.d_terrain), tbytes) ||
        !UploadDev(g.d_terrain, terrain.values.data(), tbytes)) {
        std::fprintf(stderr, "[gpu_policy] terrain upload failed\n");
        return false;
    }
    g.ter_nrow = terrain.nrow;
    g.ter_ncol = terrain.ncol;
    g.ter_ox = terrain.origin.x;
    g.ter_oy = terrain.origin.y;
    g.ter_oz = terrain.origin.z;
    g.ter_base_z = terrain.base_z;
    g.ter_scale_z = terrain.scale_z;
    g.ter_cx = terrain.cell_x;
    g.ter_cy = terrain.cell_y;

    if (!AllocDev(reinterpret_cast<void**>(&g.d_obs),
                  static_cast<size_t>(g.num_dogs) * g.obs_dim * sizeof(float)) ||
        !AllocDev(reinterpret_cast<void**>(&g.d_act),
                  static_cast<size_t>(g.num_dogs) * g.act_dim * sizeof(float)) ||
        !AllocDev(reinterpret_cast<void**>(&g.d_last_action),
                  static_cast<size_t>(g.num_dogs) * nj * sizeof(float))) {
        std::fprintf(stderr, "[gpu_policy] state buffers alloc failed\n");
        return false;
    }
    if (cudaMemset(g.d_last_action, 0,
                   static_cast<size_t>(g.num_dogs) * nj * sizeof(float)) !=
        cudaSuccess) {
        return false;
    }
    g.ready = true;
    return true;
}

bool Go2GpuPolicy::Ready() const { return g_ != nullptr && g_->ready; }

void Go2GpuPolicy::SetCommand(float vx, float vy, float wyaw) {
    g_->c.command[0] = vx;
    g_->c.command[1] = vy;
    g_->c.command[2] = wyaw;
}

void Go2GpuPolicy::ApplyGains(void* backend, float* drive_stiffness,
                              float* drive_damping, float* drive_force_limit,
                              uint32_t env_index) {
    GImpl& g = *g_;
    if (!g.ready) return;
    cudaStream_t s = MainStreamOf(backend);
    const size_t base = static_cast<size_t>(env_index) * g.links_per_env;
    const uint32_t n = g.num_dogs * g.c.njoints;
    Go2GainsKernel<<<Grid(n), kBlock, 0, s>>>(
        drive_stiffness + base, drive_damping + base, drive_force_limit + base,
        g.d_slot, g.d_force_limit, g.c.kp, g.c.kd, g.num_dogs, g.c.njoints,
        g.links_per_dog);
}

void Go2GpuPolicy::Step(void* backend, const float* link_pose,
                        const float* link_velocity, const float* q,
                        const float* qdot, const float* base_pose,
                        float* drive_target, uint32_t env_index) {
    GImpl& g = *g_;
    if (!g.ready) return;
    cudaStream_t s = MainStreamOf(backend);

    Go2DevParams p{};
    const size_t lbase = static_cast<size_t>(env_index) * g.links_per_env;
    const size_t abase = static_cast<size_t>(env_index) * g.num_dogs;
    p.link_pose = link_pose + lbase * 7u;
    p.link_velocity = link_velocity + lbase * 6u;
    p.q = q + lbase;
    p.qdot = qdot + lbase;
    p.base_pose = base_pose + abase * 7u;
    p.drive_target = drive_target + lbase;
    p.action_in = g.d_act;
    p.obs = g.d_obs;
    p.last_action = g.d_last_action;
    p.default_angles = g.d_default;
    p.slot = g.d_slot;
    p.terrain_values = g.d_terrain;
    p.ter_nrow = g.ter_nrow;
    p.ter_ncol = g.ter_ncol;
    p.ter_ox = g.ter_ox;
    p.ter_oy = g.ter_oy;
    p.ter_oz = g.ter_oz;
    p.ter_base_z = g.ter_base_z;
    p.ter_scale_z = g.ter_scale_z;
    p.ter_cx = g.ter_cx;
    p.ter_cy = g.ter_cy;
    p.lin_vel_scale = g.c.lin_vel_scale;
    p.ang_vel_scale = g.c.ang_vel_scale;
    p.dof_pos_scale = g.c.dof_pos_scale;
    p.dof_vel_scale = g.c.dof_vel_scale;
    p.cmd_scale0 = g.c.cmd_scale[0];
    p.cmd_scale1 = g.c.cmd_scale[1];
    p.cmd_scale2 = g.c.cmd_scale[2];
    p.command0 = g.c.command[0];
    p.command1 = g.c.command[1];
    p.command2 = g.c.command[2];
    p.action_scale = g.c.action_scale;
    p.obs_clip = g.c.obs_clip;
    p.action_clip = g.c.action_clip;
    p.scan_x_min = g.c.scan_x_min;
    p.scan_y_min = g.c.scan_y_min;
    p.scan_res = g.c.scan_res;
    p.scan_scale = g.c.scan_scale;
    p.scan_clip = g.c.scan_clip;
    p.scan_z_offset = g.c.scan_z_offset;
    p.njoints = g.c.njoints;
    p.proprio_dim = g.c.proprio_dim;
    p.scan_nx = g.c.scan_nx;
    p.scan_ny = g.c.scan_ny;
    p.obs_dim = g.obs_dim;
    p.act_dim = g.act_dim;
    p.num_dogs = g.num_dogs;
    p.links_per_dog = g.links_per_dog;

    Go2ProprioKernel<<<Grid(g.num_dogs), kBlock, 0, s>>>(p);
    Go2HeightScanKernel<<<Grid(g.num_dogs * g.scan_n), kBlock, 0, s>>>(p);
    g.mlp.impl_->ForwardDevice(g.d_obs, g.d_act, g.num_dogs, s);
    Go2DriveWriteKernel<<<Grid(g.num_dogs * g.c.njoints), kBlock, 0, s>>>(p);
}

}  // namespace nuka::runtime::inference
