// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — Backend / Device / RegistryEntry implementations.
//
// ggml analogy: this is the CUDA `ggml_backend` + `ggml_backend_device` +
// `ggml_backend_reg`. It owns:
//   * dispatch       — look up g_ops[op]; null slot => Status::Unsupported;
//                      else call the op fn on the backend's MAIN stream.
//   * synchronize    — sync the main stream.
//   * plan_create    — stream-capture the OpCall list on a dedicated CAPTURE
//                      stream into a cudaGraph, instantiate a cudaGraphExec,
//                      cache it in the Plan. (D1 replay vehicle.)
//   * plan_execute   — cudaGraphLaunch the cached exec on the MAIN stream.
//   * plan_free      — destroy the exec + graph.
//   * event_*        — cudaEvent new / record / wait / free.
//
// The backend owns its OWN streams (main + capture) — it deliberately does NOT
// pull in the legacy phi/owned_stream.hpp (that belongs to nuka_phi).
// ---------------------------------------------------------------------------

#include "phi/backend_cuda/cuda_internal.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/registry.cuh"

#include <cstdio>
#include <cstdlib>

namespace nuka::phi {

// NUKA_STEP_TIMING: env-gated per-op GPU-time breakdown, off by default.
// Brackets each dispatched op with CUDA events; a warmup gate skips the transient.
namespace {
const char* NkOpName(int op) {
    switch (op) {
        case 0: return "ApplyDrives"; case 1: return "AbaForward";
        case 2: return "IntegrateVelocity"; case 3: return "FkWorldPoses";
        case 4: return "IntegratePosition"; case 5: return "CrbaComputeM";
        case 6: return "CrbaFactorM"; case 7: return "ApplyImplicitDamping";
        case 8: return "BuildAabbs"; case 9: return "LbvhBuild";
        case 10: return "LbvhQueryPairs"; case 11: return "ParticleGridBuild";
        case 12: return "NarrowphasePrimitives"; case 13: return "NarrowphaseSdf";
        case 14: return "ContactTangentBasis"; case 15: return "AssembleRows";
        case 16: return "SolveRowsBlockIsland";
        case 17: return "ParticleAeroDrag"; case 18: return "ParticlePredict";
        case 19: return "XpbdProject"; case 20: return "PbfDensityLambda";
        case 21: return "PbfApplyDelta"; case 22: return "ParticleFinalize";
        case 23: return "MpmStep"; case 24: return "ReadoutContactWrench";
        case 25: return "ExportObs"; case 26: return "ResetEnvs";
        case 27: return "SnapshotState"; case 28: return "RestoreState";
        case 29: return "RandomizeMaterialBuckets"; case 30: return "RandomizeBodyParams";
        case 31: return "StepBackward"; case 32: return "ParticleParticleContact";
        case 33: return "SyncLinkBodyPose"; case 34: return "NarrowphaseBodyParticle";
        case 35: return "NarrowphaseHeightfield"; case 36: return "BuildSolveIslands";
        case 37: return "ContactWarmStart";
        default: return "op";
    }
}
struct OpProfiler {
    static constexpr int kN = 64;
    double ms[kN] = {0.0};
    unsigned long long calls[kN] = {0};
    cudaEvent_t a = nullptr, b = nullptr;
    bool seen[kN] = {false};
    long step = 0, warmup = 200;
    bool on = false, init = false;
    static OpProfiler& Get() { static OpProfiler p; return p; }
    void Ensure() {
        if (init) return;
        init = true;
        const char* e = std::getenv("NUKA_STEP_TIMING");
        on = (e != nullptr && e[0] != '0');
        const char* w = std::getenv("NUKA_STEP_TIMING_WARMUP");
        if (w != nullptr) warmup = std::atol(w);
        if (on) {
            cudaEventCreate(&a);
            cudaEventCreate(&b);
            std::atexit(&OpProfiler::Dump);
        }
    }
    static void Dump() {
        OpProfiler& p = Get();
        double tot = 0.0;
        unsigned long long steps = 0;
        for (int i = 0; i < kN; ++i) { tot += p.ms[i]; if (p.calls[i] > steps) steps = p.calls[i]; }
        if (steps == 0) return;
        std::printf("\n[NUKA_STEP_TIMING] steady-state per-op GPU ms (steps=%llu, warmup=%ld)\n",
                    steps, p.warmup);
        std::printf("  %-24s %10s %12s %8s\n", "op", "ms/step", "total_ms", "%%");
        for (int i = 0; i < kN; ++i) {
            if (p.calls[i] == 0) continue;
            const double per = p.ms[i] / static_cast<double>(p.calls[i]);
            std::printf("  %-24s %10.3f %12.2f %7.1f%%\n", NkOpName(i), per, p.ms[i],
                        tot > 0.0 ? 100.0 * p.ms[i] / tot : 0.0);
        }
        std::printf("  %-24s %10.3f %12.2f\n", "TOTAL/step",
                    tot / static_cast<double>(steps), tot);
        std::fflush(stdout);
    }
};
}  // namespace

// ---------------------------------------------------------------------------
// Concrete Plan / Event.
// ---------------------------------------------------------------------------
struct CudaPlan {
    cudaGraph_t     graph;
    cudaGraphExec_t exec;
};

struct CudaEvent {
    cudaEvent_t event;
};

namespace {

CudaBackend* AsBackend(Backend* b) { return reinterpret_cast<CudaBackend*>(b); }
CudaPlan*    AsPlan(Plan* p)       { return reinterpret_cast<CudaPlan*>(p); }
CudaEvent*   AsEvent(Event* e)     { return reinterpret_cast<CudaEvent*>(e); }

// --- Backend vtable ------------------------------------------------------

const char* BackendGetName(Backend*) { return "cuda"; }

void BackendFreeImpl(Backend* b) {
    CudaBackend* cb = AsBackend(b);
    if (cb == nullptr) {
        return;
    }
    if (cb->capture != nullptr) { (void)cudaStreamDestroy(cb->capture); }
    if (cb->main != nullptr)    { (void)cudaStreamDestroy(cb->main); }
    delete cb;
}

// Dispatch a single op onto an explicit stream (shared by dispatch + capture).
Status DispatchOn(const ModelView& model, const DataView& data,
                  const OpCall& call, cudaStream_t stream) {
    OpFn fn = GetCudaOp(call.op);
    if (fn == nullptr) {
        return Status::Unsupported;
    }
    return fn(model, data, call.params, stream);
}

Status BackendDispatchImpl(Backend* b, const ModelView& model,
                           const DataView& data, const OpCall& call) {
    CudaBackend* cb = AsBackend(b);
    // Pin the active device to the backend's device (the ported thrust grid /
    // LBVH sorts run on the default device via thrust::cuda::par.on(stream) —
    // they REQUIRE the current device to match the stream's device, which a
    // caller thread may not have set).
    (void)cudaSetDevice(cb->device_id);
    OpProfiler& prof = OpProfiler::Get();
    prof.Ensure();
    if (!prof.on) {
        return DispatchOn(model, data, call, cb->main);
    }
    const int oid = static_cast<int>(call.op);
    // Step boundary = an op id repeats since the last boundary (robust to the
    // one-off construction/seed dispatches that precede the step loop).
    if (oid >= 0 && oid < OpProfiler::kN) {
        if (prof.seen[oid]) {
            ++prof.step;
            for (int i = 0; i < OpProfiler::kN; ++i) prof.seen[i] = false;
        }
        prof.seen[oid] = true;
    }
    const bool acc = (prof.step > prof.warmup) && oid >= 0 && oid < OpProfiler::kN;
    if (acc) cudaEventRecord(prof.a, cb->main);
    const Status s = DispatchOn(model, data, call, cb->main);
    if (acc) {
        cudaEventRecord(prof.b, cb->main);
        cudaEventSynchronize(prof.b);
        float el = 0.0f;
        cudaEventElapsedTime(&el, prof.a, prof.b);
        prof.ms[oid] += static_cast<double>(el);
        prof.calls[oid] += 1;
    }
    return s;
}

void BackendSynchronizeImpl(Backend* b) {
    CudaBackend* cb = AsBackend(b);
    (void)cudaStreamSynchronize(cb->main);
}

Plan* BackendPlanCreateImpl(Backend* b, const ModelView& model,
                            const DataView& data, const OpCall* calls, int n_calls) {
    CudaBackend* cb = AsBackend(b);

    // Capture the op sequence on the dedicated capture stream into a graph.
    if (cudaStreamBeginCapture(cb->capture, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
        (void)cudaGetLastError();
        return nullptr;
    }
    bool ok = true;
    // Defensive guard: should an op ever allocate on the stream mid-capture it
    // throws std::bad_alloc (cudaErrorStreamCaptureUnsupported); catch it so
    // plan_create returns a clean nullptr (the caller falls back to Step) instead
    // of unwinding through the C ABI vtable. The capture MUST still be ended (a
    // left-open capture poisons the stream for the next StepPlanned). Every op is
    // capture-safe today (the particle-grid sort/scan draw from pre-allocated
    // scratch), so this path is not taken on the supported scenes.
    try {
        for (int i = 0; i < n_calls; ++i) {
            if (DispatchOn(model, data, calls[i], cb->capture) != Status::Ok) {
                ok = false;
                break;
            }
        }
    } catch (...) {
        ok = false;
    }

    cudaGraph_t graph = nullptr;
    cudaError_t end_err = cudaStreamEndCapture(cb->capture, &graph);
    if (!ok || end_err != cudaSuccess || graph == nullptr) {
        (void)cudaGetLastError();
        if (graph != nullptr) { (void)cudaGraphDestroy(graph); }
        return nullptr;
    }

    cudaGraphExec_t exec = nullptr;
    if (cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) != cudaSuccess) {
        (void)cudaGetLastError();
        (void)cudaGraphDestroy(graph);
        return nullptr;
    }

    CudaPlan* plan = new CudaPlan{};
    plan->graph = graph;
    plan->exec  = exec;
    return reinterpret_cast<Plan*>(plan);
}

Status BackendPlanExecuteImpl(Backend* b, Plan* p) {
    CudaBackend* cb = AsBackend(b);
    CudaPlan* plan = AsPlan(p);
    if (plan == nullptr || plan->exec == nullptr) {
        return Status::Failed;
    }
    if (cudaGraphLaunch(plan->exec, cb->main) != cudaSuccess) {
        (void)cudaGetLastError();
        return Status::Failed;
    }
    return Status::Ok;
}

void BackendPlanFreeImpl(Backend*, Plan* p) {
    CudaPlan* plan = AsPlan(p);
    if (plan == nullptr) {
        return;
    }
    if (plan->exec != nullptr)  { (void)cudaGraphExecDestroy(plan->exec); }
    if (plan->graph != nullptr) { (void)cudaGraphDestroy(plan->graph); }
    delete plan;
}

Event* BackendEventNewImpl(Backend*) {
    CudaEvent* e = new CudaEvent{};
    if (cudaEventCreate(&e->event) != cudaSuccess) {
        (void)cudaGetLastError();
        delete e;
        return nullptr;
    }
    return reinterpret_cast<Event*>(e);
}

void BackendEventRecordImpl(Backend* b, Event* e) {
    CudaBackend* cb = AsBackend(b);
    CudaEvent* ce = AsEvent(e);
    if (ce != nullptr) { (void)cudaEventRecord(ce->event, cb->main); }
}

void BackendEventWaitImpl(Backend* b, Event* e) {
    CudaBackend* cb = AsBackend(b);
    CudaEvent* ce = AsEvent(e);
    // Make the main stream wait on the recorded event.
    if (ce != nullptr) { (void)cudaStreamWaitEvent(cb->main, ce->event, 0); }
}

void BackendEventFreeImpl(Backend*, Event* e) {
    CudaEvent* ce = AsEvent(e);
    if (ce == nullptr) {
        return;
    }
    (void)cudaEventDestroy(ce->event);
    delete ce;
}

const BackendI kCudaBackendI = {
    /*get_name     =*/ &BackendGetName,
    /*free         =*/ &BackendFreeImpl,
    /*dispatch     =*/ &BackendDispatchImpl,
    /*synchronize  =*/ &BackendSynchronizeImpl,
    /*plan_create  =*/ &BackendPlanCreateImpl,
    /*plan_execute =*/ &BackendPlanExecuteImpl,
    /*plan_free    =*/ &BackendPlanFreeImpl,
    /*event_new    =*/ &BackendEventNewImpl,
    /*event_record =*/ &BackendEventRecordImpl,
    /*event_wait   =*/ &BackendEventWaitImpl,
    /*event_free   =*/ &BackendEventFreeImpl,
};

// ---------------------------------------------------------------------------
// Device.
// ---------------------------------------------------------------------------
struct CudaDevice {
    const DeviceI* iface;   // MUST be first
    int            device_id;
    char           name[256];
};

CudaDevice* AsDevice(Device* d) { return reinterpret_cast<CudaDevice*>(d); }

const char* DeviceGetNameImpl(Device* d) { return AsDevice(d)->name; }

void DeviceGetMemoryImpl(Device* d, size_t* free_b, size_t* total_b) {
    int prev = 0;
    (void)cudaGetDevice(&prev);
    (void)cudaSetDevice(AsDevice(d)->device_id);
    size_t f = 0, t = 0;
    (void)cudaMemGetInfo(&f, &t);
    (void)cudaSetDevice(prev);
    if (free_b  != nullptr) { *free_b  = f; }
    if (total_b != nullptr) { *total_b = t; }
}

Backend* DeviceInitBackendImpl(Device* d, const char* /*params_json*/) {
    CudaDevice* cd = AsDevice(d);
    (void)cudaSetDevice(cd->device_id);

    CudaBackend* cb = new CudaBackend{};
    cb->iface     = &kCudaBackendI;
    cb->device_id = cd->device_id;
    if (cudaStreamCreate(&cb->main) != cudaSuccess ||
        cudaStreamCreate(&cb->capture) != cudaSuccess) {
        (void)cudaGetLastError();
        if (cb->capture != nullptr) { (void)cudaStreamDestroy(cb->capture); }
        if (cb->main != nullptr)    { (void)cudaStreamDestroy(cb->main); }
        delete cb;
        return nullptr;
    }

    cb->device_bt.iface   = &kCudaDeviceBufferTypeI;
    cb->device_bt.backend = cb;
    cb->device_bt.is_host = false;

    cb->host_bt.iface   = &kCudaHostBufferTypeI;
    cb->host_bt.backend = cb;
    cb->host_bt.is_host = true;

    return reinterpret_cast<Backend*>(cb);
}

BufferType* DeviceGetBufferTypeImpl(Device* /*d*/) {
    // The buffer type is bound to a Backend (for its stream), not a Device. A
    // caller wanting a stream-bound type allocates after init_backend and uses
    // the backend's types (DeviceBufferType on the BACKEND in M3+). For the M1
    // surface we return a stream-less default-stream device type so the
    // BufferType handle is non-null and usable for allocation tests.
    static CudaBufferType s_default_device_bt = {
        &kCudaDeviceBufferTypeI, /*backend=*/nullptr, /*is_host=*/false};
    return reinterpret_cast<BufferType*>(&s_default_device_bt);
}

BufferType* DeviceGetHostBufferTypeImpl(Device* /*d*/) {
    static CudaBufferType s_default_host_bt = {
        &kCudaHostBufferTypeI, /*backend=*/nullptr, /*is_host=*/true};
    return reinterpret_cast<BufferType*>(&s_default_host_bt);
}

bool DeviceSupportsOpImpl(Device* /*d*/, NkOp op) {
    return GetCudaOp(op) != nullptr;
}

const DeviceI kCudaDeviceI = {
    /*get_name             =*/ &DeviceGetNameImpl,
    /*get_memory           =*/ &DeviceGetMemoryImpl,
    /*init_backend         =*/ &DeviceInitBackendImpl,
    /*get_buffer_type      =*/ &DeviceGetBufferTypeImpl,
    /*get_host_buffer_type =*/ &DeviceGetHostBufferTypeImpl,
    /*supports_op          =*/ &DeviceSupportsOpImpl,
};

// ---------------------------------------------------------------------------
// RegistryEntry. Devices are enumerated lazily on first device_count/get_device.
// ---------------------------------------------------------------------------
constexpr int kMaxCudaDevices = 16;

struct CudaRegistryEntry {
    const RegistryEntryI* iface;   // MUST be first
    bool                  enumerated;
    int                   device_count;
    CudaDevice            devices[kMaxCudaDevices];
};

CudaRegistryEntry* AsRegistry(RegistryEntry* r) {
    return reinterpret_cast<CudaRegistryEntry*>(r);
}

void EnsureEnumerated(CudaRegistryEntry* reg) {
    if (reg->enumerated) {
        return;
    }
    reg->enumerated = true;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) {
        (void)cudaGetLastError();
        n = 0;
    }
    if (n > kMaxCudaDevices) { n = kMaxCudaDevices; }
    reg->device_count = n;
    for (int i = 0; i < n; ++i) {
        CudaDevice& dev = reg->devices[i];
        dev.iface     = &kCudaDeviceI;
        dev.device_id = i;
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            // copy name, NUL-terminated
            size_t k = 0;
            for (; k < sizeof(dev.name) - 1 && prop.name[k] != '\0'; ++k) {
                dev.name[k] = prop.name[k];
            }
            dev.name[k] = '\0';
        } else {
            (void)cudaGetLastError();
            dev.name[0] = '\0';
        }
    }
}

const char* RegistryGetNameImpl(RegistryEntry*) { return "cuda"; }

size_t RegistryDeviceCountImpl(RegistryEntry* r) {
    CudaRegistryEntry* reg = AsRegistry(r);
    EnsureEnumerated(reg);
    return static_cast<size_t>(reg->device_count);
}

Device* RegistryGetDeviceImpl(RegistryEntry* r, size_t i) {
    CudaRegistryEntry* reg = AsRegistry(r);
    EnsureEnumerated(reg);
    if (i >= static_cast<size_t>(reg->device_count)) {
        return nullptr;
    }
    return reinterpret_cast<Device*>(&reg->devices[i]);
}

void* RegistryGetProcAddressImpl(RegistryEntry*, const char* /*name*/) {
    return nullptr;  // no extensions in M1
}

const RegistryEntryI kCudaRegistryEntryI = {
    /*get_name         =*/ &RegistryGetNameImpl,
    /*device_count     =*/ &RegistryDeviceCountImpl,
    /*get_device       =*/ &RegistryGetDeviceImpl,
    /*get_proc_address =*/ &RegistryGetProcAddressImpl,
};

// The singleton CUDA registry entry + a registered-once guard.
CudaRegistryEntry g_cuda_registry_entry = {&kCudaRegistryEntryI, false, 0, {}};
bool g_cuda_registered = false;

} // namespace

// ---------------------------------------------------------------------------
// Stream accessor used by cuda_buffer.cu.
// ---------------------------------------------------------------------------
cudaStream_t CudaBackendMainStream(CudaBackend* b) {
    return (b != nullptr) ? b->main : static_cast<cudaStream_t>(nullptr);
}

// ---------------------------------------------------------------------------
// Backend-bound buffer-type accessors (declared in phi/backend.hpp). These hand
// out the backend's own device_bt / host_bt, whose vtables resolve the async
// stream to backend->main (cuda_buffer.cu). The buffer-sweep consumers that run
// their ops on backend->main allocate from these (vs DeviceBufferType(device),
// the stream-0 default type). CUDA-free at the call site (BufferType* is opaque).
// ---------------------------------------------------------------------------
BufferType* BackendDeviceBufferType(Backend* b) {
    if (b == nullptr) { return nullptr; }
    return reinterpret_cast<BufferType*>(&reinterpret_cast<CudaBackend*>(b)->device_bt);
}

BufferType* BackendHostBufferType(Backend* b) {
    if (b == nullptr) { return nullptr; }
    return reinterpret_cast<BufferType*>(&reinterpret_cast<CudaBackend*>(b)->host_bt);
}

// ---------------------------------------------------------------------------
// Explicit registration entry point (the static-lib self-registration de-risk).
// Called from registry.cpp::InitBestDevice() under NUKA_PHI2_WITH_CUDA so the
// linker never drops this TU. Idempotent.
// ---------------------------------------------------------------------------
void RegisterCudaBackendEntry() {
    if (g_cuda_registered) {
        return;
    }
    g_cuda_registered = true;
    RegisterBackend(reinterpret_cast<RegistryEntry*>(&g_cuda_registry_entry));
    // M3b: explicitly register the articulation pipeline op implementations
    // (the same static-lib linker de-risk as this entry itself — explicit
    // calls, never static initializers).
    RegisterNkArticulationPipelineOps();
}

} // namespace nuka::phi
