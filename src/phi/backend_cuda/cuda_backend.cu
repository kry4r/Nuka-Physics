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
#include <new>
#include <exception>

namespace nuka::phi {

// NUKA_STEP_TIMING: env-gated per-op GPU-time breakdown, off by default.
// Brackets each dispatched op with CUDA events; a warmup gate skips the transient.
namespace {
const char* NkOpName(int op) {
    switch (static_cast<NkOp>(op)) {
        case NkOp::ApplyDrives: return "ApplyDrives";
        case NkOp::ApplyDynamicsDrives: return "ApplyDynamicsDrives";
        case NkOp::ReadoutDrives: return "ReadoutDrives";
        case NkOp::RefitParticleSurfaces: return "RefitParticleSurfaces";
        case NkOp::AbaForward: return "AbaForward";
        case NkOp::IntegrateVelocity: return "IntegrateVelocity";
        case NkOp::FkWorldPoses: return "FkWorldPoses";
        case NkOp::IntegratePosition: return "IntegratePosition";
        case NkOp::CrbaComputeM: return "CrbaComputeM";
        case NkOp::CrbaFactorM: return "CrbaFactorM";
        case NkOp::ApplyImplicitDamping: return "ApplyImplicitDamping";
        case NkOp::BuildAabbs: return "BuildAabbs";
        case NkOp::LbvhBuild: return "LbvhBuild";
        case NkOp::LbvhQueryPairs: return "LbvhQueryPairs";
        case NkOp::ParticleGridBuild: return "ParticleGridBuild";
        case NkOp::NarrowphasePrimitives: return "NarrowphasePrimitives";
        case NkOp::NarrowphaseSdf: return "NarrowphaseSdf";
        case NkOp::ContactTangentBasis: return "ContactTangentBasis";
        case NkOp::AssembleRows: return "AssembleRows";
        case NkOp::SolveRowsBlockIsland: return "SolveRowsBlockIsland";
        case NkOp::ParticleAeroDrag: return "ParticleAeroDrag";
        case NkOp::ParticlePredict: return "ParticlePredict";
        case NkOp::XpbdProject: return "XpbdProject";
        case NkOp::PbfDensityLambda: return "PbfDensityLambda";
        case NkOp::PbfApplyDelta: return "PbfApplyDelta";
        case NkOp::ParticleFinalize: return "ParticleFinalize";
        case NkOp::MpmPredict: return "MpmPredict";
        case NkOp::MpmExchange: return "MpmExchange";
        case NkOp::MpmCommit: return "MpmCommit";
        case NkOp::ReadoutContactWrench: return "ReadoutContactWrench";
        case NkOp::ExportObs: return "ExportObs";
        case NkOp::SampleObservation: return "SampleObservation";
        case NkOp::ResetObservation: return "ResetObservation";
        case NkOp::ResetEnvs: return "ResetEnvs";
        case NkOp::SnapshotState: return "SnapshotState";
        case NkOp::RestoreState: return "RestoreState";
        case NkOp::RandomizeMaterialBuckets: return "RandomizeMaterialBuckets";
        case NkOp::RandomizeBodyParams: return "RandomizeBodyParams";
        case NkOp::StepBackward: return "StepBackward";
        case NkOp::ParticleParticleContact: return "ParticleParticleContact";
        case NkOp::SyncLinkBodyPose: return "SyncLinkBodyPose";
        case NkOp::NarrowphaseBodyParticle: return "NarrowphaseBodyParticle";
        case NkOp::NarrowphaseHeightfield: return "NarrowphaseHeightfield";
        case NkOp::BuildSolveIslands: return "BuildSolveIslands";
        case NkOp::ContactWarmStart: return "ContactWarmStart";
        case NkOp::SnapshotStepVelocity: return "SnapshotStepVelocity";
        case NkOp::ParticleProjectionVelocity: return "ParticleProjectionVelocity";
        case NkOp::ParticleContactDelta: return "ParticleContactDelta";
        case NkOp::AccumulateStep: return "AccumulateStep";
        case NkOp::FkLinkVelocities: return "FkLinkVelocities";
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
        unsigned long long calls = 0;
        for (int i = 0; i < kN; ++i) { tot += p.ms[i]; calls += p.calls[i]; }
        if (calls == 0) return;
        std::printf("\n[NUKA_STEP_TIMING] per-op GPU ms (calls=%llu, warmup_groups=%ld)\n",
                    calls, p.warmup);
        std::printf("  %-24s %10s %12s %8s %12s\n", "op", "ms/call", "total_ms", "%%", "calls");
        for (int i = 0; i < kN; ++i) {
            if (p.calls[i] == 0) continue;
            const double per = p.ms[i] / static_cast<double>(p.calls[i]);
            std::printf("  %-24s %10.3f %12.2f %7.1f%% %12llu\n", NkOpName(i), per, p.ms[i],
                        tot > 0.0 ? 100.0 * p.ms[i] / tot : 0.0, p.calls[i]);
        }
        std::printf("  %-24s %10s %12.2f\n", "TOTAL", "", tot);
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
    const auto selected = cudaSetDevice(cb->device_id);
    if (selected != cudaSuccess) return CudaStatus(selected);
    OpProfiler& prof = OpProfiler::Get();
    prof.Ensure();
    if (!prof.on) {
        return DispatchOn(model, data, call, cb->main);
    }
    const int oid = static_cast<int>(call.op);
    // Repeated op ids define warmup groups, not physical steps or solver intervals.
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

Status SetExecutionError(ExecutionError* error, Status status, NkOp op,
                         cudaError_t native, const char* message) {
    if (error) {
        *error = {};
        error->status = status;
        error->failed_op = op;
        error->native_code = static_cast<int32_t>(native);
        std::snprintf(error->message, sizeof(error->message), "%s", message ? message : "");
    }
    return status;
}

Status BackendSynchronizeImpl(Backend* b, ExecutionError* error) {
    CudaBackend* cb = AsBackend(b);
    auto status = cudaSetDevice(cb->device_id);
    if (status == cudaSuccess) status = cudaStreamSynchronize(cb->main);
    return SetExecutionError(error, CudaStatus(status), NkOp::Count, status,
                             status == cudaSuccess ? "" : cudaGetErrorString(status));
}

Plan* BackendPlanCreateImpl(Backend* b, const ModelView& model,
                            const DataView& data, const OpCall* calls, int n_calls,
                            ExecutionError* error) {
    CudaBackend* cb = AsBackend(b);
    ExecutionError failure{};
    if (error) *error = {};
    if (n_calls < 0 || (n_calls != 0 && calls == nullptr)) {
        SetExecutionError(error, Status::InvalidArgument, NkOp::Count, cudaErrorInvalidValue,
                          "invalid graph operator sequence");
        return nullptr;
    }
    auto native = cudaSetDevice(cb->device_id);
    if (native == cudaSuccess)
        native = cudaStreamBeginCapture(cb->capture, cudaStreamCaptureModeThreadLocal);
    if (native != cudaSuccess) {
        SetExecutionError(error, CudaStatus(native), NkOp::Count, native, cudaGetErrorString(native));
        return nullptr;
    }
    NkOp current = NkOp::Count;
    try {
        for (int i = 0; i < n_calls; ++i) {
            current = calls[i].op;
            const auto status = DispatchOn(model, data, calls[i], cb->capture);
            if (status != Status::Ok) {
                SetExecutionError(&failure, status, current, cudaSuccess,
                                  "operator failed during graph capture");
                break;
            }
        }
    } catch (const std::bad_alloc& exception) {
        SetExecutionError(&failure, Status::OutOfMemory, current, cudaSuccess, exception.what());
    } catch (const std::exception& exception) {
        SetExecutionError(&failure, Status::Failed, current, cudaSuccess, exception.what());
    } catch (...) {
        SetExecutionError(&failure, Status::Failed, current, cudaSuccess,
                          "unknown exception during graph capture");
    }
    cudaGraph_t graph = nullptr;
    native = cudaStreamEndCapture(cb->capture, &graph);
    if (failure.status != Status::Ok || native != cudaSuccess || graph == nullptr) {
        if (failure.status == Status::Ok)
            SetExecutionError(&failure, native == cudaSuccess ? Status::Failed : CudaStatus(native),
                              current, native, native == cudaSuccess ? "empty captured graph" : cudaGetErrorString(native));
        if (error) *error = failure;
        if (graph) cudaGraphDestroy(graph);
        cudaGetLastError();
        return nullptr;
    }
    cudaGraphExec_t exec = nullptr;
    native = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    if (native != cudaSuccess) {
        SetExecutionError(error, CudaStatus(native), NkOp::Count, native, cudaGetErrorString(native));
        cudaGraphDestroy(graph);
        return nullptr;
    }
    CudaPlan* plan = new (std::nothrow) CudaPlan{graph, exec};
    if (!plan) {
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
        SetExecutionError(error, Status::OutOfMemory, NkOp::Count, cudaSuccess,
                          "graph owner allocation failed");
    }
    return reinterpret_cast<Plan*>(plan);
}

Status BackendPlanExecuteImpl(Backend* b, Plan* p, ExecutionError* error) {
    CudaBackend* cb = AsBackend(b);
    CudaPlan* plan = AsPlan(p);
    if (plan == nullptr || plan->exec == nullptr)
        return SetExecutionError(error, Status::InvalidArgument, NkOp::Count,
                                 cudaErrorInvalidValue, "graph is not initialized");
    auto status = cudaSetDevice(cb->device_id);
    if (status == cudaSuccess) status = cudaGraphLaunch(plan->exec, cb->main);
    return SetExecutionError(error, CudaStatus(status), NkOp::Count, status,
                             status == cudaSuccess ? "" : cudaGetErrorString(status));
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
    CudaBufferType device_bt;
    CudaBufferType host_bt;
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
    if (cudaSetDevice(cd->device_id) != cudaSuccess) return nullptr;

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
    cb->device_bt.device_id = cd->device_id;

    cb->host_bt.iface   = &kCudaHostBufferTypeI;
    cb->host_bt.backend = cb;
    cb->host_bt.is_host = true;
    cb->host_bt.device_id = cd->device_id;

    return reinterpret_cast<Backend*>(cb);
}

BufferType* DeviceGetBufferTypeImpl(Device* device) {
    return reinterpret_cast<BufferType*>(&AsDevice(device)->device_bt);
}

BufferType* DeviceGetHostBufferTypeImpl(Device* device) {
    return reinterpret_cast<BufferType*>(&AsDevice(device)->host_bt);
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
        dev.device_bt = {&kCudaDeviceBufferTypeI, nullptr, false, i};
        dev.host_bt = {&kCudaHostBufferTypeI, nullptr, true, i};
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
