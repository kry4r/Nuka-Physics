#pragma once
// ---------------------------------------------------------------------------
// nuka::phi::PlatformContract -- physics backend selection contract
// ---------------------------------------------------------------------------

namespace nuka::phi {

enum class PhysicsBackend {
    Cuda,
    CpuReference
};

enum class BackendSelectionPolicy {
    PreferCuda,
    ForceCuda,
    ForceCpuReference
};

struct PlatformContract {
    PhysicsBackend physics_backend = PhysicsBackend::Cuda;
    PhysicsBackend reference_backend = PhysicsBackend::CpuReference;
    BackendSelectionPolicy default_backend_policy = BackendSelectionPolicy::PreferCuda;
    bool requires_gpu_simulation = true;
    bool cpu_reference_only = true;
    bool backend_selection_layer_enabled = true;
    int cuda_device_id = 0;
};

struct BackendSelectionRequest {
    BackendSelectionPolicy policy = BackendSelectionPolicy::PreferCuda;
    int cuda_device_id = 0;
};

struct BackendSelectionResult {
    PhysicsBackend selected_backend = PhysicsBackend::Cuda;
    bool production_backend = true;
    bool reference_backend = false;
    int cuda_device_id = 0;
};

BackendSelectionResult ResolvePhysicsBackend(const BackendSelectionRequest& request);
BackendSelectionResult ResolvePhysicsBackend(const BackendSelectionRequest& request,
                                             const PlatformContract& contract);

} // namespace nuka::phi
