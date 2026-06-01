// ---------------------------------------------------------------------------
// n2_domain_randomization.cpp -- per-episode domain randomization sampling
// (v0.5 p04 Task 5.4.7 sim-to-real N2)
// ---------------------------------------------------------------------------
//
// The sampling math is header-inline (n2_domain_randomization.hpp -- the Philox
// helpers are __host__ __device__ __forceinline__). This translation unit gives
// the nuka_sensor_noise static lib a concrete object file for the N2 surface and
// hosts the non-inline convenience entry SampleEpisodeRandomizationOutOfLine,
// used to pin the symbol for separate-TU linkage / debugging.
//
// The buffer-touching half -- ApplyPerEpisodeRandomization(WorldRecord&, cfg) --
// is implemented in c_abi/noise.cpp, where the WorldRecord + runtime
// (MakeSpatialInertia, the live device buffers) are already linked; keeping it
// out of THIS lib avoids dragging all of runtime into a "sensor" lib.
// ---------------------------------------------------------------------------

#include "sensor/noise/n2_domain_randomization.hpp"

namespace nuka::sensor::noise {

// Non-inline mirror of the header SampleEpisodeRandomization. Provides a stable
// out-of-line symbol (the inline version may be fully elided at -O2); callers in
// other translation units may use either.
SampledRandomization SampleEpisodeRandomizationOutOfLine(
    const DomainRandomizationConfig& cfg, uint32_t env_idx) {
    return SampleEpisodeRandomization(cfg, env_idx);
}

}  // namespace nuka::sensor::noise
