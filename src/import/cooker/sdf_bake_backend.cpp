#include "import/cooker/sdf_bake_backend.hpp"

namespace nuka::import::cooker {

const SdfBakeBackend& DefaultSdfBakeBackend() {
  // v0.7: the deterministic host-CPU backend. The v1.0 GPU backend slots in
  // here (e.g. switch on a build flag / cook config) without touching any
  // caller — see sdf_bake_backend.hpp.
  static const CpuSdfBakeBackend kCpuBackend;
  return kCpuBackend;
}

} // namespace nuka::import::cooker
