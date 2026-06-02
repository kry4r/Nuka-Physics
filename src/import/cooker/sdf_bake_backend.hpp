#pragma once
// ---------------------------------------------------------------------------
// nuka::import::cooker – SDF bake backend SEAM (v0.7 p07)
// ---------------------------------------------------------------------------
// The swappable backend that turns one triangle mesh into a narrow-band SDF.
// v0.7 ships ONLY CpuSdfBakeBackend (deterministic host CPU — the production +
// reference path). v1.0 will add a DETERMINISTIC GPU bake backend behind THIS
// same interface, WITHOUT changing the cooked SparseSdfData format or the
// runtime p08 query (src/runtime/sdf/sparse_sdf_query.cuh). This mirrors the
// v0.5 SparseLinearSolver backend seam: a stable factory + interface that a
// later phase extends. The CPU backend stays the reference oracle that
// validates the future GPU backend (cf. Eigen-oracle vs self-written CG, SAP vs
// LBVH). Owner decision 2026-06-02: "现在CPU吧，留出接口，1.0做GPU".
// ---------------------------------------------------------------------------

#include "import/cooker/sparse_sdf_cooker.hpp"
#include "import/cooker/sparse_sdf_storage.hpp"

#include <cstdint>
#include <string>

namespace nuka::import::cooker {

// Bake one mesh's narrow-band SDF + its content-hash dedup key. Every backend
// MUST be byte-reproducible run-to-run (the cooked SDF is a golden) and emit the
// SAME SparseSdfData layout, so downstream (p08) is backend-agnostic.
class SdfBakeBackend {
 public:
  virtual ~SdfBakeBackend() = default;

  virtual SparseSdfData Bake(const float* vertices, uint32_t vertex_count,
                             const uint32_t* indices, uint32_t triangle_count,
                             const SparseSdfParams& params) const = 0;

  virtual std::string CacheKey(const float* vertices, uint32_t vertex_count,
                               const uint32_t* indices, uint32_t triangle_count,
                               const SparseSdfParams& params) const = 0;
};

// v0.7 production + reference backend: deterministic host-CPU bake. Delegates to
// the pure CookSparseSdf / ComputeSdfCacheKey functions.
class CpuSdfBakeBackend final : public SdfBakeBackend {
 public:
  SparseSdfData Bake(const float* vertices, uint32_t vertex_count,
                     const uint32_t* indices, uint32_t triangle_count,
                     const SparseSdfParams& params) const override {
    return CookSparseSdf(vertices, vertex_count, indices, triangle_count, params);
  }

  std::string CacheKey(const float* vertices, uint32_t vertex_count,
                       const uint32_t* indices, uint32_t triangle_count,
                       const SparseSdfParams& params) const override {
    return ComputeSdfCacheKey(vertices, vertex_count, indices, triangle_count,
                              params);
  }
};

// The default backend the cooker uses. v0.7 ALWAYS returns the CPU backend; the
// v1.0 GPU backend is selected HERE (build flag / cook config) once it exists.
// Returns a reference to a stateless function-local static.
const SdfBakeBackend& DefaultSdfBakeBackend();

} // namespace nuka::import::cooker
