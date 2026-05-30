# v0.3 — Shared CUDA Math + Buffer Utility Library

**Task #17, Phase 1 (EXTRACTION/AUTHORING ONLY).** Branch `v03`. Date 2026-05-30.

> **Scope guard.** Phase 1 authors the new shared headers, the duplication
> inventory, and this migration plan. It **does NOT modify any existing file**
> and **does NOT replace any call site**. Per the owner: *"先不替换，做完之后等待
> T8B 结束后再进行替换。"* Phase 2 (replacing call sites) is deferred until the
> parallel agent **T8b** finishes editing the articulation translation units.

## 0. New files created (additive only)

| File | Purpose |
|---|---|
| `src/math/cuda_vec_ops.cuh` | `__forceinline__ __device__` small-vector / quaternion primitives (`nuka::math::gpu`). |
| `src/math/cuda_spatial_ops.cuh` | `__forceinline__ __device__` spatial 6-vector / 6×6 algebra over flat `float*` arrays (`nuka::math::gpu`). |
| `src/phi/buffer_transfer.hpp` | Host templated `UploadVector` / `DownloadVector` Buffer⇄vector helpers (`nuka::phi`). |

Include root is `/root/Nuka-Physics/src` (confirmed from the build response
file), so the headers are reached as `#include "math/cuda_vec_ops.cuh"`,
`#include "math/cuda_spatial_ops.cuh"`, `#include "phi/buffer_transfer.hpp"` —
matching the existing `math/vec3.hpp` / `phi/buffer.hpp` convention.

Namespaces chosen so Phase-2 migration is *delete-local-def + qualified-call*:
- device math: `nuka::math::gpu` (the `.cu` TUs already live inside `nuka::…`,
  so a `namespace mg = nuka::math::gpu;` alias or `using` makes the call sites
  read almost identically to today).
- host buffers: `nuka::phi` (same namespace the `phi::Buffer` type lives in).

## 1. Duplication inventory

### 1.1 Device small-vector / quaternion helpers

Every CUDA physics TU re-declares these inside its anonymous namespace. `math::`
resolves to `nuka::math` by enclosing-namespace lookup. "Recipe" = the exact
float expression + op order; copies sharing a recipe are bit-identical.

| Symbol(s) | Defining files (file:line) | Recipe / notes |
|---|---|---|
| `MakeVec3(x,y,z)` | broadphase:19, contact_generation:18, sensors:22, row_solver:35, particle:93, featherstone:25, batched:152 (`__host__ __device__`) | Two forms: brace-init `{x,y,z}` (featherstone) and field-assignment (others). **Bit-identical** result; one unified `MakeVec3` (brace form) serves both. |
| `MakeQuat(w,x,y,z)` | broadphase:27, contact_generation:38, sensors:30, row_solver:61, contacts:62, batched:176 | Field-assignment (Quat ctors are host-only constexpr). Identical. |
| `Add(a,b)` | broadphase:36, contact_generation:47, invariants:43, sensors:39, row_solver:43, particle:101, batched:185, featherstone:29, world_stepper:24 | `{a.x+b.x, …}`. Identical (brace vs field-assignment, same result). |
| `Sub(a,b)` | broadphase:40, contact_generation:51, sensors:43, row_solver:47, particle:105, batched:189, jacobian:16 | Identical. |
| `Neg(v)` | batched:193 | `{-v.x,-v.y,-v.z}`. Single copy. |
| `Scale(v,s)` | broadphase:44, contact_generation:55, sensors:47, row_solver:51, particle:109, batched:197, featherstone:33, world_stepper:32 | `{v.x*s, …}`. Identical. |
| `Dot` / `Dot3` | broadphase(Dot via Cross use), invariants:47, contact_generation:59, sensors:51, row_solver:130 (+fwd-decl:55), particle:113, batched:201, jacobian:28, contacts(Dot3):35, featherstone(Dot3):37 | `x*x+y*y+z*z` left-assoc. **Identical everywhere.** Note `row_solver.cu:55` is a forward declaration; def at :130. |
| `Cross` / `Cross3` | broadphase:48, invariants:51, contact_generation:75, sensors:55, row_solver:90, particle:129, batched:205, jacobian:20, contacts(Cross3):39, featherstone(Cross3):41 | `{y*z−z*y, z*x−x*z, x*y−y*x}` fixed order. **Identical everywhere.** |
| `LengthSq(v)` | sensors:62, particle:117 | `Dot(v,v)`. Identical. |
| `Length(v)` | contact_generation:63, sensors:66, row_solver:57, batched:212 | `sqrtf(Dot(v,v))`. Identical. |
| `ZeroVec3/UnitX/UnitY/UnitZ` | contact_generation:26/30/34, batched:160/164/168/172, world_stepper(ZeroVec3):16 | Component literals. Identical. |
| `Rotate(q,v)` (vec by quat, short form) | broadphase:55, contact_generation:82, sensors:78, row_solver:96, batched:231, particle:154 | `qv=(x,y,z); t=2*(qv×v); v + w*t + qv×t`. **Identical** → `RotateShort`. |
| Quat product `Mul`/`Multiply`/`QuatMul` | broadphase:65, contact_generation:92, sensors:84, particle:169, batched(Mul):237, row_solver(Multiply):82, contacts(QuatMul):76 | Hamilton w-first, fixed op order. **Identical** → `QuatMul`. |
| `QuatIdentity` | contacts:71 | `MakeQuat(1,0,0,0)`. |

#### Normalization families — **DIVERGENT, kept separate** (see §1.4)

| Local symbol | File:line | eps | cmp | fallback | sqrt path | New symbol |
|---|---|---|---|---|---|---|
| `Normalize(Vec3)` | contact_generation:67 | 1e-6 | `<` | `UnitY()` | `sqrt`,`1/len` | `NormalizeSqrt(v,1e-6f,UnitY())` |
| `Normalize(Vec3)` | batched:216 | 1e-8 | `<` | `UnitY()` | `sqrt`,`1/len` | `NormalizeSqrt(v,1e-8f,UnitY())` |
| `Normalize(Vec3)` | sensors:70 | 1e-8 | `<` | `(1,0,0)` | `sqrt`,`1/len` | `NormalizeSqrt(v,1e-8f,MakeVec3(1,0,0))` |
| `NormalizeOrUp` | jacobian:43 | 1e-10 | `>` | `(0,0,1)` | `rsqrtf` | `NormalizeOrUp(v)` |
| `NormalizeOrUpLocal` | contacts:50 | 1e-10 | `>` | `(0,0,1)` | `rsqrtf` | `NormalizeOrUp(v)` (same body) |
| `NormalizeOr(v,fallback)` | particle:121 | 1e-12 | `<=` | param | `rsqrtf` | `NormalizeOr(v,fallback)` |

#### Quaternion-normalization families — **DIVERGENT, kept separate**

| Local symbol | File:line | eps | cmp | sqrt path | New symbol |
|---|---|---|---|---|---|
| `Normalize(Quat)` | row_solver:70 | 1e-12 | `<=` | `sqrtf`,`1/norm` | `QuatNormalizeSqrtLe(q,1e-12f)` |
| `NormalizeQuat` | batched:245 | 1e-8 | `<` | `sqrtf`,`1/norm` | `QuatNormalizeSqrtLt(q,1e-8f)` |
| `QuatNormalize` | contacts:84 | 1e-12 | `<=` | `rsqrtf` | `QuatNormalizeRsqrt(q,1e-12f)` |
| `QuatNormalizeForward` | featherstone:751 | 1e-24 | `<` (on norm_sq) | `rsqrtf` in-place | `QuatNormalizeForwardRsqrt(q,1e-24f)` |

### 1.2 Spatial 6-vector / 6×6 helpers

| Local symbol | File:line | New symbol | Notes |
|---|---|---|---|
| `Zero6` | featherstone:49 | `Zero6` | identical |
| `Zero36` | featherstone:55 | `Zero36` | identical |
| `Dot6` / `Dot6Local` | featherstone:61, contacts:266 | `Dot6` | identical |
| `Add6InPlace` | featherstone:69 | `Add6InPlace` | identical |
| `SubtractOuterProduct36` | featherstone:75 | `SubtractOuterProduct36(m,u,diag,min_diag)` | `kMinDiagonal=1e-6f` lifted to a param literal (bit-identical `fmaxf` path) |
| `Copy36` / `Copy36Local` | featherstone:258, contacts:274 | `Copy36` | identical |
| `Mat66MulVec6` / `Mat66MulVec6Local` | featherstone:248, contacts:280 | `Mat66MulVec6` | identical |
| `TransformMotion` | featherstone:200 | `TransformMotion(X,in,out)` | **takes `const float* X`** (see §1.3) |
| `TransformForceTranspose` / `…Local` | featherstone:212, contacts:316 | `TransformForceTranspose(X,in,out)` | takes `const float* X` |
| `TransformInertiaToParent` / `…Local` | featherstone:224, contacts:291 | `TransformInertiaToParent(X,inertia,delta)` | takes `const float* X` |
| `MotionCross` | featherstone:264 | `MotionCross` | builds through `Vec3 Cross/Add`; order preserved |
| `ForceCross` | featherstone:279 | `ForceCross` | same |

### 1.3 T8b-owned-type decoupling

`TransformMotion / TransformForceTranspose / TransformInertiaToParent` in the
originals take `const LinkSpatialTransform&` but read only the `float[36] X`
member. `LinkSpatialTransform` is defined in `articulation_state.hpp`, which
**T8b owns and is actively editing**. To keep the shared header free of any
dependency on a file under concurrent edit, the library versions take the raw
`const float* X`. The body is byte-identical; the Phase-2 call-site change is the
mechanical `f(transform.X, …)`.

### 1.4 Host buffer-transfer helpers

| Local symbol | Files (file:line) | New symbol | Notes |
|---|---|---|---|
| `UploadVector(const vector<T>&) → Buffer` | articulation_state.cpp:16, device_world.cu:16, particle:1712, sensors:421, batched:59, row_solver:513 | `phi::UploadVector` | 6 copies, **byte-identical** |
| `DownloadVector(const Buffer&, uint32_t count) → vector<T>` | broadphase:188, contact_generation:364, device_world.cu:36, batched:68, particle:1721, sensors:430 | `phi::DownloadVector(buf,count)` | 6 copies, **byte-identical** |
| `DownloadVector(const Buffer&, vector<T>* out) → void` | articulation_state.cpp:25 | `phi::DownloadVector(buf,out)` | **DIVERGENT OVERLOAD** — out-param, null-safe, sizes by `Size()/sizeof(T)`. Only used in articulation_state.cpp. Provided as a distinct overload; both coexist. |

> Other Upload/Download symbols (`UploadDeviceWorld`, `UploadBatchedDeviceWorld`,
> `UploadToScratch`, `DefaultedCopy`, Vulkan's `UploadBuffer`, etc.) are
> **not** generic Buffer⇄vector helpers and are **out of scope** for this lib.

## 2. New library API

### `nuka::math::gpu` (in `math/cuda_vec_ops.cuh`)
```
Vec3 MakeVec3(float,float,float);  Quat MakeQuat(float,float,float,float);
Vec3 Add(Vec3,Vec3);  Vec3 Sub(Vec3,Vec3);  Vec3 Neg(Vec3);  Vec3 Scale(Vec3,float);
float Dot(Vec3,Vec3);  Vec3 Cross(Vec3,Vec3);
float LengthSq(Vec3);  float Length(Vec3);
Vec3 ZeroVec3();  Vec3 UnitX();  Vec3 UnitY();  Vec3 UnitZ();
Vec3 NormalizeSqrt(Vec3 v, float eps, Vec3 fallback);   // sqrt + 1/len family
Vec3 NormalizeOrUp(Vec3 normal);                        // rsqrtf, 1e-10, +Z
Vec3 NormalizeOr(Vec3 v, Vec3 fallback);                // rsqrtf, 1e-12
Quat QuatIdentity();  Quat QuatMul(Quat,Quat);
Quat QuatNormalizeSqrtLe(Quat,float eps);   // sqrtf + 1/norm, `<=`
Quat QuatNormalizeSqrtLt(Quat,float eps);   // sqrtf + 1/norm, `<`
Quat QuatNormalizeRsqrt(Quat,float eps);    // rsqrtf, `<=`, fallback Identity
Quat QuatNormalizeForwardRsqrt(Quat,float eps); // rsqrtf in-place, `<` on norm_sq
Vec3 RotateShort(Quat,Vec3);
```

### `nuka::math::gpu` (in `math/cuda_spatial_ops.cuh`)
```
void Zero6(float*);  void Zero36(float*);  void Copy36(const float*,float*);
float Dot6(const float*,const float*);  void Add6InPlace(float*,const float*);
void Mat66MulVec6(const float* m,const float* v,float* out);
void SubtractOuterProduct36(float* m,const float* u,float diag,float min_diag);
void TransformMotion(const float* X,const float* in,float* out);
void TransformForceTranspose(const float* X,const float* in,float* out);
void TransformInertiaToParent(const float* X,const float* inertia,float* delta);
void MotionCross(const float*,const float*,float*);
void ForceCross(const float*,const float*,float*);
```

### `nuka::phi` (in `phi/buffer_transfer.hpp`)
```
template<class T> Buffer UploadVector(const std::vector<T>&);
template<class T> std::vector<T> DownloadVector(const Buffer&, std::uint32_t count);
template<class T> void DownloadVector(const Buffer&, std::vector<T>* out);
```

## 3. Phase-2 migration plan (ordered, mechanical) — DEFERRED until T8b done

General mechanical recipe per file:
1. Add the include (`#include "math/cuda_vec_ops.cuh"` / `cuda_spatial_ops.cuh`
   / `phi/buffer_transfer.hpp`).
2. Delete the anonymous-namespace local definitions listed in §1.
3. Optionally add `namespace mg = nuka::math::gpu;` and rewrite calls as
   `mg::Cross(...)`, or pull names in with `using mg::Cross;` etc.
4. For the spatial transform calls, change `f(transform, …)` → `f(transform.X, …)`.
5. For normalization/quat-normalize call sites, substitute the parameterized
   form from the §1.1 tables (the eps/fallback become explicit literal args).
6. Build `build-cuda128`, run the determinism / owner-golden gates, and confirm
   **byte-identical** trajectories (see §4 risk).

**Bucketing rule.** §3.1 vs §3.2 is decided strictly by the task's explicit
T8b NEVER-touch list and the live `git status`, NOT by directory feel. The
forbidden/T8b-edited set is: `featherstone_aba.{cu,hpp}`,
`articulation_contacts.cu`, `articulation_jacobian.cu`,
`articulation_state.{cpp,hpp}`, `batched_articulated_world.cu`,
`c_abi/world.cpp`, `tests/CMakeLists.txt`. At Phase-1 close, `git status`
confirms `articulation_contacts.cu`, `articulation_jacobian.cu`, and
`tests/CMakeLists.txt` are actively modified by T8b. Note
`batched_device_world.cu` is **NOT** in the forbidden list and is **not**
shown modified — so it belongs in §3.1, not §3.2.

### 3.1 Files safe to migrate first (NOT in the T8b forbidden list)

Ordered easiest→hardest:

1. `src/core/diagnostics/invariants_gpu.cu` — only `Add/Dot/Cross`.
2. `src/runtime/gpu/cuda_world_stepper.cu` — `ZeroVec3/Add/Scale`.
3. `src/sensor/gpu/cuda_sensors.cu` — vec ops + `Normalize`(→`NormalizeSqrt(.,1e-8f,MakeVec3(1,0,0))`) + `Rotate`(→`RotateShort`) + `Mul`(→`QuatMul`) + buffer helpers.
4. `src/collision/gpu/broadphase.cu` — vec ops + `Rotate/Mul` + `DownloadVector(buf,count)`.
5. `src/constraint/gpu/contact_generation.cu` — vec ops + `Unit*` + `Normalize`(→`NormalizeSqrt(.,1e-6f,UnitY())`) + `DownloadVector(buf,count)`.
6. `src/solver/gpu/row_solver.cu` — vec ops + `Normalize(Quat)`(→`QuatNormalizeSqrtLe(.,1e-12f)`) + `Multiply`(→`QuatMul`) + `UploadVector`. Remove the `Dot` forward-decl at :55.
7. `src/runtime/gpu/device_world.cu` — `UploadVector` + `DownloadVector(buf,count)` (keep its own `DefaultedCopy`).
8. `src/runtime/gpu/cuda_particle_world.cu` — vec ops + `NormalizeOr`(→`NormalizeOr`) + `Rotate/Mul` + buffer helpers.
9. `src/runtime/gpu/batched_device_world.cu` — `NormalizeQuat`(→`QuatNormalizeSqrtLt(.,1e-8f)`), `Normalize(Vec3)`(→`NormalizeSqrt(.,1e-8f,UnitY())`), the `__host__ __device__` `MakeVec3` (shared symbol is HD — see §4), vec ops, `Mul`(→`QuatMul`), `Rotate`(→`RotateShort`), buffer helpers. *(Not in the forbidden list, but it sits adjacent to T8b's `batched_articulated_world.cu` in the same target — rebuild/re-verify carefully; migrate after items 1–8.)*

### 3.2 Files GATED on T8b completion (in the forbidden list) — migrate LAST

These are explicitly off-limits in Phase 1 and must wait for T8b. Every file
here is on the task's NEVER-touch list and/or shown modified in `git status`:

- `src/runtime/articulation/featherstone_aba.cu` — the canonical home of all
  spatial helpers + `QuatNormalizeForward`(→`QuatNormalizeForwardRsqrt(.,1e-24f)`).
- `src/runtime/articulation/articulation_contacts.cu` — the `*Local` spatial
  copies + `NormalizeOrUpLocal`(→`NormalizeOrUp`) + `QuatNormalize`(→`QuatNormalizeRsqrt(.,1e-12f)`) + `QuatMul`. **(`git status`: modified by T8b.)**
- `src/runtime/articulation/articulation_jacobian.cu` — `Sub/Cross/Dot` +
  `NormalizeOrUp`. **(On the forbidden list AND `git status`: modified by T8b — do not race it.)**
- `src/runtime/articulation/articulation_state.{cpp,hpp}` — `UploadVector` +
  out-param `DownloadVector`.
- `src/runtime/gpu/batched_articulated_world.cu` — vec/quat ops as duplicated there.
- `src/c_abi/world.cpp` — on the forbidden list; inspect before touch.

> **Do not migrate any §3.2 file until T8b confirms its edits are merged.**

## 4. Optimizations: applied vs rejected-for-determinism

NVIDIA confirms FMA contracts `X*Y+Z` to a single-rounding `rn(X*Y+Z)`, whereas
non-FMA is `rn(rn(X*Y)+Z)` — two roundings — and that *"the final values … can
depend on … whether to use fused multiply-add or whether additions are organized
in series or parallel"* (Floating Point and IEEE 754 Compliance, §2.3 FMA; §5.4).
nvcc contracts by default (`--fmad=true`). Therefore anything that changes which
multiplies/adds get contracted, or the reduction/association order, can change
bits and is **rejected / gated on a golden re-verify**.

| Optimization | Status | Rationale |
|---|---|---|
| `__forceinline__ __device__` on all primitives | **APPLIED** | Inlining itself does not reorder a given float expression. BUT see risk row — re-verify golden. |
| `MakeVec3`/`MakeQuat` marked `__host__ __device__` | **APPLIED** | Only `batched_device_world.cu:152` declares `MakeVec3` HD; the shared symbol is HD so that migration compiles and host construction (if any) is covered. Vec3/Quat ctors are HD. No effect on float ops. |
| `__restrict__` on non-aliasing pointer pairs (`Copy36` src/dst) | **APPLIED** (where provably non-aliasing) | Aliasing qualifier doesn't change the arithmetic; aids the compiler. CUDA Best Practices §12.2 Memory Instructions. |
| `const`-correctness on params | **APPLIED** | No effect on emitted float ops. |
| De-duplicate identical bodies into one symbol | **APPLIED** | Same source expression ⇒ same codegen ⇒ bit-identical. |
| Collapse eps/fallback-only variants via literal params | **APPLIED** | A literal arg compiles to the same FP ops with a different immediate; bit-identical to the inlined constant. |
| Merge `rsqrtf` and `1.0f/sqrtf` Normalize variants | **REJECTED** | `rsqrtf(x)` ≠ `1.0f/sqrtf(x)` bit-for-bit; kept as separate families. |
| Merge `<` vs `<=` epsilon-boundary quat-normalize variants | **REJECTED** | Boundary input maps to different branches; kept as `…SqrtLe`/`…SqrtLt`. |
| Hand-insert `__fmaf_rn` / change `--fmad` / `--use_fast_math` | **REJECTED** | Directly alters rounding/contraction ⇒ breaks D1 golden. |
| Float atomics for reductions | **REJECTED** | Non-deterministic accumulation order. |
| Reassociate / tree-reduce `Dot`/`Dot6`/matvec sums | **REJECTED** | Changes op order ⇒ different rounding (Floating Point guide §5.4). |
| cuBLAS / cuSOLVER / CUTLASS / cuBLASDx (MathDx) for the 6×6 solves | **REJECTED (gated)** | (a) The dense ops here are tiny per-thread 6×6/6-vector kernels, not batched GEMM; library launch + layout overhead dominates and they don't fit the per-thread model. (b) Any library GEMM uses its own accumulation/FMA strategy ⇒ would not reproduce the golden. CUDA Best Practices §6.1 recommends libraries for *large* parallel primitives, which these are not. Revisit only with a fresh golden if a future batched dense-solve stage is added. |

**LINKAGE CAVEAT (Phase 2):** the original helpers live in anonymous namespaces
(internal linkage — one private copy per TU). The shared functions are in a
named namespace and are `__forceinline__ __device__` (host `MakeVec3/MakeQuat`
aside). Under whole-program / non-separable CUDA compilation this is safe: the
build uses **no** `CUDA_SEPARABLE_COMPILATION` / `-rdc=true` (grep of
`CMakeLists.txt` + `cmake/` found none), and `__forceinline__` functions are
inlined at every call so no out-of-line device symbol is emitted to clash at
device link. **If a future target enables `-rdc=true`/separable compilation**, a
non-`inline` header-defined `__device__` function included by ≥2 TUs would
multiply-define at device link — at that point mark the helpers `inline`
(zero-cost, determinism-neutral). The host buffer templates are immune (template
linkage). No ODR issue exists today; this is a forward-looking note.

**RISK / GATE (must hold in Phase 2):** force-inlining the previously separate
`__device__` calls can let nvcc contract a multiply in one former-callee with an
add in the caller into a new FMA that did not exist when they were distinct
functions — so **bit-identity is not automatic**. The Phase-2 owner-golden /
D1-determinism re-verify (`build-cuda128` + the determinism gate) is the gate
that proves each migrated file still produces a byte-identical trajectory. If a
file regresses, isolate that file's migration. Source expressions are kept
verbatim; no intrinsics, no `--fmad`/`--use_fast_math` changes are introduced.

## 5. Standalone verification performed (Phase 1)

Headers were proven to compile **without touching `build-cuda128`** — throwaway
TUs in `/tmp`, include flags derived from the build (`-I/root/Nuka-Physics/src`,
CUDA `-std=c++17 -arch=native`). All passed clean (incl. `-Wall -Wextra`, and
the strict host gate `-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror`).

- `nvcc -std=c++17 -arch=native -Xcompiler=-fPIC,-Wall,-Wextra -I/root/Nuka-Physics/src -isystem <cuda>/include -c /tmp/probe_device.cu` → exit 0 (both device headers, every symbol referenced once).
- `g++-10 -std=c++2a -fPIC -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror -I/root/Nuka-Physics/src -isystem <cuda>/include -c /tmp/probe_buffer.cpp` → exit 0 (buffer header, strict host gate, all three overloads).
- `nvcc -std=c++17 … -c /tmp/probe_buffer_cu.cu` → exit 0 (buffer header from a `.cu` TU).

Additionally, a host-side **bit-identity check** (throwaway `/tmp`,
`g++-10 -O2 -ffp-contract=off`) ran each ORIGINAL helper body against the new
header body over 200,000 randomized inputs and asserted bitwise-equal float bits
(`memcmp` on the IEEE-754 words). Covered: `Cross`, `Dot`, all three Normalize
families, `QuatNormalizeSqrtLe` / `QuatNormalizeForwardRsqrt`, `QuatMul`,
`RotateShort`, `Dot6`, `Mat66MulVec6`, `TransformForceTranspose`,
`TransformInertiaToParent`. Result: **0 mismatches** — confirms transcription
fidelity. (This proves source-body equality on host; the Phase-2 GPU owner-golden
re-verify remains the gate for device-side FMA-contraction effects per the risk
note in §4.)

## 6. Sources

- NVIDIA, *Floating Point and IEEE 754 Compliance for NVIDIA GPUs* — §2.3 The
  Fused Multiply-Add (FMA); §5.4 Verifying GPU Results. Quoted text directly
  retrieved. <https://docs.nvidia.com/cuda/floating-point/index.html>
- NVIDIA, *CUDA C++ Best Practices Guide* — Parallel Libraries (recommends
  cuBLAS etc. for large parallel primitives), Memory Instructions, and Register
  Pressure guidance. Section numbers (~§6.1 / §12.2 / §10.2.7.1) are approximate
  — the page's prose was retrieved but the exact subsection anchors were not
  individually confirmed. <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html>
