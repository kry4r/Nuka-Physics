# Pi 0.5 Grasping Failure - Root Cause Analysis

## Problem
The pi0.5 LIBERO demo cannot grasp the bowl. Finger-bowl contacts never appear in the 
exported contact telemetry (`contact_points`, `contact_normal`, `contact_force`), 
even when the gripper closes to 0mm and physically intersects the bowl geometry.

## Root Cause: Architecture Mismatch

### Internal Engine State (post-refactor)
- Commit `0663313` (2026-06-17) deleted the FUSED contact path
- **ONE general contact path remains: PairDriven**
- All contact narrowphase (prims, heightfield, SDF, particles) now writes to:
  - `ucontact_count` (uint32, per contact_slot)
  - `ucontact_point` (vec3×4, per contact_slot)
  - `ucontact_normal` (vec3×4, per contact_slot)
  - `ucontact_depth` (f32×4, per contact_slot)
- These become 12 constraint rows per slot (`urows`, per row_slot):
  - 4 manifold points × (1 normal + 2 tangent) = 12 rows
- Solved impulses stored in `lambda` (f32, per row_slot)

### Public API (stale)
The C API and Python bindings still ONLY expose legacy FUSED fields:
```c
NUKA_FIELD_CONTACT_POINTS  = 5   // per contact_slot, vec3
NUKA_FIELD_CONTACT_NORMAL  = 17  // per contact_slot, vec3
NUKA_FIELD_CONTACT_FORCE   = 18  // per contact_slot, vec3
NUKA_FIELD_CONTACT_LINK    = 19  // per contact_slot, uint32
```

These fields are declared in `fields.yaml` but **never written** by the PairDriven path.

### The Readout Bug
`src/phi/backend_cuda/ops/readout.cu:ContactForceKernel`:
```cpp
const uint32_t base = slot * kContactForceComponents;  // 3 floats per slot
out_contact_force[base+0] = lambda[base+0] * inv_dt;
out_contact_force[base+1] = lambda[base+1] * inv_dt;
out_contact_force[base+2] = lambda[base+2] * inv_dt;
```
**Bug**: `lambda` is per-row (12 rows/slot), but kernel indexes it as per-slot with 3 components.
This reads the wrong memory and produces meaningless forces.

## Evidence
1. Increasing `solver_max_pairs` from 512→2048→4096 makes **no difference**
   - Rules out pair-budget exhaustion
2. `contact_force` shows ~60 active slots, but **zero** near rim height (z>0.90)
3. Finger links reach z=0.9598m while rim is at z=0.9320m, yet no contacts export
4. The `ucontact_*` fields exist internally but are `arena: scratch` and not exposed

## Fix Options

### Option A: Expose ucontact_* fields (clean, breaking)
1. Add to `src/include/nuka/nuka.h`:
```c
NUKA_FIELD_UCONTACT_COUNT  = <next_id>
NUKA_FIELD_UCONTACT_POINT  = <next_id>
NUKA_FIELD_UCONTACT_NORMAL = <next_id>
NUKA_FIELD_UCONTACT_DEPTH  = <next_id>
```
2. Add Python bindings in `python/src/nuka_ext.cpp`
3. Update controller to read from ucontact_* instead of contact_*
4. **Breaking change**: existing code reading contact_* sees zeros

### Option B: Populate legacy fields from ucontact (compat, wasteful)
1. Add a readout kernel that copies first valid point from ucontact_* to contact_*:
```cpp
__global__ void LegacyContactReadoutKernel(
    const uint32_t* ucontact_count,
    const Vec3* ucontact_point,   // elem:4
    const Vec3* ucontact_normal,  // elem:4
    uint32_t slot_count,
    Vec3* out_contact_point,      // per slot
    Vec3* out_contact_normal) {
    const uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= slot_count) return;
    uint32_t n = ucontact_count[s];
    if (n > 0) {
        out_contact_point[s] = ucontact_point[s*4];
        out_contact_normal[s] = ucontact_normal[s*4];
    }
}
```
2. Fix `ContactForceKernel` to sum the 3 normal-row impulses (rows 0,3,6,9):
```cpp
const uint32_t row_base = slot * 12;  // 12 rows per slot
float fn = 0.0f;
for (uint32_t i = 0; i < 4; ++i) {
    fn += lambda[row_base + i*3];  // normal rows: 0,3,6,9
}
out_contact_force[slot*3 + 0] = fn * inv_dt;
out_contact_force[slot*3 + 1] = 0.0f;  // no single tangent
out_contact_force[slot*3 + 2] = 0.0f;
```
3. **Non-breaking**: existing Python code works immediately

### Option C: Delete legacy fields, document migration (aggressive)
1. Remove CONTACT_POINTS/NORMAL/FORCE from C API entirely
2. Expose ucontact_* as the canonical API
3. Force all consumers to migrate

## Recommendation
**Option B** for immediate fix: populate legacy fields from ucontact in the readout pass.
Then **Option A** as the v2 API with proper multi-point manifold access.

## Timeline Impact
Pi 0.5 grasping has been broken since commit `0663313` (2026-06-17, 2.5 months ago).
No consumer can reliably read contacts from the PairDriven path - this affects all 
post-June simulations including LIBERO, locomotion, and manipulation demos.
