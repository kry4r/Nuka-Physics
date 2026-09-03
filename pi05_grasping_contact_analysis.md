# Pi0.5 Grasping Contact Detection Issue - Root Cause Analysis

## Problem
The pi0.5 Libero black bowl grasping demo fails because finger-bowl contacts are not detected. The solver generates contact forces (10 active slots) but all contact geometry is at origin (0,0,0), indicating the narrowphase collision detection never ran.

## Root Cause
**The scene was cooked with `max_contacts_per_env = 0` in the `.nks` file.**

### Evidence Chain

1. **Pipeline capacity calculation** (`src/nk/pipeline/pipeline.cpp:95`):
   ```cpp
   const uint32_t rigid_cap = cap.max_contacts_per_env - particle_reserve;
   ```

2. **Narrowphase parameter setup** (`src/nk/pipeline/pipeline.cpp:386`):
   ```cpp
   p_np_prim_.rigid_slot_cap = rigid_cap;  // body<->body fills only [0, rigid_cap).
   ```

3. **Narrowphase kernel guard** (`src/phi/backend_cuda/ops/narrowphase_prims.cu:205`):
   ```cpp
   if (slot < live && slot < rigid_slot_cap) {
       // run collision detection
   }
   ```

4. **Result**: If `max_contacts_per_env == 0`, then `rigid_slot_cap == 0`, and NO body-body collision detection runs.

## Expected Value
For a scene with N collidable bodies, `DefaultRigidCandidatePairs` should set:
```cpp
max_contacts_per_env = collidables * kCandidatePairsPerCollidable  // = collidables * 4
```

For the Panda + bowl scene (~10 bodies), this should be **40 contact slots**, not 0.

## Fix Required
Re-cook the scene with current code (`src/scene/cook/cook_to_model.cpp:805`):
```cpp
cap.max_contacts_per_env = enable_contacts
    ? DefaultRigidCandidatePairs(cap.bodies_per_env) : 0u;
```

The existing `.nks` file at `.nuka-assets/generated/libero/libero_spatial_black_bowl.nks` was cooked with old code that set this to 0.

### Steps to Fix
1. **Rebuild Python extension** in WSL (where cmake/ninja are available):
   ```bash
   cd build-linux-python  # or build-win-python if using Windows tools
   cmake --build . --target _nuka_ext
   ```

2. **Delete stale .nks file** to force recook:
   ```bash
   rm .nuka-assets/generated/libero/libero_spatial_black_bowl.nks
   ```

3. **Load from .xml source** - World.create_from_scene will automatically cook:
   ```python
   world = nuka.World.create_from_scene(
       device,
       ".nuka-assets/generated/libero/libero_spatial_black_bowl.xml",
       env_count=1,
       dt=0.005,
       contact_family=2,  # PairDriven
   )
   ```

4. **Verify contact capacity** - After successful load, the scene should have:
   - `max_contacts_per_env = collidables * 4` (≈40 for 10-body scene)
   - `rigid_slot_cap > 0` (allowing body-body collision detection)
   - Non-zero contact geometry when fingers close on bowl

## Attempted Recook
Initial recook attempt crashed, likely because the Python extension was using stale compiled code. The diagnostic code added to narrowphase_prims.cu wasn't executing because the installed Python package dated from August 21st.

## Diagnostic Artifacts
- `diagnose_collision_pipeline.py`: Confirmed all contact geometry at (0,0,0)
- `read_narrowphase_diagnostic.py`: Attempted to read kernel diagnostics (requires rebuild)
- `diagnose_broadphase.py`: Attempted to read candidate_pairs buffer (not exposed to Python API)
