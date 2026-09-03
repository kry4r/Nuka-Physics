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

## Attempted Recook
Recook attempt crashed during world creation (CUDA error or scene load failure). The crash needs investigation before the scene can be rebuilt with correct contact capacity.

## Diagnostic Artifacts
- `diagnose_collision_pipeline.py`: Confirmed all contact geometry at (0,0,0)
- `read_narrowphase_diagnostic.py`: Attempted to read kernel diagnostics (requires rebuild)
- `diagnose_broadphase.py`: Attempted to read candidate_pairs buffer (not exposed to Python API)
