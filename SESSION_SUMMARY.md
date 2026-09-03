# Session Summary: Pi0.5 Grasping Contact Detection Investigation

## Problem Identified
The pi0.5 Libero black bowl grasping demo fails because finger-bowl contacts are not detected. The solver generates contact forces but all contact geometry is at origin (0,0,0).

## Root Cause
**The scene was cooked with `max_contacts_per_env = 0` in the `.nks` file.**

This causes:
```
rigid_slot_cap = max_contacts_per_env - particle_reserve = 0 - 0 = 0
```

The narrowphase kernel checks:
```cpp
if (slot < live && slot < rigid_slot_cap)  // Never runs when rigid_slot_cap == 0
```

Result: **NO body-body collision detection runs**, so finger-bowl contacts are never generated.

## Evidence Trail
1. `diagnose_collision_pipeline.py` confirmed all contact geometry at (0,0,0)
2. Added diagnostic code to `narrowphase_prims.cu` to read kernel parameters
3. Traced through `pipeline.cpp` → contact capacity calculation
4. Found `cook_to_model.cpp:805` should set `max_contacts_per_env = collidables * 4`
5. Confirmed the `.nks` file has stale cooked data from old code

## Fix Steps (Documented)
1. Rebuild Python extension (requires cmake/ninja in WSL or proper Windows build env)
2. Delete stale `.nks` file
3. Load from `.xml` source to trigger automatic recook with correct contact capacity
4. Verify `rigid_slot_cap > 0` and non-zero contact geometry

## Commits
- `b86b473`: Root cause analysis document
- `8f2a17e`: Added recook procedure with fix steps
- Earlier: Diagnostic scripts and CUDA kernel instrumentation

## Blocking Issue
Python extension is outdated (August 21st) and couldn't be rebuilt in Git Bash due to missing cmake/ninja. User suggested compiling in WSL ("你在wsl编译吧，wsl编译简单点"), but WSL path resolution is broken in this environment.

## Next Steps
1. Set up proper build environment (WSL with cmake/ninja, or Windows with Visual Studio)
2. Rebuild Python extension
3. Follow fix steps in `pi05_grasping_contact_analysis.md`
4. Test grasping demo with correctly cooked scene
