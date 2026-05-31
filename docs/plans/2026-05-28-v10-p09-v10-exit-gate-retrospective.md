# Nuka Physics v1.0 – Phase 9: v1.0 Exit Gate + Retrospective

> **Master plan reference:** §3 Round 13 (S2 polish demos + quarterly external output) + §7 v1.0 exit
> **Prerequisites:** v1.0 Phases 1-8
> **Blocks:** v1.5 entry
> **Exit criteria gate:** **v1.0 CLOSE**
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

Close v1.0 with three deliverables:

1. **v1.0 exit gate verification** — every master plan §7 v1.0 exit criterion confirmed green.
2. **Quarterly external output** — blog post / talk / preprint per §5.1 rhythm.
3. **Public release** — v1.0.0 tagged on GitHub; pip/PyPI publication if ready.

After this phase, the project has demonstrated S2's full demos (grasp + pour + wring) and is ready to begin S3 work in v1.5 (warehouse scaling, D3D12 interop, OpenUSD binary path).

## Tech Stack

- All v1.0 deliverables
- GitHub Releases
- PyPI (optional)

## Files to Create

- `docs/blog/2026-XX-XX-v10-launch.md` — quarterly external output
- `docs/architecture/v10-retrospective.md` — internal retrospective
- `CHANGELOG.md` — first proper changelog entry (or extension)
- `RELEASE-v1.0.md` — release notes

## Files to Modify

- `README.md` — update status to v1.0; add demo videos / links to blog
- `docs/plans/2026-05-28-nuka-physics-master-plan.md` — owner edit log entry: v1.0 closed

## Tasks

### Task 10.9.1 — v1.0 exit gate checklist

Run through every v1.0 exit criterion from master plan §7:

- [ ] **K2 + K3 cross-system coupling working for rigid↔soft, rigid↔fluid, soft↔fluid.**
  - Evidence: Phase 1 stress battery passes; Phases 7 + 8 demos work.
- [ ] **Sim-to-real noise N3 key items.**
  - Evidence: Phases 2 + 3 + 4 (lens distortion + rolling shutter + lidar beam divergence + motion blur) all operational.
- [ ] **Self-written CUDA RT pipeline operational (LBVH + traversal + intersection + shading).**
  - Evidence: v0.7 Phases 12 + 13 + this version's Phase 5 optimization. Sensor workloads hit throughput target.
- [ ] **Vulkan ↔ CUDA external memory interop (Windows priority).**
  - Evidence: Phase 6 + example Vulkan integration app + docs.
- [ ] **Demo: H1 pour water + wring towel.**
  - Evidence: Phases 7 + 8 demos run; regression tests pass.

Plus carry-forwards from prior versions (must still hold):
- [ ] V1 oracle suite passes (no regression).
- [ ] V2 invariant monitoring works.
- [ ] V3 FD check for all row classes.
- [ ] V4 demo suite extended.
- [ ] V5 lint clean.
- [ ] D1 determinism preserved everywhere.

### Task 10.9.2 — Performance baseline update

Update `docs/architecture/perf-budget.md`:
- v0.3: Go2 sub-ms / env-step on 4096-env (S1 throughput target).
- v0.7: H1 grasp 1× realtime on single env.
- v1.0: H1 pour 1× realtime; H1 wring 1× realtime with 4× substeps.
- Memory: per-env footprint at v1.0 baseline.

### Task 10.9.3 — Risk register update

Master plan §8 risks; mark which are mitigated, which remain:

- ✅ Featherstone ↔ XPBD instability — mitigated by Phase 1 stability work.
- 🟡 Diff-sim long-episode convergence — partially addressed; long episodes still hit checkpoint limits.
- 🟡 D1 perf — passes for S1 (Go2), passes for S2 (1× realtime); to be re-verified for S3 (multi-robot).
- ✅ Self-written sparse solver — core shipped v0.5 (CG/Jacobi, D1); extended v0.7 (MINRES/ILU/GMRES/AMG). No closed-source SDK ever.
- ⏳ Single H100 envelope for S3 — to be stress-tested in v1.5.
- 🟢 OpenUSD binary path — deferred to v1.5 (low risk).

### Task 10.9.4 — Quarterly external output

`docs/blog/2026-XX-XX-v10-launch.md`:

Suggested structure:

1. **Headline**: H1 robot wrings a towel and pours water — single-GPU, 1× realtime, bit-exact.
2. **What's new in v1.0**:
   - Full rigid + cloth + fluid 3-way coupling.
   - N3 sim2real (lens distortion, rolling shutter, lidar divergence, motion blur).
   - CUDA-Vulkan zero-copy interop on Windows.
   - 150 MRays/s self-written RT (no OptiX).
3. **Demo videos**: pour, wring, grasp (linked or embedded).
4. **Comparison to PhysX 5 / Isaac Lab**:
   - Same kind of demo (PhysX 5 also does grasp/pour); difference: ours is differentiable AND deterministic AND fully open source.
5. **Whats next**: v1.5 entry — multi-robot warehouse scaling.

Length: 1500-2500 words; 5-8 figures / videos.

### Task 10.9.5 — Release notes

`RELEASE-v1.0.md`:

- Headline features.
- API changes since v0.5 (any C ABI additions like `nuka_interop.h`).
- Breaking changes (likely none — additive only).
- Migration notes if any.
- Acknowledgments (contributors, libraries used).

### Task 10.9.6 — GitHub release + PyPI

- Tag `v1.0.0` in git.
- Create GitHub Release with release notes + binaries (if applicable).
- If wheel packaging stable: publish `nuka-physics==1.0.0` to PyPI.
- Post launch blog to social channels (Twitter / Mastodon / LinkedIn / Reddit r/robotics).

### Task 10.9.7 — Master plan owner edit log

Add an entry per §13 amendment process:

```
> - 2026-XX-XX – v1.0 closed. All exit criteria green. S2 phase complete. Begin v1.5 (S3 entry).
```

### Task 10.9.8 — Retrospective

`docs/architecture/v10-retrospective.md`:

Internal-facing reflection:
- What worked well in the v0.7 → v1.0 path?
- Where did stability tuning take longer than expected?
- Any spec revisions needed for v1.5+?
- Risk register adjustments.
- Lessons for v1.5+ planning.

This is for the owner; not necessarily public.

### Task 10.9.9 — Plan v1.5 entry phase specs (separate effort)

After v1.0 closure, the next workstream is **authoring v1.5 phase specs**. The master plan documents v1.5 entry exit criteria; the detailed phase breakdown gets written when v1.0 closure is complete (per master plan §5.2 — no speculative architecture; v1.5 plans are authored at the gate).

This is mentioned here so the path forward is clear; the actual specs are not part of v1.0 Phase 9 deliverables.

## Validation

- All v1.0 exit criteria green.
- All carry-forwards still pass (no regression).
- v1.0 binary builds successfully on Windows + Linux.
- Public release artifacts (blog, GitHub release) produced.
- Master plan log updated.

## Exit Criteria for Phase 9 = **v1.0 EXIT**

All master plan v1.0 exit criteria green (Task 10.9.1 checklist).
Plus rhythm: quarterly external output published (Task 10.9.4).
Plus public release: tag + release notes (Task 10.9.6).
Plus master plan log entry (Task 10.9.7).

When complete: **v1.0 CLOSED**. The S2 phase of the master plan is delivered. Next: v1.5 plan authoring.

## What This Phase Does Not Do

- No v1.5 work begins. Per master plan §5.2 strict scope discipline, v1.5 starts only after v1.0 is fully closed and v1.5 plan is authored.
- No new physics features.
- No new demos beyond closing what was promised.
