# Nuka Physics v0.5 – Phase 5: License + CLA + Open Source Release + Go2 System ID Demo (v0.5 Exit Gate)

> **Master plan reference:** §3 Round 12 (Apache 2.0 + CLA, public at v0.5) + §3 Round 13 (S4 diff-sim demo) + §7 v0.5 exit
> **Prerequisites:** v0.5 Phases 1–4
> **Blocks:** v0.7 (S2 entry)
> **Exit criteria gate:** **v0.5 CLOSE — repo goes public**
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

Close v0.5 with three deliverables that together complete the master plan §7 v0.5 exit criteria:

1. **GitHub repo public** under Apache 2.0 + CLA — repository goes public at this moment per master plan §3 Round 12 timing decision.
2. **Demo: gradient-based system identification on Go2** — recover an unknown Go2 link mass (or friction coefficient) by gradient descent through 2 s of simulation. This is the showcase that "the diff-sim infrastructure works end-to-end."
3. **Quarterly external output** — release announcement + technical blog post.

## Tech Stack

- All prior v0.5 deliverables
- GitHub (repo hosting + CI)
- Apache 2.0 + Contributor License Agreement
- Optional: arXiv preprint (technical write-up)

## Files to Create

### License + governance

- `LICENSE` — Apache 2.0 text
- `NOTICE` — copyright header + third-party attributions (CUDA, nanobind, PyTorch interop notes)
- `CLA.md` — Contributor License Agreement (Apache CLA template, owner-customized)
- `CONTRIBUTING.md` — contribution flow + CLA signing instructions
- `CODE_OF_CONDUCT.md` — standard CoC (Contributor Covenant)
- `SECURITY.md` — vulnerability reporting policy
- `.github/PULL_REQUEST_TEMPLATE.md` — checklist (lint passes, tests pass, CLA signed)
- `.github/ISSUE_TEMPLATE/bug_report.md`
- `.github/ISSUE_TEMPLATE/feature_request.md`
- `.github/workflows/cla.yml` — CLA-bot integration

### Public-facing docs

- `README.md` (rewrite from current dev-internal README to public-facing)
- `docs/getting-started.md` — install, hello-world, RL training quickstart
- `docs/concepts/architecture.md` — high-level architecture overview (links to master plan)
- `docs/concepts/diff-sim.md` — what is differentiable, how, examples
- `docs/concepts/isaaclab-compat.md` — migrating from Isaac Lab
- `docs/examples/go2_locomotion.md` — replicate the v0.3 demo
- `docs/examples/system_identification.md` — replicate the v0.5 demo (this phase)
- `docs/architecture/sim2real-noise.md` — N1+N2 noise design
- `docs/architecture/diffsim-tape.md` — checkpointing strategy

### System ID demo

- `examples/scenes/go2_system_id.usda` — Go2 with one tunable link mass
- `examples/demos/go2_system_id.py` — gradient descent on the unknown mass
- `examples/demos/go2_system_id_results/` — output directory (logs, plots)
- `tests/regression/test_go2_system_id_convergence.py` — gradient descent recovers ground-truth mass within tolerance
- `docs/examples/system_identification.md` — write-up with figures

### Quarterly external output

- `docs/blog/2026-XX-XX-v05-launch.md` — launch blog post
- (optional) `docs/papers/2026-XX-XX-arxiv-preprint.md` — technical preprint draft

## Tasks

### Task 5.5.1 — License + governance files

Standard Apache 2.0 license file from <https://www.apache.org/licenses/LICENSE-2.0.txt>. The CLA template from Apache Software Foundation (Individual CLA + Corporate CLA versions). `NOTICE` lists all third-party dependencies (CUDA Toolkit, nanobind, OpenUSD if integrated by then, etc.) with required attribution. The sparse linear solver is self-written (no closed-source SDK), so it carries no third-party attribution.

CLA-bot setup: a GitHub Action that checks every PR's author has signed the CLA before allowing merge. Free service (cla-assistant.io) or self-hosted.

### Task 5.5.2 — Repo cleanup before going public

Pre-publish checklist:

- [ ] No `.env` / credentials in history
- [ ] No internal-only notes / customer names
- [ ] All experimental branches archived or rebased
- [ ] master plan + spec files reviewed for clarity
- [ ] All AI-protected files (master plan §5.4) are clearly marked
- [ ] Issue tracker enabled
- [ ] Discussions enabled
- [ ] GitHub Pages set up to render docs
- [ ] Repository description + topics + README banner ready

### Task 5.5.3 — Public README rewrite

`README.md`:

```markdown
# Nuka Physics

A CUDA-first, GPU-resident physics engine for embodied robotics and large-scale RL training.

## Highlights
- 4096+ parallel envs on a single GPU
- Differentiable simulation through rigid + articulated dynamics
- Strong determinism (bit-exact reproducibility)
- Drop-in API for Isaac Lab's RL training subset
- Zero-copy PyTorch + JAX tensor interop

## Quick start

    pip install nuka-physics                            # PyPI (when released)
    python -m nuka.examples.train_go2_ppo --num_envs 4096

## Architecture
See [docs/concepts/architecture.md](docs/concepts/architecture.md) for the
big picture and [docs/plans/2026-05-28-nuka-physics-master-plan.md](docs/plans/2026-05-28-nuka-physics-master-plan.md)
for the long-term plan.

## Status
v0.5 — initial public release. Differentiable rigid + articulated; soft body / fluid / coupling coming in v0.7.

## License
Apache 2.0 with CLA. See [LICENSE](LICENSE) and [CONTRIBUTING.md](CONTRIBUTING.md).
```

### Task 5.5.4 — System ID demo

`examples/scenes/go2_system_id.usda`:
- Go2 with all links at known nominal mass except one link (e.g., front-left thigh) with mass treated as a tunable parameter.

`examples/demos/go2_system_id.py`:

```python
import nuka
import torch

# Ground truth (only used to generate target trajectory; not visible to optimizer)
GROUND_TRUTH_MASS = 1.42

# Scene with mass as a learnable parameter
dev = nuka.Device(0)
world_gt = nuka.World(dev, "examples/scenes/go2_system_id.usda", num_envs=1)
world_gt.set_link_mass("front_left_thigh", GROUND_TRUTH_MASS)

# Generate target trajectory: 2 s of PD-driven motion, observe joint trajectory
target_obs = []
for _ in range(480):  # 2s @ 240Hz
    actions = generate_test_actions()
    world_gt.write_actions(actions)
    world_gt.step()
    target_obs.append(world_gt.observations.clone())
target_obs = torch.stack(target_obs)

# Optimizer: gradient descent on mass parameter
world = nuka.World(dev, "examples/scenes/go2_system_id.usda", num_envs=1)
tape = nuka.Tape(world, checkpoint_interval=50)
mass_param = torch.nn.Parameter(torch.tensor([1.0]))   # initial guess
optimizer = torch.optim.Adam([mass_param], lr=0.05)

for iteration in range(200):
    world.set_link_mass("front_left_thigh", mass_param.item())
    world.reset()
    tape.reset()

    sim_obs = []
    for _ in range(480):
        actions = generate_test_actions()
        obs = nuka.differentiable_step(world, tape, actions)
        sim_obs.append(obs)
    sim_obs = torch.stack(sim_obs)

    loss = (sim_obs - target_obs).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"iter {iteration} mass={mass_param.item():.4f} loss={loss.item():.6e}")

# Final mass should be within 1% of GROUND_TRUTH_MASS
recovered_err = abs(mass_param.item() - GROUND_TRUTH_MASS) / GROUND_TRUTH_MASS
assert recovered_err < 0.01, f"System ID failed: recovered mass error {recovered_err:.2%}"
```

The demo produces:
- `examples/demos/go2_system_id_results/loss_curve.png` — loss vs iteration
- `examples/demos/go2_system_id_results/mass_curve.png` — recovered mass over iterations
- `examples/demos/go2_system_id_results/run_log.txt` — full log

### Task 5.5.5 — Regression test for the demo

`tests/regression/test_go2_system_id_convergence.py`:

```python
def test_system_id_converges_to_true_mass():
    # Run the demo with fixed seed
    # Assert final mass within 1% of ground truth
    # Assert loss monotonically decreases (with some tolerance for SGD noise)
```

### Task 5.5.6 — Launch blog post

`docs/blog/2026-XX-XX-v05-launch.md`:

Suggested outline:

1. **What is Nuka Physics?** — one-paragraph elevator pitch
2. **What we shipped in v0.5** — bullet list of capabilities
3. **The demo: gradient-based system ID on Go2** — figure of loss curve, before/after gait, recovered mass
4. **What makes it different from Isaac Lab / Brax / MJX**:
   - Full analytical adjoint (vs Brax stop-grad-only)
   - Strong determinism (vs PhysX/Isaac Lab non-deterministic)
   - Direct C++ embedding (vs Isaac Lab Python-only)
   - Coming in v0.7: rigid+soft+fluid coupling
5. **Roadmap** — link to master plan §7 phase table; emphasize S2 / S3 / S5 anchors
6. **How to contribute** — CLA, lint, master plan amendment process

Length: 1500-2000 words; include 4-6 figures.

### Task 5.5.7 — Public CI

Move CI from local-only to GitHub Actions:
- Lint on every PR
- Build + test on RTX-class self-hosted runner (if available) OR on CPU-only test path for code that doesn't require GPU
- Wheel build matrix (Linux + Windows; Python 3.10, 3.11, 3.12)
- Docs site build (GitHub Pages)
- Optional: pip publish workflow (manual trigger for v0.5.x releases)

### Task 5.5.8 — v0.5 exit gate checklist

- [ ] Phase 1: V3 FD adjoint check passes for all v0.1 row classes
- [ ] Phase 2: tape + checkpointing operational; replay bit-exact
- [ ] Phase 3: self-written sparse solver integrated; IFT path operational (D1)
- [ ] Phase 4: PyTorch autograd complete + JAX custom_vjp; FD agreement
- [ ] Phase 4: Sim-to-real N1 + N2
- [ ] Phase 5: License + CLA in place
- [ ] Phase 5: GitHub repo public
- [ ] Phase 5: Go2 system ID demo converges (regression test green)
- [ ] Phase 5: Launch blog post published
- [ ] All v0.1 / v0.3 regressions still passing (no v0.5 regression)
- [ ] Quarterly external output (§5.1 rhythm) ✓

When all checked: **v0.5 CLOSED**. The project is now public; v0.7 (S2 entry — XPBD soft, PBF fluid, SDF coupling) begins.

## Validation

- Demo converges: recovered mass within 1% of ground truth in < 200 iterations.
- Loss decreases monotonically (modulo SGD noise).
- Determinism: same seed produces same convergence trajectory.
- Reverse pass time < 3× forward pass time at K=50.
- Build wheel installs cleanly on Linux + Windows.
- GitHub Actions CI green.

## Exit Criteria for Phase 5 = **v0.5 EXIT (PROJECT GOES PUBLIC)**

Per master plan §7 v0.5:

1. ✅ GitHub repo public under Apache 2.0 + CLA.
2. ✅ Diff-sim end-to-end through rigid + Featherstone.
3. ✅ `torch.autograd.Function` adjoint FD check passing for all base row classes (Phase 1 + 4).
4. ✅ JAX `custom_vjp` operational (Phase 4).
5. ✅ Sim-to-real noise N1 + N2 (Phase 4).
6. ✅ Self-written deterministic sparse solver integrated for IFT (Phase 3, D1).
7. ✅ **Demo: gradient-based system identification on Go2** (Phase 5 Task 5.5.4).

Plus rhythm: ✅ Quarterly external output published (Phase 5 Task 5.5.6).

## What This Phase Does Not Do

- No soft body or fluid (v0.7).
- No cross-system coupling rows (v0.7).
- No advanced sparse-solver methods yet — the self-written deterministic CG + Jacobi/Block-Jacobi core ships in v0.5 Phase 3 for the IFT path; MINRES/ILU/GMRES/AMG are added in v0.7+.
- No CUDA RT pipeline (v1.0).
- No Vulkan/D3D12 interop (v0.7 / v1.0).
- No sim-to-real real-hardware deployment (v3.0).

---

After this phase closes, **stop** before opening v0.7. Per §5.2 scope discipline:
- All v0.5 exit criteria must check.
- Quarterly output published.
- Review master plan once more — does anything need amendment in light of v0.5 learnings?
- Then write v0.7 phase plans (this document set covers up to v0.5; v0.7 plans are authored at that gate).
