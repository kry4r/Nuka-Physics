# Nuka Physics v1.0 — Exit-Condition Expansion (owner directive 2026-06-02)

> **Status:** **CONFIRMED 2026-06-02 (中文).** Render bar = **A1** (self-written CUDA RT → 真实感 tier). Demo set = **all four** new demos (convex-pile collapse · dam-break-over-obstacles · buoyancy · domino/Newton's cradle) + existing H1 pour/wring kept. Implementation is v1.0-time; this doc + the §7 rewrite (§E) await owner §13 application.
> **Master plan reference:** §7 v1.0 exit (`:932-937`, the protected "constitution") + §3 Round 9 (renderer pillar) + §3 Round 13 (S2 polish demos).
> **Authored:** 2026-06-02, mid-v0.7 (engine spine at p08). This is a **v1.0 (=`v10`) deliverable** — capture + confirm now, implement at v1.0-time. It does **not** block the in-flight v0.7 phases.
> **🔒 §7 is PROTECTED.** This doc is a *proposal*. The §7 rewrite (§E) must be applied by the owner via the master-plan §13 Amendment Process — do not edit the master plan from an agent.

## 0. Owner directive (verbatim)

> 现在需要对1.0结尾条件进行修改，不只是机器人倒水demo，还需要增加一些刚体碰撞，
> 经典的流体+刚体的demo，并且所有demo都要渲染出真实的画面，放在github主页上展示

Decoded:
1. v1.0 exit is **no longer just the H1 pour-water (robot) demo**.
2. **Add** classic **rigid-body collision** demo(s).
3. **Add** classic **fluid + rigid-body** coupling demo(s).
4. **All** demos must render **真实的画面 (realistic / photorealistic imagery)**.
5. Showcase on the **GitHub homepage** (README / project front page).

## 1. Current v1.0 exit criteria (master plan §7 `:932-937`)

```
### v1.0 – S2 Polish
- K2 + K3 cross-system coupling working for rigid↔soft, rigid↔fluid, soft↔fluid.
- Sim-to-real noise N3 key items.
- Self-written CUDA RT pipeline operational (LBVH + traversal + intersection + shading).
- Vulkan ↔ CUDA external memory interop (Windows priority).
- **Demo: H1 pour water + wring towel.**
```

Existing v1.0 (`v10`) phase plans: p01 coupling-stability, p02-04 sim2real-N3, **p05 CUDA-RT optimization**,
p06 Vulkan↔CUDA interop, **p07 H1 pour-water**, **p08 H1 wring-towel**, **p09 exit-gate + retrospective + public v1.0.0**.

Key fact for framing: the self-written CUDA RT (built v0.7 p12/p13, throughput-tuned v10-p05) is today a
**sensor renderer** — RGB/depth for the policy at ~50→150 MRays/s. **Nothing in the current plan targets
photorealistic *beauty* renders.** The renderer is bound by a master-plan **pillar** (Round 9, decision 4:
"Self-written CUDA ray tracer; no OptiX").

## 2. The two owner-level decisions (do not self-decide)

### A. Render-quality bar — what must the homepage *prove*? → **DECIDED: A1**

The real axis is **not** "which renderer" — the self-written-RT-no-OptiX pillar + the project's
"from-scratch everything" identity make the plain reading decisive: **the homepage should prove Nuka
renders its own showcase.** Owner set the **quality bar**:

- ✅ **A1 (CHOSEN):** self-written CUDA RT → **"真实感" tier** = PBR materials + shadows + AO / soft-GI
  + denoise + anti-aliasing + tonemap. Pillar-consistent. Medium scope → **a new `v10` offline-render phase**
  layered on the sensor RT. *(This is the homepage-rendering path for ALL demos.)*
- ⛔ A2 (not chosen): self-written CUDA RT → film-grade path tracing (full GI, caustics). Large scope; v1.0-schedule risk.
- ⛔ A3 (not chosen): external offline renderer (Blender/Cycles) for beauty shots. Deviates from the self-written-RT pillar.

Owner's answer to the discriminator ("must the homepage demonstrate Nuka's *rendering*, or only its
*physics*?"): **rendering — Nuka renders its own showcase.**

### B. Demo set → **DECIDED: all four new demos + keep H1 pour/wring**

- ✅ **Rigid collision — convex-pile / stack collapse** (flagship). Exercises v0.7's **p04 LBVH + p06 V-HACD + contacts**.
- ✅ **Rigid collision — domino chain / Newton's cradle** (classic supplement; visually legible side-by-side).
- ✅ **Fluid + rigid — dam-break-over-obstacles** (canonical PBF benchmark; **p10 PBF + p11 K2/K3**).
- ✅ **Fluid + rigid — buoyancy / float-sink** (Archimedes; **p11** two-way coupling stability + volume conservation).
- ✅ **Keep** existing **H1 pour water + wring towel** (re-rendered through the A1 path).

## 3. Cross-cutting requirement (NOT just "add two scenes")

"所有 demo 都要真实画面" retroactively **raises the render bar on every demo that lands on the homepage** —
including the **v0.7 grasp-cup** demo (p16) and the **v10 pour-water / wring-towel** demos, not only the two
new ones. This is a **render-arm requirement** touching v0.7 **p12/p13** + **v10-p05** (RT optimization), and
it adds an offline/beauty-render path (per decision A). Treat it as a horizontal exit gate, not a phase add.

## 4. Proposed new `v10` phases (feed `v10-p09` exit gate)

> Phase numbers indicative; final numbering when v1.0 specs are authored. Each carries the §5.6 GPU-only
> hard-constraint callout and a named downstream consumer (integration-debt discipline).

| New phase (proposed) | Content | Exercises (consumer mapping) |
|---|---|---|
| `v10-pXa` rigid-collision showcase | (i) convex-pile / stack-collapse + (ii) domino-chain / Newton's-cradle scenes + demos + regression (no interpenetration; rest-state; momentum transfer) | p04 LBVH, p06 V-HACD, contact rows |
| `v10-pXb` fluid+rigid showcase | (i) dam-break-over-obstacles + (ii) buoyancy float-sink scenes + demos + volume/energy regression | p10 PBF, p11 K2/K3 coupling |
| `v10-pXc` photoreal render path (**A1**) | offline/beauty-render mode over the self-written CUDA RT (PBR + shadows + AO/soft-GI + denoise + AA + tonemap); renders ALL homepage demos incl. re-rendering grasp-cup + pour/wring | v0.7 p12/p13 RT, v10-p05 |
| `v10-pXd` GitHub homepage showcase | README hero gallery / GIFs / short reel of all demos; CI-reproducible render command | all of the above |

## 5. Proposed §7 v1.0-block rewrite (FOR OWNER §13 APPLICATION — do not auto-edit)

```
### v1.0 – S2 Polish
- K2 + K3 cross-system coupling working for rigid↔soft, rigid↔fluid, soft↔fluid.
- Sim-to-real noise N3 key items.
- Self-written CUDA RT pipeline operational (LBVH + traversal + intersection + shading).
- Photorealistic "真实感" render path over the self-written CUDA RT (PBR + shadows + AO/soft-GI    ← NEW (decision A1)
  + denoise + AA + tonemap); EVERY homepage demo rendered through it.
- Vulkan ↔ CUDA external memory interop (Windows priority).
- **Demos (all rendered photorealistically via the A1 path, showcased on the GitHub homepage):**  ← CHANGED
  - H1 pour water + wring towel                          (existing; re-rendered through A1)
  - Rigid collision: convex-pile / stack collapse        (NEW)
  - Rigid collision: domino chain / Newton's cradle      (NEW)
  - Fluid + rigid: dam-break-over-obstacles              (NEW)
  - Fluid + rigid: buoyancy / float-sink                 (NEW)
```

## 6. Open follow-ups
- After owner confirms decision A + trims the demo set: finalize this doc, author the new `v10-p*` phase specs,
  and hand the §5 §7-rewrite to the owner for §13 application.
- Reconcile with the already-known v1.0 doc-discrepancies (entry-plan §6-3): the stale "OptiX" row at master
  plan `:535` / §10 (renderer is self-written, not OptiX) — fold into the same §13 amendment.
