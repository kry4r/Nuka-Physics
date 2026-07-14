# BDX 感知重训 — 后续任务清单与记忆索引

更新：2026-07-14。状态：**WORKTREES CONSOLIDATED**。所有 BDX/scene/T1/T2/SDF
工作已进入本地主干 `windows-editor`，清理后只保留 `/root/Nuka-Physics`；未推送。

---

## 0. 当前主干快照

| 项 | 状态 |
|---|---|
| 主干 | `windows-editor`，包含原主干 WIP、oneshot scene v2、T1、T2、SDF/solver 优化与真物理 probes |
| T1 BDX 建模 | 已合入：body-local collision frame 通用修复、MJCF placeholder mass 修复、质量/惯量审计报告 |
| T2 media 提速 | 已合入：按需 contact-wrench readout、MpmXpbd MPM slice 不占 contact slots、training-coarse profile |
| SDF/solver 优化 | 已合入：SDF host mirror 去冗余、紧凑 row/arena、稀疏 P2G/stress 预计算、稳定排序、island/Jacobian 通用优化 |
| 场景 | 主入口 `examples/scenes/bdx_oneshot.nks/.nka`；训练粗粒入口 `bdx_oneshot_training_coarse.nks/.nka`；rigid-only 入口 `corridor_nomedia.nks/.nka` |
| 构建 | `/data/xtzhang25/_work/activate/build-main-consolidated`；Python staging `/data/xtzhang25/_work/activate/python-main-consolidated` |
| GPU 规则 | 只允许物理 GPU1；所有 GPU 工作严格串行 |

主干 fresh 验证：

- full scenario：`103` 项，`102 passed / 1 asset-gated skipped / 0 failed`；
- cloth FNV：`777503423208024307`；
- Go2 owner value：`0.923080623`（既有 tolerance 红语义，数值未动）；
- MPM：sand35 `h=0.0395 / r=0.0740 / angle=28.3`，footprint
  `0.0214→0.0215`，cohesion `h=0.1195`；
- BDX coarse 8-step SHA：
  - pos `95b30dd9f9a52a198f22a03d9276637d9bdc512db6b4a3a5ae9b487cff36ff7c`
  - vel `69ae2be5cc643cae69e48ab75eb23dde36c3f38bc3a6f912d34450b680c2ec09`
  - link `faee5e272126dd9d8915f9dbf0ebdf9f3c99cd8a7ed70ce0f93f864c4d217c65`
- coarse 600-step：NaN/grid escape/lane escape 均为 `0`；
- coarse N=1/64/256 fresh timing：`13.69 / 26.82 / 69.58 ms/step`；N=256
  CUDA delta 约 `4.73 GiB`。

## 1. 下一主线：只在主干继续

1. **T3 传感器（depth + RGB 共设计）**
   - depth-only primary-hit fast path；
   - MPM/media 进入传感 BVH（每 medium 一个粒子球 BLAS/等价通用表示，不能逐粒 instance）；
   - BDX 真机头部相机：训练 64×48 depth，评估/出片/蒸馏 640×480 RGB；
   - obs 接线与 RL media scene build 入口；
   - 开工前复测真走廊 sensor 吞吐、instance cap、rigid-only 场景资产配对。
2. **T4-A rigid corridor depth policy**：N=4096，直接在锐边刚体走廊训练；gate 为
   真闭环下完两个 5 cm 台阶、不摔、均速 ≥0.12 m/s、到达 x≥3.8 m。
3. **T4-B full-media fine-tune**：N=64–256 warm start；碎石没脚走、留下持久足迹，
   rigid 段无回退。
4. **T4-C RGB student**：teacher → DAgger；640×480 → 224² → frozen
   SigLIP-B/ViT-S embedding，复用到 π0.5/IHI 轨道。
5. **T5 demo 完善与终版视频**：不规则碎石、光滑绳管、好看的双绳石板、提亮远端、
   鸭材质去塑料/去噪、封天带；真物理连续一镜到底，逐帧检查。

## 2. SDF 后续方向（不得 BDX 特化）

- 以 mesh bytes + voxel/band/decomposition/scale + cooker/format version 为 cache key，
  预烘焙进 `.nka` chunk 或 content-addressed cache；
- 同 mesh/参数跨 instance、env、world 共享一个 device payload；
- 基于真实 query trace 比较现 sparse layout 与 Morton 8³/16³ brick atlas/page table；
- FP16/定点量化必须先测 SDF、normal、depth/force 误差并重跑冻结门；默认 FP32；
- 当前真实共享 payload 约 9.27 MB，不因 1.1 GB host 估算盲目重写；先证 query/cache
  瓶颈，再决定 sparse brick/quantization。

## 3. 持续硬规则

- 不调用 skill；不再创建 worktree；只用 GPU1，GPU 作业串行；
- 所有编译、cache、日志、渲染与临时产物放 `/data/xtzhang25/_work/activate/`；
- git 前 `export PATH=/root/.nuka-toolchain-gcc14/bin:$PATH`；commit body 含
  `[skip ci]`，禁 `Co-Authored-By`，不推送；
- solver/MPM/SDF 只接受通用 ONE-path/provider/geometry contract，禁 BDX、
  MpmXpbd 或 demo mode 特化；
- 不新增 host/unit test；只保留可反复运行的 scenario/pipeline/regression gate；
- 冻结金丝雀未经 owner 明确批准不得 regen。

## 4. 路径速查

| 用途 | 路径 |
|---|---|
| 主仓 | `/root/Nuka-Physics` |
| 主构建 | `/data/xtzhang25/_work/activate/build-main-consolidated` |
| Python staging | `/data/xtzhang25/_work/activate/python-main-consolidated` |
| 验证日志 | `/data/xtzhang25/_work/activate/tmp/main_consolidated_*` |
| v9 checkpoint | `/data/xtzhang25/_work/activate/out/bdx_walk_v9/nn/last_bdx_walk_v9_ep_2000_rew_447.30893.pth` |
| 场景 author | `examples/demo/bdx_oneshot_author.py` |
| 真闭环 harness | `examples/demo/bdx_oneshot_realsim.py` |
| 性能 pipeline | `examples/demo/bdx_oneshot_perf.py`、`sdf_world_pipeline_profile.py`、C++ `nuka_solver_arena_profile` |
| 分镜/规格 | `docs/storyboard/nv-BDX/` |
| 批准任务书 | `docs/plans/2026-07-09-bdx-perception-retrain-plan.md` |

## 5. 新会话恢复指令

继续 BDX 感知重训与一镜到底 demo。先完整读：

1. `/root/.claude/projects/-root-Nuka-Physics/memory/bdx-oneshot-demo.md`
2. `/root/Nuka-Physics/NEXT-TASKS.md`
3. `/root/Nuka-Physics/docs/plans/2026-07-09-bdx-perception-retrain-plan.md`

所有 worktree 已收敛，只在 `/root/Nuka-Physics` 的 `windows-editor` 主干继续；先核
`git status` 与最近提交，再从 T3/BDX 真物理 demo 当前停点推进。遵守 GPU1 串行、
ONE-path、pipeline-only tests、产物只进 `/data/.../activate/`、不推送等硬规则。
