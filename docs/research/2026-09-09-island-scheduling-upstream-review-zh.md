# 岛调度与迭代提交 review

现行物理实现已提交 `b0c2816`。下文保留此前候选和失败的历史；当前性能分母及实施方向以末节为准，旧物理轨迹不用于计算新一批收益。

日期：2026-09-09。已复核总模块、总细化和性能方向 spec 的 T01c/T07/T10b/T11/T13/T21；完整 goal 保持 active，Editor 暂缓。

再次通过网络查询三家 HEAD，均未变化：Newton `3f54b4a16c8b7125e6559f2668cce2a6a7d929c5`、MuJoCo `297f5fc6592e89fc41960251e88b6cf94fd7573e`、Genesis `0ce793b42c945b6848aad624501b2ca756d3e244`。查询时间与文件 SHA256 见 `out/validation/island_scheduling_20260909/upstream_review.json`；源码沿用 `.nuka_cache/contact-index-review-20260909/` 的相同 revision。

## 上游与官方资料

| 实现 | 本批采用 | 不直接采用 |
| --- | --- | --- |
| Newton Kamino `linalg/sparse_operator.py` 的 active rows/cols 与 block sparse 接口 | 将有效执行集合与稳定行身份分离；共同算子消费明确的有效范围 | 不替换求解模型、精度或框架来声称加速 |
| MuJoCo `engine_solver.c` 的 `solPGS`、`nefc`/`efclist` 和 island 参数 | 仅遍历求解所需的有序行，保留块约束整体更新与岛内顺序 | 不照搬 CPU 循环或修改接触松弛/投影来降低质量 |
| Genesis `rigid/constraint/island.py` 的确定 root、DOF/constraint spans | 以 canonical 行集合验证并行岛；各岛保持独立写入范围 | 不引入其单岛专用路径，不以 append 顺序作为物理行序 |
| NVIDIA CUDA Best Practices 与 Blackwell tuning guide | 依据实际 kernel/block/shared memory 查询可驻留 block 数，使用通用 grid-stride 遍历 | occupancy API 是静态资源上限，不能宣称为实测 occupancy；不假定 RTX 5080 等同 B200 |

官方资料链接与原文 SHA256 见 [全路径 review](2026-09-09-performance-directions-upstream-review-zh.md)。本批未引入新的外部库；CUB 继续承担已有排序/索引。cuSPARSE/cuSOLVER 不能减少未调用矩阵库的无效 row dispatch；需要质量算子基线后再评估。

## 冻结基线与归因

当前基线为 `8211bdf` 加 benchmark 的计时外 island 观测，冻结于 `out/validation/island_scheduling_20260909/baseline_frozen/`。source SHA256 为 `438f6c364a6f58918b3bcddf8ed471e669c784736592d66768baaf0baf20a5d4`，benchmark 为 `c23fb4a7369a93653e43880ed9ab21aaeb7a465e2d324020a119a9e757a0507d`；物理库仍为 `d88839ba5f05a581c996a44aec0839cefd0551e0def34d73812adfc7eb288dfe`，与上一批 final 完全相同。原接触索引之前的基线继续保留，用于检查尚未解决的 eager 回退。

现有 Systems trace 的 E=256 block 岛求解为 gridX=72,704、blockX=32，154 registers/thread、56 B dynamic shared、1.612 ms/step。新增完整 pipeline 观测在 251–450 步的九个采样点均为 256 个岛，每岛 33–60 行、一个 articulation；所有 active rows 被同环境岛恰好覆盖一次且严格保持升序。初验整步为 5.349 ms，质量和全部 450 步 wrench 身份通过；这是观测初验，不替代五进程性能验收。

`launch_attribution.json` 按既有完整 API trace 的 event marker/correlation ID 统计：新版每步 XPBD distance/bend 分别 144/336 次 launch；两者 host launch API 合计 7.672 ms，GPU kernel 合计 0.933 ms。旧版相同部分为 7.753 ms host 与 0.919 ms GPU。这定位了当前主要提交成本，但没有证明未 profiling 的 eager 回退根因。单次最长 launch 约 4 ms，不能将其全部当作 kernel 自身计算。

## 改造前决定

1. 采用通用 bounded grid + grid-stride 岛遍历。网格上限由 kernel 的静态资源和设备 SM 数取得，不能写场景/机器人尺寸，也不把 live count 下载到 host。大于网格的岛由下一轮遍历处理，跨岛共享内存复用必须有 block barrier。测量判断空 block 的真实成本，不能由 72,704/256 的比值直接推导加速比。
2. 采用求解前一次构造的有序执行行集合。块约束的切向行由 normal row 整体更新，现有循环在每次迭代重复进入这些不执行更新的行；将其排除于执行顺序，保留完整 island rows、稳定 row ID、Jacobian/lambda 索引、接触读出和 pseudo 清零。静态、动态、scalar 与 warp 调度使用同一选择规则，不增加求解器或按 side 种类增加路径。
3. 优先复用现有、在动态调度中未使用的 `pd_solve_scratch`，按每个 island 的原 row span 分配不重叠的执行行范围；命名并记录索引域，禁止硬编码机器人或点数量。所有行运算表达式、A→B 次序、warm-start、速度/位置迭代次数保持不变。
4. 本批统一构建及完整 pipeline 验证，补必要的多岛/巨岛、static reference、reset、公共 API/渲染及 memcheck；不新建一批镜像实现的单测。五进程比较 graph/eager E=1/16/256，保存失败候选；回退超过 5% 必须处理，低于初始 10% 收益门槛时继续剩余热点。
5. XPBD 的提交开销仍需自己的通用执行架构设计及网格尺寸基线。保留 eager/graph 明确语义，不用隐式 graph 或某一布料尺寸的 fused fast path 获得表面收益。有效 row/J 复用、残差及完整 T11/T13 尚未完成。

## 同批候选检查与后续决定

调度候选 `candidate_v1_frozen` 的 source 为 `a5175d3825477165f25a867eb11c821bb430ba3f3b3e2a763c69a4520113f75c`。初验 graph E=1/16/256 为 2.421/2.709/5.233 ms，状态和 450 步 wrench 与旧版身份相同；这些是单进程初验。最新 Systems 中岛求解为 1.452 ms，占 kernel 时间 30.31%，静态寄存器从 154 增为 165/thread。仅减少无效调度尚未达到整步 10% 门槛，因此继续本批 CUDA 内存读取优化。

改动前决定：将 articulation dot 的连续 Jacobian/qdot 操作数分配给现有 warp 读取，随后 shuffle 两个原始操作数到 lane 0，按原索引顺序执行相同的 FP32 累加。不能先计算乘积再求和，也不能换为树形 reduction；标量执行宽度 1 使用同一个算法，保持实际/位置速度与两侧顺序。warm-start 不需要速度残差，可跳过没有结果消费者的点积。这是对现有通用行算子的读取方式优化，不新增 solver 或按 DOF/机器人分类的路径。继续以完整轨迹身份和实际性能确认收益，不能由访存形式直接宣称提速。

联合验证扩大到 50 个已有场景项：49 passed；20 机器人台阶用例报告 `worst_sink=-0.0892994255 m`。冻结基线单独重跑得到完全相同的失败和值，证据为 `terrain_baseline.*`。该用例当前不能作为有效性能负载，保留在接触/地形测量方法排查中，不调整阈值或隐藏失败。主 robot-cloth-fluid 有效基线不变。

有序合并读取的第二个候选 `candidate_v2_frozen` 虽然保持物理身份，但初验 graph 为 2.437/2.723/5.265 ms，未取得额外净收益，因此舍弃这部分源码。首个候选保留为可恢复的调度改造，尚未声称完成正式五进程验收。

## XPBD 执行方向的补充审查

再次联网核对后，Newton 更新到 `1128af71407e002d15a8cfb180b35158cf64c42f`；MuJoCo/Genesis 仍为上文 revision。已 fetch 并阅读 Newton 的完整增量：小尺寸 MJCF sphere/cylinder/capsule/ellipsoid 的显式质量缩放参考下限改为体积相关下限，并补相应导入验证；XPBD/稀疏算子文件 SHA256 未变。此次不改变 Nuka 导入或接触质量模型。精确查询命令、源码和官方网页 SHA256 见 `out/validation/island_scheduling_20260909/xpbd_upstream_review.json`。

Newton `SolverXPBD.step` 在迭代内发射不同 family、累加 `particle_deltas` 后应用；其执行顺序和更新模型与 Nuka 颜色 GS 不同。Genesis `pbd_solver.py` 的 stretch/bending/volume 将迭代放在 family kernel 内，使用 `dpos` 累加和 apply；借鉴缩减提交边界的思路，不复制其累加方式和弯曲公式。MuJoCo `solPGS` 的约束块与有序有效行继续作为保留物理顺序的参照。这些是源码审查，未实测外部引擎速度。

NVIDIA [Cooperative Groups](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html) 明确 grid 同步要求 cooperative launch，全体线程参与并保证同步前写入可见；[Runtime execution](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html) 要求网格不超过 occupancy × SM 的驻留 block 上限，且设备支持 cooperative launch。本机已实查支持。采用同一通用设备颜色迭代模板，保持 family/颜色/迭代顺序和投影公式；先补大网格原实现基线，再实施，具体文件、能力/失败契约及矩阵已写入性能方向 spec 第 4.3 节。

## 正确物理基线与 CUDA 实施

统一粒子工作位置和材料/接触交替求解已完成公共边界验收，见 [物理记录](2026-09-09-particle-topology-review-zh.md)。冻结版本为 `particle_contract_closure_frozen`，source `818337451b329cd25f166a3a8c7c17a69e2fd72ee574864d2d7af1aba88d58f4`，benchmark `54141e539ba920caf7649784bea6fd799934d7a5e25aec58ef3bdcd8bb2ca5f5`，library `7e3de5e12e5a87e63db63008ea42140c082bdf653a8cbd30792934372454034f`。五个独立进程的原始数据位于 `out/validation/island_scheduling_20260909/particle_contract_baseline/`。

| 环境数 | graph 中位数 / 范围 ms | eager 中位数 / 范围 ms |
| --- | --- | --- |
| 1 | 3.3656 / 3.3542–3.3887 | 15.1817 / 8.0212–15.3465 |
| 16 | 3.6718 / 3.6710–3.6762 | 10.8485 / 7.3652–13.0165 |
| 256 | 6.4022 / 6.3983–6.4126 | 13.9409 / 12.4253–14.5610 |

全部进程的完整 450 步质量门、最终状态、逐步 wrench、canonical island 身份与重放通过。最大长度应变 1.838344%，最坏每环境 RMS 0.203045%，原预算 5%/1% 不变。eager 波动很大，候选必须用交错进程对照，不能只与这里的中位数相减。

新的 Systems trace 只取 200 个 timed steps。E=256 graph kernel busy 为 6.0191 ms/step，其中岛求解 1.9667 ms（32.67%）、bend 0.6951 ms、distance 0.2587 ms。实际岛 launch 仍为 gridX=72,704、blockX=32、157 registers/thread、56 B dynamic shared；连同 static shared 为 72 B。E=1 eager 的 host API 为 9.7272 ms/step，其中常规 kernel launch 为 9.2039 ms；GPU kernel busy 为 3.1500 ms。这些 profile 数据用于归因，不能替代无 profiler 的性能结果。

本批重新 review 三份 spec 和三家最新源码记录，revision 为 `1128af7` / `297f5fc` / `0ce793b`，与 `particle_topology_upstream_review.json` 相同。采用通用拓扑、工作量与设备资源决定的调度，禁止 demo 参数或隐性质量降级：

1. `backend_cuda/ops/solve_rows.cu` 的 block 以 grid-stride 处理 live islands，设备驻留资源限制静态 grid，上次岛写回后同步再复用 shared。保留岛内行序和 `continue_impulses`，没有复制旧候选的有序行缓存或合并读取改动。
2. `backend_cuda/ops/particles.cu` 提取原四类投影公式为 device projector；一个模板按原颜色/迭代顺序执行，每次 op 每个非空 family 一次 cooperative launch。所有线程参加颜色 barrier，不跨材料/接触交替边界合并 24 轮。
3. `backend_cuda/launch_grid.cuh` 缓存按 kernel/device/block/shared memory 查询的驻留上界；能力、输入和资源检查在 XPBD 写入前完成，所有 launch 失败显式传播。`launch.cuh` 为普通与 cooperative launch 提供统一 stream 入口。核心模型与公共 API 不引入 CUDA 类型或能力要求。

当前 CUDA 批次已完成 [联合验收](2026-09-09-cuda-execution-validation-zh.md)：eager 五进程中位数下降 39.02%–55.21%，graph 增加 0.22%–2.38%，内存不变。完整轨迹/长度门、现有材料、公开 graph/reset/渲染及完整 memcheck/synccheck 通过。E=2048 eager 的 2048 个 live islands 跨越 1008-block 驻留网格，完整状态与逐步 wrench 一致。soft-tet 既有失败和大环境 ContactWarmStart graph capture 失败如实留存；仍需继续 GPU 热点与完整引擎功能。
