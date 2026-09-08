# 全路径性能方向与上游 review

日期：2026-09-09。对应用户要求：先冻结基线、确定方向，再实施架构与 CUDA 优化。已 review 两份总 spec 的基线、执行/内存、接触、粒子/MPM、API/传感器及 M11；Editor 暂缓。

## 已有证据与适用范围

当前改造前基线 `ea807fe` 和执行/workspace 候选分别保存在 `out/validation/execution_workspace_20260908/baseline_frozen/`、`candidate_frozen/`。候选源码身份 `a2b0a2493d482d1b19e9d616007a1e05d018c154dcd4605c84e999e4071fed30`，benchmark SHA256 `a0b5b1acfc4d9f2379e2a364e46bd423098422653127b29166cb68e9f09b6a6e`。

固定 robot-cloth-fluid 为 Go2、自由刚体、XPBD 布料与 PBF 流体；每次 250 步预热、200 步计时，另完整重放 450 步。历史 1/16/256 环境的五进程 graph GPU 步时中位数分别为 3.066/3.382/6.872 ms，原 eager 基线为 11.565/11.891/16.807 ms。后续容量检查发现 E=256 的共同副本错误，该组数据仅保留为历史，不能计算同质量加速比。修正基线、候选和最终结果以 [执行/workspace 验收](2026-09-09-execution-workspace-validation-zh.md) 为准。小环境 eager 的历史范围重叠，不能将其中位数差异当作已消除噪声的结论。

这些数据只覆盖一种耦合环境。修正后的正式对照已完成：新 graph E=1/16/256 为 3.073/3.367/6.499 ms，相对正确旧基线为 4.053×/3.536×/2.408×；完整状态/缓存、新旧版本和容量物理等价通过。MLS-MPM、SDF、光追、异构多树及完整 T01 仍需各自基线；不得把 PBF 或 headless 图像正确性替代这些性能结果。具体实施顺序见 [细化方向 spec](../plans/2026-09-09-performance-directions-detailed-spec-zh.md)。

## 最新上游及决定

以下 revision 于 2026-09-09 通过 `git ls-remote ... HEAD` 查询，相关源码以 `git show revision:path` 归档到 `.nuka_cache/performance-review-20260909/upstream/`，manifest 记录每个文件 SHA256。Newton 相对前次 review 仅 `uv.lock` 改变；MuJoCo 包含碰撞 pair 表达及 Warp/JAX graph 缓存契约更新，已读取新版本，未仅查看旧工作树。

| 上游 | Revision | 读取实现 | 采用与限制 |
| --- | --- | --- | --- |
| [Newton](https://github.com/newton-physics/newton/tree/3f54b4a16c8b7125e6559f2668cce2a6a7d929c5) | `3f54b4a16c8b7125e6559f2668cce2a6a7d929c5` | Kamino `linalg/sparse_operator.py` 的 active rows/cols、`solvers/warmstart.py` 的索引；Style3D `collision/bvh/bvh.py` 的分配/build/rebuild/refit 分离 | 采用稳定身份与活跃执行索引分离、明确 workspace 生命周期。保持 Nuka 现有约束/摩擦运算顺序，不能照搬不同求解器的结果契约。 |
| Newton | 同上 | `implicit_mpm/solver_implicit_mpm.py` 的活跃网格、容量失败和 scratch；`xpbd/solver_xpbd.py` 的约束/粒子迭代 | 借鉴网格生命周期和通用调度。隐式 MPM 与 Nuka 显式 MLS-MPM 的本构/迭代不同，仅作架构参考，速度不可直接比较。 |
| Newton | 同上 | `geometry/sdf_texture.py:89` 附近的两类插值及稀疏 subgrid | 优先减少重复角点读取和空间无效工作。硬件线性过滤使用有限精度权重，不能未经接触误差验证就替换 FP32 插值。 |
| [MuJoCo](https://github.com/google-deepmind/mujoco/tree/6b5c05d11facd04d5e8c8496052652ec54169568) | `6b5c05d11facd04d5e8c8496052652ec54169568` | `engine_core_constraint.c:418` 的 `nefc`/CSR Jacobian，`engine_solver.c` 与 `engine_core_smooth.c`；`engine_collision_driver.c` 的显式 pair 分量/容量 | 采用有效长度、稀疏算子契约和明确索引单位。CPU 算法不能冒充 CUDA 同性能实现，16 位 pair 编码不照搬到通用 Nuka 容量。 |
| MuJoCo | 同上 | `mjx/.../warp/_src/jax/ffi.py` 中 graph 地址复用、staging 和缓存键 | 所有影响图结构/输入地址的属性属于执行计划身份。staging 有额外显存与每次复制，须与已有稳定 arena 对照后选择。 |
| [Genesis](https://github.com/Genesis-Embodied-AI/Genesis/tree/0ce793b42c945b6848aad624501b2ca756d3e244) | `0ce793b42c945b6848aad624501b2ca756d3e244` | `rigid/constraint/island.py` 的 DOF span 和 `_sort_island_contacts` | 采用活跃岛/接触范围和确定次序。其逐岛 insertion sort 不直接用于 Nuka 大接触集合；不能让并发 append 次序决定 GS 顺序。 |
| Genesis | 同上 | `mpm_solver.py:135` 的 dirty-grid 生命周期、材料需要 SVD 的判断；`couplers/legacy_coupler.py` 的界面反作用 | 采用明确的活跃节点生命周期和数据驱动材料能力。其前向 sparse reset 不覆盖梯度状态，Nuka 必须单列 tape/checkpoint 契约；不照搬按 solver 组合分开的耦合路径。 |
| Genesis | 同上 | `vis/raytracer.py` 的 LuisaRender 适配与刚体/粒子更新 | 仅作资源更新边界参考；LuisaRender 的采样/着色与 Nuka 不同，未实跑等质量输入前不声称渲染加速比。 |

## CUDA 与加速库资料

官方网页的原始内容、文本、下载时间和 SHA256 保存在 `.nuka_cache/performance-review-20260909/manifest.json`。旧 CUB API 网页返回 404，已保存该失败，并改为读取官方 CCCL 精确 revision 的头文件。

| 资料 | 对优化的约束与使用决定 |
| --- | --- |
| [Blackwell tuning guide 13.3](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html) | RTX 5080 的计算能力为 12.0，不能套用 B200/10.0 的 shared memory/warp 上限。以实际设备属性、寄存器、occupancy、合并访存和内存吞吐选择 block/tile；不因降低寄存器数而引入更多 spill。 |
| [CUDA best practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) | 先减少全容量工作与传输，再调局部指令；小环境区分 host launch 与 GPU 执行，批量环境区分带宽、计算和串行依赖。 |
| [Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html) | Systems 定位完整关键路径，Compute 采集热点的 roofline、带宽、寄存器、stall、occupancy。replay 改变耗时和 cache 状态，不能把 profiler 计时当正式五进程结果。已确认本机 Compute 2026.2.1 可调用，硬件计数器权限仍需实际采样证明。 |
| [CCCL DeviceSegmentedSort](https://github.com/NVIDIA/cccl/blob/8c6fc52fc780c778e2e4ca41fcf9d5a0cd4379ce/cub/cub/device/device_segmented_sort.cuh) | revision `8c6fc52fc780c778e2e4ca41fcf9d5a0cd4379ce`。稳定 scan/select/sort 用于活跃索引、端点 CSR 和网格集合；workspace 在创建时查询并分配，捕获中不分配。最新接口变化不能自动放宽当前其他设备数组的索引上限。 |
| [cuSPARSE](https://docs.nvidia.com/cuda/cusparse/index.html) | 候选用途是已构建的通用 CSR/BSR 算子。需同时计入结构构建、格式转换、workspace 和调用成本；文档对 transpose 的逐位再现性无保证，不能直接替换严格 D1 的顺序累加。 |
| [cuSOLVER](https://docs.nvidia.com/cuda/cusolver/index.html) | 仅在相同质量矩阵/残差契约下评估 batched 分解。小矩阵调用成本可能超过自身 CUDA；需记录确定性模式及库版本，不默启 TF32/混合精度。 |
| [OptiX](https://developer.nvidia.com/rtx/ray-tracing/optix) | 可评估 RT Core 遍历能力，但现有实现是自研 CUDA tracer。先建立相同几何/AOV/采样/光照的基线并优化现有结构；SDK 可用性、构建成本、stream/传感器交付和图像误差齐全后再决定是否接入，当前未引入依赖。 |

## 实施选择

1. 先扩展固定耦合环境的 benchmark 配置，冻结 MLS-MPM、SDF、渲染及端到端结果，补足每步物理质量和活跃量观测。基线采集可改工具，不能先改被测物理算法再声称是原基线。
2. 第一组共享架构方向是稳定 active-row/endpoint 索引、确定性读出 gather、活跃岛调度及其 workspace/失败传播。当前 `LinkContactWrenchKernel` 每个 link 扫整个 row capacity，容量扩张成本提供了直接依据；最终优先级由最新完整 profile 确认。
3. MLS-MPM、XPBD、SDF 和渲染分别按自身热点推进，相关数据结构和生命周期成批改造。每组在代码修改前填清采用方案、反例、预期量化指标和停用条件。
4. 保持一个通用物理求解路径；场景 fixture 只用于输入构造，不进入引擎 dispatch 或容量捷径。任何迭代顺序、本构、采样或精度变化须作为质量契约变化单独验证。

修正后最终候选的 Systems 记录通过 correlation ID 精确选择 901 次 graph launch 中预热后的 200 次定时调用。kernel busy 合计 6.226 ms/step，其中岛求解 1.644 ms（26.40%）、接触力读出 1.262 ms（20.28%）、XPBD bend/distance 合计 0.911 ms（14.63%）。这是归因数据，不替代无 profiler 的五进程计时；旧 v3 归因留在原证据文件，不作为最终质量基线。

Compute 实际采样返回 `ERR_NVGPUCTRPERM`，日志见 `candidate_v3_wrench_counters.log`。当前驱动不允许读取硬件性能计数器，未获得带宽/stall/实际 occupancy；Systems 已给出资源静态信息及时间。此限制不阻塞已有充分证据的活跃数据结构优化，后续 CUDA 指令级结论仍需补齐计数器或其他直接证据。

本 review 不是全路径性能验收，也不是外部引擎实测报告。每轮达到初始 10% 门槛后继续处理已测出的主要热点；未建立基线的模块保留待完成状态。

## 容量检查发现的基线缺陷与修复决定

严格比较发现：E=256、默认容量中 env=15 的关节/粒子状态与其他相同环境不同，扩容后恢复相同。冻结旧版与候选在各自同容量下逐字节一致，因此该问题早于本批排序/graph 改造。`capacity_identity/comparison.json` 保留字段差异；旧 benchmark 的 valid 门仅检查有限性、env_status、reset/重放和两类接触存在，未检查环境副本等价，不能将它视为完整质量证明。

源码定位到 `PairSortFillKernel`：排序项数为 `E * rigid_slot_cap`，快照写入却使用 `E * full_slot_stride` 域的索引。E=256、rigid_cap=32、slot_stride=1108 时，快照之后的 prefix 相对范围为 `[132096,133120)` 字节；env=15 的快照从 132960 字节开始，和 prefix 重叠。原世界为整体 scratch 分配了更大内存，因此该错误在同一个 allocation 内部，既有 memcheck 不一定发现。

改造前决定：快照与 permutation 都改为紧凑的 rigid-sort 索引域；快照按两个 u32/排序项划分，创建预算直接读取 pipeline 的实际排序容量，并将 workspace bytes 绑定到算子检查。保持输出候选/接触身份与排序次序。完整 benchmark 增加物理字段的环境副本等价门；保存旧失败，分别冻结仅修正此缺陷的基线与最终候选，再进行相同质量比较。该修复先于下一批性能改造。

此前 E=1/16 的比较仍是历史证据；E=256 原五进程结果不能用来宣布同质量加速。新冻结版本已经完成 E=1/16/256 五进程、容量×1/2/4 权威状态逐字节等价、公共渲染和完整 pipeline memcheck，最终身份与限制见执行验收。空容量扩张后的 E=256 步时仍为 18.721/35.315 ms，支持继续优化活跃数据结构。
