# Nuka-Physics 引擎完善目标

状态：active。创建：2026-09-08。依据用户最新要求，后续成批修改多个相关模块，再统一验证；Editor 暂缓。

## 目标与范围

完成 [模块 spec](../plans/2026-09-07-physics-module-specs-newton-port-zh.md) 与 [细化 spec](../plans/2026-09-08-physics-optimization-detailed-spec-zh.md) 中尚未完成的引擎功能和性能工作，持续回写规格、进度、TODO、实际结果和限制。旧目标的 demo 调优、规格细化及开始实施已验收；本目标接续其全部引擎剩余项，不以首批完成代表整体完成。

已验收基线：π0.5/G1 demo 与主页、完整环境 reset、三轴重力和自由体外力（`0953516`）。本批新增稳定自由旋转、创建拓扑防护、确定性风阻功能、required op/首错停止及邻域容量池基础；见 [动力学整批验收](../research/2026-09-08-dynamics-pipeline-validation-zh.md)。原 demo、reset、外力轨迹和冻结库保留。

当前进度：执行/workspace 批次主 pipeline 33 passed、0 skip，公共耦合/多环境/相机通过，完整 16 环境 graph memcheck 0 errors。公共 graph、失败缓存、LBVH workspace、活跃 cache 排序/merge 和字段预算已实现。容量检查发现旧版也存在的 pair snapshot 覆盖，现已通用修复；原 E=256 分母保留为失败证据。修正后 65 个独立进程、容量物理等价和 8 对公共渲染图通过，新 graph E=1/16/256 为 3.073/3.367/6.499 ms，结果见 [执行验收](../research/2026-09-09-execution-workspace-validation-zh.md)。

## 执行批次

粒子物理批次：结构邻接/静止距离过滤、统一投影位置、材料与接触交替求解已实现；既有 24 次 XPBD 和 48 次速度预算不变，投影/接触增量只计一次，固定粒子保持不动。E=1/16/256 × eager/graph 完整 450 步的最大应变为 1.838344%、最坏每环境 RMS 为 0.203045%，原 5%/1% 门通过。已有场景及 C/Python/reset/渲染验收收口，E=16 graph 完整 memcheck 为 0 errors；修正既有压板几何/表面速度和流体三维量测，原失败保留、物理预算不放宽。物理修复已提交 `b0c2816`，新源码/二进制已冻结，六组五进程分母采集完成；graph E=1/16/256 为 3.366/3.672/6.402 ms，eager 中位数为 15.182/10.848/13.941 ms，其噪声需交错对照。此批是物理算法修复，不能用旧错误物理作为性能分母。独立架构隔离已提交 `6953f4a`，无 CUDA 核心构建通过；不代表 CPU 求解功能已完成。证据见 [粒子 review](../research/2026-09-09-particle-topology-review-zh.md)。

当前 CUDA 批次已完成 [验收](../research/2026-09-09-cuda-execution-validation-zh.md)：eager 五进程中位耗时下降 39.02%–55.21%，graph 增加 0.22%–2.38%，内存相同。保持材料/接触交替、原预算和公式；完整质量/身份、既有材料、公共 API/reset/渲染、memcheck/synccheck 通过。E=2048 eager 验证跨驻留网格的完整循环；soft-tet 既有失败和 ContactWarmStart 的大环境 graph capture 失败继续处理。GPU 岛求解仍是热点，不把减少 host 提交写成全部性能完成。

当前接触索引的功能与 graph 子集已验证，见 [review](../research/2026-09-09-contact-index-upstream-review-zh.md) 和 [验收报告](../research/2026-09-09-contact-index-validation-zh.md)。有效 row/endpoint、稳定 link gather、预算/reset 契约已实现；主 pipeline 34 passed，完整 graph 和读出/reset memcheck 为 0 errors，公共报告与两组各 8 张渲染图逐字节一致。graph E=1/16/256 从 3.081/3.381/6.516 降至 2.551/2.837/5.367 ms；容量×2/4、逐步 wrench 和权威物理状态等价通过。

eager 性能仍未验收：首次五进程和两组交错采样存在超过 5% 的回退，固定 CPU 后仍未消除。普通/完整 API trace 显示 GPU kernel busy 降低、host 主要耗在 launch，尚未证明原始回退根因；不能以 profiler 数据替代正常计时或据 graph 收益关闭此项。下一批先复核有效岛/J 调度及 host 提交契约，再继续通用路径优化。

改造前已核对 spec 与 Newton/MuJoCo/Genesis，见 [执行 review](../research/2026-09-08-execution-workspace-upstream-review-zh.md) 和 [9 月 9 日性能方向 review](../research/2026-09-09-performance-directions-upstream-review-zh.md)。当前批次验收完成，后续按 [全路径细化方向](../plans/2026-09-09-performance-directions-detailed-spec-zh.md) 补 MLS-MPM/SDF/光追/端到端基线，并推进已测出的 active row/endpoint、读出、岛求解及其余热点。T06 全部错误传播、T08 密度、refit、异构、双浮基重叠和其他未完成物理/API/可微工作继续保留。

| 顺序 | 合并实施范围 | 统一验收重点 |
| --- | --- | --- |
| 基础正确性 | T02c 自由旋转、惯量/冲量/pseudo 一致性；T04a 拓扑防护；T05a 确定性风阻；T06a 必需算子/失败传播与必要诊断 | robot-cloth-fluid 完整 pipeline，角动量/能量/步长收敛，风阻功与 D1，非法拓扑与缺算子失败，API/reset/渲染/memcheck |
| 内存与执行 | T06 容量/错误契约、T09a LBVH workspace、T07 graph、T08 cache、T10 内存预算 | 真正 graph/eager 对照，接触 graph→reset→graph，溢出可读，地址稳定，完整计时边界 |
| 接触与多树 | T04b 异构映射、T12 残差与病态块、T13 共享质量算子、T15 几何/材料及 T22 关节/CCD | 多树反作用、抓取滑移与释放、薄壁/高速碰撞，质量和几何误差可解释 |
| 多物理与高级能力 | T16–T23 剩余粒子/耦合/介质能力，T24–T25 API/传感器/可微 | 同一时间层、线/角动量、确定性、接口生命周期与匹配前向的梯度 |
| 性能收敛 | T01 完整基线及 T11/T14/T19/T26 等依赖已满足的优化 | 同输入/质量的五进程对照，GPU 完成时间、显存及噪声区间；外部引擎性能结论须实跑 |

表中是组织顺序，模块 spec 的依赖和验收要求仍然有效。发现跨模块共因时成批修复；必要的首次反例和失败定位可独立运行，不为维持批次而保留已知错误。

## 持续约束

- 按用户最新要求，重心放在生产代码分析与成批改造，再用已有完整 pipeline 统一验证。已有证据足够定位时立即推进实现，不持续扩充独立测试、对照脚本或重复测量；新增覆盖只针对具体未解决失败。
- 原引擎不是物理真值。字节一致只证明确定性或既定运算次序；还须独立验收约束误差、接触/滑移/穿透、守恒和时间步收敛。发现共同错误时保留失败、修复通用契约，再冻结正确性能分母。
- 物理验收可参考 Newton、MuJoCo、Genesis 的实跑仿真结果。先对齐几何、质量/惯量、材料、约束、dt/substep、控制和坐标/单位，记录版本、原始轨迹与差异；不等价模型只比较共同可解释的物理量。参考引擎分歧用解析解、不变量和收敛证据判断，不能据旧版相同或多数一致宣布正确。
- 目标是 multi-backend。物理契约、模型/拓扑、算子依赖和公共 API 保持后端无关；架构优化、CUDA 实现调优和物理算法变化分别标注、验证与提交。CUDA occupancy/cooperative launch/stream/graph/intrinsics/库只能属于 CUDA 后端，不成为核心模型要求；无 CUDA 构建检查只证明依赖隔离，不代表其他后端物理能力已完备。
- 每批修改前 review 对应 spec，再读取最新 Newton、MuJoCo、Genesis 相关实现，记录 revision、采用与差异。
- 一个通用物理求解路径；禁止机器人、抓取、场景专用分支和固定布局捷径。
- 少新增单测；固定主环境覆盖 cook、创建、控制、step、接触/耦合、reset、读出，图像受影响时包含 headless 渲染。
- 相关修改合并构建和 pipeline 验证，避免每个小改动都重复全量测试。物理不变量和无法由主环境测量的失败才增加必要解析补充。
- 保存失败日志、源码与二进制身份、配置、质量、确定性、耗时和内存证据；缺测数据明确列出。
- 本地 commit 简洁、无合作者；保留原有工作区文件。Editor 等引擎完善后再设计。

## 性能模块的重点

MLS-MPM 已完成体积硬截断修复和 [粒子归属/容量架构验收](../research/2026-09-09-mpm-ownership-validation-zh.md)。保持原 bunny-water 输入，五进程落水 eager/graph 从 50.865/47.616 降至 48.378/44.633 ms/步，数据区从 1.64 GB 降至 71.22 MB。CUDA 传输、活跃工作和反作用读取已成批修改，继续独立验证；局部大 J、无 SDF、多 owner、有限质量和 articulation 子步反馈均未据此关闭。

MLS-MPM 按用户最新补充提升为当前 CUDA 调度批次之后的主要工作，覆盖从 P2G/应力到活跃网格、双向刚体/机器人反作用及 G2P/F 的完整路径。主负载复用 `examples/demo/mpm_water_drop_demo.cpp` 的 bunny-water（MLS-MPM），辅以已有 jelly、rigid/articulation-on-MPM 和 transfer 场景；`water_pool_demo.cpp` 的同名 PBF 场景不能替代。先记录当前输入、完整质量与性能基线/profile，再成批改生产代码，不新增独立测试框架。

性能批次后紧接着完善 MPM 耦合：无 SDF collider 的通用几何查询，多个碰撞端点的有限质量约束，刚体/关节子步反作用与速度刷新，以及 MPM↔XPBD/其他介质的完整交换链路。SDF 是几何表示之一，不应成为通用耦合的必需条件；解析 primitive、凸体和网格查询生成同一接触契约，不增设场景求解器。当前 bunny 的 SDF、独立网格地板、single owner 和延迟 articulation deposit 等边界必须随性能结果报告，不以未覆盖能力无限推迟已有有效路径的优化，也不把局部性能验收当作完整多体耦合。

按用户追加要求，进入性能模块后以极致性能优化为主要工作：先用 GPU 完成时间和 profiler 定位关键路径，再成批优化 launch/graph、调度、数据布局、带宽、访存合并、寄存器与 occupancy、活跃工作压缩和 workspace/显存复用。分别优化少环境延迟与大批量吞吐，记录瓶颈如何迁移；达到初始 10% 门槛后继续处理仍有实测收益的热点，不把门槛当终点。

优化始终遵守通用求解路径与同等物理质量。采用五个独立进程的对照，保存 GPU 设备/时钟条件、完整执行边界、p50/p95/p99、吞吐、显存和质量指标；microbenchmark 用于归因，最终收益以完整生产 pipeline 为准。每轮给出实际增益、回退和下一个主要瓶颈，尚未测量的上限不写成已达成结果。

性能范围按用户追加要求明确覆盖多环境、刚体、柔体、MLS-MPM 流体、机器人多体耦合、SDF 接触求解和光追渲染。逐项建立少环境延迟、批量吞吐与容量/显存曲线，并以固定耦合环境验证跨模块收益。物理、渲染及物理→传感器端到端分别计时；渲染保持分辨率、相机数、采样数、材质和光照一致。当前 robot-cloth-fluid benchmark 的流体为 PBF，不能据此宣称 MLS-MPM 或光追性能已覆盖。具体矩阵见细化 spec 第 3.4 节。

顺序约束：每条待优化路径先冻结通过质量检查的原基线，再用 profiler 和最新上游/官方 CUDA 资料确定方向，最后实施架构、CUDA 或库优化。发现共同正确性错误时保留失败证据、修复通用契约并重新冻结；不得从 invalid 运行中导出收益。Compute 硬件计数器当前受驱动权限限制，Systems 和 GPU 完成证据可继续使用，缺测指标保持待采集。

## 完成定义

每个剩余 spec 项均有已实现的通用契约与可重放验收，必要的性能比较达到对应质量要求，公开 API 和文档一致。接触 graph、双浮基重叠、异构、gyro、风阻、容量、薄壁、耦合和高级能力等当前限制必须逐项解决并记录；未完成或仅有间接证据时继续保持目标 active。
