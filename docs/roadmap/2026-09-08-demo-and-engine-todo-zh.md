# pi0.5 demo 与引擎优化 TODO

更新：2026-09-08。接续 pi session `01a07a9d-0d1e-7670-9978-7b2ebc41cf0c`。

当前 active goal 已接续为 [引擎完善目标](2026-09-08-engine-completion-goal-zh.md)：覆盖 spec 全部剩余项，相关模块成批修改、统一验证。旧 demo/细化规格/开始实施目标已验收关闭。

按用户指定顺序推进：先完成 pi0.5 demo 的调优、代码修复和渲染，再细化 9 月 7 日引擎功能/性能优化规格，最后按细化规格开始执行。固定 seed 的 demo、reset、外力/重力、稳定自由旋转和共享风阻功能已验收，现接续执行、内存与性能模块。

## pi0.5 demo

- [x] 恢复原会话并用原动作逐位复现搬运脱落。
- [x] 修复 MJCF 显式关节 damping/armature 覆盖，并完成前后对照测试。
- [x] 修复 OSC 补偿抵消被动阻尼，并完成解析对照测试。
- [x] 保存失败的接触参考速度实验的源码、二进制和日志。
- [x] 撤回未通过场景回归的接触参考速度实验，恢复可比较的构建。
- [x] 完成实际推理链路审计，明确缺陷、风险与排除项。
- [x] 修复自由刚体质心、惯量坐标和角速度积分的一致性；CUDA 前后对照通过。
- [x] 修复 PairSortScratch 分配越界；39 项回归通过，CUDA memcheck 从 78 个错误降为 0。
- [x] 修复 OSC 中关节干摩擦缺失；统一约束行与耦合双滑块解析测试通过。
- [x] 修正接触时间离散与有效质量问题；解析回归、51 项场景、8 项 memcheck 通过。
- [x] 验证接触双方读出与更严格的抓取、搬运、释放、支承、稳定性判定。
- [x] 量化 demo 碗盘碰撞盒重叠、接触负载和夹持滑移，修复排序容量等已证实的通用缺陷；完整 CCD/残差/overflow 公共诊断列入引擎规格。
- [x] 校准策略观测、动作、控制频率与渲染输入；保留真实 pi0.5 闭环控制。
- [x] 验证抓取、搬运、放置和松爪后的稳定性，保存轨迹、接触证据与成功判定。
- [x] 输出经过画面检查的完整 demo 视频、关键帧和复现命令。

## 引擎优化细化规格

- [x] 核对 2026-09-07 的引擎审计/优化规格与当前代码，区分已修复和仍缺失能力。
- [x] 按功能契约、数据流、通用算法、失败语义、兼容性细化优化点。
- [x] 定义正确性/性能基线、指标、采集规则和验收门槛；未采集数据不冒充实测。
- [x] 输出可实施的 [细化 spec](../plans/2026-09-08-physics-optimization-detailed-spec-zh.md)、依赖关系、执行顺序与风险。

## 按规格实施

- [x] 收口粒子时间层基础修复：统一工作位置、材料/接触交替、累计 lambda 与一次提交；六组完整长度门、已有场景/API/reset/渲染及 E=16 graph memcheck 通过。最大应变 1.838344%、最坏 RMS 0.203045%；完整接触刷新与 MPM 子步反馈另行继续。
- [x] 单独完成构建架构隔离（`6953f4a`）：后端注册和 nk 核心脱离 CUDA 配置门，CUDA 源码与库按能力附加；无 CUDA 核心构建通过，不代表完整 CPU solver 已实现。
- [x] 补充持续约束：原版不作为物理真值；参考 Newton/MuJoCo/Genesis 的匹配仿真、解析解/守恒/约束误差/收敛；multi-backend 下分开架构优化、CUDA 调优和物理算法变化。
- [x] MuJoCo 静止结构参照及主环境逐步长度质量门已加入，原始参数/结果留存；Newton 单三角运行失败、Genesis 未实跑，不宣称完整三引擎动态对照。
- [x] 当前新增模型字段的无 CUDA 核心构建及公共 C/Python/reset/渲染链路通过；非 CUDA solver 能力仍未完成。
- [x] 修复结构近邻球接触与距离约束冲突，保留原 152.74% 应变失败；冻结新物理源码/二进制，见 [证据与方向](../research/2026-09-09-particle-topology-review-zh.md)。
- [x] 布料压板输入修正：间隙计入球径/粒子厚度，表面速度与位姿一致；原各 10 mm 门通过，下压/恢复约 29.0/28.9 mm。原失败记录保留。
- [x] 收口已有流体验收：三维 control-relative 量测覆盖侧向流动，原 3 mm/0.02 L1 预算通过，接触反作用同时通过；没有新增测试集合。
- [ ] 冻结新物理六组五进程性能分母后，继续 CUDA 岛调度与每次 XPBD op 内的设备颜色循环；保留材料/接触交替边界，后端调优独立提交。

以 [模块 spec 第 14 节](../plans/2026-09-07-physics-module-specs-newton-port-zh.md) 跟踪进度。每项改造前 review 对应 spec，再对照最新 Newton/MuJoCo/Genesis 实现并记录 revision 和决策。

验证约束：少新增单测，以固定机器人＋布料＋流体环境的完整 pipeline 为主；必要解析反例辅助定位，影响渲染时验证图像链路。

批次约束：首批合并 T02c/T04a/T05a/T06a 与必要 API/状态适配，完成整批后统一构建和 pipeline 验收，不在每个小修改后重复全量测试。

性能约束：进入性能模块后主要投入极致性能优化，以 profiler 和 GPU 完成时间驱动，分别压榨少环境延迟与大批量吞吐；达到初始收益门槛后继续处理有实测收益的热点。完整 pipeline 同质量五进程对照为最终依据，细节见 [goal](2026-09-08-engine-completion-goal-zh.md)。

- [ ] 多环境与刚体：优化环境批处理、宽相/窄相、接触缓存、岛调度与求解；分别报告延迟、吞吐、容量和显存扩展曲线。
- [ ] 柔体与 MLS-MPM 流体：优化 XPBD 约束调度、邻域、P2G/G2P、活跃网格、应力及边界处理，保持本构、迭代预算和确定性要求。
- [ ] 机器人多体耦合与 SDF：优化异构映射、质量算子、Jacobian、双向反作用、SDF 采样/梯度/接触生成，验证接触完整性及物理质量。
- [ ] 光追渲染：优化多环境/多相机、加速结构更新、ray traversal、着色和传感器数据交付；在相同图像质量下量化 GPU 时间、显存和物理→渲染端到端时间。
- [ ] 扩展固定耦合 pipeline 的性能矩阵并完成各路径五进程对照；PBF 结果不替代 MLS-MPM，物理吞吐不替代渲染性能。

- [x] 整理并提交 π0.5/G1 demo、GIF、运行说明与主页 logo/两列布局（`9b8a779`、`efb20b2`）。
- [x] 完成 T03 的 [spec 与最新上游源码 review](../research/2026-09-08-reset-upstream-review-zh.md)。

- [x] 实施 T01a/T03：基线记录、E×K snapshot/reset、环境隔离、重复/非法 ID、FK/proxy、MPM/自由体和无接触图重放。
- [x] 主 pipeline、C ABI/Python 与 CUDA memcheck 验收；对比五进程 reset 成本与 arena bytes，见 [报告](../research/2026-09-08-reset-validation-zh.md)。
- [x] 将 reset 验收与限制回写模块 spec、细化 spec 和 TODO。
- [x] T02b：三轴重力、COM force/torque 消费和新公共字段，完成 [上游 review](../research/2026-09-08-rigid-dynamics-upstream-review-zh.md) 与 [pipeline/API/渲染验收](../research/2026-09-08-rigid-dynamics-validation-zh.md)。
- [x] 修复创建/reset 位姿差异、无接触 feet 容量冲突及 Python context manager 复制 owning handle。
- [x] T02c：稳定 gyro、世界角动量/能量、残差、RK4/dt 收敛与接触/pseudo 一致性，见 [整批验收](../research/2026-09-08-dynamics-pipeline-validation-zh.md)。
- [x] T04a/T06a：创建拓扑与 DOF 防护、required op、readout demand 失败保留、首错停止；Go2+H1、fixed+floating 和非法 dt 公共错误通过。
- [x] T05a 功能：共享时间层面 impulse、固定顺序 CSR gather、耗散/dv 上界/D1、MPM+XPBD 偏移通过。
- [x] T17a 基础：环境容量池、真实/保留计数、64 位 scan、稠密邻域及小容量隔离；修复 soft 预测位置导致的假邻居。
- [x] 本批主 pipeline 逐步零环境错误、reset/公共 API/渲染；12 项 memcheck 为 0 errors，冻结源码/库/日志并回写规格。
- [ ] T05a 成本：带宽/occupancy 与同质量五进程性能对照；新增内存与 dispatch 成本已记录，不宣称加速。
- [ ] T06b/c：完整 checked-u64、容量/状态、copy/memset/alloc/capture/completion 错误传播。
- [x] T09a/T07a：稳定 LBVH workspace、公共 eager/graph 与失败缓存、接触 graph → reset → graph、控制/demand 变化联合检查通过，见 [执行验收](../research/2026-09-09-execution-workspace-validation-zh.md)。
- [x] 修复 pair snapshot 索引域覆盖，加入物理副本质量门；冻结仅修复此缺陷的旧架构基线与最终候选，保留旧 E=256 无效证据。
- [x] 收口修正基线的五进程、容量×1/2/4 物理等价、公共渲染与最终 profile；65 个独立进程通过，新 graph E=1/16/256 为 3.073/3.367/6.499 ms，见执行验收。
- [ ] 基于正确基线继续活跃 row/endpoint、确定性 link wrench gather、J/质量算子与岛调度；MPM/SDF/光追补各自基线，不能套用 PBF 加速比。
- [x] 接触索引改造前冻结逐步 wrench 观测基线；graph E=1/16/256 各五进程、450 步读出轨迹通过，见 [本批 review](../research/2026-09-09-contact-index-upstream-review-zh.md)。
- [x] 接触索引的功能/graph 子集：主 pipeline 34 passed，每步 wrench 与完整末帧逐字节一致；容量×1/2/4、graph/demand/reset、公共 API/渲染和 memcheck 通过，默认 graph 降时 16.08%–17.64%，见 [报告](../research/2026-09-09-contact-index-validation-zh.md)。
- [ ] 接触索引 eager 性能：分组五进程与两组交错对照仍有回退；固定 CPU 未消除。完成 host launch/调度归因并处理，不把 graph 加速写成全部模式通过。
- [ ] 依据新 profile 继续有效岛/J：岛求解占 kernel 时间 32.12%，XPBD bend/distance 占 17.98%；检查容量网格、active spans 和临时内存。MPM/SDF/光追仍先补各自完整基线。
- [ ] 岛调度批次：已补 canonical island/活跃行观测，试验资源限制网格与有序执行行、Jacobian 合并读取；正式性能尚未收口，见 [review](../research/2026-09-09-island-scheduling-upstream-review-zh.md)。
- [ ] 20 机器人台阶脚部下沉：扩大验证报告 -0.0892994255 m，冻结旧版同样失败且值相同；核对实际碰撞几何、地形与测量方法，保存 `out/validation/island_scheduling_20260909/terrain_baseline.*`，不改容差掩盖。
- [ ] T08a/T10a 剩余：真实接触密度曲线、完整容量水位及显存峰值；活跃 cache 排序/merge 和字段预算已实现。
- [ ] T17 剩余：公共容量配置、投影位移触发的刷新/skin、边界密度和完整粒子时间层。
- [ ] M04/M06：定位双浮基机器人重叠不分离；冻结旧版同样失败，尚未归因。
- [ ] M03/M11：masked reset 分项 profile；完整 benchmark 不由 reset 成本替代。
- [ ] 完整 T01 benchmark/profiler、T04b 异构映射、T09b refit、T10b 活跃 row/J、T11 岛调度；按 [方向 spec](../plans/2026-09-09-performance-directions-detailed-spec-zh.md) 先有对应路径的有效基线，再成批优化。

Editor 按用户 2026-09-08 指示暂缓，待物理引擎完善后再设计；Go2 `.nks` 加载限制只保留记录，不阻塞引擎功能工作。

## 证据位置

- 排查记录：`docs/research/2026-09-07-pi05-grasp-contact-debug.md`。
- 引擎审计：`docs/research/2026-09-07-physics-engine-audit-zh.md`。
- Demo 验收：`docs/research/2026-09-08-pi05-demo-validation-zh.md`。
- 细化规格：`docs/plans/2026-09-08-physics-optimization-detailed-spec-zh.md`。
- 实验与日志：`out/libero/pi05_contact_repair/`。
- 失败候选归档：`out/libero/pi05_contact_repair/reference_candidate_archive_20260908/`。
- 内存安全且包含刚体坐标修复的比较构建：`out/libero/pi05_contact_repair/safe_frames_engine/`。

最新实跑：`chunk8_pi05_12s`，OSC/8 步分块/12 秒/seed=20260828；抬升 12.172 cm，搬运滑移 0.845 mm，碗盘动态重叠 2.086 mm、稳定后 0.322 mm，已释放并稳定支承。`chunk50_pi05_16s` 因 71.403 mm 搬运滑移判为失败，其旧视频 badge 不能作为成功证据。

所有物理修复使用同一通用路径，不添加碗、盘子或抓取场景专用求解分支。
