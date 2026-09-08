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

以 [模块 spec 第 14 节](../plans/2026-09-07-physics-module-specs-newton-port-zh.md) 跟踪进度。每项改造前 review 对应 spec，再对照最新 Newton/MuJoCo/Genesis 实现并记录 revision 和决策。

验证约束：少新增单测，以固定机器人＋布料＋流体环境的完整 pipeline 为主；必要解析反例辅助定位，影响渲染时验证图像链路。

批次约束：首批合并 T02c/T04a/T05a/T06a 与必要 API/状态适配，完成整批后统一构建和 pipeline 验收，不在每个小修改后重复全量测试。

性能约束：进入性能模块后主要投入极致性能优化，以 profiler 和 GPU 完成时间驱动，分别压榨少环境延迟与大批量吞吐；达到初始收益门槛后继续处理有实测收益的热点。完整 pipeline 同质量五进程对照为最终依据，细节见 [goal](2026-09-08-engine-completion-goal-zh.md)。

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
- [ ] T09：LBVH workspace 后补接触 graph → reset → graph；当前 Unsupported。
- [ ] T17 剩余：公共容量配置、投影位移触发的刷新/skin、边界密度和完整粒子时间层。
- [ ] M04/M06：定位双浮基机器人重叠不分离；冻结旧版同样失败，尚未归因。
- [ ] M03/M11：masked reset 分项 profile；完整 benchmark 不由 reset 成本替代。
- [ ] 完整 T01 benchmark/profiler、T04b 异构映射、T07 公共 graph、T08 cache、T10 活跃内存预算；按实测瓶颈成批推进。

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
