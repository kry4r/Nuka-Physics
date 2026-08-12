# P2：MuJoCo 同资产 Go2 接触 Oracle

状态：需求已确认，待实现
依赖：[P1 接触与 Row/Block 基线](2026-08-12-contact-row-block-foundation-zh.md)

## 1. 目标

建立 Nuka 与 MuJoCo 的同资产、同控制输入、同事件定义对照。MuJoCo 用于判断动作和接触包络是否物理可信，不要求逐 timestep 状态完全相同。

Newton/Newton Physics 继续作为算法机制参考，本期不建立三方 parity 工程。

## 2. 对照合同

- 使用同一 Go2 质量、惯量、关节轴、关节上下限、effort limits、碰撞几何和材料对语义。
- 对齐重力、控制频率、物理 `dt/substep` 和 torque clamp；求解器内部迭代不要求一致。
- 输入为同一 timestamped torque trace；禁止一侧使用 PD target、另一侧使用直接 torque。
- 坐标系、四元数顺序、关节顺序和接触脚命名在导出时显式记录。
- 比较事件与统计分布，不用逐步 position epsilon 作为主 gate。

## 3. 指标

### 共用接触指标

- 支撑脚 wrench、切向/法向冲量与累计 slip；
- 起跳总冲量、airtime、首次接触和落地峰值/积分冲量；
- 躯干角动量与能量变化；
- joint-limit active 时间、最大越界和 effort saturation；
- 非支撑脚误触地、机身碰撞和恢复稳定时间。

### 技能级指标

- 单前腿倒立跳：FL/FR 分开统计入式时间、75 度保持、5 次离地/恢复、最终四足落地。
- 后空翻：起跳、全脚离地、累计后向旋转、首次落地、四足稳定恢复。

## 4. 建议实现

1. 定义版本化 `Go2OracleTrace`，记录 state、torque、contacts、events 和 profile hash。
2. 为 Nuka 和 MuJoCo 各写一个 exporter，输出同一列定义。
3. 写离线 comparator，按事件对齐并生成 CSV/Markdown 报告。
4. 先用 open-loop torque/短轨迹验证资产与符号，再接 policy rollout。
5. 把关键 envelope 固化为回归阈值；阈值来自实测，不在实现前臆定逐项数值。

## 5. 验收标准

- [ ] 两端资产摘要可机器比较，质量/惯量/limit/effort/材料差异会直接失败。
- [ ] 同一 torque trace 可在两端重放，事件检测不依赖 policy 内部状态。
- [ ] 报告至少包含上述共用指标，并对 FL/FR 分栏。
- [ ] 固定 seed Nuka trace 保持 D1；MuJoCo 只要求自身可重复。
- [ ] 对照差异能归类为资产、时间离散、接触模型、控制或事件检测，而不是只给总分。

## 6. 非目标

- 不把 MuJoCo solver 复制进 Nuka。
- 不要求浮点逐帧相同，也不以视觉相似代替指标。
- 不在此任务调训练奖励或生成最终视频。
