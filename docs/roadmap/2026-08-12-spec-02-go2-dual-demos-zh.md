# Spec 02：Go2 双 Demo

状态：需求已收敛，等待 Spec 01
顺序：阶段 2/3
输入：[Spec 01 物理补全](2026-08-12-spec-01-physics-completion-zh.md)、[单前腿倒立连续跳](2026-08-12-go2-single-front-handstand-hop-zh.md)、[站立后空翻](2026-08-12-go2-standing-backflip-zh.md)
输出：两个通过批量物理 gate 的独立 policy、冻结轨迹与 oracle 报告

## 1. 目标

在冻结的物理/资产基线上重新设计并交付两个互不替代的 Go2 demo：

1. 同一 policy 按 `FL/FR` 指令完成单前腿倒立五连跳。
2. 独立 policy 从正常站立自主完成完整后空翻。

两个 demo 共用真实 effort limit 的直接扭矩合同、无 phase clock 的正式执行边界、批量随机 gate、D1 replay 和 MuJoCo 事件级 oracle。训练可以使用参考轨迹、优化结果和 curriculum，正式执行不能使用 scripted assist。

## 2. 共用合同

### 2.1 前置冻结

- 必须记录并校验 Spec 01 输出的 physics ABI、asset、canonical material、contact profile、joint limit 和 oracle schema hash。
- 任一 hash 变化使 checkpoint/gate 显式失效或标记需重训，不允许静默沿用。
- 正式 reset 使用正常四足静止站立，不 teleport 到近目标姿态。

### 2.2 控制与观测

- action：逐关节 `[-1, 1] * effort_limit` 的直接扭矩。
- actuator clamp 只使用真实资产 effort limit，不因训练困难扩大 authority。
- observation：本体状态、接触、上一动作及技能必要命令；不包含 episode time、phase clock 或隐藏阶段编号。
- 正式执行的唯一控制源是 policy torque；禁止 scripted PD、姿态 target、外部 impulse、手工 takeoff 或 landing assist。

### 2.3 训练许可

- 允许 reference pose/velocity/contact schedule、trajectory optimization、imitation、关键姿态和从简到难 curriculum。
- curriculum 初态和 reference 仅用于训练，不改变正式 reset 与成功定义。
- 奖励按物理事件、持续时间和安全边界组织，不能按固定时刻给阶段分。

### 2.4 正式 Gate

- 每个技能 `N >= 256` 并行环境。
- 小范围随机化覆盖初始基座/关节、摩擦/soft-contact 和质量/惯量容差，范围随冻结 profile 版本记录。
- 每个技能成功率至少 80%；倒立跳还必须分别报告 FL/FR，任一侧不能被总体均值掩盖。
- 固定 seed 的初态、事件序列、success bitmap 和失败分类 D1 可重放。
- 输出 contact ID 命中率、脚滑移、冲量/airtime、limit active、最大越界、torque saturation 和非有限计数。

## 3. Demo A：单前腿倒立五连跳

### 3.1 动作定义

同一 policy 接收 `FL` 或 `FR` 指令，从正常站立自主进入倒立，指定前腿成为唯一承重足，躯干 elevation 至少 75 度；连续完成 5 次约 2-5 cm 的低幅跳跃，前 4 跳恢复到同一指定前腿的单腿倒立，第 5 跳后四足稳定落地。

```text
Stand
  -> EnterHandstand
  -> SingleFrontHold
  -> Hop1Air -> Hop1Recover
  -> Hop2Air -> Hop2Recover
  -> Hop3Air -> Hop3Recover
  -> Hop4Air -> Hop4Recover
  -> Hop5Air
  -> FourFootLanding
  -> Stable
```

### 3.2 事件语义

- `SingleFrontHold` 使用连续窗口确认 75 度姿态、指定前腿承重和另外三脚离地，不接受单帧触发。
- 每跳必须出现四脚全离地、最小 airtime 和约 2-5 cm 的基座/质心垂直位移，接触抖动不得计数。
- Hop 1-4 中另一脚承重时按唯一规则失败或重置，不能按场景临时选择。
- 最终成功要求四脚有效接触、机身不触地、关节不越限，并在连续窗口内保持低姿态/速度/滑移误差。

### 3.3 独立验收

- [ ] 一个 policy 覆盖 FL/FR，训练与评估样本左右平衡。
- [ ] 每个成功 episode 恰有 5 个有效 airborne event。
- [ ] 前 4 跳恢复指定单前腿，第 5 跳后四足稳定。
- [ ] 总体与 FL/FR 分栏成功率均满足 gate，失败分类可追踪。
- [ ] MuJoCo 报告覆盖支撑脚 wrench/slip、每跳冲量/airtime 和最终落地包络。

旧双后腿倒立任务仅作为失败样本，不能 warm-start 正式 policy 或充当成功依据。

## 4. Demo B：站立后空翻

### 4.1 动作定义

独立 policy 从正常四足站立自主下蹲和起跳，在空中完成完整后向 360 度旋转，以四足稳定落地。

```text
Stand
  -> SelfSelectedCrouch
  -> Takeoff
  -> Airborne
  -> BackwardRotationHalf
  -> BackwardRotationComplete
  -> FirstLandingContact
  -> FourFootLanding
  -> Stable
```

### 4.2 事件语义

- 自主起跳后必须出现四脚全离地。
- 旋转使用连续展开姿态/角速度积分，能区分不足 360 度、完整一周和过旋；不能只比较首尾四元数。
- 首次有效落地后必须形成四足支撑；非脚先着地、机身触地、关节越限均失败。
- stable window 同时约束姿态、角速度、平移速度和脚滑移。
- 失败至少分为未起跳、旋转不足、过旋、非脚先着地、非四足稳定落地、越限、torque saturation、非有限和超时。

### 4.3 独立验收

- [ ] 正常站立 reset 下自主完成一次完整后空翻并四足稳定落地。
- [ ] 执行 observation 不含 phase clock，action 不叠加脚本或外力。
- [ ] 旋转判据和失败分类在合成轨迹单测中正确。
- [ ] 批量随机 gate 成功率至少 80%，固定 seed D1。
- [ ] MuJoCo 报告覆盖起跳冲量、airtime、角动量、累计旋转、落地冲量和恢复时间。

## 5. 交付物

每个 demo 独立交付：

- versioned task/event/reward 配置和训练入口；
- policy checkpoint、训练配置、physics/asset/profile hash；
- `N >= 256` gate 的 seed、success bitmap、失败分类和 telemetry；
- 固定 seed D1 replay artifact；
- Nuka/MuJoCo oracle CSV/Markdown 报告；
- 通过 gate 后导出的冻结关节/基座/接触/相机时间线，供 Spec 03 离线渲染。

冻结轨迹是渲染输入，不是技能正确性的替代证据。两个 demo 必须分别通过 gate 后，Spec 03 才制作对应最终视频。

## 6. 非目标

- 不复活双后腿倒立，不做超过 5 跳、高跳或长距离移动。
- 不做连续多周翻转、侧空翻、跳台或障碍辅助。
- 不通过放宽碰撞、joint limit、effort limit 或随机范围提升成功率。
- 不把训练 reference、curriculum 或最终电影视频当成正式执行控制源或唯一验收。
