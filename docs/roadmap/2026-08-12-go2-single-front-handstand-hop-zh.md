# P3：Go2 单前腿倒立连续跳

状态：需求已确认，待重新设计
依赖：[P1 物理基线](2026-08-12-contact-row-block-foundation-zh.md)、[P2 MuJoCo Oracle](2026-08-12-mujoco-go2-contact-oracle-zh.md)

## 1. 目标动作

同一直接扭矩策略接收 `FL` 或 `FR` 指令，从正常四足静止站立自主进入单前腿倒立，使用指定前腿作为唯一支撑连续完成 5 次低幅跳跃，并以四足稳定落地结束。

旧 `go2_handstand` 是双后腿支撑，动作朝向、初态、奖励和控制假设均不适用。它只用于记录失败模式，不能 warm-start 新策略或充当正式 gate。

## 2. 正式事件状态机

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

状态由物理事件推导，不向 policy 暴露阶段编号或 episode time。

### 2.1 Stand

- 正常四足静止初态，禁止预先抬高、预旋转或 teleport 到近倒立姿态。
- 允许训练 curriculum 使用简化初态，但正式 gate 必须从 Stand 开始。

### 2.2 SingleFrontHold

- 躯干 elevation 至少 75 度。
- 指定 `FL`/`FR` 是唯一承重足，另外三脚离地。
- 使用连续时间窗口确认 hold，不能以单帧越过阈值计成功。
- 允许有限摆动；窗口长度与允许角速度需通过后续数据收敛，不先用任意单帧常量锁死。

### 2.3 每次 Hop

- 四脚全部离地，不能把接触抖动记为 hop。
- 质心或基座垂直位移在约 2-5 cm 可见范围，并设置最小 airtime。
- Hop 1-4 后必须回到同一指定前腿的 SingleFrontHold；中途另一脚承重即失败或重置计数，规则必须唯一。
- Hop 5 后进入四足落地，不要求再次单腿 hold。

### 2.4 FourFootLanding

- 四脚建立有效接触，机身不触地，关节不越限。
- 在连续稳定窗口内保持姿态、速度和滑移低于 gate 阈值。
- 不能只以“四脚曾在同一帧接触”判定成功。

## 3. 策略与训练合同

- action：每关节 `[-1, 1] * effort_limit` 的直接扭矩。
- observation：本体状态、接触、上一动作和左右前腿指令；不包含 phase clock。
- 同一 policy 覆盖 FL/FR，训练采样和评估样本数左右平衡。
- 允许 reference pose/velocity/contact schedule、trajectory optimization、imitation 和 curriculum。
- 正式评估只能执行 policy torque；不叠加 scripted PD、起跳 impulse 或姿态控制器。
- 奖励按真实事件和持续时间组织，不能因近目标 IC 或固定时刻给分。

## 4. 正式 Gate

- 并行环境数 `N >= 256`。
- 小范围随机化至少覆盖初始关节/基座扰动、摩擦/soft-contact 参数和质量/惯量容差；具体范围由 P2 同资产和训练稳定性确认。
- 总成功率至少 80%，并分别报告 FL 与 FR 成功率；任一侧不能被总体均值掩盖。
- 固定 seed 可 D1 重放，事件序列和成功 bitmap byte-identical。
- 输出接触 ID 命中率、support-foot slip、每跳冲量/airtime、落地冲量、limit 活动和 torque saturation。

## 5. 验收标准

- [ ] 单一 policy 从正常站立完成 FL 和 FR 两种指令。
- [ ] 每个成功 episode 有且只有 5 个满足定义的 airborne event。
- [ ] 前 4 跳均恢复到指定单前腿；第 5 跳后四足稳定落地。
- [ ] 不读取 episode phase，不执行 scripted torque/PD assist。
- [ ] 批量随机 gate 总体及左右两侧均达到约定成功率。
- [ ] 与 MuJoCo 对照报告支撑脚 wrench/slip、起跳/落地冲量和成功包络。
- [ ] 物理 baseline 变化触发技能回归，而不是静默沿用旧 checkpoint。

## 6. 非目标

- 不复活双后腿倒立。
- 不追求高跳、长距离移动或连续超过 5 跳。
- 不把最终电影视频作为策略正确性的唯一证据。
- 不在任务代码中加入 shape ID、scene name 或单侧腿特化的物理分支。
