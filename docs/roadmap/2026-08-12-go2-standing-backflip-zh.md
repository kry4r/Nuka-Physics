# P4：Go2 站立起跳后空翻

状态：需求已确认，待重新设计
依赖：[P1 物理基线](2026-08-12-contact-row-block-foundation-zh.md)、[P2 MuJoCo Oracle](2026-08-12-mujoco-go2-contact-oracle-zh.md)

## 1. 目标动作

Go2 从正常四足静止站立自主下蹲和起跳，在空中完成完整 360 度后向旋转，以四足稳定落地。

现有后空翻任务中的深蹲 teleport、固定阶段时钟和手工起跳 probe 不能作为新正式方案的核心。旧代码仅用于提取 actuator authority、观测接口和失败样本。

## 2. 正式事件状态机

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

- 事件全部由姿态、接触、速度和累计旋转推导。
- policy observation 不含 episode time、phase clock 或预设接触 schedule。
- 累计旋转必须使用连续展开的姿态/角速度积分，不能只比较首尾四元数，因为 360 度首尾姿态相同。

## 3. 策略与训练合同

- action：与 P3 相同的真实限幅直接扭矩。
- observation：本体状态、四脚接触、上一动作；可以有动作命令，但无阶段输入。
- 正式 gate 从正常 stand reset 开始，禁止深蹲初态、跳台、外部 impulse 和 scripted takeoff。
- 训练允许 reference trajectory、关键姿态、contact schedule imitation 和从简到难 curriculum。
- reference 只参与 loss/curriculum；执行时唯一控制源是 policy torque。

## 4. 成功与失败

成功必须同时满足：

1. 自主起跳后出现四脚全离地事件；
2. 空中累计后向旋转达到完整一周，且没有以前向旋转绕阈值；
3. 首次有效落地后形成四足支撑；
4. 机身不触地，关节不越限；
5. 连续稳定窗口内姿态、角速度、平移速度和脚滑移符合 gate。

失败至少分类为：未起跳、旋转不足、过旋、非脚先着地、单/双脚不稳定落地、关节越限、torque saturation、数值非有限和超时。

## 5. 正式 Gate

- `N >= 256` 并行环境，小范围 IC 与物理随机化。
- 成功率至少 80%。
- 固定 seed 的 event trace、success bitmap 和失败分类 D1 可重放。
- 与 P3 共用 torque、contact、limit 和 soft-contact telemetry。
- 与 MuJoCo 比较起跳冲量、airtime、角动量、累计旋转、落地冲量和恢复时间。

## 6. 验收标准

- [ ] 从正常站立自主完成一次完整后空翻并四足稳定落地。
- [ ] observation/actor 输入不包含 phase clock。
- [ ] 执行阶段没有 scripted action、姿态 target 或外部 impulse。
- [ ] 旋转判据能区分 360 度、旋转不足和过旋。
- [ ] 批量随机 gate 成功率达到约定门槛，并输出失败分类分布。
- [ ] Nuka/MuJoCo 同 torque trace 的事件级指标可比较。
- [ ] P1 baseline 或资产 hash 变化会使旧策略/gate 明确失效或标记需重训。

## 7. 非目标

- 不做连续多周翻转、侧空翻或障碍物辅助。
- 不为了视频观感放宽碰撞、limit 或 effort limit。
- 不使用旧任务的固定 phase reward 作为正式成功定义。
