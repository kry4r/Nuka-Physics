# Spec-03: 后空翻任务的旋转状态观测与目标课程

日期: 2026-08-25
状态: 已实现
参考: juyoung020/go2-backflip-rl (GitHub)、ziyanx02/Genesis-backflip、
      Stage-Wise Reward Shaping for Acrobatic Robots (arXiv 2409.15755)、
      Learning Impact-Rich Rotational Maneuvers via Centroidal Velocity Rewards
      (Kang et al., 2025)

## 背景

调研同类开源后空翻实现后发现两个结构性缺口:

1. 策略观测中不含任何旋转进度信号 —— 落地判据依赖累计转角, 但策略看不到它,
   任务是非马尔可夫的, 条件化奖励(收腿时机/展开时机)不可学。
2. 单一 2π 目标对探索不友好; 同类实现均按目标转角分级(90°→180°→360°),
   以实测物理量(陀螺积分)作为晋级依据。

## 需求决策 (grill-me)

| 问题 | 决策 | 依据 |
|---|---|---|
| 墙钟 phase 进 obs? | 否 | 引擎合同禁止阶段输入 |
| 物理积分转角进 obs? | 是 | 仿真状态推导, 非时钟, 合同允许 |
| 目标转角是命令吗? | 是 | 与手倒立 θ_cmd 同类, 走命令块 |
| 分级目标? | {0.5π, π, 1.5π, 2π} 四级 | 参考项目同级设计 |
| 晋级依据? | 实测 \|rot_acc\|≥0.92·target 且直立着陆窗口完成 | 实测物理量, 非迭代数 |
| 非对称 critic? | 暂缓 | 收益小, rl_games 接入成本高 |
| 动作空间改 PD 目标? | 否 | 引擎合同规定纯扭矩为唯一控制源 |
| 旧 checkpoint? | 作废 (obs 维度变化), 当前策略仅 rew≈8 | 重训成本低 |

## 观测变更

48 维基础 obs 之后追加 2 维:

```
[0] target_rot / 2π    当前环境的 目标转角 命令
[1] rot_acc / 2π       本局累计俯仰转角 (负=后向), 物理积分
```

## 课程

每环境独立维护 target_rot, 重置时在当前级别采样, 晋级条件:

    landed_upright ∧ |rot_acc| ≥ 0.92·target 连续满足 3 局 → 升一级

落地奖励的翻转判定同步改为相对 target: ||rot_acc| − target| < 0.3 rad。

## 奖励增补 (对齐参考实现的塑形项)

| 项 | 形式 | 权重 |
|---|---|---|
| r_flip_rate | 空中门控的后向 ω_y 速率 (已有) | 3.0 |
| r_roll | −g_y² 滚转抑制 | 2.0 |
| r_contact | 大腿链接触地罚 (新增读 LINK_CONTACT_WRENCH 大腿槽位) | 0.5 |
| r_rate | ‖a_t − a_{t−1}‖² 动作平滑 (需缓存上一步动作) | 0.01 |
| r_drift | 落地窗口内 ‖p_xy − p₀‖² 位置保持 | 2.0 |
| r_sym | 左右腿动作镜像对称罚 | 0.05 |

起跳速度项维持 clamp ±6 m/s; 总奖励硬包络 [-5, 25] 不变。
