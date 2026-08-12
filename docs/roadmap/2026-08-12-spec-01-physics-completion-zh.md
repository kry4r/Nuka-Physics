# Spec 01：物理补全

状态：需求已收敛，待实现
顺序：阶段 1/3
输入：[Row/Reaction 生产矩阵](2026-08-12-row-reaction-production-matrix-zh.md)、[接触与 Row/Block 基线](2026-08-12-contact-row-block-foundation-zh.md)、[MuJoCo Go2 Oracle](2026-08-12-mujoco-go2-contact-oracle-zh.md)
输出：可供 Go2 双 demo 训练和回归的冻结物理基线

## 1. 目标

在训练高动态 Go2 技能前，补齐跨系统约束与反作用边界、接触稳定性、真实关节限位和外部物理对照。阶段完成后，刚体、关节链和粒子的外部接触具有一致、可诊断、可重放的生产合同；XPBD、PBF、MLS-MPM 内部算法继续保留专用求解器。

本阶段不是“把所有物理 Row 化”。统一对象是外部接触、限位、island、reaction 与诊断边界，不改变驱动经 ABA 广义力执行的路径。

## 2. 冻结范围

### 2.1 Row/Reaction 边界

- 刚体-刚体、关节链-刚体/静态、刚体/关节链-粒子外部接触统一进入 canonical contact descriptor、shared island 与 Row/Block solve。
- drive/torque 保持 generalized force，经真实 effort limit clamp 后由 ABA 执行，不迁入 Row。
- XPBD 形变、PBF 密度投影、MLS-MPM P2G/Grid/G2P 和粒子局部 correction 不强制迁移。
- MPM-body 反作用写固定容量 `ReactionRecord`，按 stable key 排序、分段归约并应用；不为 grid node 生成 Row，不以无序浮点 atomic 作为 D1 生产路径。
- analytic-only body 缺少 MPM 所需 cooked SDF 时返回明确 capability/diagnostic，不允许静默单向耦合。

### 2.2 接触身份与 Warm Start

```text
ContactId = canonical(shapeA, shapeB)
          + featureA + featureB
          + manifold_slot

fallback = quantized(local_point_A, local_point_B)
```

- canonical pair 同时固定法线方向、两侧局部点和切线基语义。
- feature ID 优先；仅缺少稳定 feature 时使用局部量化点 fallback。
- manifold matching 使用固定容量、确定性代价和 lexicographic tie-break。
- 按 `ContactId` 持久化整个 `[lambda_n, lambda_t1, lambda_t2]`，并纳入 reset、checkpoint、replay 与状态 hash。
- topology、shape/material version、法线翻转、接触消失和重新匹配触发版本化衰减或失效规则。

### 2.3 `ContactBlock3`

```text
lambda = [lambda_n, lambda_t1, lambda_t2]
lambda_n >= 0
sqrt((lambda_t1 / mu1)^2 + (lambda_t2 / mu2)^2) <= lambda_n
```

- 用固定三维椭圆锥 block 替换生产四个 friction spokes。
- 首期支持 `mu1 == mu2`，ABI 保留各向异性 `mu1/mu2`。
- canonical normal 生成确定性切线基；近轴向使用固定 tie rule。
- contact block 与 bilateral/limit scalar row 共享 island、reaction、容量和 telemetry。
- 固定迭代数，不引入 data-dependent early exit。

### 2.4 Canonical Soft Contact

- MJCF `solref/solimp` 与 Nuka `tau/zeta` 或 `stiffness/damping` 是 authoring 入口。
- cook 阶段统一转换为版本化 canonical contact profile；运行时 assembly 不解释源格式。
- 材料对规则明确优先级、默认值、单位、摩擦/软接触组合、正则化下限及 `dt/substep` 缩放。
- 至少独立配置 Go2 脚垫-地面、机身-地面与默认材料对，禁止 scene-name 分支。

### 2.5 生产 Joint Limits

- 从 URDF/MJCF/SceneIR cook lower、upper 与 effort limit 到生产模型。
- 上下限使用单边 limit row，进入共同排序、island、正则化和 reaction 路径。
- 角度限位与 actuator effort clamp 分开实现和验证。
- Go2 与 MuJoCo oracle 使用同一关节范围、质量、惯量、碰撞几何和材料摘要。

## 3. 实施分段

### P0：生产覆盖审计转运行证明

1. 为生产矩阵每一行绑定源码 emitter、consumer、capacity 和运行测试。
2. 为保留局部求解的系统增加边界测试，防止误迁移。
3. 固定 reaction record、排序、归约和诊断格式。

出口：生产矩阵从“静态审计”更新为“运行验证”，并能机器检查每条边界。

### P1：接触与限位基线

1. canonical descriptor、stable ID 和 manifold matching。
2. warm-start cache 生命周期及 checkpoint/replay。
3. `ContactBlock3` 投影、solver integration 与旧 spokes 移除。
4. soft-contact cooker 和材料对合并。
5. joint-limit row 与 effort clamp 接线。
6. 堆叠、斜坡、单脚支撑、body-particle 和 MPM reaction 组合回归。

出口：固定 seed 下状态、事件、contact cache 和 reaction 记录可 D1 重放。

### P2：MuJoCo 同资产 Oracle

1. 定义版本化 `Go2OracleTrace`：asset/profile hash、state、timestamped torque、contacts、events 和 limit/effort telemetry。
2. Nuka/MuJoCo exporter 输出同一列定义和坐标/关节顺序元数据。
3. 离线 comparator 按事件和统计包络比较，不要求逐 timestep 数值一致。
4. 先用 open-loop torque/短轨迹校验资产，再用于 Go2 policy rollout。

出口：同 torque trace 的资产差异能直接失败，接触/起跳/落地/限位差异能归类。

## 4. 必需测试与指标

- stable contact ID 序列、cache 命中/失效、cache age 和 checkpoint replay。
- 椭圆锥内/外、零法向、切线基旋转及各向异性 ABI 单测。
- fixed-iteration residual、support-foot slip、normal/tangent impulse 和 cone projection 次数。
- joint lower/upper sweep、最大越界、limit active 时间和 torque saturation。
- rigid/articulation/particle contact block 及 MPM reaction 排序/归约。
- MJCF/Nuka 等价 authoring cook 后 canonical profile byte-equal。
- MuJoCo 事件级指标：wrench/slip、起跳冲量、airtime、累计角动量、首次接触、落地冲量与恢复时间。

## 5. 阶段验收 Gate

- [ ] 外部接触统一进入约定 Row/Block 边界，局部 XPBD/PBF/MPM 算法未被误改写。
- [ ] drive 仍经 generalized force/ABA；joint limits 与 effort clamp 分离且均使用真实资产值。
- [ ] 生产不再发出四个 friction spokes，`ContactBlock3` 解析投影通过。
- [ ] warm-start cache 进入 reset/checkpoint/replay/hash，固定 seed D1 不因 cache 丢失而伪通过。
- [ ] MPM reaction record 排序和归约 D1，缺 capability 显式失败。
- [ ] Go2 脚垫、机身和默认材料对使用 canonical soft-contact profile。
- [ ] MuJoCo 同资产摘要可机器比较，同 torque trace 可生成事件/统计报告。
- [ ] 堆叠、斜坡、单脚支撑和全关节 sweep 无非有限值、非法穿透或越限。

只有全部 gate 通过并冻结 physics/asset/profile hash 后，Spec 02 才能启动正式训练和验收。任何 P1 合同变化都使既有 Go2 checkpoint 标记为需重训。

## 6. 非目标

- 不实现 general TOI CCD、sleeping、contact topology gradient、rolling/torsional friction。
- 不统一所有子系统的内部迭代次数或数学形式。
- 不复制 MuJoCo/Newton solver；Newton 仅作机制参考。
- 不在本阶段训练或以最终视频代替物理 gate。
