# P0：Row/Reaction 生产覆盖矩阵

状态：需求已确认，静态源码审计完成，运行验证待执行

## 1. 决策

“统一 Row 求解器”定义为统一跨系统约束和反作用边界，不要求所有物理子系统共享同一种内部数学。

生产事实以 `src/nk/pipeline/pipeline.cpp` 的 op 顺序为准。`src/constraint/row.hpp` 中更宽泛的 `constraint::Row` 类型不能证明它已接入 `nk::Pipeline`；生产接触记录是 `src/nk/solve/nk_row.hpp` 中的 `nk::NkRow`。

## 2. 当前生产矩阵

| 系统/约束 | 当前生产路径 | 是否共享 `NkRow`/solve | 收敛要求 |
|---|---|---:|---|
| 刚体-刚体接触 | PairDriven -> `AssembleRows` -> islands -> `SolveRowsBlockIsland` | 是 | 保持单一路径，增加 stable ID/block/warm start |
| 关节链-刚体/静态接触 | 同上，articulation side 使用预计算 `M^-1 J^T` | 是 | 保持 ABA inverse-mass operator |
| 刚体/关节链-粒子外部接触 | body-particle narrowphase -> `NkRow` | 是 | 与其他接触共享 ID、材料和 warm-start 合同 |
| 生产 joint limits | 未发现 emitter；旧 runtime 仅有 scaffold | 否 | cook 上下限并生成统一 limit rows |
| joint drive/torque | `ApplyDrives` -> generalized force -> ABA | 否，且不要求 | 明确保留广义力，不迁入 Row |
| XPBD 内部约束 | `XpbdProject` 局部位置投影 | 否 | 保留专用 solver，只统一外部接触 |
| PBF 密度约束 | `PbfDensityLambda` / `PbfApplyDelta` | 否 | 保留专用 solver，只统一外部接触 |
| 粒子-粒子接触 | `ParticleParticleContact` 局部 correction | 否 | 本轮不强制迁移，补充边界说明和诊断 |
| MLS-MPM 内部连续介质 | `MpmStep` 的 P2G/Grid/G2P | 否 | 保留专用 solver |
| MLS-MPM-body 反作用 | grid projection/react 直接写 rigid velocity 或 articulation `qdot` | 否 | 统一 deterministic reaction record/provider，不生成 grid-node rows |

## 3. 统一合同

### 3.1 Row/Block 侧

- 一个 canonical contact descriptor 是 collision 到 solver 的唯一输入边界。
- descriptor 至少包含 canonical pair、feature IDs、局部点、法线、确定性切线基、距离、材料对和 manifold slot。
- 标量 bilateral/limit row 与固定大小 contact block 可以共存；“统一”是共同调度、island、reaction 和诊断，不是强迫所有记录等宽。
- `NkRow`/block ABI 必须版本化，容量与排序在 cook 时确定。

### 3.2 Reaction 侧

统一 reaction record 至少表示：

```text
target kind: rigid | articulation | particle | static
target id
linear/angular Jacobian or generalized impulse reference
impulse delta
stable source/contact key
```

- rigid、articulation、particle 应使用同一 provider dispatch 语义。
- MPM reaction 先写固定容量记录，再按 stable key 排序/分段归约；禁止以无序浮点 atomic 作为 D1 生产路径。
- analytic-only body 缺少 cooked SDF 导致 MPM 单向耦合的情况必须显式报 capability/diagnostic，不能静默退化。

## 4. 不足清单

1. `NkRow` 当前仍按每个 contact point 展开一个 normal 加四个 friction spokes，不是已决定的三维椭圆锥 block。
2. 当前 contact slot/row slot 主要是瞬时容量位置，不构成跨帧 stable identity。
3. 生产路径没有持久 contact impulse cache，固定迭代下无法利用上一帧解。
4. 生产 joint limit rows 缺失，高动态策略可能学习非法姿态。
5. canonical soft-contact 的公共 authoring、材料对合并、单位和 timestep scaling 尚未形成端到端合同。
6. MPM reaction 与普通 row reaction 共享状态 sink，但没有统一记录、排序、诊断和回归接口。
7. 抓持现有测试只证明接触生成，不证明稳定 hold；不能作为摩擦质量验收。
8. 文档中的 “universal constraint rows” 容易被理解为 XPBD/PBF/MPM 内部算法也已 Row 化，需要始终附带边界限定。

## 5. 验收标准

- [ ] 为矩阵每一行建立源码入口和运行测试的对应表。
- [ ] rigid-rigid、articulation-ground、body-particle 均证明走相同 assembly/island/solve 入口。
- [ ] XPBD、PBF、MPM 内部局部解保持 byte-identical，除非各自任务明确修改。
- [ ] MPM reaction record 在重复运行中排序和归约 byte-identical。
- [ ] 驱动未迁移为 Row，并有架构测试/文档防止以后误判。
- [ ] joint limit、contact block 和 warm-start 的新 ABI 均能在诊断中区分。

## 6. 非目标

- 不合并所有局部迭代次数或收敛准则。
- 不在本任务实现 general TOI CCD、sleeping 或 contact topology gradients。
- 不用 legacy/reference `constraint::Row` 替换生产 `NkRow` 而不经过显式迁移计划。

## 7. 依赖与后续

本审计是 [P1 接触与 Row/Block 基线](2026-08-12-contact-row-block-foundation-zh.md) 的输入。P1 完成后，P0 矩阵需要更新为“已验证实现”，而不是保留为静态分析。
