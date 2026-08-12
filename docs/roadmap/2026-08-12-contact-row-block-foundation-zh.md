# P1：接触身份、Warm Start 与 Row/Block 基线

状态：需求已确认，待实现
依赖：[P0 Row/Reaction 生产矩阵](2026-08-12-row-reaction-production-matrix-zh.md)

## 1. 目标

在训练两个 Go2 高动态技能前，建立最小可信物理基线：稳定 contact identity、持久 warm start、三维椭圆锥 contact block、canonical 材料对软接触和生产 joint limit rows。

## 2. 已确认设计

### 2.1 Contact identity

```text
ContactId = canonical(shapeA, shapeB)
          + featureA + featureB
          + manifold_slot

fallback = quantized(local_point_A, local_point_B)
```

- shape pair 顺序 canonical 化后，法线方向和两侧 local point 也必须随之 canonical 化。
- mesh/convex 有稳定 feature 时优先使用 feature；量化点只作为缺少 feature 的兼容路径。
- tiny manifold 匹配使用固定上限、确定性代价与 lexicographic tie-break。
- topology、shape/material 版本或 feature 语义变化时，按明确规则失效缓存。

### 2.2 Warm start

- 按 `ContactId` 持久化整个 `[lambda_n, lambda_t1, lambda_t2]` block。
- warm-start cache 是显式模拟状态，必须进入 reset、checkpoint、固定 seed replay 和 hash。
- 固定迭代数不因残差提前退出；warm start 对截断求解可见，因此不能被当作纯性能缓存。
- 消失接触、重新匹配、法线翻转和材料变化的衰减/失效规则必须测试。

### 2.3 ContactBlock3

```text
lambda = [lambda_n, lambda_t1, lambda_t2]
lambda_n >= 0
sqrt((lambda_t1 / mu1)^2 + (lambda_t2 / mu2)^2) <= lambda_n
```

- 首期至少支持各向同性 `mu1 == mu2`，ABI 为以后各向异性保留两个系数。
- 切线基由 canonical normal 确定，近轴向时使用固定 tie rule。
- block 与标量 bilateral/limit rows 共享 island 和 reaction dispatch。
- 迁移后删除生产四棱锥 spokes，不保留按场景切换的第二套摩擦物理。

### 2.4 Canonical soft contact

外部入口：

- MJCF：`solref/solimp` 及材料属性。
- Nuka：面向用户的 `tau/zeta` 或 `stiffness/damping` 表达。

两者在 cook 时转成版本化 canonical profile。生产 assembly 不解释源格式，只消费 canonical coefficients。

材料对合并必须定义：优先级、默认值、单位、摩擦组合、软接触组合、正则化下限、`dt/substep` 缩放和序列化。首期至少提供：

- Go2 脚垫 x 摄影棚/训练地面；
- Go2 机身 x 地面；
- 未显式配置的默认材料对。

### 2.5 Joint limit rows

- 从 URDF/MJCF/SceneIR cook 真实 lower/upper 与 effort limit 到生产 `nk::Model`。
- 到达 lower/upper 时生成单边 limit row，使用相同排序、island、正则化与 reaction 路径。
- limit row 不冒充 actuator clamp；角度限位和扭矩限幅分别验证。
- Go2 与 MuJoCo 对照必须使用相同关节范围。

## 3. 建议拆分

| 子任务 | 内容 | 独立 gate |
|---|---|---|
| P1.1 | canonical contact descriptor、stable ID 与 manifold matching | 缓慢滑动/滚动接触跨帧 ID 稳定 |
| P1.2 | cache 生命周期、reset/checkpoint/replay 接线 | cache 恢复后固定步数 byte replay |
| P1.3 | `ContactBlock3` ABI、analytic projection、solver integration | 锥内/锥外/零法向/各向异性单测 |
| P1.4 | canonical contact profile 与材料对 cooker | MJCF/Nuka 等价输入 cook 后 byte-equal |
| P1.5 | production joint-limit rows | 单关节撞上下限和 Go2 全关节 sweep |
| P1.6 | 组合回归与性能容量更新 | 堆叠、斜坡、单脚支撑、body-particle |

## 4. 验收标准

- [ ] 相同场景重复运行 contact ID 序列 byte-identical。
- [ ] warm start 开启后在相同固定迭代预算下残差/滑移不劣于关闭，并保持 D1。
- [ ] checkpoint 包含 cache；省略 cache 的负向测试能检测 replay 差异。
- [ ] 椭圆锥投影满足解析约束，旋转切线基不会改变各向同性结果。
- [ ] 生产路径不再发出四个 friction spokes。
- [ ] MJCF 和等价 Nuka 参数 cook 到同一 canonical profile。
- [ ] 脚垫和机身可以按材料对使用不同 profile，无 scene-name 分支。
- [ ] Go2 所有关节无法越过 cooked limits；effort clamp 保持真实规格。
- [ ] rigid、articulation 和 particle side 继续使用同一 contact block solve。

## 5. 失败诊断

至少输出：active contact 数、ID 命中/失效率、cache age、normal/tangent lambda、cone projection 次数、support-foot slip、limit row active 数、最大限位误差和每阶段 GPU 时间。

## 6. 非目标

- 不加入速度阈值驱动的 static/dynamic friction 离散切换。
- 不做 torsional/rolling friction；等 stable frame 与 block ABI 稳定后再评估。
- 不以 data-dependent early exit 换取速度。
- 不在 P1 中训练最终策略；技能训练由 P3/P4 承担。
