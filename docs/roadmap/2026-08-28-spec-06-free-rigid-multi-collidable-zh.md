# Spec-06: 自由刚体的多碰撞体与碰撞体局部变换

日期: 2026-08-28
状态: 设计
参考: robosuite `PandaGripper` / LIBERO `akita_black_bowl`(约 40 个带偏移的薄盒),
      既有 multi-geom link proxy 机制 (`cook_to_model.cpp` `ProxyCollidableSpec`),
      `tests/scenario/multi_geom_cook.cpp`

## 背景

pi0.5 LIBERO 黑碗任务在感知与控制全部对齐官方后仍无法抓取。逐项排除得到的结论是
场景几何保真度不足, 而根因在引擎的 cook 阶段, 有两个叠加缺陷。

### 缺陷 1: 自由刚体的碰撞体局部变换被静默丢弃

判别实验: 把黑碗碰撞盒的 `pos` 从 `0 0 0.028` 改为 `0 0 0.100`, 静置高度**完全不变**
(两次都是 `0.925979`)。而 `桌面 0.900 + 半高 0.026 = 0.926` 正是"代理盒以 body 原点为
中心"的预期值 —— 局部偏移未进入碰撞路径。

渲染路径**使用**该偏移, 于是视觉与物理错开 2.8cm: 视觉上物体悬空。

### 缺陷 2: 自由刚体只保留第一个碰撞体

`cook_to_model.cpp` 每个 body 只取其第一个碰撞形状写入 shape_table 行, 其余丢弃。

### 共同根因

`cook_to_model.cpp:538`:

```cpp
for (uint32_t link = 0; link < m.link_count; ++link) {
    if (host.link_body[link] == body) { owner_link = link; break; }
}
if (owner_link == ~uint32_t(0)) {
    continue;              // body is not an articulation link
}
```

"一个刚体挂多个碰撞体 + 每体局部变换"的通用机制**已经存在并在运行**, 但入口被这道
`continue` 限制在 articulation link 上, 自由刚体走不进去。既有设施:

| 设施 | 位置 |
|---|---|
| 额外碰撞体 -> 追加 collidable body row | `ProxyCollidableSpec` / `cook_to_model.cpp` |
| 每 proxy 的局部变换 (完整 Transform) | 模型字段 `body_collidable_local` |
| 每步 `owner_pose ∘ local` 合成世界位姿 | `sync_body_pose.cu` `SyncProxyCollidablePoseKernel` |
| 首碰撞体的局部偏移 | `link_geom_local`, `SyncLinkBodyPoseKernel` 内 compose |
| 排除对继承 (不与 owner / 兄弟自碰) | `cook_to_model.cpp` proxy 块 |
| 接触反作用力解析回 owner | `assemble_rows.cu` `ResolvePairSide` 经 `body_to_link` |

Panda 手指之所以正常, 是因为每根手指是独立 link 各挂一个 pad, 从未触发该限制。

### 为什么这挡住了抓取

黑碗网格 (scale 0.7) 实测 `0.1118 × 0.1113 × 0.0526 m`, 碗沿半径 `0.0562`。Panda 单指
行程 `0~0.04` 即最大开口 `0.08`, 可夹外径上限约 `0.075` (半径 `< 0.037`)。

策略伸向半径约 `0.05` 的碗沿 —— 这是正确的抓握策略, 不是误差。单个居中盒子无法同时
满足"出现在可见碗沿位置"与"塞进夹爪行程": 做成真实碗沿宽度就夹不上, 做成可夹宽度
碗沿处就没有碰撞体。薄壁必须由多个带偏移的碰撞体表达, 而这正是上述限制所禁止的。

## 需求决策 (grill-me)

| 问题 | 决策 | 依据 |
|---|---|---|
| 扩 shape_table 行 + 改各 narrowphase kernel? | 否 | 实测既有 proxy 机制已具备全部要素, 重写属重复建设 |
| 改动切片 | 把 proxy 的 owner 从"link"推广到"任意刚体" | 让自由刚体复用 link 在用的同一条通路, 符合 ONE general path |
| 自由刚体首碰撞体的偏移怎么办? | 全部碰撞体都变 proxy, owner row 置为不碰撞 | 首/次形状无特例, 语义统一 |
| owner body row 是否仍参与动力学? | 是 | 它是自由刚体的动力学状态与积分对象, 只是不再碰撞 |
| 单碰撞体场景是否变化? | 不变, 不生成 proxy | 保持既有场景 cook 产物 byte-identity |
| 新增窄单测? | 否 | agent.md 禁止; 折进 LIBERO demo + 既有 scenario 回归 |
| 验证判据 | 官方 `(On akita_black_bowl_1 plate_1)` 语义 | 真实物理状态, 不用抬升启发式冒充 |
| 碰撞体类型限制 | 放开当前仅 Sphere/Box/Capsule 的入口过滤 | 该过滤同在缺陷入口处, 一并处理 |

## 设计

### 1. cook: owner 从 link 推广到刚体

`ProxyCollidableSpec` 增加自由刚体 owner 的表达 (owner_link 保持 `~0u`, 新增
owner_body 语义为"位姿源 body row")。入口循环去掉 `owner_link == ~0u` 的 `continue`:

- owner 是 articulation link: 行为完全不变 (首形状折进 link_geom, 其余成 proxy)。
- owner 是自由刚体, 且该 body 的碰撞形状数为 1 且局部变换为恒等: 行为不变, 不生成
  proxy (保证既有场景 byte-identity)。
- 其余情况: 该 body 的**每一个**碰撞形状都成为 proxy row, owner row 的
  `contype/conaffinity` 置 0 使其退出 broadphase 配对, 动力学积分不受影响。

proxy row 沿用既有布局: `inv_mass = 0`、`body_collidable_local = shape.local_transform`、
`contact_profile_index = shape.material_bucket`、排除对继承同现有逻辑。

### 2. 模型字段: 记录 proxy 的 owner body

新增 `body_collidable_body` (`per:body`, u32, 模板局部 body row, `~0u` 表示非 proxy),
与既有 `body_collidable_link` 并列。`fields.yaml` 加一行后重跑
`tools/codegen/fields/gen_fields.py` 重生成 `src/nk/model/generated/*`,
并在 `model.cpp` 的字段 staging 与 `MoveModelMembers()` 中补齐。

### 3. 位姿同步: 支持 body owner

`sync_body_pose.cu` 的 proxy 位姿 pass 增加 body-owner 分支, 由
`body_pose[owner] ∘ body_collidable_local` 合成。该 op 位于 `FkWorldPoses` 之后、
`BuildAabbs` 之前, 此时 owner 的 `body_pose` 已是本步待用位姿, 无需新增 op。

需同时放宽 `OpSyncLinkBodyPose` 的前置守卫: 当前 `link_pose == nullptr` 或
`links_per_env == 0` 会整体早退, 使纯刚体场景 (无 articulation) 的 body-proxy 同步被
跳过。改为 link pass 与 body-proxy pass 各自独立判定。

### 4. 反作用力: proxy 重定向到 owner

这是本切片中唯一需要触碰 `assemble_rows.cu` 的地方, 与最初预估不同。

`ResolvePairSide` 的自由刚体分支目前返回 `env * bodies_per_env + local_body`, 对
proxy row 即其自身。下游 `body_inv_mass[idx]` 与 `com = body_pose[idx].position` 都会
取到 proxy 行 —— 而 proxy 的 `inv_mass = 0`、位置也不是 owner 的, 结果是反作用力被
静默丢弃。link proxy 不受影响是因为它经 `body_to_link` 走 artic 分支。

处理: 把 `body_collidable_body` 传入该 kernel, 在 `ResolvePairSide` 做 body-row 查找
之前先把 proxy row 重定向到 owner row。link proxy 重定向到 owner body 后仍经
`body_to_link` 解析到同一 artic/link, 行为不变。这是通用重定向, 不是特例分支。

## 验证

按 agent.md, 不新增窄单测。

1. **主验证 — LIBERO demo 真实 rollout**: 黑碗碰撞体改为带偏移的多盒薄壁 (碗沿可夹),
   跑 4s OSC rollout, 以官方 `(On akita_black_bowl_1 plate_1)` 语义判定: 黑碗被夹住、
   离开桌面、最终与盘子接触且 XY 距离 `< 0.03m`。
2. **回归 — 既有 scenario**: `multi_geom_cook`、`pairdriven_*`、`vproof_go2_ground`、
   `vproof_h1_grasp`、`island_byte_identity`、`multi_dog_costep` 必须保持通过。
3. **byte-identity**: 单碰撞体场景的 cook 产物不得变化。

同时修正本次排查中发现的场景侧问题: 黑碗代理盒半高应为 `0.0263` (真实碗高 `0.0526`),
先前写成 `0.0425`; demo 成功判据改用官方 `On` 语义。

## 风险

- 多碰撞体自由刚体会增加 body row 数, 进而抬高 LBVH 叶子数与 candidate 预算。薄壁碗
  约 40 个碰撞体即 +40 row/碗, 需实测 4s rollout 的步耗与显存。若代价过高, 退化方案是
  用更少的盒子近似碗沿 (例如 8~12 个), 保真度与开销折中。
- `contype/conaffinity` 置 0 的 owner row 仍留在 LBVH 中作为叶子, 只是配对被过滤。若
  实测发现空转开销显著, 再考虑从叶子集合中剔除。
- 排除对数量随 proxy 数增长 (owner + 兄弟两两), 40 个碰撞体约产生 800 对。
  `max_excluded_pairs` 按实测值增长, 需确认排除表查询是线性扫描还是有序二分。
