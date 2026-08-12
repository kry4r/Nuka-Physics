# Nuka Physics 物理与渲染收敛路线

状态：本批需求已收敛，待实现
来源：Pi session `019ff3a6-06f4-7d9c-a5e4-0ceed3077a43`
日期：2026-08-12

## 1. 目标

暂停多环节 gallery/corridor 展示扩展，把近期工作收敛到三条产品主线：

1. 完成跨系统 Row/Reaction 边界，优先提升接触稳定性与高动态可信度。
2. 以两个重新设计的 Go2 独立技能作为物理压力测试。
3. 先提升离线光追速度，再扩展 RT 后端；3DGS 作为独立静态查看模块。

本批文档记录已经拍板的决策、由源码确认的事实和可以独立执行的小任务。物理、Go2、离线光追、PHI RT 多后端与静态 3DGS 的首期边界均已冻结；后续扩展须另行版本化确认。

实施以三份阶段 spec 为主，细分 roadmap 是其设计依据和验收附录：

1. [Spec 01：物理补全](2026-08-12-spec-01-physics-completion-zh.md)
2. [Spec 02：Go2 双 Demo](2026-08-12-spec-02-go2-dual-demos-zh.md)
3. [Spec 03：渲染补强](2026-08-12-spec-03-rendering-enhancement-zh.md)

三个阶段按 gate 串行：Spec 01 冻结物理/资产/profile，Spec 02 产出通过批量 gate 的冻结轨迹，Spec 03 只消费已通过的轨迹制作离线结果。

## 2. 已确认边界

### 2.1 物理统一边界

- 统一跨系统交互边界，不把所有内部算法改写成 Row。
- 刚体、关节链、刚体/关节链与粒子的外部接触继续汇入共享 `NkRow -> SolveRowsBlockIsland` 路径。
- XPBD 内部形变约束、PBF 密度投影、MLS-MPM P2G/Grid/G2P 保留专用求解器。
- 驱动保留为广义力，经 ABA 进入系统；Row 负责接触、摩擦、限位、等式约束和跨系统耦合。
- MPM 不按网格节点生成 Row；其刚体/关节链反作用统一到确定性的 `ReactionProvider`/广义冲量合同。
- 当前 `NkRow` 的生产覆盖与缺口见 [Row/Reaction 审计](2026-08-12-row-reaction-production-matrix-zh.md)。

### 2.2 接触基线

- 第一优先级是稳定 contact identity、持久 warm start 和摩擦质量。
- 接触 ID 采用特征优先复合 ID：canonical shape pair、两侧 feature、manifold slot；无法取得 feature 时才回退到量化局部点。
- 摩擦从四棱锥标量 spokes 演进为固定三维椭圆锥 `ContactBlock3 = [normal, tangent1, tangent2]`。
- 软接触采用“双入口、单一 cook”：MJCF `solref/solimp` 与 Nuka 直观参数都编译为 canonical cooked contact profile，运行时只读取 canonical 数据。
- canonical contact profile 按材料对配置，至少能区分脚垫-地面、机身-地面和默认接触。
- 生产关节限位必须在两个技能训练前进入统一 Row/Block 基线。

### 2.3 Go2 技能

- 旧双后腿倒立和旧后空翻任务不可作为训练起点或验收依据，只能保留为失败经验。
- 新建两个独立技能、独立 gate，共享相同的直接扭矩 actuator 合同。
- 训练允许参考轨迹、优化结果和关键姿态；正式评估仅允许 policy 输出扭矩，不允许脚本辅助。
- 正式 gate 使用不少于 256 个并行环境、小范围初态与物理随机化，成功率至少 80%，并支持固定 seed 的 D1 重放。
- 外部物理对照使用同资产 MuJoCo，比较事件和统计指标，不要求逐步数值相同；Newton 只作为机制参考。

### 2.4 渲染与 3DGS

- Go2 视频使用纯摄影棚，不把 3DGS 绑定为背景。
- 物理 rollout 通过 gate 后冻结轨迹，离线独立重渲染。
- RTX 4000 Ada 20GB 上，摄影棚 1920x1080 快速迭代档目标约 1 秒/帧，包含软阴影、基础反射和开源/自写降噪。
- 这不是实时 viewport 目标；最终成片按质量决定，不设硬时间上限。
- 允许按资源版本持久化静态 BLAS、纹理/HDRI 和灯光采样结构；每帧只更新动态实例、TLAS 和必要数据。
- 可引入 OptiX 与 Vulkan RT；普通 CUDA kernel 不能通过公开 CUDA 指令直接调用 RT Core。
- 自写 CUDA/MUSA 保留为 sensor/query、reference 和 fallback；离线 beauty 允许 RT Core provider 成为性能主线。PHI 仍由 Nuka 自写并统一资源、调度和能力查询，内部可调用 OptiX/Vulkan RT/厂商 RT API。
- RT 能力通过 `phi::Device` 的版本化 `nuka.phi.rt.v1` 扩展发现；持久 BLAS/TLAS/trace 不进入固定物理 `NkOp` 表。
- RT provider 按 workload 和 required capabilities 自动选择，并允许显式 override；强制 provider 缺能力时直接报错，自动模式只在允许 fallback 时显式降级。
- RT provider 采用两级能力：必选 `trace_query_v1` 统一 AS、ray/hit 与基础 AOV；可选 `native_beauty_v1` 允许 OptiX/Vulkan RT 等融合完整离线渲染路径。
- RT Core provider 按 OptiX 优先实施：先接 `trace_query_v1`，再做 `native_beauty_v1`；Vulkan RT 在 ABI 稳定后作为第二 provider。
- OptiX 首期 geometry 仅覆盖 triangle mesh 与 rigid instances；analytic primitive 先确定性 tessellation，custom primitives 延后。
- OptiX 首期只持久化静态/刚性 BLAS并逐帧更新 TLAS transforms；deforming BLAS 与 motion blur AS 延后。
- OptiX `native_beauty_v1` 首期采用同一套可配置多 bounce 摄影棚 PBR 积分器；preview 有界，final 可提高 diffuse/specular/transmission bounce。
- 首期 path sampling 使用 NEE + MIS、灯光/HDRI importance sampling 与 Russian roulette；ReSTIR 等历史采样延后。
- preview 首期同时提供无 history 的单帧空间降噪与受控时域累积/降噪；时域路径使用冻结轨迹的 camera/rigid-instance motion，并对 disocclusion、运动边缘和资源/profile/provider 变化精确失效 history。
- 降噪归属 provider-independent 的共享 post-process；自写/开源实现是基线，OptiX 等原生 denoiser 仅通过适配器可选加速，所有路径共用 canonical raw AOV、history、失效与 telemetry 合同。
- 首期共同灯光支持 directional、point、spot、rectangle、disk 与 HDRI；显式面积灯参与 NEE/MIS，通用 emissive mesh 与其三角形功率分布延后。
- 首期 transmission 支持薄片与受限实体 dielectric；实体限定为闭合、单层、均匀 IOR，支持 Fresnel、全反射及 Beer-Lambert 吸收，嵌套介质、体散射和色散延后。
- 首期 canonical AOV 支持 linear HDR color、albedo、shading normal、linear depth、motion、instance/material ID、sample count/variance 与 validity mask，并只按 profile 请求分配；motion 固定为无 jitter 的 `previous_pixel - current_pixel`。
- 1080p 约 1 秒 preview 的具体 spp/bounce/RR/降噪参数先由 RTX 4000 Ada 固定 benchmark 收敛，再冻结为可复现、不可同名覆盖的版本化 preset；不使用逐帧动态画质冒充该 gate。
- OptiX provider 是可选构建模块：构建时使用外部 SDK headers，运行时动态发现并校验 driver ABI、同一 CUDA device 与 capabilities；不可用时只按显式 fallback policy 处理，不影响 core PHI/self-CUDA 构建运行。
- RT 子系统不承诺同 provider 或跨 provider D1；固定输入/seed 只用于诊断，验收采用 hit/AOV 容差、ID/拓扑规则、多次统计画质和序列稳定性 gate。物理与 Go2 rollout 的 D1 合同不受影响。
- 首期支持二值 alpha mask/cutout，并对 camera、shadow 和 secondary ray 使用相同纹理/cutoff 可见性；连续或随机 alpha、排序混合延后。
- 3DGS 首期只加载 Graphdeco 风格 PLY 的静态预训练场景，cook 为 Nuka 缓存，提供 CUDA tile renderer 和 color/depth/opacity 输出；不训练、不编辑、不与 mesh 合成。

## 3. 实施依赖

```text
Spec 01 物理补全
  P0 Row/Reaction -> P1 接触/限位 -> P2 MuJoCo oracle
       |
       v
Spec 02 Go2 双 Demo
  P3 单前腿倒立五连跳 + P4 站立后空翻
       |
       v frozen trajectories
Spec 03 渲染补强
  R1 持久场景/benchmark -> R2 OptiX 优先 RT

G1 静态 3DGS 可独立推进，不阻塞三阶段主 gate。
```

物理基线先行，随后两个技能可以并行。离线光追基准应先于后端替换，以便区分 Nuka 自身浪费与 RT Core 的结构性收益。

## 4. 小任务文档

| ID | 文档 | 交付 |
|---|---|---|
| S1 | [Spec 01：物理补全](2026-08-12-spec-01-physics-completion-zh.md) | Row/Reaction、接触/限位、MuJoCo oracle 的阶段合同 |
| S2 | [Spec 02：Go2 双 Demo](2026-08-12-spec-02-go2-dual-demos-zh.md) | 两个独立 policy、批量 gate、冻结轨迹 |
| S3 | [Spec 03：渲染补强](2026-08-12-spec-03-rendering-enhancement-zh.md) | 持久场景、OptiX 优先 RT、多 bounce、降噪与成片 |
| P0 | [Row/Reaction 生产矩阵](2026-08-12-row-reaction-production-matrix-zh.md) | 定义什么必须统一、什么保留局部求解 |
| P1 | [接触与 Row/Block 基线](2026-08-12-contact-row-block-foundation-zh.md) | stable ID、warm start、椭圆锥、软接触、限位 |
| P2 | [MuJoCo 对照验证](2026-08-12-mujoco-go2-contact-oracle-zh.md) | 同资产事件级与统计级 oracle |
| P3 | [Go2 单前腿倒立连续跳](2026-08-12-go2-single-front-handstand-hop-zh.md) | FL/FR 随机指令、5 跳、四足落地 |
| P4 | [Go2 站立起跳后空翻](2026-08-12-go2-standing-backflip-zh.md) | 无 phase clock 的直接扭矩策略 |
| R1 | [离线光追提速](2026-08-12-offline-raytracing-performance-zh.md) | 基准、阶段剖析、持久场景、1 秒预览 gate |
| R2 | [PHI RT 多后端能力](2026-08-12-phi-rt-multibackend-capability-zh.md) | 冻结的能力/ABI 边界、OptiX 优先、多 bounce 与双路径降噪 |
| G1 | [静态 3DGS 查看器](2026-08-12-static-3dgs-viewer-zh.md) | PLY 导入、cooked cache、CUDA tile splat、AOV |

## 5. 全局非目标

- 不继续新增 corridor/gallery 环节、场景或策略展示。
- 不把专用连续介质内部迭代强行 Row 化。
- 不把旧 Go2 技能改名包装为新 gate。
- 不做实时光追目标，不用降低分辨率或移除核心效果冒充优化。
- 不把 3DGS、Go2 和 mesh/GS 合成绑成一个首期任务。
- 不在完成阶段剖析前以更换 RTX 5080 代替软件优化。

## 6. 全局完成标准

- 每个实现任务有独立测试或可自动运行的 gate。
- 物理改造保持固定调度、固定迭代和 D1 重放；warm-start 状态进入 checkpoint/replay 合同。
- 性能报告同时记录场景、GPU、分辨率、spp、bounce、降噪、构建、遍历、着色和传输耗时。
- 所有失败明确归因于实现、物理模型、任务定义、资产对照或硬件上限，不只保留最终视频。
