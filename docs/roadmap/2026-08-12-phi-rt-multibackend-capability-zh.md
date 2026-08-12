# R2：PHI 下的 RT 多后端能力设计

状态：需求已收敛，待实现
依赖：[R1 离线光追性能基线](2026-08-12-offline-raytracing-performance-zh.md)

## 1. 已确认事实

1. 普通 CUDA kernel 没有公开的 CUDA 指令/API 可以直接把 ray 提交给 RT Core。
2. NVIDIA RT Core 的公开入口是 OptiX、Vulkan RT 或 DXR 等光追运行时/着色语言路径。
3. OptiX device program 使用 CUDA 风格代码和 `optixTrace`，但它是由 OptiX 管理、编译和启动的管线，不等于任意现有 CUDA kernel 内直接调用 RT Core。
4. Vulkan RT 同样通过 acceleration structure 和 trace-ray/ray-query 接口触发硬件遍历。
5. CUDA 可以和 OptiX 在同一 CUDA context/device 上共享指针、stream 和 event；CUDA 与 Vulkan 可通过 external memory/semaphore 或同设备 staging/interop 同步。
6. “高质量实时”不只来自 RT Core：低 spp、重要性采样、wavefront/queue 调度、时空复用和降噪同样决定最终表现。
7. R1 中重复 scene build/upload 属于 Nuka 可先修的软件浪费；修复后若 traversal/secondary rays 占主导，自写 CUDA 与 RT Core provider 才呈现结构性差距。

## 2. 当前仓库边界

- `phi::BackendI` 管理物理 op dispatch、plan、event 和 synchronize。
- `phi::DeviceI::supports_op` 只查询 `NkOp`，没有 RT scene/build/trace 能力描述。
- `RegistryEntryI::get_proc_address` 可作为扩展发现机制，但当前没有版本化 RT extension ABI。
- `ActiveBackend()` 表达物理与渲染共用设备/主 stream 的意图。
- `render::RtBackendI` 已经抽象 scene build/render/free，但当前工厂 `CreateCudaRtBackend()` 直接绑定唯一实现，能力粒度不足。
- `RenderWorld` 是 backend-neutral host scene，适合作为共同上游，但需要补齐灯光、相机、材料和增量资源语义，不能把当前 adapter 的信息丢失固化为跨后端 ABI。

## 3. 推荐的架构概念

RT 能力作为 `phi::Device` 的版本化扩展被发现，不建模为普通 `NkOp`：

```cpp
void* ext = RegistryEntryGetProcAddress(registry, "nuka.phi.rt.v1");
```

实际实现可以把查询入口下沉到 device，但扩展必须绑定明确的 `phi::Device`/物理设备，不能依赖进程级隐式全局 GPU。扩展返回带 `abi_version` 和 `struct_size` 的纯 C vtable；调用方先协商版本，再枚举该设备上的 RT provider。

这个边界的原因是：

- `NkOp` 表达固定物理 step 中对 `ModelView/DataView` 的短生命周期操作；
- RT scene、BLAS 和资源缓存跨很多物理 step/渲染帧持久存在；
- render graph、AOV 和 provider 能力是可变的，不应扩大所有物理 backend 的 `NkOp` 表；
- RT 扩展仍复用 PHI 的 device、buffer、event 和同步合同，而不是形成第二套设备发现。

```text
SceneIR / RenderWorld
        |
        v
Versioned RT Scene IR + Render Profile + AOV Contract
        |
        v
RT capability selection on a PHI device
        |
   +----+-------------------+
   |                        |
self-CUDA provider     RT-Core provider(s)
CUDA cores             OptiX / Vulkan RT
   |                        |
   +---- CUDA compute ------+
        shading / queues / denoise / compose
```

共同上层应只看：

- geometry/instance/material/light/camera 语义；
- scene/resource version 与增量 update；
- acceleration-structure build/update 能力；
- trace mode、ray type、payload/AOV；
- device-memory ownership、stream/event/synchronization；
- feature/capability query、随机采样配置和性能 telemetry。

## 4. 候选能力维度

### 4.1 已确认的扩展职责

`nuka.phi.rt.v1` 至少负责：

- 枚举绑定到当前 `phi::Device` 的 RT providers；
- 查询 provider capabilities、名称、厂商和版本；
- 创建/销毁 provider context；
- 创建、更新、压缩和销毁 BLAS/TLAS 或等价 opaque scene objects；
- 提交 trace/ray-query 工作并写入 PHI device buffers；
- 与 PHI event/stream 或显式 external semaphore 建立同步；
- 返回结构化 status、unsupported reason 和阶段 telemetry。

scene handle、AS scratch 和 provider context 由产生它们的 provider 所有，不能跨 provider 混用。上层 `RenderSession` 持有这些 opaque handles，物理 pipeline 不持有它们。

ABI 采用 append-only 演进：`abi_version` 标识语义版本，`struct_size` 允许旧调用方忽略尾部字段。任何必需函数缺失或版本不兼容都显式返回 unsupported，不回退到未声明实现。

### 4.2 Provider 身份

- `self_cuda`：自写 BLAS/TLAS、CUDA-core traversal、交点和 shading。
- `optix`：首个 RT Core provider；OptiX acceleration structure 与 `optixTrace`，CUDA compute 可承担前后处理。
- `vulkan_rt`：Vulkan acceleration structure 与 trace ray/ray query，通过 interop 接 CUDA 计算。
- 未来厂商 provider：例如 MUSA 自写遍历或厂商硬件 RT API；不假设其 API 与 OptiX 同构。

### 4.3 必须查询而非猜测的能力

- triangle/custom primitive、motion transform、instance count/capacity；
- BLAS compact/update、TLAS refit/rebuild；
- ray pipeline 与 inline ray query；
- callable/custom intersection、payload/attribute 限制；
- same-device zero-copy、external memory、semaphore/event interop；
- device-side output、AOV 格式、denoiser availability；
- build/traversal 的已知数值与调度差异；
- profiling timestamp 和阶段计时支持。

### 4.4 已确认的两级 Provider 能力

所有 RT provider 必须实现基础层，可选实现 native beauty 层。

**必选 `trace_query_v1`：**

- 创建/更新 acceleration structures；
- 接收批量 rays 或 provider 支持的等价 trace dispatch；
- 输出 canonical hit records 与请求的基础 AOV；
- 支持 device-resident input/output、显式同步和阶段 telemetry；
- 可被共享 sensor/query 与 wavefront renderer 使用。

canonical hit record 至少包含 hit/miss、ray index、distance、instance/primitive ID、barycentric 或等价局部坐标、geometric normal 和 material reference。字段布局需要版本化；provider-specific payload 不得泄漏到共同上层。

**可选 `native_beauty_v1`：**

- provider 内部可融合 ray generation、closest-hit/miss、材质、灯光、路径积分和队列调度；
- 可利用 OptiX/Vulkan RT 的 payload、shader binding、SER 或厂商专有调度能力；
- 必须消费共同 RT Scene IR、Render Profile 和材质/灯光语义，并输出共同 AOV 合同；
- provider 可声明不支持的材质/灯光特性，但不能静默近似。

选择规则：

- sensor/query 与 reference workload 只要求 `trace_query_v1`，默认走 self-CUDA/self-MUSA；这表示实现定位，不构成 D1 承诺。
- offline beauty 优先选择满足 profile 的 `native_beauty_v1`。
- 没有合适 native beauty 时，可由 `trace_query_v1` 加共享 wavefront shading 完成；是否允许该 fallback 仍遵循 session 的 fallback policy。

这不是要求每个 provider 复制整套 renderer。基础 trace/hit 语义必须统一，native beauty 只在能带来实质性能或平台收益时实现。

### 4.5 OptiX 首期 Geometry 范围

首期只覆盖 triangle mesh BLAS 与 rigid instances/TLAS：

- Go2 每个视觉 link mesh 和摄影棚静态 mesh 都编译为 triangle BLAS；
- 同 geometry 的多个实例共享 BLAS；
- analytic sphere/capsule 等先通过现有或补充的确定性 tessellation 转为 triangle，不在首期写 OptiX custom intersection program；
- tessellation 参数、坐标、法线、UV 和 source hash 进入 scene/resource version，确保 provider 间使用同一几何；
- SDF、procedural primitive、curve 和 deforming vertex buffers 不属于首期 OptiX gate；
- capability 必须精确广告 `triangles` 与 `rigid_instances`，不得用首期 tessellation fallback 假装支持 `custom_primitives`。

这样首个对比直接覆盖 Go2 摄影棚 workload，并把变量限制在 AS 生命周期、RT Core traversal、共享 shading 和同步开销。

### 4.6 OptiX 首期动态 AS 范围

首期只支持静态/刚性 BLAS 与逐帧 TLAS instance transform 更新：

- 摄影棚静态 geometry 和 Go2 各 link 的 triangle BLAS 按 geometry version 构建一次；
- 每帧只上传或直接读取 link instance transforms，并 update/rebuild TLAS；
- TLAS 选择 update 还是 rebuild 由 capability 和 benchmark 决定，但必须分别计时并写入 metadata；
- 不因 instance transform 变化重建对应 mesh BLAS；
- 首期不支持 cloth/soft-body 等 deforming vertex buffer 的 BLAS update/rebuild；
- 首期不支持 transform/vertex motion blur acceleration structures；离线 motion blur 可在后续通过时间采样或 motion AS 单独收敛。

scene/resource version 必须区分 geometry、instance layout 和 per-frame transforms：transform 变化只使 TLAS/frame state 失效，geometry 变化才使对应 BLAS 失效。

### 4.7 OptiX `native_beauty_v1` 首期视觉范围

首期实现一套可配置、多 bounce 的摄影棚 PBR path integrator，preview 与 final 不分叉成两套 renderer。

共同能力：

- metallic/roughness PBR；
- albedo、normal、roughness textures；
- HDRI 与多灯光直接采样；
- area-light soft shadows；
- diffuse、specular 与 transmission path continuation；
- configurable maximum bounce、Russian roulette 起始深度和每类 lobe 开关；
- color、depth、normal、albedo、motion/ID 等降噪或输出所需 AOV，其中 motion AOV 只有在不依赖 motion AS 的相机/instance transform 语义明确后启用。

profile 约束：

- `offline_preview` 使用较低 spp 与有界 bounce，首个性能 gate 至少保留软阴影、基础反射和最多一跳漫反射 GI；
- `offline_final` 使用同一积分器，可提高 diffuse/specular/transmission bounce 与 spp，不设固定帧时限；
- 多 bounce 是首期积分器能力，但不要求高 bounce 配置达到 1080p 约 1 秒/帧；
- provider metadata 必须记录各类最大 bounce、实际 path-depth 分布、spp 和 Russian roulette 设置。

材质或 profile 要求超出 provider capability 时显式 unsupported 或按允许策略 fallback，不能静默关闭 transmission/GI。

### 4.8 首期路径采样

首期使用无历史状态的经典 path-tracing 基线：

- BSDF importance sampling；
- 每个非 delta bounce 执行 next-event estimation；
- analytic/area lights 与 HDRI 使用持久化 importance distributions；
- light sampling 与 BSDF sampling 使用 MIS，权重启发式和 PDF 单位写入版本化积分器合同；
- delta lobe、emissive hit、environment miss 和 transmission 的 MIS 计权必须有独立测试；
- 深路径使用 Russian roulette，起始深度与 survival probability 由 Render Profile 记录；
- 随机数基于显式 frame/sample/pixel/bounce/lobe key，不能依赖线程调度或全局递增状态。

ReSTIR DI、temporal/spatial reservoir reuse 与其他历史采样不进入首期正确性基线。它们可在 NEE/MIS、AOV 和 history invalidation 合同稳定后作为 `offline_preview` 可选能力加入。

### 4.9 首期 Preview 降噪

首期同时提供两条可选择的 preview 降噪路径：

- 单帧空间降噪是无历史依赖的必备 fallback；单独渲染任意帧、history 无效或时域路径不受支持时仍能产出结果；
- 受控时域累积/降噪使用冻结轨迹中相邻帧的 camera 与 rigid-instance transforms 重建 motion vectors，并结合 depth、normal、albedo、material/instance ID 做重投影与拒绝；
- disocclusion、屏幕外重入、深度/法线不连续和运动边缘必须拒绝错误 history，不能用拖影换取表面稳定；
- geometry、material、light、camera cut、Render Profile、积分器版本、provider 或影响采样结果的 scene/resource version 变化时，按资源依赖精确失效 history；无法证明局部失效正确时失效整帧；
- history 状态属于 `RenderSession` 与确定的 provider/profile 组合；随机访问帧和 session/provider 切换不得隐式复用旧 history；
- 每帧 metadata 记录降噪模式、history 有效性/长度、失效原因、motion-vector 版本与 denoise 时间。

时域降噪只消费当前帧及历史 AOV，不改变 4.8 的无历史 path-sampling 基线，也不提前引入 ReSTIR 或 reservoir reuse。

降噪定义为 provider-independent 的共享 post-process 能力，归属 `RenderSession` 的渲染/合成阶段，不进入 `trace_query_v1`、`native_beauty_v1` 或物理 `NkOp` ABI：

```text
trace_query_v1 + shared shading  --+
                                  +-> canonical raw color/AOV
native_beauty_v1 ----------------+             |
                                                v
                                 shared denoise scheduler/history
                                   |          |             |
                              self/open   open-source   provider-native
                                   +----------+-------------+
                                              |
                                      denoised color + metadata
```

- `native_beauty_v1` 必须能输出未降噪的 canonical color 与请求 AOV；不能只返回 provider 私有降噪后的图像；
- 共享调度器根据 denoise profile、可用 AOV、设备互操作和显式选择挑选实现，并统一管理 history、invalidation、同步和 telemetry；
- 开源/自写的单帧与时域实现构成首期必备基线；provider-native denoiser 通过适配器实现同一输入/输出合同，只是可选加速；
- 原生适配器缺少必需 AOV、时域能力或零拷贝互操作时，显式返回 unsupported；仅在降噪 fallback policy 允许时切换到共享基线，并记录原因；
- benchmark 必须可设 `denoise=off` 比较 raw output，也能在相同降噪模式下比较 provider，避免把原生降噪差异归因于 traversal/integrator；
- provider 可广告 denoiser capability，但其私有模型、scratch 与状态由适配器封装，不扩散进共同 RT Scene IR。

### 4.10 首期灯光合同

共同 RT Scene IR 与 `native_beauty_v1` 首期覆盖以下灯光：

- `directional`：方向、辐照度以及可选角直径；角直径大于零时产生可采样软阴影；
- `point`：位置、辐射强度和有限作用范围；首期按 delta light 处理，不用非物理半径冒充面积光；
- `spot`：位置、方向、强度、内外锥角和有限作用范围；首期按 delta light 处理；
- `rectangle`、`disk`：变换、尺寸、单/双面、辐射亮度；作为显式面积灯参与 NEE、可见性与 BSDF/light MIS；
- `HDRI`：环境纹理、旋转和强度；建立基于亮度与球面测度的持久化 importance distribution。

所有灯光使用版本化、单位明确的 canonical 参数，provider 不得把多灯截断为第一盏 point light。light-set topology、参数和 importance-distribution version 分开记录；参数或 HDRI 变化只重建受影响的灯光采样表并使相关降噪 history 失效。

任意 triangle mesh 的 emissive material、mesh-light extraction、按三角形功率建表与动态 emissive mesh 不进入首期。`rectangle`/`disk` 可以被路径直接命中并执行 emissive-hit MIS，但这不等于支持通用 emissive mesh。遇到通用 emissive mesh 请求时必须显式 unsupported 或按允许策略 fallback，不能只让它在相机可见却不参与照明。

### 4.11 首期透明与折射合同

首期材质把透明定义为物理 BSDF transmission，而不是依赖绘制顺序的 alpha blending，并覆盖两种明确模式：

- `thin_walled`：用于玻璃面板等无可解析内部厚度的薄片；参与 Fresnel reflection/transmission，但不改变后续 ray 的介质状态，不应用基于内部距离的体吸收；
- `solid_dielectric`：仅用于闭合、可定向、单层且具有均匀 IOR 的 triangle mesh；支持入射/出射介质切换、Snell 折射、Fresnel、全反射以及按内部路径长度计算的 Beer-Lambert 吸收。

canonical material 至少记录 transmission weight、IOR、thin/solid mode，以及实体介质的 absorption coefficient 或等价 attenuation color/distance；单位、颜色空间和参数换算必须版本化。几何 cook 对 `solid_dielectric` 验证闭合性、一致绕序和退化面，并记录验证结果；不能验证的网格不得静默按实体玻璃渲染。

首期假设 ray 在任一时刻至多位于一层非空气介质中，不支持嵌套/重叠 dielectric、介质优先级、体散射、非均匀介质、色散或 spectral rendering。遇到超出范围的 scene/material 时显式 unsupported 或按允许策略 fallback，不允许忽略吸收、把实体改成薄片或强制使用固定 IOR。

alpha cutout/mask 是独立的几何可见性能力，不能借 `thin_walled` transmission 代替；其首期合同如下：

- 仅支持二值 `alpha_mask` mode；canonical material 记录 opacity texture、通道选择、UV set/transform、sampler/mip/filter 与 `[0,1]` cutoff；
- opacity 按版本化颜色空间/通道规则采样，比较规则固定为 `opacity >= cutoff` 保留命中，否则继续 traversal；纹理缺失、UV 无效和非有限采样值有明确 validation/error 语义；
- camera、shadow、diffuse/specular/transmission secondary ray 均执行相同 mask 判定，不能出现相机可见但不投影、或 shadow ray 把 cutout 当实心面的差异；
- self-CUDA 使用交点过滤，OptiX 使用 any-hit 或等价可验证路径；provider 必须广告 `alpha_mask` capability，缺少该能力时显式 unsupported 或按 policy fallback；
- mip/filter 与 ray-footprint/LOD 规则进入 canonical sampler ABI；首期可以使用版本化的近似 LOD，但不同 ray 类型不得无声明地强制 LOD 0；
- mask 纹理及 sampler version 进入 material/resource cache 和 history invalidation；改变 cutoff 或纹理不能复用陈旧降噪 history。

随机 alpha-to-coverage、stochastic transparency、连续 alpha、排序式半透明 blend 和 colored opacity 不进入首期 path-tracing 合同。mask 通过/拒绝是离散几何可见性，不贡献部分 transmission，也不改变 solid/thin dielectric 介质规则。

### 4.12 首期 Canonical AOV 合同

所有首期 beauty provider 必须具备以下 AOV 生成能力，但仅为 Render Profile 请求的通道分配、写入和传输资源：

| AOV | Canonical 语义 |
|---|---|
| `linear_hdr_color` | tone map、曝光和降噪前的场景线性 RGB radiance；必须可保留 raw noisy 结果 |
| `albedo` | primary ray 首个有效表面的线性 RGB base color，已采样纹理但不含灯光、Fresnel、曝光或 tone map |
| `shading_normal` | 首个有效表面的 world-space、单位长度 shading normal；经过 normal map，并统一朝向规则 |
| `linear_depth` | 从无 jitter 相机 view space 得到的正向深度；miss 为 `+Inf`，不使用非线性的 depth-buffer 值 |
| `motion` | 当前像素到上一帧对应位置的二维像素位移，即 `previous_pixel - current_pixel` |
| `instance_id` | `RenderSession` 内稳定的 `uint32` instance ID；miss 使用保留的 invalid sentinel |
| `material_id` | `RenderSession` 内稳定的 `uint32` canonical material ID；miss 使用保留的 invalid sentinel |
| `sample_count` | 当前 raw color 估计实际累计的每像素样本数，不用 profile 请求值冒充实际值 |
| `radiance_variance` | 与 raw color 同一线性、曝光前空间内的每像素 luminance 样本方差；样本不足时由 validity 标记无效 |
| `validity_mask` | 版本化 bit mask，至少区分 primary hit、finite color/depth、motion valid、variance valid 与 history eligible |

格式由版本化 AOV descriptor 明确并允许设备友好的等价存储，但语义和精度下限固定；ID 不得经过 float round-trip。扩展诊断通道如 world position、geometric normal、primitive ID、roughness 和 direct/indirect decomposition 可以后加，不属于首期必选 capability。

几何/材质类 AOV 统一取 primary camera ray 的首个有效表面，不随后续 diffuse/specular/transmission bounce 改变；这也适用于 thin/solid dielectric 的第一个界面。多样本像素按合同执行一致的 filter/resolve，ID 与 validity 不能做数值平均。

motion-vector 语义进一步固定为：

- 使用无采样 jitter 的当前/上一帧 view-projection 矩阵，以及同一 stable instance ID 的当前/上一帧 rigid transform；
- 对当前首个表面的 object-space 位置分别投影到当前帧与上一帧，以像素为单位计算 `previous - current`；图像坐标原点、像素中心和 Y 轴方向由 AOV ABI 固定；
- 静态物体仍包含相机运动，刚体同时包含相机与 instance motion；HDRI/background miss 只使用相机旋转重投影，不虚构有限深度或相机平移视差；
- 上一帧不存在对应 instance、投影落在有效域外、矩阵不连续、camera cut 或无法建立刚体对应时清除 `motion_valid/history_eligible`；
- motion AOV 不依赖 motion-blur AS，也不表达曝光区间内的速度；deforming geometry 在首期能力外。

这些通道属于 `trace_query_v1` + shared shading 与 `native_beauty_v1` 的共同输出合同，也是共享降噪器唯一可依赖的输入语义。provider-private AOV 可以存在，但不得替代 canonical 通道。

### 4.13 Preview Preset 冻结策略

不在实现和基准数据之前硬编码首期 spp/bounce 数值，也不按单帧耗时动态改变画质。先在固定 Go2 摄影棚 benchmark 与 RTX 4000 Ada 20GB 上执行参数 sweep，再把通过性能和画质 gate 的组合冻结为版本化 preset。

不可调低的 preview 语义下限：

- 1920x1080，目标约 1 秒/帧；时间口径包含 trace、shade、共享降噪与必要 compose，不用隐藏的预热后处理规避总时间；
- 保留多灯/HDRI 的软阴影、基础 specular reflection 和最多一跳 diffuse GI；
- 保留共同 metallic/roughness PBR、首期灯光/材质语义与 canonical AOV/时域有效性，不通过删除核心效果或改变冻结轨迹达标；
- 单帧空间 fallback 与受控时域路径分别过画质 gate；不能只依靠长 history 掩盖单帧不可用。

sweep 可以调整 spp、diffuse/specular/transmission 最大 bounce、Russian roulette 起始深度/概率、采样分配、空间/时域降噪参数和内部 queue/batch 设置。每个候选在相同帧集合上记录帧时间分位数、阶段时间、峰值显存、rays/path-depth 分布、raw/denoised 指标以及闪烁、拖影、disocclusion 和运动边缘结果。

冻结的 preset 至少包含 `preset_id`、语义版本、benchmark/scene hash、目标 GPU、分辨率、积分器/采样/降噪参数、required capabilities 与画质 gate 版本。改变任何影响结果的参数都产生新 ID/版本，不能同名覆盖。provider 可以在相同语义下使用不同内部调度，但若 spp/bounce/采样或降噪参数不同，则必须形成 provider-specific preset variant 并单独验收和报告。

约 1 秒是 benchmark gate，不是运行时 adaptive-quality 控制器。首期正式 preset 对同一输入固定参数，保证可重复比较；未来如增加动态预算模式，必须使用不同 profile 名称且不能替代该 gate。

### 4.14 OptiX 构建与运行时依赖

OptiX provider 作为可选构建模块交付，不成为 Nuka、PHI 或 self-CUDA 的强制依赖：

- 构建开关默认可自动探测，也必须支持显式 `ON/OFF`；显式 `ON` 但找不到兼容 OptiX SDK headers 时配置阶段失败，`OFF` 时不编译或链接任何 OptiX provider 代码；
- SDK 路径由用户或构建环境显式提供/发现，仓库不复制未获许可的 SDK headers；构建产物记录所用 OptiX header/ABI、CUDA toolkit、编译器与 device-program 格式版本；
- host provider 使用 SDK 定义的函数表/ABI，运行时动态加载 NVIDIA driver 提供的 OptiX 实现，并在注册前检查 loader、ABI/driver 兼容性、目标 PHI CUDA device 与所需 capabilities；
- device programs 作为版本化 PTX 或 OptiX IR 构建产物随 provider 交付；module/pipeline cache key 包含代码 hash、OptiX/driver ABI、GPU architecture、compile options 与 payload/AOV layout，不能跨不兼容环境误用；
- OptiX context 必须绑定 provider 选择时的同一 CUDA device/context，不得自行切换 GPU；初始化和 teardown 遵守 `RenderSession`/provider context 生命周期。

运行时发现结果至少区分：

```text
not_built
loader_missing
version_mismatch
unsupported_device
initialization_failed
capability_missing
available
```

只有 `available` 的 OptiX provider 才进入自动候选集合。未构建或运行时不可用时，`auto + fallback=allow` 可按既定规则选择 self-CUDA，并把原始状态和原因写入日志/metadata；`fallback=forbid` 或显式 `--rt-provider=optix` 必须返回结构化错误，不能静默换 provider。self-CUDA 构建、测试和运行不得要求 OptiX SDK 或 NVIDIA OptiX loader 存在。

CI 至少包含无 OptiX SDK 的核心构建，以及有 SDK/兼容驱动环境中的 provider build 与 smoke/conformance；没有 RT GPU 的普通 CI 可以只验证编译和 discovery error paths，不能伪报硬件测试通过。

## 5. 自写计算路径与 RT Core 的定位

已确认采用按工作负载分层：

- 自写 CUDA/MUSA kernel 负责 sensor/query、算法 reference 和无硬件 RT API 时的 fallback。
- 离线 beauty 允许 OptiX、Vulkan RT 或未来厂商 RT API 成为性能主线。
- 共同上层共享 Scene IR、材质、灯光、相机、Render Profile 和 AOV 合同，不要求不同 provider 的内部算法或输出跨 API byte-equal。
- PHI 的实现与生命周期仍由 Nuka 自己掌握；调用 OptiX/Vulkan RT/厂商 RT API 不等于把 PHI 交给外部 SDK。

这里必须区分两个概念：

1. **自写 PHI backend**：Nuka 自己实现设备发现、资源所有权、能力查询、调度、同步、provider 选择和 fallback。
2. **自写 traversal kernel**：光线遍历只运行在 CUDA/MUSA 通用计算核心上。

前者可以在内部调用厂商光追 API 使用 RT Core；后者不能凭普通 CUDA/MUSA kernel 自动获得 RT Core。CUDA PHI 可通过 OptiX 或 Vulkan RT provider 使用 NVIDIA RT Core。MUSA PHI 只有在摩尔线程提供并验证了公开硬件 RT API 时才广告对应 capability；否则必须返回 `unsupported` 或选择 self-MUSA，不得假定 CUDA/OptiX 接口可直接移植。

## 6. 后端公平比较

- 相同 frozen scene、camera、materials、lights、profile 和 requested AOV。
- 分开报告 AS build/update、trace、shade、denoise、interop 和 copy。
- 比较 raw AOV 与 tone-mapped output，不能让不同采样/降噪掩盖 traversal 差距。
- RT 子系统不提供 D1：同一 provider、版本、设备、preset、场景和 seed 的重复运行也不承诺 byte-equal 或完全相同的 ray/hit/AOV 顺序与数值。
- 固定 scene/profile/input/seed 仍用于缩小诊断变量，但 seed 只进入 metadata，不升级为确定性合同。
- canonical hit/AOV conformance 使用明确的绝对/相对容差、ID/拓扑一致性规则和异常值检查；随机 beauty 使用多次运行的统计误差、画质区间与序列稳定性 gate。
- 不要求 self-CUDA、OptiX、Vulkan RT 跨 provider byte-equal，也不能只凭单次随机图像逐像素相等判定正确性。
- provider 不支持必要能力时返回显式 unsupported，不静默切换材质、灯光或输出。

## 7. Provider 选择策略

已确认采用按工作负载自动选择，并允许显式 override。

选择请求至少包含：

```text
workload: sensor_query | offline_preview | offline_final | reference
required capabilities
preferred provider (optional)
fallback policy: allow | forbid
device identity
```

默认策略：

- `sensor_query` 与 `reference` 优先 self-CUDA/self-MUSA。
- `offline_preview` 与 `offline_final` 优先满足全部 required capabilities 的硬件 RT provider。
- 自动模式下，候选按版本化固定优先级和 stable provider ID 选择；选择结果写入日志与输出 metadata。
- 用户通过 API/CLI 显式指定 `self_cuda`、`self_musa`、`optix`、`vulkan_rt` 或未来 provider 时，如果能力不满足则直接报错，不静默换 provider。
- 自动模式只有在 `fallback=allow` 时才能降级；降级必须报告缺失能力、原候选和最终 provider。
- provider 选择在 `RenderSession` 创建时完成；同一 session 中不逐帧漂移。切换 provider 需要创建新 session，并重建 provider-owned scene/AS handles。

建议的显式配置语义：

```text
--rt-provider=auto|self_cuda|self_musa|optix|vulkan_rt|<vendor-id>
--rt-fallback=allow|forbid
```

环境变量可以作为调试入口，但公共 API/配置是正式合同，不能只依赖环境变量。

## 8. 首期建议验证顺序

1. 完成 R1 持久场景并获得 self-CUDA 阶段基线。
2. 先实现 OptiX `trace_query_v1`：同 CUDA device/context、device pointer 与 stream，输出 canonical hit/AOV，不先重写 beauty。
3. 验证 OptiX AS/trace 与共享 CUDA shading/denoise 的同步、零拷贝和阶段性能。
4. 在统一 benchmark 证明收益后实现 OptiX `native_beauty_v1`。
5. 冻结 `trace_query_v1` 最小 ABI；`native_beauty_v1` 作为独立可选子扩展演进。
6. Vulkan RT 作为第二 RT Core provider，在 ABI 稳定后实现 interop 和 conformance，不与首个 OptiX provider 并行抢占 R1 投入。

## 9. 验收标准

- [ ] 普通 CUDA、OptiX、Vulkan RT 的 RT Core 调用边界有最小可运行证明。
- [ ] PHI device 能枚举 RT provider 与精确 capabilities。
- [ ] `nuka.phi.rt.v1` 通过版本和 `struct_size` 协商，旧调用方可忽略新增尾字段。
- [ ] RT scene/AS handle 的 provider ownership 明确，跨 provider 误用可检测。
- [ ] BLAS/TLAS 生命周期不进入物理 `NkOp` schedule。
- [ ] provider selection 不依赖 scene name 或编译期唯一全局实现。
- [ ] 自动选择对相同 device/capability/workload 输入产生稳定结果，并把选择写入 metadata。
- [ ] 显式 override 缺能力时失败，自动 fallback 则输出结构化降级原因。
- [ ] session 内 provider 不逐帧变化，切换 provider 会重建其所有 opaque handles。
- [ ] CUDA PHI 能同时广告 self-CUDA 与可用的 RT Core provider，MUSA PHI 按实际厂商能力广告 self-MUSA/硬件 RT provider。
- [ ] 每个 provider 至少通过 `trace_query_v1` 的 canonical ray/hit/AOV conformance tests。
- [ ] `native_beauty_v1` 缺失时可以按 policy 使用共享 wavefront 路径；存在时通过共同材质/灯光/AOV conformance tests。
- [ ] native beauty 不支持某项 profile feature 时显式拒绝或触发允许的 fallback，不静默近似。
- [ ] physics buffers 与选择的 RT provider 有明确 ownership/sync 生命周期。
- [ ] 所有 provider 消费同一版本的 RT Scene IR 和 Render Profile。
- [ ] unsupported feature 显式失败或使用用户选择的 fallback，不静默降级。
- [ ] R1 benchmark 可在至少 self-CUDA 和一个 RT Core provider 上运行。
- [ ] 首个 RT Core gate 由 OptiX 完成，Vulkan RT 不阻塞 OptiX/R1 交付。
- [ ] OptiX 首期以 triangle/instance conformance 通过；analytic primitive 的 tessellated 输入在 self-CUDA 与 OptiX 使用同一 geometry hash。
- [ ] Go2 动作序列中 BLAS build 次数不随帧数增长，TLAS update/rebuild 与 trace 分别计时。
- [ ] deforming geometry 和 motion AS 请求返回显式 unsupported 或按允许策略选择其他 provider，不静默冻结变形。
- [ ] preview/final 使用同一积分器和材质实现；仅 profile 参数不同。
- [ ] final profile 可运行多 bounce diffuse/specular/transmission，preview gate 则报告其有界 bounce 配置。
- [ ] 关闭或缺失某 lobe 时由 profile/capability 明确表达，不由 provider 静默近似。
- [ ] NEE/MIS 对 directional/point/spot、rectangle/disk、HDRI、显式面积灯命中、delta lobe 和 transmission 有 reference tests，且 PDF/MIS 权重无 NaN/Inf。
- [ ] 多灯不会被 adapter 截断；canonical 单位、锥角、面积、单/双面与 HDRI 球面 PDF 在 self-CUDA/OptiX conformance 中一致。
- [ ] 通用 emissive mesh 请求显式 unsupported 或按 policy fallback，不产生“可见但不照明”的静默近似。
- [ ] thin-walled 与 solid dielectric 使用独立 capability/material mode；provider 不根据 mesh 名称或厚度猜测模式。
- [ ] solid dielectric 对闭合性、绕序、入射/出射、Snell/Fresnel、全反射与 Beer-Lambert 路径长度有 reference tests。
- [ ] self-CUDA 与 OptiX 在相同 transmission scene 上通过容差化 conformance；嵌套/重叠介质、体散射与色散请求显式 unsupported 或按 policy fallback。
- [ ] 非闭合实体玻璃、静默固定 IOR、忽略吸收或实体转薄片均被测试捕获。
- [ ] alpha mask 的通道、颜色空间、UV/sampler、mip/LOD、cutoff 与 `>=` 比较规则已版本化，所有 ray 类型使用同一可见性语义。
- [ ] self-CUDA intersection filtering 与 OptiX any-hit/等价路径在 camera、shadow 和 secondary-ray cutout scene 上通过容差化 coverage/conformance。
- [ ] 缺纹理/UV/能力、非法 cutoff 与非有限 opacity 显式失败；provider 不得把 alpha-mask geometry 静默当不透明。
- [ ] 所有 beauty provider 可按需生成 linear HDR color、albedo、shading normal、linear depth、motion、instance/material ID、sample count、radiance variance 与 validity mask；未请求通道不产生无用 D2H。
- [ ] AOV descriptor、invalid sentinel、坐标系、颜色空间、精度下限与 validity bits 已版本化；ID 不经过 float round-trip。
- [ ] motion 在静态相机、仅相机运动、仅刚体运动、组合运动、背景、出屏、instance 出生/消失与 camera cut case 上通过 reference tests。
- [ ] self-CUDA 与 OptiX 对 primary-surface AOV 通过逐通道容差/精确比较；ID、mask 与 sample count 要求精确一致。
- [ ] preview 参数 sweep 使用同一冻结场景/帧集合、画质 gate 和计时口径，不能靠移除软阴影、基础反射、一跳 diffuse GI 或 canonical AOV 语义达标。
- [ ] 首个 OptiX preview preset 在 RTX 4000 Ada 20GB 上经 benchmark 冻结，配置和输出 metadata 可完整重建其 spp、各 lobe bounce、RR、降噪和 capability 要求。
- [ ] preset 参数或 gate 改变会生成新版本；运行时不为追逐 1 秒预算逐帧隐式改变 spp/bounce/denoise。
- [ ] 无 OptiX SDK/loader 的环境仍可构建和运行 core PHI 与 self-CUDA；显式启用但缺 headers 时在配置阶段给出明确错误。
- [ ] OptiX provider 仅在 loader、ABI/driver、同一 CUDA device 与 required capabilities 全部验证后注册为 `available`。
- [ ] OptiX device-program/module cache 包含代码、ABI、GPU architecture、compile options 与 payload/AOV layout key，不复用不兼容缓存。
- [ ] `auto + fallback=allow`、`fallback=forbid` 与显式 OptiX 对每类 discovery failure 均有测试，选择结果和原因进入 metadata。
- [ ] RT API、CLI、文档和 metadata 不广告 D1 或 deterministic workload；固定 seed 仅作诊断输入，不承诺重复运行 byte-equal。
- [ ] canonical hit/AOV conformance 定义逐字段容差、ID/拓扑规则及 NaN/Inf/越界检查；随机 beauty 通过多次运行的统计画质区间，而非单次逐像素相等。
- [ ] ReSTIR/temporal reuse 不污染首期无 reservoir 的采样 baseline；降噪 history 单独按既定 invalidation 合同验证。
- [ ] 任意帧在无 history 时可通过单帧空间路径独立完成，且不依赖 provider-native denoiser。
- [ ] 时域路径使用版本化 motion/depth/normal/albedo/ID 合同，并在 disocclusion、运动边缘和 camera cut 上通过拖影/闪烁序列 gate。
- [ ] material、light、profile、provider 与相关 scene/resource version 变化会产生可观测且正确的 history invalidation。
- [ ] self/open-source 降噪基线可消费 self-CUDA 与 OptiX 产生的共同 AOV，且不依赖 `native_beauty_v1` 的私有输出。
- [ ] provider-native denoiser 通过共享 post-process 适配器选择和同步；关闭降噪时仍能保存 canonical raw output。
- [ ] 原生降噪器缺能力时按独立 denoise fallback policy 报错或显式降级，不影响已选择的 RT provider。

## 10. 非目标

- 不声称 CUDA kernel 可直接发 RT Core 指令。
- 不要求首期同时完成 OptiX 与 Vulkan RT 的全功能 beauty renderer。
- 不把 OptiX denoiser 作为唯一降噪实现。
- 不把“自写 PHI”误写成“所有 traversal 都必须由通用计算 kernel 自写”。
- 不为尚未验证的 MUSA 硬件 RT API 声明 capability。
- 不为 self-CUDA、self-MUSA、OptiX、Vulkan RT 或跨 provider 输出承诺渲染 D1。

## 11. 冻结边界与变更规则

R2 首期需求已于 2026-08-12 确认冻结。实现应按本文能力、失败语义、fallback policy、benchmark 与验收项推进，不再以访谈中的候选方案覆盖已确认合同。

以下能力保持后续范围，不得为实现便利无声并入首期或以不完整近似冒充支持：

- Vulkan RT 的完整 beauty renderer、尚未验证的 MUSA 硬件 RT provider；
- custom/deforming geometry、motion blur AS 与通用 emissive mesh；
- 嵌套/体积/spectral 介质、连续或随机 alpha；
- ReSTIR、reservoir reuse 与运行时 adaptive-quality profile；
- 跨 provider 或同 provider 的渲染 D1。

若实现证据证明首期合同不可行，须新增版本化变更记录，包含阻塞证据、受影响的 Scene IR/ABI/preset/gate、兼容与迁移方案，并重新确认后再修改本文；不得同名覆盖已冻结 capability、AOV 或 preset 语义。
