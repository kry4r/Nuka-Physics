# Spec 03：渲染补强

状态：需求已收敛，等待 Spec 02 冻结轨迹
顺序：阶段 3/3
输入：[Spec 02 Go2 双 Demo](2026-08-12-spec-02-go2-dual-demos-zh.md)、[离线光追性能基线](2026-08-12-offline-raytracing-performance-zh.md)、[PHI RT 多后端](2026-08-12-phi-rt-multibackend-capability-zh.md)、[静态 3DGS](2026-08-12-static-3dgs-viewer-zh.md)
输出：OptiX 优先的可扩展离线渲染管线、版本化 preview preset 和两条 Go2 成片

## 1. 目标

对通过物理 gate 的 Go2 轨迹进行纯摄影棚离线重渲染。先消除每帧重复 scene build/upload 等结构性浪费，再通过 PHI 的版本化 RT 扩展接入 RT Core；首个性能主线是 OptiX，Vulkan RT 在 ABI 稳定后跟进。

RTX 4000 Ada 20GB 上，固定 Go2 摄影棚 benchmark 的 1920x1080 preview 目标约 1 秒/帧，必须包含软阴影、基础反射、最多一跳 diffuse GI 和开源/自写降噪。这不是实时 viewport 目标；final 使用同一多 bounce 积分器提高 spp/质量，不设固定时限。

3DGS 是独立静态查看功能，不作为 Go2 背景，也不阻塞 mesh RT 或双 demo 成片。

## 2. 阶段 R1：性能基线与持久场景

### 2.1 固定 Benchmark

- 版本化摄影棚、冻结 Go2 轨迹、相机、材料、灯光/HDRI、requested AOV 和 profile。
- 报告 scene/trajectory/renderer hash、GPU/driver、分辨率、spp、bounce、降噪和 provider。
- 分阶段测量 adapter、upload、BLAS、instance update、TLAS、primary/shadow/secondary rays、shading、denoise、compose、copy/sync、wall time、显存和 rays/s。

### 2.2 持久资源合同

```text
CreateSession(render_world, profile_family)
UpdateInstances(frame_state)
RenderFrame(camera, profile, requested_aovs)
DestroySession()
```

- geometry/version 缓存静态与刚性 BLAS；texture/HDRI/light version 保持设备驻留及采样表。
- 每帧只更新 Go2 rigid instance transforms、必要动态参数和 TLAS；transform 变化不得重建 mesh BLAS。
- framebuffer、ray queue、scratch 和 device AOV 跨帧复用；只下载 profile 请求的通道。
- 分别观测 TLAS update/rebuild，不因 backend 策略隐藏成本。

### 2.3 Preview Preset

- 不预设 spp/bounce 常量；在固定 benchmark 上 sweep spp、lobe bounce、RR、采样、降噪和 queue 参数。
- 不得通过降分辨率、移除软阴影/反射/一跳 GI、改变轨迹或 AOV 语义达标。
- 通过画质与约 1 秒 gate 后冻结版本化 preset；改变结果的参数产生新 ID，不能同名覆盖。
- 正式 preset 固定画质，不做逐帧 adaptive-quality。

## 3. 阶段 R2：PHI RT 多后端

### 3.1 扩展与 Provider

RT 不进入物理 `NkOp`，通过绑定具体 `phi::Device` 的版本化 C ABI `nuka.phi.rt.v1` 发现。ABI 使用 `abi_version + struct_size`，scene/AS/context handle 由创建它的 provider 所有。

首期 provider 能力分两级：

- 必选 `trace_query_v1`：AS 生命周期、批量 ray、canonical hit/AOV、device-resident I/O、显式同步与 telemetry。
- 可选 `native_beauty_v1`：provider 内融合完整 path tracing，但必须消费共同 Scene IR/Render Profile/材质/灯光并输出 canonical raw AOV。

选择请求包含 workload、required capabilities、显式 provider、fallback policy 与 device identity。`sensor_query/reference` 优先 self-CUDA/self-MUSA，offline beauty 优先满足能力的硬件 RT provider。显式 provider 缺能力直接失败；自动模式仅在 `fallback=allow` 时降级并记录原因；session 内 provider 不逐帧变化。

### 3.2 OptiX 优先顺序

1. 完成 R1 self-CUDA 基线。
2. 实现 OptiX `trace_query_v1` 并验证同 CUDA device/context、指针、stream/event 和零拷贝。
3. 冻结基础 hit/AOV ABI。
4. 在 benchmark 证明收益后实现 OptiX `native_beauty_v1`。
5. ABI 稳定后接 Vulkan RT；不阻塞首个 OptiX gate。

OptiX 是可选构建模块：构建时使用外部 SDK headers，运行时动态加载 NVIDIA driver ABI 并检查版本、同一 CUDA device 与 capabilities。发现状态至少区分 `not_built`、`loader_missing`、`version_mismatch`、`unsupported_device`、`initialization_failed`、`capability_missing`、`available`。无 SDK/loader 环境必须能构建运行 core PHI/self-CUDA。

### 3.3 Geometry 与 AS

- 首期 OptiX 只支持 triangle mesh BLAS 与 rigid instances/TLAS。
- analytic sphere/capsule 确定性 tessellation，geometry hash 跨 provider 一致；custom primitive 延后。
- 摄影棚和 Go2 link BLAS 按 geometry version 构建一次，每帧 update/rebuild TLAS。
- deforming BLAS、curve/SDF procedural geometry、motion blur AS 延后并显式 unsupported。

## 4. 多 Bounce Beauty 合同

### 4.1 积分器与采样

- preview/final 共用 metallic/roughness PBR、多 bounce diffuse/specular/transmission 积分器。
- BSDF importance sampling、每个非 delta bounce 的 NEE、灯光/HDRI importance sampling、MIS 和 Russian roulette。
- random key 至少包含 frame/sample/pixel/bounce/lobe；RT 不承诺同 provider 或跨 provider D1。
- conformance 使用 hit/AOV 容差、ID/拓扑规则和异常检查；随机 beauty 使用多次统计画质与序列 gate。

### 4.2 灯光与材质

- 首期灯光：directional、point、spot、rectangle、disk、HDRI；多灯不得截断，rectangle/disk 可被命中并参与 MIS。
- 通用 emissive mesh 延后；不得出现“相机可见但不照明”的静默近似。
- transmission 支持 `thin_walled` 和闭合、单层、均匀 IOR 的 `solid_dielectric`；实体支持 Snell/Fresnel、全反射与 Beer-Lambert 吸收。
- 嵌套/重叠介质、体散射、非均匀介质、色散延后。
- 二值 alpha mask 使用版本化 opacity 通道、UV/sampler/mip/filter 和 cutoff；`opacity >= cutoff` 保留命中，camera/shadow/secondary ray 语义一致。
- 连续/stochastic alpha、排序 blend 和 colored opacity 延后。

### 4.3 Canonical AOV

provider 必须具备但仅按 profile 请求分配：

- linear HDR color、albedo、world-space shading normal、linear view depth；
- motion、`uint32` instance/material ID；
- sample count、luminance radiance variance、validity mask。

几何/材质 AOV 取 primary ray 首个有效表面。motion 使用无 jitter 的当前/上一帧相机和 rigid transform，定义为 `previous_pixel - current_pixel`；背景只按相机旋转重投影，camera cut、出屏和无对应 instance 时标记无效。

### 4.4 共享降噪

- 降噪属于 provider-independent `RenderSession` post-process，不进入 `trace_query_v1`、`native_beauty_v1` 或物理 `NkOp`。
- 自写/开源实现是必备基线：单帧空间 fallback + 受控时域累积/降噪。
- provider-native denoiser 仅通过适配器可选加速，必须共用 raw AOV、history、失效、同步与 telemetry 合同。
- disocclusion、运动边缘、camera cut、材质/灯光/profile/provider/资源版本变化精确失效 history；无法证明局部正确时整帧失效。
- benchmark 可关闭降噪保存 raw output，并可固定同一降噪模式公平比较 provider。

## 5. 独立静态 3DGS

- 输入为 Graphdeco 风格 PLY 的静态预训练场景，cook 为版本化 Nuka cache。
- 首期使用 CUDA tile renderer，输出 color/depth/opacity。
- 不训练、不编辑、不做动态 GS，不与 mesh/Go2 合成，不作为摄影棚背景。
- 3DGS 失败或延期不阻塞 R1/R2 与双 demo 视频。

## 6. 阶段验收 Gate

- [ ] 第二帧起静态/刚性 BLAS、texture、HDRI、light tables 不重复 build/upload；TLAS 与 trace 分开计时。
- [ ] self-CUDA 与 OptiX 在同一 scene/profile/AOV benchmark 上可比较，fallback 帧不计入 OptiX 结果。
- [ ] `nuka.phi.rt.v1`、provider ownership、capability query、显式 override 和 fallback error paths 通过测试。
- [ ] OptiX triangle/rigid-instance、multi-light、transmission、alpha mask 和 canonical AOV conformance 通过。
- [ ] preview/final 使用同一积分器；final 可提高 diffuse/specular/transmission bounce。
- [ ] 单帧与时域降噪分别通过 raw/denoised、闪烁、拖影、disocclusion 和运动边缘 gate。
- [ ] RTX 4000 Ada 20GB 上 1080p preview 达到约 1 秒/帧，或用完整阶段报告说明结构性瓶颈及差距；不得降低冻结视觉合同。
- [ ] 通过 gate 的 preview preset 可由 metadata 完整复现且不使用逐帧 adaptive quality。
- [ ] 两个 Go2 冻结轨迹各自完成 preview 与 final 离线重渲染；视频只使用已通过 Spec 02 的物理轨迹。
- [ ] 3DGS PLY/cooked cache/CUDA renderer/color-depth-opacity 通过独立 gate。

## 7. 冻结非目标

- 不做实时 viewport 帧率承诺。
- 不要求首期完成 Vulkan RT 全功能 beauty 或未验证的 MUSA 硬件 RT provider。
- 不做 custom/deforming geometry、motion AS、通用 emissive mesh、嵌套/体积/spectral 介质。
- 不做 ReSTIR/reservoir reuse、运行时 adaptive-quality 或渲染 D1。
- 不把 OptiX denoiser 作为唯一降噪路径，不把“自写 PHI”解释为通用 CUDA/MUSA kernel 必须自写 traversal。
