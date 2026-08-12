# R1：离线光追性能基线与持久场景

状态：需求已确认，待实现
基准硬件：NVIDIA RTX 4000 Ada Generation 20GB

## 1. 目标

先消除当前离线光追管线的结构性浪费，再判断自写 CUDA 遍历的性能上限。首个硬 gate 是 Go2 摄影棚场景 1920x1080 快速迭代档约 1 秒/帧，包含软阴影、基础反射和开源/自写降噪。

该目标不是实时 viewport。最终成片继续使用同一套 tracer，只提高 spp、bounce 和质量参数，不设固定渲染时限。

## 2. 已确认的当前问题

- `rhi::offline::OfflineRenderer::Render` 每次调用都执行 `RenderWorldToTwoLevelScene -> BuildScene -> graph -> FreeScene`，因此静态 BLAS 和场景资源无法跨帧持久。
- `RenderWorld` 已是 backend-neutral 数据产品，但缺少稳定的资源版本/增量更新合同。
- two-level tracer 本身区分 BLAS 与 per-frame TLAS；上层生命周期没有利用这个边界。
- beauty host 输出可选择 AOV，但需要实测是否仍有无用 device-to-host copy、临时分配和同步。
- 当前只有 per-call spp averaging，没有正式的离线序列降噪/时域稳定性合同。
- 当前 RT adapter 只保留第一盏灯并把它当 point light；性能达标不能以继续丢失摄影棚灯光语义为代价。

## 3. 基准先行

### 3.1 固定场景

建立一个版本化 Go2 摄影棚 benchmark：

- 静态地面、背景和摄影棚灯光；
- 一条冻结的 Go2 动作轨迹；
- 固定相机、材质、纹理/HDRI 和输出 AOV；
- 固定 preview profile 与 quality profile；
- 场景、轨迹、renderer commit 和 GPU 信息进入报告。

### 3.2 阶段计时

至少独立测量：

1. `RenderWorld`/adapter host lowering；
2. mesh/texture/HDRI upload；
3. static BLAS build；
4. dynamic instance transform update；
5. TLAS refit/rebuild；
6. primary traversal/intersection；
7. shadow rays；
8. secondary reflection/GI/transmission rays；
9. shading/texture/light sampling；
10. denoise 与 tone map；
11. AOV copy 和 host synchronization；
12. 总 wall time、峰值显存与 rays/s。

不得只报告总帧时间，也不得在未剖析前假设 traversal 是主瓶颈。

### 3.3 Preview Preset 收敛

- 不预设 spp/bounce 常量；先在 RTX 4000 Ada 20GB 和固定 Go2 benchmark 上 sweep，再冻结通过 gate 的版本化 preset。
- sweep 可调整 spp、各 lobe bounce、Russian roulette、采样分配、降噪和 queue/batch 参数，但不得移除 1080p、软阴影、基础反射、最多一跳 diffuse GI 或共同材质/AOV 语义。
- 约 1 秒是固定 preset 的 benchmark gate，不是逐帧动态画质目标；正式运行对相同输入保持参数不变。
- preset 记录场景/benchmark hash、目标 GPU、完整积分器/降噪参数、required capabilities 与画质 gate 版本；任何影响结果的改变都生成新版本。
- 同时报告帧时间分位数、各阶段、峰值显存、raw/denoised 画质与序列稳定性，不能只选择最快单帧。

## 4. 持久场景合同

### 4.1 一次构建

- 按 geometry content/version 缓存静态 mesh BLAS。
- 按 texture/environment version 保持设备驻留。
- 按 light set/version 缓存重要性采样分布。
- 为静态摄影棚实例建立可复用的 scene segment。

### 4.2 每帧更新

- 只更新 Go2 link instance transforms 和必要的动态材质/灯光数据。
- TLAS 根据后端能力选择 refit 或 rebuild，但选择和耗时必须可观测。
- 首期 OptiX 不更新刚性 link 的 mesh BLAS，也不承担 deforming geometry 或 motion-blur AS。
- 直接写 device AOV；只下载 profile 请求的最终通道。
- 资源版本不变时不得重复上传 mesh、texture 或 HDRI。

### 4.3 生命周期 API

建议把单次 `Render(world, ...)` 拆为等价能力：

```text
CreateSession(render_world, profile_family)
UpdateInstances(frame_state)
RenderFrame(camera, profile, requested_aovs)
DestroySession()
```

具体类名不是本任务的约束，但 session 必须能被后续 self-CUDA、OptiX 和 Vulkan RT provider 共同表达。

### 4.4 OptiX 可选交付

- OptiX provider 是可选构建模块；无 SDK headers 或运行时 OptiX loader 的环境仍须完整构建、测试和运行 core PHI/self-CUDA。
- 构建时由用户/环境提供 SDK headers；运行时动态加载 NVIDIA driver 的 OptiX ABI，并在注册 provider 前验证版本、目标 CUDA device 与 capabilities。
- provider 未构建或不可用时，自动模式仅在 `fallback=allow` 时转 self-CUDA；显式 OptiX 或禁止 fallback 时返回结构化 discovery/init 错误。
- OptiX module/pipeline cache 按代码、ABI/driver、GPU architecture、编译选项和 AOV/payload layout 隔离。

## 5. 优化顺序

1. 建立可重复 benchmark、CUDA event/Nsight 时间线和图像质量基线。
2. 落实持久 BLAS/资源与增量 TLAS，消除重复分配、上传和同步。
3. 移除无用 AOV/D2H，复用 framebuffer、ray queue 和临时 scratch。
4. 分解自写 CUDA traversal 的 divergence、occupancy、register spill、cache hit 和 BVH quality。
5. 分解 secondary-ray 数量与采样效率，增加低 spp 质量策略和降噪。
6. 在相同 scene/profile/AOV 合同下对比 RT Core provider；由测量决定自写 CUDA 后续投入。

## 6. 画质与降噪合同

- 允许开源或自写降噪，不允许闭源降噪成为唯一可运行路径。
- 首期 preview 同时提供单帧空间降噪和受控时域累积/降噪；单帧路径是无 history、随机访问和能力缺失时的必备 fallback。
- 时域路径由冻结轨迹的相邻帧 camera/rigid-instance transforms 生成 motion vectors，并使用 depth、normal、albedo 与 material/instance ID 拒绝错误重投影。
- disocclusion、屏幕外重入、深度/法线不连续、camera cut 和运动边缘必须显式失效或降低 history 权重。
- material、light、profile、provider、积分器或相关 scene/resource version 变化时按依赖失效 history；无法证明局部失效正确时失效整帧。
- provider-native denoiser 只可作为可选加速，不能替代开源/自写的可运行路径。
- 降噪由 `RenderSession` 的共享 post-process 层调度；`trace_query_v1`、共享 shading 与 `native_beauty_v1` 都先产出 canonical raw color/AOV，再进入同一降噪合同。
- `native_beauty_v1` 不得只输出原生降噪结果；benchmark 必须能关闭降噪、保存 raw output，并能固定同一降噪模式公平比较 provider。
- 原生降噪器通过适配器接入，共享层统一管理 history、失效、同步和 telemetry；缺能力时按独立 denoise fallback policy 显式失败或降级。
- 每个结果记录原始 spp、bounce、raw/noisy AOV、denoise 时间和总时间。
- 每帧额外记录降噪模式、history 有效性/长度、失效原因和 motion-vector 版本。
- 序列 gate 检查闪烁、拖影、disocclusion 和运动边缘，不能只看单帧截图。
- preview 和 final 共用材质/灯光语义；降采样数不能改变几何、相机或物理轨迹。
- preview 和 final 共用同一套可配置多 bounce 积分器；preview 限制 spp/bounce，final 可提高 diffuse/specular/transmission bounce。
- 首期采样基线使用 BSDF importance sampling、NEE、灯光/HDRI importance sampling、MIS 与 Russian roulette；ReSTIR/历史 reservoir 延后。
- 首期共同灯光覆盖 directional、point、spot、rectangle、disk 与 HDRI；显式 rectangle/disk 面积灯可被路径命中并参与 MIS，通用 emissive mesh 延后。
- 多灯不得被 adapter 截断；灯光参数、集合拓扑、HDRI 与 importance table 分别版本化，只重建受变化影响的采样资源。
- 首期 transmission 同时支持不进入介质栈的 `thin_walled`，以及闭合、单层、均匀 IOR 实体的 `solid_dielectric`；实体支持 Snell/Fresnel、全反射与 Beer-Lambert 吸收。
- 嵌套/重叠介质、体散射、非均匀介质与色散延后；超出能力不得静默改成薄片、忽略吸收或固定 IOR。
- canonical AOV 能力首期覆盖 linear HDR color、albedo、world-space shading normal、linear view depth、motion、instance/material ID、sample count、luminance radiance variance 与 validity mask；通道具备能力但仅按 profile 请求分配和输出。
- motion 使用无 jitter 相机矩阵和刚体前后帧 transform，定义为 `previous_pixel - current_pixel`；背景只做相机旋转重投影，camera cut、出屏和无对应 instance 时标记无效。
- 几何/材质 AOV 取 primary ray 首个有效表面；ID/mask 不做数值平均，raw color 与 variance 保持在线性曝光前空间。
- RT 子系统不承诺 D1，即使同 provider/device/preset/scene/seed 也不要求重复运行 byte-equal；固定 seed 仅作为诊断 metadata。
- hit/AOV 使用容差化 conformance、ID/拓扑规则与异常值检查；随机 beauty 和降噪使用多次运行的统计画质区间及序列稳定性 gate。
- 首期支持二值 alpha mask：纹理通道、UV/sampler/mip/filter 和 cutoff 进入 canonical material；`opacity >= cutoff` 保留命中，camera/shadow/secondary ray 使用同一交点过滤语义。
- alpha mask 是离散几何可见性，不等于 dielectric transmission；连续/stochastic alpha、排序混合与 colored opacity 延后。

## 7. 验收标准

- [ ] 同一 frozen trajectory 可重复生成结构化阶段报告。
- [ ] 第二帧起 static BLAS、texture、HDRI 和 light tables 不再重建/重传。
- [ ] 资源版本变化只失效对应缓存，且没有使用陈旧资源。
- [ ] RTX 4000 Ada 上 1080p preview 达到约 1 秒/帧，或报告清楚证明结构性瓶颈及差距。
- [ ] 达标 preview preset 经固定 benchmark sweep 后版本化冻结，结果 metadata 足以完整复现，不使用逐帧隐式 adaptive quality。
- [ ] preview 包含软阴影、基础反射与降噪，图像和时序 gate 通过。
- [ ] 任意帧可在无 history 时通过单帧空间路径独立渲染，时域路径在 camera cut、disocclusion 和运动边缘不产生不可接受拖影。
- [ ] scene/material/light/profile/provider 变化会使相关 history 正确失效，且失效原因进入 metadata。
- [ ] self/open-source 共享降噪可处理 self-CUDA 与 OptiX 的 canonical AOV；原生降噪开关不改变 raw output 合同。
- [ ] 摄影棚 benchmark 同时覆盖多盏显式面积灯与 HDRI，不依赖通用 emissive mesh，且各 provider 使用同一灯光单位和采样语义。
- [ ] transmission reference scene 分别覆盖薄片、实体入射/出射、全反射和吸收，且非法实体或嵌套介质明确失败。
- [ ] benchmark 可逐项开关 canonical AOV 并报告显存、写入和传输成本；未请求 AOV 不发生无用 D2H。
- [ ] motion/ID/validity 在冻结轨迹的相机与刚体运动边界通过时域重投影 gate。
- [ ] RT benchmark 和报告不声称 D1；统计 gate 记录重复次数、随机配置、误差区间与通过阈值，不使用单次逐像素相等。
- [ ] cutout benchmark 覆盖相机、软阴影与 secondary ray，并分别报告 self-CUDA intersection filter 和 OptiX any-hit/等价路径成本。
- [ ] benchmark metadata 记录 OptiX build/runtime ABI、driver、GPU 和 pipeline cache identity；fallback 帧不计入 OptiX 性能结果。
- [ ] final profile 没有被 preview 优化破坏。
- [ ] 自写 CUDA 与未来 RT Core provider 使用相同 benchmark、scene semantics 和输出比较器。

## 8. 非目标

- 不做实时光追或交互 viewport 帧率承诺。
- 不先实现新的复杂渲染效果来掩盖基础生命周期问题。
- 不通过降低到 720p、去掉阴影/反射或改用 RTX 5080 宣称完成。
- 不在本任务决定 OptiX、Vulkan RT 谁是默认后端；该决策属于 R2。
