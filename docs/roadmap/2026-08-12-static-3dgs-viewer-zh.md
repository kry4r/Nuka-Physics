# G1：静态预训练 3DGS 查看器

状态：需求已确认，待实现
依赖：无；与 Go2、mesh/GS 合成和 RT 后端解耦

## 1. 目标

加载 Graphdeco 风格 PLY 的静态预训练 3D Gaussian Splatting 场景，验证资产规范化、GPU 驻留、CUDA tile splatting、标定相机和 color/depth/opacity 输出。导入后可 cook 为 Nuka 自有紧凑缓存，避免每次运行解析文本/通用 PLY 属性。

## 2. 首期范围

### 输入

- Graphdeco 常见 PLY property：position、scale、rotation、opacity、SH coefficients。
- 明确支持的 SH degree、float encoding、quaternion order 和 scale/opacity activation。
- 可选相机元数据由独立 manifest/JSON 提供；PLY 本身不被假定含完整相机标定。

### Cooked asset

- 版本化 header、coordinate convention、SH degree、Gaussian count 和 source hash。
- SoA 设备友好布局，字段范围和有限性校验。
- cook 输出确定性；同一源资产重复 cook 得到 byte-identical cache。
- 运行时优先加载 cooked cache，source/版本变化时明确失效。

### Renderer

- CUDA tile binning、depth sort/compositing 和椭圆 footprint rasterization。
- 标定 pinhole camera，至少支持 `fx/fy/cx/cy` 与 world transform。
- color、depth、opacity AOV；输出尺寸和格式显式配置。
- 记录 binning、sort、raster、AOV copy 和显存占用。

## 3. 数据合同

```text
Graphdeco PLY + camera manifest
  -> validate/normalize
  -> Nuka cooked splat asset
  -> device-resident Gaussian buffers
  -> camera-dependent tile lists
  -> CUDA splat/composite
  -> color/depth/opacity
```

深度语义必须定义为 composited expected depth 或 first/median contribution 中的一种；首期实现前选定并写测试，不能只叫 `depth`。opacity 使用前向 alpha compositing 的累计不透明度。

## 4. 建议拆分

| 子任务 | 内容 | Gate |
|---|---|---|
| G1.1 | PLY schema parser 与 validator | 合法/缺字段/非有限/不支持 SH fixtures |
| G1.2 | canonical transform 与 cooked cache | 重复 cook byte-identical、round trip |
| G1.3 | CPU 小规模 reference compositor | 低数量 Gaussian 的像素 oracle |
| G1.4 | CUDA tile bin/sort/raster | 与 reference 的 color/alpha/depth 对照 |
| G1.5 | camera/AOV/viewer 接线 | 标定相机截图与 headless export |
| G1.6 | 性能与容量报告 | 不同 Gaussian 数和分辨率 sweep |

## 5. 验收标准

- [ ] 能加载至少一个原版 Graphdeco PLY，而不是只支持 Nuka 自造 fixture。
- [ ] 非法属性、非有限值、错误 quaternion/scale 明确报错。
- [ ] cooked cache 版本和 source hash 可验证，重复 cook byte-identical。
- [ ] CUDA output 与小规模 CPU reference 在约定误差内一致。
- [ ] color/depth/opacity 三个 AOV 都有语义测试。
- [ ] 相机移动不会重新上传全部 Gaussian；设备资源跨帧持久。
- [ ] headless 渲染和 viewer 使用同一 renderer，不维护两套 splat 数学。

## 6. 非目标

- 不从图像训练或在线优化 Gaussian。
- 不做增删 Gaussian、天气编辑或生成式编辑。
- 不与动态 mesh 做 depth/alpha 合成。
- 不作为 Go2 摄影棚背景，也不阻塞 R1/R2。
- 不在首期统一 3DGS splat 与 triangle RT acceleration structure。
