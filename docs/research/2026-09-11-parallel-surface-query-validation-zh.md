# CUDA 并行表面查询验收

日期：2026-09-11。分类：CUDA 后端实现。公共子步时钟的物理变化已在[独立报告](2026-09-10-coupled-substep-clock-validation-zh.md)验收；本文只比较相同时间层、接触预算与几何的查询调度。

## 结论与范围

通用 `PairDrivenSdfKernel` 将每个候选对的表面采样分配给一个 128 线程块，保留双向查询、Sphere 中心查询、真实三角/SDF/解析表面与原 top-K 接触顺序。完整落水 graph 的五进程 GPU 平均步时中位数从 317.824138 降至 30.246762 ms，减少 **90.48%（10.51 倍）**。物理轨迹、完整质量结果、末态及八张渲染图保持一致，Model/Data 大小不变。

固定 robot-cloth-fluid E16 管线没有获得同等收益：eager 中位数回退 2.16%，graph 回退 0.12%。不据落水结果宣称所有负载加速，也不把本次查询调优计作物理修复或渲染提速。

## 实现与边界

- 每线程保留局部 top-K；全局 top-K 必在这些局部集合的并集中。CUB block reduction 逐个选取最深接触，深度相等时用原 side/sample 访问序号稳定排序。
- 接触容量来自 `ContactManifold::kMaxPoints`，不减少采样、接触点、子步或求解迭代。两侧遍历及实际距离/法线公式保持不变。
- 按 rigid candidate slot 容量发射，检查计数、地址和索引范围。线程块一致返回，查询错误归并、归约广播、共享存储复用及尾槽清理均有对应同步。
- 线程组织、CUB 和 launch 资源全部留在 CUDA 后端，无 core/schema/model/public API 变更，无新增持久 Data 字段。

改造前核对上游实现 revision：Newton `07f2fc4afbe60d420d0b3058b8a2497c5954a06f`、MuJoCo `51d4338a229364269820a127468f4b5e13d3bf8f`、Genesis `b3c6c73a7a671fc486df5d69c9f481a91b1d57b6`。MuJoCo 新增 flex collision 拆分及近奇异惯量诊断，不改变本次调度方案；没有移植另一套求解路径。版本与差异保存在证据目录 `upstream/`。

## 五进程完整管线对照

环境为 RTX 5080（84 SM）、驱动 610.88、CUDA 13.3、WSL Ubuntu 24.04。旧新交错运行，各五个独立进程。表内数值为各进程平均步时的中位数；区间为五个进程均值的最小/最大值。普通事件计时与 Nsight profile 分开采集。

落水使用完整 240 步静置和 420 步落水，同一 graph 执行配置。

| 指标 | 公共子步基线 | 并行查询 |
| --- | ---: | ---: |
| 落水 GPU ms/步 | 317.824138 | 30.246762 |
| 落水 GPU 区间 ms/步 | 311.158113–321.033238 | 30.049180–30.455014 |
| 落水同步墙时 ms/步 | 326.954719 | 30.867971 |
| 静置 GPU ms/步 | 16.065169 | 15.983000 |
| Model B | 653,568 | 653,568 |
| Data B | 142,000,384 | 142,000,384 |

所有进程的 trajectory digest 为 `1a69dd75b9df2ddf`，完整 quality 相等，首进程末态 SHA 跨版本相同。静置收益很小，不能将落水的采样成本占比推广至所有场景。

固定 E16 robot-cloth-fluid 使用 250 步预热和 200 步量测，分别比较 eager/graph。

| 模式 | 旧 GPU ms/步 | 新 GPU ms/步 | 变化 | 旧墙时 ms/步 | 新墙时 ms/步 |
| --- | ---: | ---: | ---: | ---: | ---: |
| eager | 5.462261 | 5.580305 | +2.16% | 5.465545 | 5.580695 |
| graph | 3.650974 | 3.655255 | +0.12% | 3.947674 | 3.957295 |

eager GPU 区间分别为 4.754411–5.531056、5.534654–5.821407 ms；graph 为 3.643420–3.656597、3.652542–3.659377 ms。Data 均为 59,741,952 B，state digest 为 `5a3ceadd0852e2c5`，wrench 为 `fe2e407b0ba91b45`，岛调度为 `ab425ef722e9878ca618b1040a37689c99505478b3d393312d03311bd0665349`。所有进程质量和跨版本身份检查通过，回退继续保留。

## 正确性、公共接口与图像

完整 graph/eager 落水、64 项既有代表场景、公共 C ABI coupled/multi-env/camera（8/5/4）、12 项 Python 与公共 graph/reset 通过。E16 N=1/N=2 完整管线的末态和逐步 wrench 与公共时钟冻结版一致。已有解析/守恒/子步门沿用公共时钟和材料验收，旧引擎字节一致只用于确认此次 CUDA 调度没有改变结果。

640×360、8 spp 的八个生产渲染帧与冻结版逐字节相等，拼图已经目视检查。定向 memcheck 和 synccheck 均为 **0 errors**，复用 `FreeBoxRestsOnStaticGround` 与 `CookedConcaveAndOpenSurfacesSupportBodiesAndParticles`，覆盖部分满块、相等深度、双向顺序、非凸/开放面、Sphere 查询和 reset。未新增独立测试集合。

未解决的物理限制保持原样：bunny 最大瞬时穿入 1.339626 mm，min J=0.929009、max J=76.190758，最大总体积误差 1.993272%。局部大 J、连续薄面碰撞、多 owner 有限质量与完整介质交换仍需处理。

## Profile 与资源

Nsight Systems 使用相同 240 步静置加 150 步落水窗口，graph 节点内共有 15,600 次查询。

| PairDrivenSdfKernel | 旧 | 新 |
| --- | ---: | ---: |
| 平均 µs/次 | 1,564.536941 | 22.692322 |
| 全部 kernel 时间占比 | 78.0826% | 5.4968% |
| registers/thread | 117 | 113 |
| static shared/block B | 0 | 92 |
| reported local/thread B | 0 | 0 |
| grid X / block X | 1 / 128 | 16 / 128 |

新窗口内主要热点为 MpmP2GCells（16.27%）、MpmPrepareTransferInput（12.68%）、SolveRowsScalarIslands（7.44%）及 MpmGridBodyReact（7.12%）。Nsight Compute 2026.2.1 对旧新冻结版均因 `ERR_NVGPUCTRPERM` 失败；命令与日志保留。没有实测带宽、scheduler stall 或 achieved occupancy，静态资源数不替代这些指标。

## 可重放证据

根目录：`out/validation/parallel_surface_query_20260910/`。`acceptance_v1.json`、各 `*_command.json`/日志保存确切命令与返回码；五进程数据分别在 `five_process_water_graph/`、`five_process_pipeline/`。`water_graph_v1_identity.json` 记录质量、末态及全部图像相等；`candidate_graph_profile.nsys-rep/.sqlite`、`kernel_resources.json` 保留 profile 与资源结果。

冻结目录 `candidate_v1/` 记录构建配置、全部源码清单及二进制：

| 身份 | SHA-256 |
| --- | --- |
| source | `e336fd7c04fd34bcfb8a5be0bcf9edbcffd62ae5c3e092d2c2cbd0808cbcc694` |
| libnuka.so.0 | `80e6273bcea1f5d884634a7c70a2c0cdd68dc26dd8ea643a5fdaeca9c8dea2c0` |
| water demo | `9517dda0c977ca4db8a1e770958f6ac1b8a42185900eda89c72bf68d613d87c9` |
| pipeline benchmark | `0e1fb60b7608d5c3767608c05290939a92ba5feba50c5f79d8c680bfbbd6274d` |

从仓库根目录在 WSL 中重放完整落水：

```bash
build-linux/tests/nuka_mpm_water_drop_demo --execution graph \
  --width 640 --height 360 --samples 8 --video --start 75 --video-stride 45 \
  --png-dir out/validation/surface_replay/frames \
  --perf-json out/validation/surface_replay/water.json \
  --state-output out/validation/surface_replay/water.state
```

固定完整管线：

```bash
build-linux/src/nuka_pipeline_benchmark --envs 16 --execution graph \
  --substeps 2 --warmup 250 --steps 200 \
  --perf-json out/validation/surface_replay/pipeline.json \
  --state-output out/validation/surface_replay/pipeline.state \
  --wrench-output out/validation/surface_replay/pipeline.wrench
```

按最新排程，接下来先完成同画质渲染提速，再进行弹塑性体通用集成、真实夹爪 demo 及主页替换；demo 后优先继续性能优化和多体耦合补齐。整体目标保持 active。
