# 自动凸覆盖 cook 验收

闭合 TriMesh 的 Auto cook 现已接入 GPU 辅助半空间凸覆盖、版本化缓存和源三角 BVH 分组。几何误差预算、预算回退及缓存损坏重建已验证；覆盖只组织完整源三角，最终接触仍查询原始表面。多体有限质量和子步耦合尚未完成。

## 实现与物理边界

后端无关层使用距离单元、Lipschitz/固定三角距离上界、支撑平面与凸性冲突选择切分。子块继承父块已裁剪多边形，候选切分复用凸多面体边；同次切分的候选顶点及边界探测合并成批查询。CPU 与 CUDA 提供同一真实三角距离接口，CUDA 类型和运行时依赖限于后端实现及构建接线。

默认预算为相对误差 0.03、128 块、每块 64 平面、250,000 距离单元和 1,024 次操作。开放表面、Skip 和禁用碰撞形状保留原始表示。数值失败、预算耗尽或后端失败均保留完整三角/BVH，不将部分凸块当作成功结果。显式 Force 保留原有 V-HACD 语义。

`.nukasurf` 的版本为 3。键包含完整几何、参数、语义和查询后端，文件包含校验和、尺寸和树结构校验；坏缓存重新生成，后端失败不缓存。相同几何可共享 CookedBlob 存储，动力学 owner、形状与材料仍独立。

每个源三角恰好归属一个 BVH 叶子，父节点包围完整子树。分组后仅在总节点面积成本降低时采用新树；未认证的凸覆盖从不用于排除真实表面查询。`ConvexCoverStatus::Complete` 表示浮点搜索完成，不是严格覆盖认证，也不授权用凸块替代最终 collider。

分类：半空间搜索、缓存、几何共享及查询批处理属于后端无关架构；批量距离 kernel 属于 CUDA 实现。本批未改变求解器或本构，不以既有物理缺陷作为性能收益。

## 独立几何检查

资产为 Newton assets 的 `manipulation_objects/cup/model.usda`。C++ 生产 `LoadUsd → CookScene` 输出 46 块，143,692 距离单元、187,498 查询点。独立检查先按完全相同的浮点坐标合并重复顶点，再使用 SciPy 凸包不等式和 trimesh 5.1.0/rtree 距离；不使用生产距离查询作为覆盖 oracle。

| 指标 | 结果 |
| --- | ---: |
| 原始实体体积 | 0.000101009867 m³ |
| 各凸块体积之和，含重叠 | 0.000118526856 m³ |
| 误差预算 | 2.538002 mm |
| 源表面/内部采样 | 17,012 |
| 其中内部采样 | 8,040 |
| 未被覆盖的采样 | 0 |
| 凸覆盖输出采样 | 31,205 |
| 最大采样外扩 | 2.504504 mm |

源网格闭合且绕序一致。有限采样不构成严格证明。每个独立进程还检查 8,192 个查询点的距离、面号和 feature 与原始 BVH 一致。该杯子的分组树未降低面积成本，因此保留原树，不能宣称该资产运行时加速。

本次本地版本 3 缓存清单含 75 个原表面结果、30 个完成结果、4 个预算回退和 4 个数值回退；其中 15 个结果采用分组树。清单含测试和不同参数的缓存，不代表 113 个不同生产资产，更不代表全部网格均已分解成功。

## 性能与内存

RTX 5080，驱动 610.88，CUDA 13.3，WSL Ubuntu 24.04。杯子使用每次独立缓存的五进程交错对照；冷 cook 包含 CUDA 首次使用开销，warm 是同一进程内的完整再次 cook。

| 实现 | 冷 cook 中位数 | 缓存 cook 中位数 | 仅精确表面 cook 中位数 | 进程峰值 RSS 中位数 |
| --- | ---: | ---: | ---: | ---: |
| 反复裁剪源面 | 2063.324 ms | 4.249 ms | 1.791 ms | 158,852 KiB |
| 继承裁剪结果/复用边 | 1472.616 ms | 6.688 ms | 1.799 ms | 158,672 KiB |
| 合并候选查询，最终实现 | 544.751 ms | 4.290 ms | 1.747 ms | 158,812 KiB |

最终冷 cook 相对首版减少 73.60%，相对继承裁剪版减少 63.01%。自动覆盖仍明显比只建精确表面昂贵，缓存不能掩盖这项创建成本。

公共创建/控制/reset/render 脚本的首次运行从首版 189.242 s 降至 103.625 s，再降至 31.384 s；只建原始表面的历史运行是 5.863 s。最终缓存运行为 6.627 s。这是各版单次完整脚本墙时，包含创建契约中加载的大型机器人网格，不是五进程纯 cook 比较。

Nsight Systems 的最终杯子冷运行记录 479 次距离 kernel、合计 149.222 ms；H2D/D2H 实际传输分别为 2.544/9.750 MB、0.226/0.614 ms。`cudaMemcpy` API 合计 204.748 ms 包括等待，不能与 kernel 时间相加解释为独立成本。显式 CUDA 分配瞬时峰值 1,793,584 B，最高常驻查询缓冲状态为 1,151,488 B，结束后全部释放；不含驱动上下文。距离查询、主机往返和距离单元遍历仍有优化空间。

另试验独立 stream 和可复用锁页缓冲区，五组交错的插桩版本冷 cook 中位数为 545.234 → 563.903 ms，增加 3.42%；未采用。其源码、二进制和结果保留在 `buffer_comparison/`、`profile_v4/`。

完整固定 robot-cloth-fluid、E=16、graph、warmup=250、steps=200 的五进程结果如下。流体为 PBF，此处不代替 MLS-MPM 验收。

| 指标 | 原始精确表面基线 | 最终自动 cook |
| --- | ---: | ---: |
| GPU 完成时间中位数 | 3.657677 ms/步 | 3.655993 ms/步 |
| GPU 范围 | 3.645269–3.665193 ms | 3.651343–3.664125 ms |
| 同步墙时中位数 | 3.956993 ms/步 | 3.952764 ms/步 |
| Model arena | 2,475,264 B | 2,475,264 B |
| Data arena | 59,741,952 B | 59,741,952 B |

步时与内存持平。五进程状态、逐步 wrench 和 island schedule 身份一致，状态 `5a3ceadd0852e2c5`，wrench `fe2e407b0ba91b45`；不将约 0.05% 波动写成求解性能提升。

## 完整验收与保留问题

import 156 项通过，覆盖 CPU/CUDA、真实闭合 L 形、owner 分离、几何共享、缓存损坏和预算回退。water/jelly、transfer、rigid/articulation-on-MPM、混合粒子和固定管线场景共 63 项通过；C ABI coupled/multi-env/camera 为 8/5/4 项，Python coupled/reset 为 12 项。公共 graph 管线通过，定向 CUDA memcheck 为 0 errors，无 CUDA 的 cooker/core 构建通过。

bunny-water 的 8 张 640×360、8 samples 渲染图逐张目视检查，图像与原始精确表面基线逐字节一致。完整物理质量 JSON 和轨迹一致，Data/Model arena 为 141,999,616/653,568 B。既有最大瞬时穿入 9.521 mm、局部最大 J=74.4324112 仍存在；渲染和字节一致不能证明这些物理问题已解决。

下一步继续统一多端点 Jacobian/有效质量和子步时间层，随后推进连续柔体表面、CCD 与全量介质交换。提袋装球、双夹爪拧毛巾、MLS-MPM 修复、弹塑性体和主页 demo 仍为未完成路线。

## 证据与复现

本地证据根目录：`out/validation/automatic_surface_cook_20260910/`。最终源码 SHA-256 为 `a63e4e8f5ca1b7423294dd9f45600003f1451e5757c57f652baffd55e9631898`，完整源码清单、补丁、修改文件归档和二进制身份位于 `final_verified/manifest.json`。失败/回退分母分别保存在 `repeated_clipping_baseline/`、`incremental_clipping_baseline/` 和 `batched_query_baseline/`。

`*_command.json` 保存逐项命令与返回码；`acceptance_v5.json`、`five_process_cook.json`、`pipeline_five_{baseline,final}/`、`cup_final_p0_independent_quality.json`、`final_summary.json` 和 `cook_cuda.nsys-rep` 保存验收、物理、计时与 profile。辅助脚本和逐帧图只在本地保存。

构建：`cmake --build build-linux --target nuka_import_test nuka_scenario_test nuka_pipeline_benchmark nuka_mpm_water_drop_demo nuka_coupled_world_cabi_test nuka_multi_env_world_test nuka_camera_sensor_cabi_test -j 8`。公共验收：`python tools/validation/rigid_inputs_pipeline.py --output out/validation/automatic_surface_replay --execution graph`。五进程复现使用 `tools/perf/run_pipeline_sweep.py` 和上述环境/步数。

本轮 review 的 Newton/MuJoCo/Genesis revision 为 `4129afde7102e8749dc06198de6ff771bf187aad`、`bbcd494b3f03b2aed875826cfa5a38a062d0d49f`、`b3c6c73a7a671fc486df5d69c9f481a91b1d57b6`。Newton 的 joint-owned mimic 和较旧 CUDA/设备的 paired SDF texture 兼容更新未改变本算法；这里不使用 texture sampling。MuJoCo 两个 revision 的内容差异仅为 Unity 默认关节轴文档，不能把提交列表中的历史物理提交当作本次新增。原始内容 diff 保存在 `upstream/`；本批没有新增三引擎的匹配物理实跑对照。
