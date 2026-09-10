# 真实三角表面生产验收

2026-09-10。分类：后端无关几何架构与物理表示修复；CUDA 接线不作为性能优化。该批完成真实静态/刚体网格的表面存储和三个接触消费者接线，自动凸分解及全量耦合继续执行。

原来的 Auto/Skip 会把非凸网格当成凸体，三角拓扑在模型侧丢失。本批保留源顶点、原始三角编号、面内特征和重心坐标，生成确定性的无栈 BVH。闭合实体使用射线奇偶确定距离符号；开放表面双面查询。显式 ConvexHull 检查闭合性与凸性，显式 Force 才允许既有 V-HACD 近似。缺失、越界或非有限的碰撞网格拒绝 cook，错误定位到 shape/body。

刚体采样、body-particle 和 MPM 共用表面查询。几何与动力学 owner 分离，复合形状绑定自身表面；原始网格不再进入只适用凸面的旧 SDF baker，可视网格不隐式替换 collider。源三角顺序不随 BVH 排序改变，最近特征的并列选择稳定。现有场景覆盖 `.nks/.nka` 保存、加载、recook、world、接触及 reset。

## 验收身份与命令

本地证据：`out/validation/triangle_surface_20260910/`。每个 `<label>_command.json` 保存完整命令、退出码和墙时，日志及原始 JSON 同目录；源码、二进制、构建参数、扩展库和硬件身份冻结在 `final_verified/manifest.json`。

- 起始 commit：`bc7b06668dd930e56ea23264b50441ac789df021`。
- 最终源码 SHA-256：`9d6777caa320c3c459197e22f222134c7346091a9f789002cfb3935474848159`。
- `libnuka.so.0` SHA-256：`100fdcf21f65d0921f80161843498f2d43d7c3a6bba67989c9e2754379300f96`。
- 硬件：RTX 5080，CUDA 13.3；WSL Ubuntu-24.04。
- 上游 review：Newton `31f585713a631d9a87acb5bb332b6a8ba61e2410`、MuJoCo `c04c9c726c93852ee3b5b58211ce085f80d98b5d`、Genesis `b3c6c73a7a671fc486df5d69c9f481a91b1d57b6`。

统一入口为 `.nuka-runs/run_execution_batch.py --output-directory out/validation/triangle_surface_20260910 <label> <command...>`，使用 `/root/nuka-vla/bin/python`。完整批次调度保存在 `.nuka-runs/triangle_surface_acceptance_batch.py`，复跑必须使用新 label 保留原始结果。

| 证据 label | 范围与实际结果 |
| --- | --- |
| `build_v7` | scenario/import/pipeline/water demo/公共 C ABI 相关目标合并构建成功 |
| `import_all_v4` | 155 passed，0 skip；真实 Newton cup、非凸/开放表面及缺失几何 |
| `scenarios_v3` | 63 passed，16 suites；刚体、粒子、MPM water/jelly 等现有场景 |
| `coupled_cabi_v2` | 8 passed |
| `multi_env_cabi_v1` | 5 passed |
| `camera_cabi_v1` | 4 passed |
| `python_api_v1` | 12 passed；记录实际加载库身份，coupled/reset 子集 |
| `public_graph_v1` | 公共控制、外力、graph、读出与 reset 完整运行，退出 0 |
| `pipeline_e16_v2` | robot-cloth-fluid，E=16，graph，warmup=250、steps=200，质量门通过 |
| `render_graph_v1` | 640×360、8 samples，8 PNG；已检查落水、飞溅与末态四帧拼图 |
| `mesh_water_memcheck_v1` | 凹体/开放面及三个 water 场景，共 4 passed；ERROR SUMMARY: 0 errors |
| `host_core_v1` | 无 CUDA 的 `nuka_nk nuka_phi2` 构建通过，只证明依赖隔离 |

固定 E16 管线的有限性、状态码、跨环境一致性、reset 一致性、定时轨迹重放和 link wrench 全部通过。采样到 cloth rows=240、fluid rows=864；GPU 完成均时 3.745793 ms，model 2,475,264 B、data 59,741,952 B。这是一次完整运行，未采集五进程对照，不能作为性能收益结论。

## 物理边界与保留失败

新增的凹体/开放面场景使用原生产 `pos_iters=4` 和 6 mm 门；恢复默认位置修正后最大压入约 5.784 mm。先前设置 `pos_iters=0` 的 12.454 mm 失败、importer fixture/固定临时路径失败及修复前源码均保留。MuJoCo 2.3.7 的匹配 20 ms 软接触运行亦有约 10–12 mm 瞬态压入，参数与轨迹见同目录参考输出；这不豁免本引擎的薄层/CCD 验收。

落水渲染记录：最小 bunny 表面高度 −9.520624 mm，末态 −0.157017 mm；总体最大体积比误差 1.8074%，局部 `J` 范围约 0.9224–74.4324，状态有限且 env status=0。总体体积通过不代表局部大 J 已合理。完整 geometry/owner 修复不能替代 MPM 有限质量与每子步反馈。

尚未完成：自动 GPU 凸分解 cook、任意 mesh-mesh/box-mesh 面交叉、薄壳有限厚度、CCD、柔体多端点质量响应、MPM 多 owner 与同时间层交换、完整多介质矩阵，以及提袋装球/拧毛巾。继续按凸分解与全量耦合 → MLS-MPM 修复 → 弹塑性体调研/集成与机械臂挤压渲染 → 主页 Go2 展示替换 → 原 spec 剩余项推进。
