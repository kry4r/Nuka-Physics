# 公共物理子步时间层验收

日期：2026-09-10。公共子步时间层已接入生产 Pipeline；全量多体/多介质耦合仍未完成。本批属于后端无关调度架构与物理时间积分修复，CUDA 仅实现对应算子。改变后的接触轨迹不能计作性能收益。

## 生产行为

过去 ABA、自由刚体及 XPBD/PBF 每外步推进一次，MPM 在私有循环中细分时间，组内刚体位姿和 articulation 表面速度不随反馈刷新。现在 `Pipeline` 取 `N=max(1, SolverConfig.substeps, 实际 MPM 的 mpm_substeps)`，以 `h=dt/N` 重复同一完整管线。`MpmStep` 只推进一个公共区间，拒绝额外的内部细分。

每区间重新执行动力学、碰撞、材料/接触与积分。自由体外力/力矩保持到外步末再清空，避免分成 N 次后只消费 1/N 的总冲量。MPM 接触前按更新的广义速度刷新 link 速度，复用当前 ABA 的运动变换，不重复推进动力学。

外步读出累计 MPM 线冲量和关于世界原点的角冲量，再换算到可见 owner 原点；link contact wrench 按区间冲量累加并除以外步 dt。关节上下限分别累计，错误按位 OR。接触槽保留最后区间快照，不能跨不同 contact identity 累加。link 原点沿既有 `link_pose` 契约，是最后接触区间的 FK 原点；q 已积分到末态，但没有额外末态 FK。自由体读出原点为末态 COM。

N=1 且存在 MPM/body 时也执行角冲量参考点转换。无 MPM 的 N=1 保持原 op 调度和 scratch 容量。参数存储在构建后保持地址稳定，eager/graph 使用同一列表；新增 scratch 在外步首区间覆盖，reset 不依赖其旧值。创建检查包含 dt、子步 dt 及其倒数、索引和调用数边界。

## 验收证据

本地根目录为 `out/validation/coupled_substep_clock_20260910/`，所有命令、日志、失败和原始结果均保留。

| 范围 | 结果与边界 |
| --- | --- |
| 导入、场景整批 | 156 项导入、选定的 64 项生产场景通过，包含 water/jelly、刚体/关节与 MPM 支承、混合粒子及固定机器人–布料–流体；不是整个 spec 的完成证明 |
| 公共接口 | coupled C ABI 8、multi-env 5、camera 4、Python 12 项通过；public graph 与 reset 通过 |
| 子步组合律 | 两环境、非零 COM 偏移、外力/力矩下，一次 25 子步与 25 次完整小步的 body pose/velocity、particle position/velocity/F/C 逐字节一致；线冲量和世界原点角冲量收支一致 |
| 外力与重力 | N=1/5、eager/graph、旋转协变及外力消费一次通过；半隐式位移与解析离散解一致 |
| 固定 E16、N=2 | eager/graph 完整 450 步通过，状态与逐步 wrench 一致；布料最大应变 0.499141%、最坏每环境 RMS 0.041639%，保持原 5%/1% 门；有布料和流体对 articulation 的接触 |
| GPU 内存 | 子步组合律与外力检查、E16 N=2 graph 完整管线均为 0 errors，后者 memcheck 墙时 680.898 s |
| 多后端边界 | 不启用 CUDA 的 cooker/nk/phi 核心构建通过 |
| 最后源码检查 | 子步倒数溢出检查和 profiler 注释更正后增量构建通过，12 项既有时间层/管线契约检查通过；最终 N=2 graph 再验并与先前输出比较 |

首次 N=2 benchmark 失败保存在 `pipeline_substeps_2_eager_v1.*`。原因是 fixture 校准使用待测的子步配置，导致 rear foot 从约 `(-0.3126,0.0281,0.2587)` 移至 `(-0.3406,0.0550,0.2949)`，据此重放置水池改变了比较输入，未采到流体–articulation 接触。现固定使用原 fixture 的 N=1、dt=1/240 校准；待测 World 仍使用指定子步。没有改变水池参数或放宽质量门。`v2` 与最终重放通过，失败 v1 不覆盖。

主整批验收对应 `candidate_v1`；随后仅更正 benchmark 固定输入、子步倒数检查及 profiler 注释。最终源文件差异清单和二进制保存在 `final_verified/`。未把这三个收尾修改描述为又一次全量物理验收。

## 独立时间积分参考

源代码 review：Newton `07f2fc4afbe60d420d0b3058b8a2497c5954a06f`，MuJoCo `0f87d5de2878bba80946f9ba6008f509a3e53e52`，Genesis `b3c6c73a7a671fc486df5d69c9f481a91b1d57b6`。相对上一批，Newton 更新 Kamino body acceleration 和 conveyor 的 linesearch warning filter；MuJoCo 更新灯光参数编辑和 mjData X macros 完整性测试；Genesis 没有变化。它们不是新耦合求解算法。

实跑 MuJoCo 3.12.0 Euler float64、Newton 1.7.0.dev0 CPU XPBD 无约束粒子，并与 Nuka 的均匀、无应变 MPM patch 比较。Newton 实跑使用固定 revision `1128af71407e002d15a8cfb180b35158cf64c42f`，不混同最新源码 review。初始位置 `(0.25,0.25,0.25)`、速度 `(0.7,-0.3,1.1)`、重力 `(0,0,-1)`，总时长 1/60 s，细分 1/2/4。

Nuka 相对连续解析解的位置误差为 `1.38886637e-4 / 6.9447226e-5 / 3.46977181e-5 m`，符合半隐式一阶收敛。各引擎同时符合相应离散解和速度解。记录在 `uniform_acceleration_references_v2/`；v1 的 Newton 中间 NumPy 视图引用了复用状态，v2 改为独立快照并逐个时间点核对，原记录保留。Genesis 本批没有实跑环境，只做源码 review。这个参考只验证时间推进，不证明材料或接触耦合。

## 性能与内存

N=1 固定 E16，250 warmup + 200 计时步及完整质量重放，旧/新版本交错，每种模式各五对独立进程。GPU 指完整 step completion，墙时含同步，原始范围和硬件状态见 `five_process_n1/`。

| 模式 | 旧 GPU 中位 ms | 新 GPU 中位 ms | 旧墙时中位 ms/step | 新墙时中位 ms/step |
| --- | ---: | ---: | ---: | ---: |
| eager | 5.881865 | 5.822728 | 5.882312 | 5.807099 |
| graph | 4.171131 | 4.118576 | 4.234349 | 4.212091 |

状态 digest `5a3ceadd0852e2c5`、逐步 wrench digest `fe2e407b0ba91b45` 与岛调度跨版本/模式/进程一致；Data 59,741,952 B、Model 2,475,264 B 相同。约 1% 的中位差异落在明显噪声范围内，不作为加速结论。N=2 Data 为 59,749,120 B，增加 7,168 B；单次 eager/graph 分别为 10.928511/8.217270 ms，不是五进程性能结论。

真实 bunny-water，180,000 粒子、40 公共子步、240 settle + 420 drop，完整渲染运行的 drop 平均 GPU step 从旧 19.401034 ms 增至 321.189313 ms，settle 从 11.601174 增至 16.083421 ms。新 Data 为 142,000,384 B，比旧增加 768 B。场景批次墙时也从约 74 s 增至 179.56 s。这是严重性能回退，完整物理区间重复工作的成本必须继续处理。

针对最终冻结版采集 Nsight Systems：240 settle + 150 drop，共 15,600 次公共区间；仅统计 graph kernel，`PairDrivenSdfKernel` 占总 kernel 时间 78.0826%，平均 1.564537 ms/call、最大 20.259079 ms。它以单个线程串行遍历整对形状的采样点；bunny 有 35,947 点。该 trace 证明通用表面窄相是当前主要瓶颈，短窗口带 profiler 的时长不替代完整性能分母。下一批单独实施 CUDA 并行查询和稳定归约，保持全部几何与物理预算。

## 渲染与未完成物理

已渲染并目视核对第 75、120、165、210、255、300、345、390 步的 8 帧，原始图在 `render_graph_v1/`，新旧对照为 `render_comparison_0.png`、`render_comparison_1.png`。

| 量测 | 公共子步结果 |
| --- | ---: |
| 最大瞬时穿入 | 1.339626 mm，旧约 9.521 mm |
| 末态最低表面 z | −0.211500 mm |
| min / max J | 0.929009 / 76.190758 |
| 最大总体积相对误差 | 1.993272%，原 5% 门通过 |

子步改善了当前离散接触响应，但没有建立 CCD 证明。局部 max J 比旧 74.4324 更大，仍是未解决的物理问题；总体积门通过不代表局部材料有效。本批仍保留 deepest-single-owner 网格边界、无限质量投影、延后到共享行之前的 articulation deposit，以及缺失的柔体插值面端点与直接介质交换。MPM 冲量仍使用独立读出字段，link contact wrench 反映共享行接触，尚未统一成全部介质总 wrench。完整有限质量、同 owner 交叉项、多 owner、连续表面/自碰撞及全量介质矩阵继续按总 spec 实施。

## 冻结与接续

最终源码 SHA256：`eef5d50398689e2292f03bc89db1f45c1b7d4a4c58107bec89528f3bac934944`；`libnuka.so.0`：`72a50e04de415339899e74a169ca0375dfc39cb5e75f359bc7d8352981a2c9ae`。硬件为 RTX 5080、驱动 610.88、CUDA 13.3。完整逐文件身份、构建配置、源码差异与选定二进制在 `final_verified/manifest.json`；原 `candidate_v1/` 和自动 cook 冻结继续保留。

先处理上述通用 CUDA 串行查询瓶颈，再继续可插值端点/有限质量/多 owner 与直接介质耦合。随后按用户顺序推进 MLS-MPM 材料修复、弹塑性体集成、真实 ABA 挤压与提袋/拧毛巾 demo、渲染和主页替换，再返回原 spec 的其余工作。整体 goal 保持 active。
