# 粒子接触拓扑排除审查

日期：2026-09-09。对应 T17/T21b 与性能 spec 的质量前置条件。原执行优化暂缓，先修复新观测揭示的共同正确性问题；不把旧失败轨迹作为优化分母。

## 证据

`extended_baseline_frozen` 使用原岛求解与原 XPBD 提交，source 为 `11d8d580ca9413f08b7eba191c7a9b3ead055bc732c142f889649f5630e80627`。Nx=13/25/37 均通过既有 finite/env_status/reset/接触门，但新增长度约束观测揭示该门尚不足以接受布料形变。默认环境最大拉伸应变达到 1.5274，RMS 约 0.3793。

`trace_xpbd_baseline.json` 通过同一公共 OpCall 列表，在计时外对边界取样。最后完整状态 FNV `5e1dd855bb4b0e12` 与冻结原轨迹相同。初始最大应变为 `5.48e-8`；第 301 步 distance 后为 `0.037712`、bend 后 `0.038465`，ParticleFinalize 后仍为 `0.038465`，ParticleParticleContact 后升至 `1.527402`。因此不是观测索引错误，也不是 XPBD 颜色投影本身引入这次大幅拉伸。

通用粒子接触目前遍历所有邻居，缺少结构邻接排除；默认布料边长 0.016 m，小于同一模型的接触分离距离 0.030 m。距离约束和粒子接触对这些结构近邻提出互不兼容的位置要求。保留现有失败文件和原 benchmark 的旧 `valid` 字段，并在本报告解释其质量门缺口，不能重写旧结果掩盖问题。

## 上游对照与采用方向

最新精确 revision/文件 SHA256 见 `particle_topology_upstream_review.json`。Newton `tri_mesh_collision.py` 通过顶点/边/三角形的 CSR 邻接构造 n-ring collision filter；MuJoCo `engine_collision_driver.c` 将显式排除及 flex 自碰撞候选规则放在统一窄相之前。Genesis PBD 的碰撞实现含材料类别筛选，不能照搬为 Nuka 的 soft/soft 禁用分支。

采用通用结构关联和静止几何规则：仅当两个粒子共享已声明的结构元素，并且静止距离小于接触分离距离时，排除两者的粒子球接触。距离边、弯曲 stencil、tet、shape-match cluster 和表面三角形都编译为同一种粒子→结构元素 CSR，运行时不按 family 或材料分类。没有共同结构元素的近邻继续接触；同一大 cluster 中静止时相距较远、后来折叠靠近的粒子也继续接触，避免把 shape-match 的全部成员关系误当成禁用整体自碰撞。

CSR 和静止位置为共享的单环境拓扑模板；设备使用局部粒子 ID 查询，独立于实际环境数。存储量随结构 incidence 线性增长，不展开大 cluster 的 O(N²) 排除表。固定顺序构建、去重、u64 预算及 u32 设备索引检查；非法索引/不完整字段在颜色构建前拒绝。

## 成批落点与验收

本批分类为后端无关物理/拓扑契约修复及 CUDA 消费端适配，不是纯性能优化。CSR、静止位置和排除语义须能被其他后端消费；CUDA launch/执行细节不进入核心类型。另做无 CUDA 构建隔离检查，其通过不代表所有后端都已实现本求解路径。

- `src/nk/model/fields.yaml` 及生成视图：追加只读 model 字段，保留旧 FieldId。
- `src/nk/model/model.hpp/.cpp`：结构输入防护、共享 CSR/静止位置上传、元素计数及 move 生命周期。
- `src/phi/backend_cuda/ops/particles.cu`：在原粒子接触 gather 中应用同一谓词；没有第二个接触求解器，不修改半修正、质量、迭代或双向关系。
- `tools/perf/pipeline_benchmark.cpp`：保留完整 pipeline 的应变/固定点观测，修复后建立新的质量基线。

统一构建后运行固定 robot-cloth-fluid、现有粒子/多环境/reset/API 场景；一个必要的物理反例验证结构近邻保持静止、无结构关系的同类粒子仍然分离、远处共享 cluster 成员仍能碰撞及环境隔离。主环境记录所有接触/反作用、固定点和最终长度误差，不能只检查 finite。旧性能与修复后的数字不作加速比；正确基线冻结并五进程测量后，才恢复岛调度和设备颜色迭代候选。

本修复只处理粒子球与结构约束的冲突。完整 vertex-face/edge-edge、连续自碰撞、tet inversion、邻域刷新及强耦合收敛仍按原 spec 继续，不能据此宣称完整 T21b 已完成。

## 当前实跑与待判定项

默认 E=1/16/256 的完整 450 步 graph/reset/replay 初验：最大长度应变 0.023776、最大 RMS 0.003007、固定点位移 0，env_status 为 0，副本与重放一致。结构近邻、无结构关系接触、远端 cluster 接触、质心/环境隔离反例通过。尚未加入应变 valid gate、冻结正式分母或完成所有公共边界验证，不能宣称整批验收完成。

联合场景 61 项中 60 通过，`ClothPresserTwoWay.ClothDipsUnderPresserThenRecovers` 失败。修复版和冻结原版均为 rest_min_z=0.1072346291、pressed_min_z=0.102115802、recovered_min_z=0.106793806；现有下压/恢复各 10 mm 要求未满足。该用例的 CookXpbdParticles 模式没有启用粒子球接触，本次拓扑过滤不是此差异的原因；仍须核对真实几何/控制/量测并判断物理，不改容差。原版同失败只定位来源，不能作为正确性证明。

后续物理验收纳入匹配的 Newton/MuJoCo/Genesis 仿真参照，记录参数和 solver 不等价范围，并结合解析约束、不变量和时间步收敛判断。字节身份另行报告，不能替代独立物理证据。

新增独立实跑：MuJoCo 3.12.0 的 flex edge equality 与 Nuka 同一零外力三角形，16 mm 边、15 mm 半厚度、每节点 10 g、120 步、dt=1/240。两者位移、长度变化和总动量均为 0；参照只支持静止结构邻接契约，不表示动态柔体或完整机器人场景一致。Newton 最新 VBD 的首次运行因单三角形没有 edge mesh 而拒绝创建，并伴随设备分配错误，保留原日志，不计为成功仿真。Genesis 尚未实跑。本批不继续扩展对照工具，先改生产代码。

默认环境的完整逐步长度门失败：首次第 217 步、最大应变 0.055211。`trace_xpbd_transient.json` 仍保持完整原轨迹身份，217 步投影后为 0.002722，ParticleFinalize 后为 0.051365。源码显示接触求解没有消费内部投影产生的速度，且窄相在投影前执行；后续先统一这条时间层，再用已有 pipeline 验证。原字节一致不用于免除该缺陷。

## 接触与材料交替求解

仅将投影速度发布到接触前仍不够。候选 source `79c39142cfc63ed175ac2b862018188203f2dfc19622b2a0a111923f0dd75923`、benchmark `410f72c8bacf910a19070d46709bde0eba9eafdea70ade4cc2fcc230fed76e22` 的全段最大长度误差为 18.7073%、最大每环境 RMS 2.3395%，第 95 步首次越界。该步 distance/bend 后为 0.2265%，finalize 后为 7.6094%，PP 没再增加；最终稳态误差约 0.219% 不能代表全段通过。证据保留在 `projection_velocity_candidate_frozen`、`projection_velocity_quality_probe` 与 `trace_projection_velocity_s95.json`。

生产代码因此改为共用粒子工作位置及交替算子序列。XPBD、PBF 和混合粒子共享 predict、投影速度提交、contact delta 和 finalize；MPM 粒子由明确的所有权范围排除。每轮材料投影后向同一个 row solver 发布速度，再把本轮新增接触位移应用到工作位置，供后续材料投影消费。现有 24 次 XPBD 和 48 次速度迭代按轮分配，未增加总预算。XPBD 与 contact lambda 在本步跨轮保留，warm-start、重力、外力和 MPM 推进均只应用一次。

共同状态修复同时使 Coupled/PBF 的步初位置来源一致，纯 PBF finalization 能消费 pseudo 位移；固定粒子不受预测、密度修正、边界 clamp 或流体速度 polish 改写。帧末统一提交，pseudo 只进入位置。已有 `pbf_predicted_pos` 字段现在保存所有 row-coupled 粒子的工作位置，保留原字段数值身份。

本次再次查询 Newton/MuJoCo/Genesis HEAD，分别仍为 `1128af7`、`297f5fc`、`0ce793b`。阅读 Newton XPBD 的同轮粒子/刚体修正与速度更新、MuJoCo PGS 的累计约束力/残差、Genesis PBD 的投影后速度更新；采用共同状态与累计增量，不复制接触数加权或按材料禁用碰撞。具体算子与边界见性能细化 spec。

六组 E=1/16/256 × eager/graph 完整 450 步已通过原 5%/1% 长度门，最大应变 1.838344%、最坏每环境 RMS 0.203045%；固定点、finite、env_status、reset/replay、副本及逐步 wrench 身份通过。候选 source `6fd819587a5eef92da91364afb38d705613b2005418ba04f11d3e55f26176508`、benchmark `690d5676fd3961eb1254cd2bca2acf4cb8a3e7bf59f50603be395274d3baf07e`、library `deb039311e6d1d4e1db4cffa8020501565aa78721a254cf1c94a6657be6b9fa9`；这不是五进程性能验收。

扩展场景 `particle_alternation_pipeline` 为 58/61，coupled C ABI 为 7/8；multi-env 与 camera C ABI 通过。粒子接触反例仍按旧 ParticlePos 时间层直接 dispatch，已改为复用真实 pipeline 的 predict/grid/contact/projection/finalize 子序列及完整预算。压板旧 80 mm 间隙与 120 mm 球/30 mm 粒子厚度冲突，且位姿驱动未给出表面速度；修正输入后继续要求原下压/恢复各 10 mm。主环境流体门原只接受表面升高或口袋降低，未量测控制相对三维运动；改为同粒子相对各自初态的三维 RMS，保留 3 mm 门。

公开 `FluidOnlyDesc` 精确输入通过现有 Python API 观察 400 步：脚边/远处控制的最终 x/y L1 为 36.3876/23.9074 m，z L1 约 1.92e-6 m；最大单粒子差异 1.6563 m，平均 0.2270 m，两个世界 env_status 始终 0。该场景只有无侧壁、零摩擦地板，流体可以侧向铺开并回到同一高度，因此最终只比较 z 会遗漏实际耦合。原始 xyz、link poses/wrench 与参数保存在 `fluid_motion_observation/`；C ABI 已有位移量测扩展到三轴，原 0.02 L1 门不变。这一观察只归因于旧量测局限，不证明完整流体本构正确。

`build_particle_contract_closure` 合并构建通过，收口复用既有场景的 17 项全部通过，coupled/multi-env/camera C ABI 分别 8/8、5/5、4/4；没有新增测试数量。修正压板输入后 rest/pressed/recovered 高度为 0.1131/0.0842/0.1130 m，下压约 29.0 mm、恢复约 28.9 mm，原各 10 mm 门通过；动态压板有布料/无布料最终高度 0.1603/0.0376 m、接触 lambda 0.007971。主环境 fluid control-relative displacement RMS 为 0.276407 m，流体 normal lambda 0.013923、机器人 qdot L1 差异 0.759586；流体单介质 C ABI 三维 L1 为 60.2950 m。

公开 Python 完整 eager/graph 链路通过，覆盖创建、DLPack、三轴外力/重力、控制、读出、masked reset、重放及 headless 渲染；原有 reset 子集 10 passed。创建与 reset 图像相同，目视检查 `particle_contract_public_graph/render_roundtrip.png` 无异常。`particle_contract_memcheck` 对冻结版本的 E=16 graph 完整主环境报告 0 errors，返回码 0。正式计时另行运行，不引用 sanitizer 耗时作为性能。

最终物理快照 `particle_contract_closure_frozen`：source `818337451b329cd25f166a3a8c7c17a69e2fd72ee574864d2d7af1aba88d58f4`、benchmark `54141e539ba920caf7649784bea6fd799934d7a5e25aec58ef3bdcd8bb2ca5f5`、library `7e3de5e12e5a87e63db63008ea42140c082bdf653a8cbd30792934372454034f`。源码、完整 diff、构建配置与验收二进制身份在该目录的 manifest/archive；日志命令位于同级 `*_command.json`。六组五进程性能分母正在独立采集。

独立共享核心/CUDA 构建隔离已提交 `6953f4a`，无 CUDA 核心构建通过。body/particle 接触仍使用本步首轮线性化；接触刷新、CCD、MPM 子步反馈和完整残差收敛仍未完成。本改造属于物理算法与时间层修复，不计作纯性能收益。旧大网格、地形脚部下沉及双浮基重叠失败继续保留，不由默认环境通过替代。
