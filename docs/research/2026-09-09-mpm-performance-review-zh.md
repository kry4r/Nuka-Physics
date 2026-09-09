# MLS-MPM bunny-water 性能与耦合 review

主负载采用 `examples/demo/mpm_water_drop_demo.cpp` 的 MLS-MPM bunny-water。`water_pool_demo.cpp` 使用 PBF，不能作为 MLS-MPM 的性能分母。当前工作按用户顺序先完成有效路径的性能优化，随后接续无 SDF、有限质量、多 owner 和子步多体反馈；全部 goal 保持 active。

## 规格和上游

已复核总模块 spec 的 M08/M09、细化 spec 的 T18/T19/T23 和性能方向 spec。原始源码、SHA256 与 revision 查询保存在 `out/validation/mpm_performance_20260909/upstream_review*.json`。

| 上游 | 最新查询 revision | 采用与边界 |
| --- | --- | --- |
| Newton | `1128af71407e002d15a8cfb180b35158cf64c42f` | ImplicitMPM 的 active function spaces、scratch 生命周期、容量/错误契约；隐式算法不是 Nuka 显式 MLS-MPM 的等价性能分母 |
| MuJoCo | `297f5fc6592e89fc41960251e88b6cf94fd7573e` | 有限质量端点、约束残差和共同质量算子；MuJoCo 没有对应 MPM solver |
| Genesis | `4efac5f92ac444c9a606379ecc1c0a6b4f20874c` | MPM dirty-grid 与活跃节点生命周期；其 atomic scatter 与 Nuka 稳定 gather 的求和顺序不同，不直接照搬 |

Genesis 从 `0ce793b` 更新后的 MPM solver 与 legacy coupler 内容 SHA256 均未变。相关增量主要移除 solver 中的渲染 transform staging，并调整渲染接口；已读取 FEM/rigid 相关差异，不需要重复全文审查相同 MPM 文件。

容量改造前于 10:37 UTC 再次查询：Newton 更新为 `90c56c3be73e35a34ccb37ab593a529e8a3d18dd`，MuJoCo 更新为 `f4959d9dc173cbe61adcb9fb5bfb76040264bccd`，Genesis 保持 `4efac5f`。已 fetch 并检查增量：Newton 只改资产下载重试；MuJoCo 为发布/版本更新，`engine_support.c` 仅改版本常量。相关求解实现未变，证据为 `ownership_upstream_review.json`、`ownership_upstream_delta.json` 及原始 diff。

## 真实输入与首次观测

原工作区缺少 `.nuka-assets/stanford/bunny.obj`。从 [Stanford 原始扫描](https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz) 的 `bunny/reconstruction/bun_zipper.ply` 恢复 35,947 顶点、69,451 三角形，仅转换 xyz 与三角索引，没有简化、封洞或改拓扑。OBJ SHA256 为 `205b80d216a4c4fd0a7e9a313e1ac37af0df0df068f1291cb969d4028136075c`，档案与转换身份见 `input_identity.json`。这是有完整身份的新输入，不声称与缺失的历史本地资产逐字节相同。

保持原 demo 的 180,000 粒子、110,400 网格节点、40 子步、dt=1/240 s、dx=0.011 m、Tait 系数 200,000、gamma=7、viscosity=0.4、2.5 kg bunny、240 步静置及 420 步下降。复用普通 `World` 生产链路与原 mesh SDF cook，不添加专用物理路径。

现有 demo 已补充明确的 eager/graph CLI、GPU 完成事件、步骤失败/下载检查、每步 position/velocity/F 与刚体反作用轨迹、完整末帧状态、内存与体积观测。物理计时排除读回、质量扫描和渲染，创建与 graph capture 单列。CUDA 光追 demo 不依赖 Vulkan，构建目标已从 Vulkan gate 移到 CUDA 目标范围，原 executable 路径保持。

冻结初验为 `baseline_frozen`：source `adf150e798a6b7f71ef62922363f3e5cf8336a62622902bd3b5057abaaa78191`，demo `f42a579598bed4c319f79afc7eb71f259e0edc4a55ed31ac0405a50c9bc64fa3`。单进程 eager 下降阶段平均 51.958 ms/步；数据区 1,641,139,200 B，模型区 75,101,952 B。最小 J=0.928397、最大 J=2.99999952，静置体积比 1.003959，完整体积比最大偏差 1.57359%；有 splash、减速与正向反作用。

原 probe 的 PASS 不是最终性能分母验收：源码将流体 J 隐式截断到 [0.3,3]，观测达到上限；环境状态 16 表示无 SDF 地板，由相同几何位置的独立网格 floor BC 实际承担流体碰撞。完整通用耦合仍未实现。保留原始报告，先处理体积截断，再冻结性能分母，不把不同物理轨迹相减当加速。

## 物理契约先行

当前 EOS 是 `p = B * max(J^-gamma - 1, 0)`，不是 `B/gamma * (J^-gamma - 1)`。因此参考态声速为 `sqrt(gamma*B/rho0)`，本输入约 37.42 m/s，单独波速的 `c*dt_sub/dx` 约 0.3543。旧文档中 14 m/s、0.09 dx、四倍 CFL 余量的说法不符合实际代码，需纠正参数口径；不能悄悄把 EOS 改软来降低成本。

体积演化应遵从已有散度更新 `J_next = J*(1+dt*tr(C))`。正的有限结果不应被硬截断；非正或非有限结果必须保留失败状态，不作为有效仿真。正压截断对应当前材料的无拉应力假设，与任意数值体积截断不同。此修复单独标作物理算法/失败契约变化，之后的 CUDA 或架构结果均对比修复后的同一分母。

沿用原输入的反作用、splash 与淹没检查；物理验收看所有步骤的正且有限 J、完整体积变化预算 5%、无逃逸及有限状态，并复用已有 rest、transfer、jelly 和耦合场景。原 J≤3 判据直接镜像实现的截断上限，不能继续用它证明物理正确。体积预算不会根据候选结果放宽。

### 体积更新修复与边界

已移除数值 J 截断。无效 J 不提交到 F，设置原有 `MpmGridEscape` 失败位；有效 J 按原散度公式更新。参数、Tait 刚度、子步数、无拉应力材料假设均保留，CFL 说明按实际 EOS 纠正。这是物理算法/失败契约修复，不能把前后时间差记为优化收益。

`volume_contract_frozen` 的 source 为 `4bcb0d50432468cefe4c4589732641c96eddca9ed224fd15636ba7d25405642f`，demo 为 `4c4f4dcb80c41bdc144ad6cbf666fca33ca70824d389186183025bacbef37a8d`，library 为 `0f1c5e72bf55ae6f914c72d84360fae4728828c7173fa37bf1c8e91e9a6b4ddd`。完整 eager/graph 均通过既定门：最小 J=0.935678，最大体积比偏差 1.667317%，splash、减速、反作用及淹没检查通过，无非有限状态/网格逃逸。全程轨迹 FNV 为 `6cf2c38c57c8f590`，十个末帧字段在 eager/graph 中相同，状态 SHA256 为 `80d6a2404b91bd38afd478b7cd3b8ccd92e0105a2392fb12bab2875a04d8f872`。已有 28 项 transfer、静水、jelly、刚体、关节、多环境和混合介质场景通过，没有增加单测集合。

局部稀疏膨胀仍需审查：全程最大 J=79.488075；末帧 J 的 p99/p99.9 为 1.25160/1.82691，180,000 粒子中 33 个 J>3、5 个 J>10。总体积门不能证明这些粒子的局部液体密度正确。后续按无拉应力材料、自由表面稀疏采样和耦合界面分别归因；不重新增加任意截断掩盖它。该输入只作为明确质量预算内的当前材料/耦合路径性能分母，不是完整不可压水或通用耦合的验收。

单次下降阶段 eager/graph 为 51.367/47.521 ms/步，仅用于初验；五进程结果另报。原无 SDF 地板状态 16 和 single owner/有限质量/关节子步反馈限制仍保留。完整命令、质量、状态比较与场景结果位于 `volume_contract_*`，旧截断版本的证据不覆盖。

### 已采集的热点与采用方向

`baseline_bunny_profile_attribution.json` 根据完整步骤的 CUDA event 与 launch correlation 分开静置/下降。下降阶段 GPU kernel busy 为 47.000 ms/步，其中 P2G 30.752 ms（65.43%）、反作用 gather 3.658 ms（7.78%）、CellKeys/逐粒子节点标记 3.287 ms（6.99%）；静置 P2G 为 26.670 ms（68.07%）。P2G 每线程 71 registers、无 local spill，不能把当前瓶颈称为已证实的寄存器溢出。该 profile 采自保留的旧截断版本，只用于选方向，性能收益仍对比体积修复后的分母。

Nsight Compute 返回 `ERR_NVGPUCTRPERM`，没有获得硬件计数器；保留命令与失败输出，不推断实际带宽/occupancy。采用以下可直接由数据依赖和重复工作证明的 CUDA 方向，再由完整 pipeline 检验净收益：

- 每子步按稳定排序的粒子索引预计算质量与 B-spline 权重，保持 P2G 内原 APIC/应力分项及固定累加顺序。
- 由已排序 cell run 的唯一首项标记 27 节点 stencil，消除同一占用单元内各粒子的重复整数原子操作。
- 活跃索引、计数、cell ranges、排序节点值和权重记录使用后端私有命名 workspace；网格质量和反作用字段保持各自类型。延伸活跃集合至 grid update/body project，保留网格场的清零语义。
- 每个刚体的节点数据并行分块加载，线/角反作用仍按原稳定节点顺序累加；保持自由体反馈与关节 deposit 的现有时间层，时间层修复另做。
- 检查 CUB/workspace 与 launch 错误。记录额外 workspace 成本，不用删除 eager 完成检查、改变材料或减少子步取得收益。

首个 CUDA 候选 `cuda_transfer_v1_frozen` 保持完整轨迹/十个末帧字段相同，既有 33 项场景通过，但不接受其静置回退：单次 eager/graph 静置为 44.307/39.747 ms，高于容量基线的 41.471/36.965 ms。profile 中 P2G 静置从 26.670 增至 33.838 ms，尽管 registers 从 71 降至 54；缓存完整权重/64 位 base 没有取得净收益。反作用分块读取降至下降阶段 0.419 ms，cell run 标记为 0.259 ms，这两项保留。原候选、状态和 profile 完整留存，不进入正式性能验收。

下一实现取消完整 B-spline 记录：稳定 cell run 已给出 base，P2G 直接计算当前 stencil 的单个轴权重，避免重复 floor、64 位 base 读取与无用轴权重。位置、质量、速度、C、应力及参考体积按稳定粒子次序连续打包，替换每个贡献项的多数组间接读取；仅复制原 f32 属性，原 APIC/应力分项及累加顺序保持。其他 workspace/活跃集合/错误传播与反作用改动不变，再进行联合验证。

第二版单进程 eager 静置/落水为 28.393/30.154 ms，graph 为 23.617/26.189 ms；33 项既有场景通过，但完整轨迹与分母不同，未进入正式五进程验收。代码审查发现按 cell 推导 base 会丢弃越界粒子的部分 stencil，且内联权重的坐标乘减存在不同的收缩舍入机会。下一版在连续记录中保留已检查范围的三轴 int32 base，恢复部分 stencil 贡献及其逃逸状态；单权重计算显式保留坐标减法舍入。仍不缓存完整权重，也不改变材料、迭代或反作用公式。原第二版结果完整保留，数值差异需验证归因，不能只凭时间改善接受。

本次更新前再次查询并 fetch：Newton `90c56c3be73e35a34ccb37ab593a529e8a3d18dd`，MuJoCo `319cf22fdf6fabb803205fcafbb9398707736c8d`，Genesis `8325a478f0d52a1c28bb58c95dfe519fcf38fc2b`。Newton 无增量，MuJoCo 仅阻抗说明/图与渲染包初始化变化；Genesis 更新异构 default armature、树内 inverse weight 刷新及 MJCF actuator force range，MPM/coupler 源码未变。本批不改惯量或驱动力契约；后续有限质量和多体耦合须包含转子惯量、整个关节树的有效质量刷新，不能照搬忽略锁定轴的简单 slider 权重。版本与原始增量保存在 `cuda_transfer_v3_upstream_review.json` 及各引擎 diff。

最终第三版通过 [五进程与完整 pipeline 验收](2026-09-09-cuda-mpm-transfer-validation-zh.md)：落水 eager 48.270 → 32.812 ms（−32.02%），graph 44.611 → 28.809 ms（−35.42%）；静置也改善，十对完整状态和质量相同。Data arena 增加 24.48 MB 至 95.70 MB。33 项既有场景、固定 E=16 graph pipeline 及针对性 memcheck/synccheck 通过。最终 profile 中 P2G/记录准备仍占落水 kernel busy 的 62.78%/13.26%，继续保留其优化方向与网格主导容量的内存成本；本批不关闭完整耦合或全路径性能目标。

## 性能方向与分类

1. **共享架构**：纯 MPM 与混合 MPM/XPBD 应使用相同的粒子所有权定义。当前 `RowExemptParticles` 仅识别混合模式，纯 MPM 的 180,000 粒子被分配 720,000 个不会发射的 body-particle slots，使 contact/cache/row workspace 显著膨胀。统一 model/cook/pipeline 对网格粒子范围的认识，保留真实刚体接触预算、全部网格反作用及混合场景 XPBD 接触；不得通过关闭接触获得减量。
2. **CUDA 后端**：在 profile 确认成本后延伸 active nodes 到网格准备、更新、SDF 投影及稳定反作用排序，减少重复标记和全网格发射。活跃索引、计数、cell ranges 和 CUB 临时区使用命名的私有 workspace 生命周期，不继续把 `grid_body_dp`/`grid_mass` 解释成索引。保留粒子/节点求和顺序、全部子步、应力、本构与反作用。
3. **CUDA 后端**：P2G 已缓存每粒子应力，不能把该已完成工作再次作为优化。新方向需根据完整 profile 决定是否缓存重复 B-spline 权重、整理稳定粒子读取或改驻留调度；不采用 demo 阈值、材料专用融合求解器或浮点原子替换固定 gather。
4. **错误/执行**：检查 CUB 返回值、workspace 查询、索引乘积与 launch；正常 eager 的完成/失败语义不能为了计时删除。MPM stage timer 每阶段同步，仅用于归因；正式结果使用未开启 stage timer 的完整 step。

每项先确定有效分母和主热点，再成批修改，复用现有 pipeline 联合验证。五进程交错结果、轨迹/物理量、显存及回退分别记录，架构收益和 CUDA 收益不混称。

## 性能之后的完整耦合

- 几何：SDF、解析 primitive、凸体距离/穿透与网格 BVH 查询输出共同接触数据；网格单面/双面与厚度必须明确，检查复合形状和 proxy 到实际 body/link 的映射。SDF 不应成为通用耦合的必要条件。
- 约束：从 deepest single owner 变为稳定的多端点约束流；刚体世界惯量、articulation `J*M^-1*J^T` 和网格节点质量共同决定冲量，双方使用同一增量。
- 时间层：每子步 deposit 后刷新 rigid/articulation 速度；外力、姿态预测、外层积分和反作用只能作用一次。现有自由体逐子步反馈、关节全部子步结束才 deposit 的不一致需要消除。
- 完整链路：多个自由体、多个机器人、混合树和 MPM↔XPBD 直接界面不能只通过某个刚体间接表现耦合。按接口残差、质量比、线/角动量、固定边界外部冲量和步长收敛验收。

这些是后续必须执行的功能任务，不能由当前 SDF bunny 的单个响应或性能结果代替。
