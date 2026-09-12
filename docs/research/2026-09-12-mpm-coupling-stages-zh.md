# MPM 耦合调度与状态提交

本批将一个公共物理区间内的 MPM 网格预测、边界交换和粒子提交拆为显式生产算子，为网格端点进入共同接触求解器保留状态。它属于后端无关的调度架构改动；没有改变材料、本构精度、网格传输、接触公式、迭代预算或 CUDA kernel 的线程块选择。多 owner 有限质量 `ContactBlock` 尚未实现，当前交换仍采用 deepest-owner 投影与反作用。

## 对照与依赖

已 review 原模块 spec M02/M08/M09、性能细化 spec §8.1 和全量耦合 spec 的共同时间层/质量契约。2026-09-12 重新查询官方主干：Newton `e30e1dee2958ecc195ee11f1aaf42bf8a1ab84b0`、MuJoCo `ba57cabefde8580158266a0f76ac321da19d110d`、Genesis `58f25d87a5cd8dd1c3882bd0eaf3b2714035f07a`，证据位于 `out/research/mpm_stages_20260912/`。

Newton 的隐式 MPM 独立保存输入/输出状态，并通过 coupling interface 声明端点质量、状态刷新、代理反作用收集；Genesis MPM 在 `substep_pre_coupling` 中 P2G，在 `substep_post_coupling` 中 G2P，网格跨越中间耦合点。采用明确的状态所有权与前后边界，不复制代理力回卷或 legacy 速度投影作为有限质量证明。MuJoCo 的前向外力、约束质量响应及最终积分顺序用于核对依赖；材料不同，不视为 MPM 物理 oracle。本批是源码 review，没有新增三引擎匹配仿真。

## 生产契约

| 操作 | 读取 | 写入及生命周期 |
| --- | --- | --- |
| `MpmPredict` | 区间起点粒子位置、速度、APIC C、F、材料与外力 | 初始化本区间网格、排序/active scratch、应力与预测速度，保留原静态地板投影；粒子持久状态保持不变 |
| `MpmExchange` | 预测网格、当前刚体/关节表面速度与质量响应 | 更新网格及 owner 速度、区间反作用；反作用只累计一次 |
| 共同 row solve | 自由/耦合后的端点速度、共同质量与接触行 | 求解已有刚体/关节/粒子接触；MPM 网格保留到提交 |
| `MpmCommit` | 完成交换的网格速度、区间起点粒子数据 | G2P、位置积分、弹塑性 F/历史更新一次 |

Pipeline 为每个公共子步重复完整清单，所有 MPM 操作共用同一 `MpmParams` 和 h。网格/transfer scratch 不与中间 row solve 复用内存。关节反作用在共享求解前写入广义速度；其外步累计由 `MpmExchange` 后的 `AccumulateStep` 负责。错误状态仍在完整区间结束时收集。eager 保留每区间一次的完成同步，graph 的失败仍通过 replay/公开同步报告；移除同步属于后续执行优化。

内部 `NkOp::MpmStep` 由三个操作取代，不保留第二条 umbrella 求解路径。公开 C/Python 的 World stepping 接口与保存的场景格式不变。CUDA 注册、错误名称、provider、公共子步展开和已有直接 transfer 不变量使用同一组新操作。

## 验证

整批验证通过，汇总及完整命令位于 `out/validation/mpm_stages_20260912/acceptance.json`。环境为 Ubuntu 24.04 / WSL、Release、g++-10、CUDA 13.3、RTX 5080，驱动 610.88。候选源码与二进制在 `stages_frozen/` 冻结；补充 water 可执行文件在 `stages_execution_frozen/`，两份源码摘要相同。

| 验证边界 | 实际结果 |
| --- | --- |
| 合并构建 | `nuka`、两支弹塑性 demo、pipeline benchmark、scenario、coupled-world C ABI 通过；water demo 补充构建通过 |
| 现有 MPM/混合场景 | `*Mpm*:*MlsMpm*:RobotClothFluidCoResident.*` 39 项通过，包含 transfer、材料、刚体/关节支承及混合系统；两个原有 granular 诊断/吞吐探针仍 disabled |
| 调度不变量 | 扩展既有 transfer 检查：预测/交换后粒子位置、速度、F 不变；网格注入已知冲量后，G2P 获得相同线动量且只积分一次。常速度重现、子步收敛及拒绝算子私自细分仍通过 |
| 公共 C/Python | coupled-world C ABI 8 项通过；Python eager/graph 覆盖 cook、创建、控制、step、接触/反力、读出、运行时参数、选择性 reset、重放和减半 dt |
| 固定 robot–cloth–fluid | E=16、graph、N=1/2，各 250 预热＋200 测量步；完整状态与每步 wrench 对 `autograd_contract_20260912` 逐字节一致，450 步布料质量门通过 |
| 渲染 | 公共 pipeline 的创建/步进/reset/状态恢复及细步长帧全部生成，eager/graph PNG 哈希一致，reset 图像逐字节恢复；步进画面完成本地检查 |
| CUDA 内存 | transfer、公共子步组合律、MPM/XPBD 共驻、多环境 articulation deposit 四项 memcheck 通过，0 errors |
| 多后端边界 | 无 CUDA 的 `nuka_nk` 核心构建通过；核心没有新增 CUDA 类型或设备布局要求，不代表完整 CPU solver 已实现 |
| 弹塑性完整捕获 | 压缩和 bunny 各运行冻结基线/候选一对；各自原物理检查全部通过，`positions.bin`、`metrics.csv` 以及 bunny 的 `material_state.bin` 逐字节一致 |

候选源码 SHA-256：`3d39af6e3c6feea49da7df10a9ba3a0c14ca26ab45d6c17532f13625ba73fe2c`；`libnuka.so.0` SHA-256：`51ae747386cfb32496be1d8ce48fca75604a04bfa2c04101622c13235ecfd108`。冻结时 HEAD 为 `8ed2e96`，manifest、源码增量及各可执行文件哈希共同标识本批。最终汇总再次检查工作源码与冻结内容一致、live library 与冻结库一致。

两支 demo 的基线来自 `out/validation/material_performance_20260912/division_frozen/`，与已发布捕获的身份衔接见 `out/validation/material_performance_20260912/timing_instrumentation_acceptance.json`。新捕获在 `stages_compression/`、`stages_bunny/`；原 MP4/GIF 未重编码或替换。bunny 四元数经独立归一化和 acos 量测出现 `5.16e-8 rad` 的舍入量，原始位姿文件哈希完全相同。

物理检查独立于旧版身份：压缩的轻载卸载高度误差 0.06484%，塑性残余压缩 33.133%，历史单调且体积检查通过；bunny 的自由落体误差、线/角动量、非增能和 reset 检查通过，最大接触穿入 2.393 mm、稳定支承力 24.501 N。压缩的未解释能量损失约 0.209 J，bunny 约 1.904 J，保留原 demo 能量账本及限制，不把轨迹相同当作全部接触物理正确。bunny 的压入量仍是带载值。

首次 memcheck 启动因 PATH 未包含可执行工具而返回 126，未执行任何 CUDA 检查；失败命令及日志保留为 `stages_memcheck*`。随后使用 `/usr/local/cuda/bin/compute-sanitizer` 重跑，成功结果为 `stages_memcheck_cuda*`，没有覆盖失败证据。

## 计时、容量与限制

这是调度架构验收，未进行五进程性能采纳。以下为上述完整捕获各一对的均值，仅记录观察和回退；GPU 边界包围完整 `World::StepConfigured`，控制区间墙时包含已有上传、反力/状态读出及完成等待，不包含粒子分析、写捕获和渲染。

| 输入（graph） | GPU 完整步 ms，基线 → 候选 | host 调用 ms，基线 → 候选 | 控制区间墙时 ms，基线 → 候选 |
| --- | --- | --- | --- |
| 压缩 | 0.525722 → 0.535315（+1.82%） | 0.086763 → 0.097158（+11.98%） | 50.5465 → 51.1903（+1.27%） |
| bunny | 3.378387 → 3.382981（+0.14%） | 0.118586 → 0.094069（−20.68%） | 241.4577 → 244.3478（+1.20%） |

两对的模型、持久区、workspace、tape 和计时事件数量均相同；压缩 Data arena 为 57,851,136 B，bunny 为 209,031,424 B。本批未新增模型字段或设备容量。设备可用内存为含其他进程的观测，CUDA event 内部分配大小不可查询，不以这些值宣称已测独占显存峰值。host 波动与单进程结果不作性能结论，材料/CUDA 算术候选也未随本批混入。

后续有限质量接触需在此边界接入所有活跃 `(node, owner, feature)`，将网格质量、刚体世界惯量和同 articulation 交叉项纳入共同 `ContactBlock`；再完成连续柔体端点、外迭代及全量介质矩阵。移除 eager 同步需单独验证完成/错误语义和同质量五进程性能。本次提交后先按 [功能 TODO](../roadmap/2026-09-08-demo-and-engine-todo-zh.md#后续功能清单待确认) 请用户确认下一批，不继续自动开启实现。
