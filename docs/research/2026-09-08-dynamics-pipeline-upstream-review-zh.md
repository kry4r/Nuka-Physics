# 自由旋转、风阻和执行契约 review

日期：2026-09-08。依据模块 spec 的 M01/M02/M04/M07 及细化 spec 的 T02c/T04a/T05a/T06a，合并实施和验证。Editor 暂缓。此前外力/重力基线是 `0953516`，冻结结果保留。

## 上游身份与结论

本次再次使用官方 origin `ls-remote HEAD`，并以 `git show <revision>:<path>` 读取缓存源码，避免把旧 checkout 当最新实现。

| 引擎 | revision | 读取位置及采用理由 |
| --- | --- | --- |
| Newton | `853c4fef9543fae380de59464cb804465c041721` | `newton/_src/solvers/solver.py` 的 COM/惯量坐标和 `tau-w×Iw`；其显式更新不直接用作稳定 gyro。`semi_implicit/kernels_particle.py` 从固定速度读、向独立 force 写；采用读写分离，D1 不采用原子汇总。`featherstone/solver_featherstone.py` 用实际 articulation start/end 和矩阵 offsets，不能用 max DOF 代替真实映射。 |
| MuJoCo | `0c3b94a6ed6528bb9a175428cd1e883d347ea8ba` | `src/engine/engine_forward.c` 的隐式速度导数、局部 LU 和不支持组合检查；`engine_passive.c` 的相对空气速度与局部阻力坐标。当前源码未找到与 Genesis 相同的自由体 midpoint 实现，不把 Genesis 注释当作 MuJoCo 的实现证据。 |
| Genesis | `0ce793b42c945b6848aad624501b2ca756d3e244` | `genesis/engine/solvers/rigid/abd/forward_dynamics.py` 的惯性帧 midpoint、Newton/backtracking、相对动量残差和姿态时间层；采用数值思想，不移植 free-joint 特化调度及 composite/armature augmentation。`pbd_solver.py` 的粒子二次空气阻力与面各向异性模型不同。 |

MuJoCo 相对上一 review 的 `dbe1e2e2c336684cf60674a19ba96ec62b839c16` 只更新 CI/通知文件。本表是源码设计对照，未实跑三引擎，不提供外部性能结论。

## 自由体时间层

现有顺序为外力 kick → MPM reaction → 共享接触求解 → pose drift。所有冲量在同一个当前 pose/world inverse inertia 下作用；保留此顺序，在 drift 中求解无外力自由转动。惯性主轴坐标下：

```text
m0 = I*w0
mm = (m0+m1)/2
wm = inverse(I)*mm
r = m1-m0 + h*(wm × mm) = 0
R1 = R0*Cayley(h*wm)
```

用归一化动量变量、float Newton 和回溯，初始最大 12 次 Newton、8 次回溯、相对残差门槛 `1e-6`。3×3 有主元消元，不使用零 inverse 掩盖奇异问题。姿态增量是归一化四元数 `[1,h*wm/2]`，与离散动量方程配对：收敛时同时保持能量和世界角动量；单独改变 omega 后继续旧姿态公式没有该保证。完整外力/接触分裂仍是一阶，只有无力 drift 宣称二阶。

pseudo 仅提供几何修正，不增加真实世界角动量。先保存 physical L，再应用 pseudo rotation，按新姿态惯量重建 physical omega；因此 omega 数值可以随姿态惯量变化，真实 L 不变。pseudo 可能改变转动能量，此处不宣称同时保能量。每个 body 输出残差、迭代和状态；失败保留 drift 前 pose/omega、设置环境失效位，绝不静默改显式更新。此前 kick 和其他 body 的工作不回滚；整个受影响环境不能当作有效轨迹。

## 风阻时间层与耗散界

一面一线程只读 step-start 位置/速度，写独立每面 impulse；第二核按 cooked CSR 的 tri ID 顺序汇总，每粒子写一次速度。创建时从最终粒子索引构建 CSR，因此 MPM+XPBD 的 offset、SoftFluid 的附加粒子及多环境复制共用同一逻辑。无 aero 时不分配粒子邻接字段。

保留当前静止空气、法向/切向二次阻力和 rest area。固定顶点作为静止支点，不读其无效速度参与平均，也不写速度。每面每顶点原 impulse 为 g；令 d_i 为顶点 incident face 数、W=Σ_dynamic v_i·g≤0、S=Σ_dynamic inverse_mass_i*d_i。统一面缩放满足：

```text
alpha <= min(1, -W/(S*|g|^2))
alpha <= max_dv/(|g|*max_i(inverse_mass_i*d_i))  [max_dv > 0]
delta_K <= Σ_faces (alpha*W + 0.5*alpha^2*S*|g|^2) <= 0
```

Cauchy 上界处理共享顶点的总和，避免每面分别 clamp 后总 dv 超预算。小 dt 未触发缩放时保持原力；多面共享相同面缩放，不按顶点质量分别破坏面耗散。没有新增 wind API；非零气流在后续参数契约中另行设计。

## 创建与执行

创建前检查实际 articulation spans、局部 parent/root、joint type、真实 DOF、共享上限及 DOF map 一致性。当前尚无完整异构 map，拒绝不同 topology/joint layouts 的混合树；不按机器人名判断。单树手工 Model 可使用已有缺省 span。host/device 共用关节枚举和容量常量，避免重复 magic number。

Pipeline Build 返回缺失 required op，完整列出缺失项；任何生产物理 op 不可静默删除。World 在 upload/arena 前进行校验和能力检查，包含初始化/reset 所需 op。Step 遇首个 host/launch 失败停止，记录 op 和状态；未 Ready 的空结果不能算成功。readout demand 先验证能力，成功回填后才更新 pipeline/demand 并失效 graph；失败不能破坏已有 forward pipeline。设备数值状态显式查询，不在正常步强制 D2H，也不把异步 launch 成功当成数值收敛。

## 验收及性能边界

首轮整批运行的 22 项中 21 通过、1 项已知 contact graph 跳过，但主环境 `env_status=2` 暴露邻域截断，因此不能作为完整有效验收。将 T17a 的容量池基础部分并入本批：逐粒子真实计数、64 位 scan、环境私有池和实际 CSR offset，所有 PBF/粒子接触消费者共用。保留固定 arena 地址和容量溢出状态，公开 attempted/retained；不扩大一个硬编码 slice 后称问题解决。

补充 review 再次查询三引擎 HEAD，revision 未变化。Newton `semi_implicit/kernels_contact.py:68` 用 hash-grid iterator 遍历候选，无私有 32 条截断；Genesis `sph_solver.py:248` 调 `for_all_neighbors` 累积密度；MuJoCo `engine_collision_driver.c:2025` 分配碰撞 arena，失败时报告容量问题，其 manifold reduction 与流体邻域不是同一个模型。Nuka 采用容量与活跃邻域分离，不照搬即时 iterator 导致现有累积顺序变化。

容量默认沿用每粒子 32 条的总字节预算，允许单个粒子使用更多邻居；由 `ModelCapacities` 提供显式每环境 pool 预算，可调而非布局常量。真实计数超过池容量时环境失效，其他环境的预算不被抢占。查询计数不再创建局部 32 元素 scratch；填充按粒子 ID 固定排序。容量充足时与 brute-force 邻域对照，必须无截断、同环境内双向邻域一致。刷新/skin、完整公共容量配置及新边界密度模型仍属 T17 剩余工作。

容量池首轮报告 attempted=30996、retained=8608、max=168。继续追查发现 `SoftFluidPredictKernel` 的 soft 分支未写 `pbf_predicted_pos`，共享网格将全部 169 个布料粒子视为原点，产生 28392 条假邻域记录并耗尽池。对照上述 Newton 同一 `particle_x` 查询/测距以及 Genesis 同一 `particles_reordered.pos` 查询/密度输入，修复为动态和固定 soft 粒子均写入统一预测时间层；主 pipeline 首步检查该位置数组。不扩大容量掩盖错误；邻域刷新和边界密度的其余要求仍保留。

扩展现有 robot-cloth-fluid 主环境，启用风阻和非等方自由旋转，保留控制、耦合、reset、D1。只补主场景不能隔离的角动量/能量与高精度 dt 收敛、共享面耗散、非法拓扑、缺 op 和中途失败 oracle。整批一次构建后统一运行，失败再定点重跑。保存日志、源码/二进制身份、耗时和 model/data bytes；新增诊断与第二风阻核的成本明确列出。进入性能模块后按用户要求集中压榨关键路径性能，同质量完整 pipeline 为最终收益依据。

最终功能验收见 [报告](2026-09-08-dynamics-pipeline-validation-zh.md)：主环境无假邻居和截断、全程环境有效，25 passed/1 个已知 graph skip；12 项 memcheck 为 0 errors，公共 API/reset/渲染通过。性能带宽/profile 和五进程对照未采集，明确留到后续性能批次。
