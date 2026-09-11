# 大变形弹塑性材料与夹爪耦合

2026-09-11。Hencky J2 材料、公共参数、塑性历史、reset/readout 和动态连续表面已完成本批验收。有限质量共同接触与真实夹爪 demo 继续实施；材料落地画面不代表动态夹持已经完成。

## 资料与适用边界

| 资料 | 本次核对与取舍 |
| --- | --- |
| Jiang、Schroeder、Selle、Teran、Stomakhin，[APIC，2015](https://disneyanimation.com/publications/the-affine-particle-in-cell-method/) | 仿射速度携带子粒子角动量，不能只计算粒子平移动量就判定角动量守恒。沿用已有二次 B-spline APIC 传输。 |
| Hu、Fang、Ge、Qu、Zhu、Pradhana、Jiang，[MLS-MPM/CPIC，2018](https://yuanming.taichi.graphics/publication/2018-mlsmpm/) | 核对原文式 17/18 的形变更新与参考体积乘 Kirchhoff 应力；补充材料说明插值的零阶/一阶再现。CPIC 的切分与颜色场解决另一项不连续性问题，不能把共享单速度网格说成多材料分离。原文投影并回传最近刚体冲量，也不证明任意质量比下的共同有限质量求解。公开参考实现 MIT。 |
| Stomakhin、Schroeder、Chai、Teran、Selle，[Snow，2013](https://disneyanimation.com/publications/a-material-point-method-for-snow-simulation/) | 核对 `F = Fe Fp`、参考体积、弹塑性势能和主伸长返回。论文明确其伸长屈服及硬化经过简化，不能将该经验雪模型替代近等体积延性塑性。 |
| J. C. Simo，[乘法塑性返回映射，1992](https://doi.org/10.1016/0045-7825(92)90123-2) | 已核实 Crossref 书目信息；没有取得全文，不声称逐式复现。采用乘法分解与一致应力/屈服度量的原则，下面完整写出所选模型。 |
| [MOOSE Simo-Hughes J2 实现](https://github.com/idaholab/moose/blob/master/modules/solid_mechanics/src/materials/lagrangian/ComputeSimoHughesJ2PlasticityStress.C) | 已读取公开源码，LGPL-2.1。其等体积弹性左 Cauchy-Green 状态、有效塑性应变和非线性返回映射适于 FEM；它的 Neo-Hookean 应力与本次 Hencky 能量不等价。仅参考状态和残差契约，不复制代码。 |

论文、提取文字、相关页渲染及下载身份保存在 `out/research/elastoplastic_20260911/`，不进入版本控制。APIC 旧作者目录 404、MOOSE 文档页 403 的失败保留，分别改读 Disney 原论文页和公开源码。

三家最新源码已 fetch，记录在同目录 `upstream.json`：Newton `811b7b1ac803e193f063819d6e5d085cebdcddf6`（Apache-2.0）、MuJoCo `6783903ffece5bcbc2f8c73ba44975df86f33b9e`（Apache-2.0）、Genesis `bee8d8b0f5ffd044738da65e57744fa9e8375a13`（Apache-2.0）。Newton/MuJoCo 相对渲染批次没有本次相关的求解器源码变化。Genesis 更新了显式运动学树表及按树折叠惯量、稀疏约束调度；材料代码未变，后续多树质量响应与性能改造需核对其边界。

Genesis 的 `MPM.ElastoPlastic` 以主对数应变偏差范数做径向返回，使用 `yield_stress/(2 mu)` 半径，且 `Jp` 不演化；默认应力却是 corotated。其参数不是下面标准等效 Kirchhoff 应力的同一个数值约定，不能直接用同名参数比较完整轨迹。Newton 隐式 MPM 在函数空间求解应变率/应力和塑性增量，有不同的时间离散和硬化；MuJoCo 无等价体积材料，仅适用于机器人和刚体接触侧参考。匹配运行必须分别声明这些差异，解析不变量仍是材料的主要判据。

## 方法选择

首个模型为显式 MLS-MPM 上的各向同性、等体积、率无关 J2 塑性，采用二次 Hencky 弹性能和线性各向同性硬化。它可以分别量测小载荷回复、屈服后残余形变和再加载硬化，保留已有传输和统一接触求解路径。

有限应变 FEM 适合连续边界与高刚度小弹性应变，但当前通用 FEM 体积本构、重网格和大变形接触尚不齐全；把它设为首个展示的必经步骤会引入另一套未验收离散。MPM 已有完整 cook/world/step/graph 和水池/jelly 路径，适合先实现材料。显式方案仍受波速/CFL 约束；近不可压、高刚度的隐式扩展保留原 spec，不能减少子步或隐藏截断换速度。

J2 不模拟颗粒内摩擦、压密、损伤、黏塑性、黏附或湿材料出流。Drucker-Prager 的压力相关屈服与雪的体积塑性保留独立材料含义。当前 granular 仍含主伸长下界及 `0.15` 对数应变截断，且已有 `particle_plastic` 未实际积累材料历史；这些限制保留，不能套用到新 J2 材料。

## 冻结的材料契约

单位为 m、kg、s、Pa。输入为 `E > 0`、`-1 < nu < 0.5`、`rho > 0`、初始屈服应力 `sigma0 > 0`、硬化模量 `H >= 0`，均须有限。派生 `mu = E/[2(1+nu)]`、`K = E/[3(1-2nu)]` 也必须有限且为正。

持久状态为 `Fe`、完整 `Fp` 和累计等效塑性应变 `alpha >= 0`；初始为 `I, I, 0`。保留完整 `Fp` 可独立检查总形变及塑性体积，不用粒子位移或目标形状缓存代替材料历史。无该状态需求的材料不分配额外张量。

令 `Fe = U diag(exp(e)) V^T`、`e_dev = e - tr(e)/3`。每单位参考体积的弹性能为 `psi_e = mu ||e_dev||^2 + K tr(e)^2/2`，硬化储能为 `psi_h = H alpha^2/2`。Kirchhoff 应力为 `tau = U diag(2 mu e_dev + K tr(e)) U^T`，Cauchy 应力为 `tau / det(Fe Fp)`。MPM 力使用 `V0 tau`，不能再额外乘一次当前体积比。

屈服函数 `f = q - (sigma0 + H alpha)`，其中 `q = sqrt(3/2) ||dev(tau)||`。这是 Kirchhoff 应力度量下的模型；在体积明显变化时不等价于以 Cauchy 应力定义的恒定屈服。

一个真实子步只提交一次：

1. 先用已提交状态计算应力并做 P2G、外力、共同接触解和 G2P，得到本区间的仿射速度 `C`。
2. `Fe_trial = (I + h C) Fe_old`；分解得到 `U, e_trial, V`。不钳制奇异值或体积。倒置、非有限值、分解未收敛或无法表示的结果标记本构失败，保留先前持久状态。
3. `delta_alpha = max(0, [q_trial - sigma0 - H alpha_old]/[3 mu + H])`。若屈服，`e_dev_new = e_dev_trial [1 - 3 mu delta_alpha/q_trial]`；`tr(e_new) = tr(e_trial)`。
4. `Fe_new = U diag(exp(e_new)) V^T`，`Fp_new = V diag(exp(e_trial-e_new)) V^T Fp_old`，`alpha_new = alpha_old + delta_alpha`。由此 `Fe_new Fp_new = Fe_trial Fp_old`，且塑性流动保持 `det(Fp)`。

不可逆耗散为 `sigma0 delta_alpha >= 0`，硬化部分计入储能。总能量验收还须单列 APIC 传输/时间离散、摩擦、重力和控制器做功，不把全部能量下降归为塑性。现有 `mpm_particle_stress` 是 P2G 起点的 scratch；末态材料应力由持久 `Fe` 重新求值，不能混用两个时间层。

共享头定义材料方程、状态和失败结果；CUDA 仅调度粒子计算。作者接口与 `.nks` 保存有单位的命名参数；C 接口采用新增函数或有明确大小的描述符，保持既有结构体 ABI。world 初始状态、snapshot、整世界/per-env reset 和张量读出覆盖全部新状态。尚未支持的反向求导必须明确报错。

## 实施前验收预算

材料点独立检查用于完整管线无法隔离的本构不变量：对角伸长的屈服/硬化与上述闭式解相对误差不超过 `2e-5`；任意刚体转动的等效应力、弹性能及塑性增量相对误差不超过 `2e-5`；`Fe Fp` 重构及 `det(Fp)=1` 相对误差不超过 `2e-4`；弹性卸载和零载旋转不增加 `alpha`（绝对预算 `2e-6`）。失败不提交部分状态，NaN/倒置不能被截断后当作成功。

完整生产验证复用 jelly/water、robot-cloth-fluid、C/Python 和 reset；增加有明确触发的材料生命周期场景。机械臂 demo 的几何、质量比、厚度和时间步确定后，在运行前补充接触/形变预算；当前不编造尚未选定场景的通过阈值。完整有限质量反馈、多 owner 与共享 ContactBlock 是 demo 前置，deepest-owner 的无限质量投影不作为可接纳夹持。

渲染沿真实粒子状态重建表面并检查法线、夹爪遮挡、轮廓连续性。物理通过后才录制精选 GIF/视频并替换 README 的 Go2 展示；之后优先继续性能优化与完整耦合矩阵。任何材料算法变化都重新冻结质量通过的性能分母。

## 已实现的生产契约

共享 `nk/material/hencky_j2.hpp` 实现应力和返回映射，CUDA MPM 调度调用同一方程。持久状态保持 float，Jacobi 分解及谱计算使用 double；没有形成 `F^T F`、裁剪奇异值或放宽原精度门。`particle_F` 保存 Fe，新增完整 Fp，`particle_plastic` 保存 alpha。仅含 J2 材料的世界分配 Fp 及其 snapshot，每个粒子共增加 72 B。

`Soft.ElastoPlastic(yield_stress=..., hardening_modulus=...)` 经 authoring、`.nks`、cook 和 world 接入。C ABI 新增 `nuka_scene_add_media_ex` / `nuka_scene_add_mpm_fill_ex` 与带 `struct_size` 的塑性扩展，保持原描述符布局和旧入口。公开 Fe/Fp/alpha 字段使用 env-major、矩阵 row-major；本构失败置 `ENV_STATUS_CONSTITUTIVE_FAILURE`，不提交失败粒子的 Fe/Fp/alpha。整世界及分环境 reset 恢复材料历史、清除对应错误。J2 世界创建反向 tape 明确返回 NOT_SUPPORTED。

公共 beauty bridge 对连续 MPM 介质使用现有密度等值面重建器，颗粒介质仍显示独立粒子。重建核支撑半径为采样间距的 2 倍，体素边长为间距的一半，以参考采样体积归一化密度，默认等值面为参考密度的 0.5。它仅决定渲染表面，不改变物理状态或本构。批量 sensor 的动态粒子表面仍未接入。

## 本批验收与保留失败

全部原始命令、日志、二进制和数据在 `out/validation/elastoplastic_material_20260911/`。材料与生命周期的 `candidate_v2_frozen` 源码摘要为 `c557037f8d2868c0a5fffe4d5a2bd1434beb851e53bc65986e488b41dc34e969`；最终渲染接线及身份由 `candidate_v3_frozen` 和 `acceptance_v3.json` 记录。

| 边界 | 结果 |
| --- | --- |
| 独立材料不变量 | 三项通过：闭式屈服/硬化/卸载、转动客观性与等体积塑性、非法状态/参数不提交；原 `2e-5` / `2e-4` 预算不变 |
| 生产场景 | 82 项通过、2 项既有 disabled 保留；包含 MPM water/jelly/granular、ABA/刚体支承、XPBD 共存、多树 reset 与 robot-cloth-fluid |
| J2 生命周期 | 两环境、graph、160 步；peak alpha `0.78097314`，最大 `abs(det(Fp)-1)` 为 `6.0200691e-06`；整世界/分环境 reset 通过 |
| Python 公共管线 | 13 passed、38 deselected；authoring → `.nks` → recook → graph，Fe/Fp/alpha 读出及注入非法 Fp 的环境隔离、失败保留、reset 通过 |
| C ABI | coupled 8 项、camera 4 项、多环境 5 项通过；表面桥接改动后 coupled 8 项再次通过 |
| 完整固定管线 | E16、N=1/2、250 预热＋200 测量步通过；state/wrench 与上一渲染批次逐字节一致 |
| 传感器 | E16×2、256²、4 spp 完整 physics→sensor 通过，所有 AOV 与渲染批次一致 |
| CUDA 内存 | J2 两环境生命周期 memcheck 0 errors |
| 无 CUDA | 核心、材料不变量及 `nuka_render` 构建通过；不代表已实现完整 CPU solver |
| 动态 beauty | 480×360、8 spp，initial/24/48/120/reset 五帧；形变帧图像不同，reset 与 initial 完全一致；物理数组与表面接线前一致，并实际检查初末画面 |

保留的失败及修复：

1. FP32 分解的转动弹性能误差超过冻结门限。内部分解/谱计算改 double，闭式参考使用实际 float 输入的 double log；失败在 `material_invariants_v1.*`，通过在 `material_invariants_v2.*`。
2. 零容量 Fp 的 arena 地址仍非空，导致非 J2 世界 snapshot 误复制。`Data::FillView` 统一将所有零字节字段绑定为 nullptr；保留 `candidate_v1_frozen` 和 Python v1 的创建失败，完整 v2 回归验证修复。
3. 无 CUDA 目录关闭测试目标，首次 `host_core_v2` 找不到 `nuka_math_test`。显式开启测试后，`host_core_v3` / `host_material_v3` 通过。
4. 动态画面首次只显示粒子；增加连续表面后又发现 bridge 的粒子下载条件遗漏新表面集合，导致画面停在原点。失败画面保存在 `render_v3` / `render_v4`，修复后 `render_v5` 增加形变帧图像差异检查并通过。更早的 `render_v2` 是检查脚本误用字段名，修正为 `Field.PARTICLE_POSITION`。

本批属于物理材料及公共生命周期扩展，连续表面属于渲染功能接线，不宣称性能收益。后续先拆分 MPM 预测/提交，让有限质量网格端点和所有 owner 进入共同 ContactBlock；随后完成夹爪加载/卸载、反力/能量、dt/分辨率收敛、正式视频和主页替换。完整三引擎匹配实跑、granular 历史审计、连续柔体接触与介质矩阵仍在原规格范围。
