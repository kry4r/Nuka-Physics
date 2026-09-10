# MLS-MPM cell 共享传输验收

状态：当前性能批次已收口，采纳最终版本并进入完整耦合改造。日期：2026-09-10。原 bunny-water 五进程落水 eager/graph 步时减少 20.03%/43.25%，新增 Data arena 47.69 MB。water/jelly、固定 pipeline、内存/同步和渲染检查通过。小水池墙时及历史压痕回退保留；按用户最新授权，不等待穷尽性能热点。

## 实现与分类

改动位于 `src/phi/backend_cuda/ops/mpm.cu`。每个 occupied cell 共享粒子记录，计算二次 B-spline 的 27 个节点质量/动量部分和，节点按固定 z/y/x cell 次序合并。本构、40 个子步和 APIC/应力传输项不变；浮点累加分组变化，按**数值累加算法变化及 CUDA 实现**验收，不能称为跨版本字节等价优化。

32 个 CUDA lane 协作读取原 116 B 记录，27 个 stencil lane 按稳定粒子顺序累加质量、APIC 和应力冲量。无效 lane 仍参与同步，保留真实 base、clipped stencil 和环境切片。复用 active-node 列表枚举 occupied cells，跳过空区间；occupied cell 自身总在该列表中。部分和索引使用较小的单射域：P<N 取稳定粒子区间起点，否则取 cell key，容量严格为 `min(P,N) × 27 × 16 B`；只改变地址，不切换算法。

最终版合并节点部分和汇总、动量归一化、重力及原静态边界速度更新；逐粒子本构应力与记录准备也合并执行，并保留原 stress 字段读出。删除重复 Unique 筛选和多余阶段，全部输入走同一通用路径。共享模型、公共 API 和调度没有新增 CUDA 要求；无浮点原子、场景阈值或质量预算缩减。

```text
m_i,c = sum(p in c) w_ip m_p
q_i,c = sum(p in c) [w_ip m_p (v_p + C_p (x_i-x_p))
                     - dt w_ip V0_p (4/dx²) tau_p (x_i-x_p)]
m_i = sum(c) m_i,c
q_i = sum(c) q_i,c
```

相消误差用贡献绝对值之和与分组求和误差界衡量，不除以接近零的最终动量。保留数学项不等于物理验收；独立传输矩、解析收敛和生产场景共同作为依据。

## 输入、性能与内存

主输入为 `examples/demo/mpm_water_drop_demo.cpp` 的原 MLS-MPM bunny-water：180,000 粒子、110,400 节点、40 子步、dt=1/240 s、dx=0.011 m、240 静置步及 420 落水步。总体积误差预算保持 5%，未用同名 PBF demo 替代。

分母为 `out/validation/mpm_capacity_20260910/capacity_frozen`；最终版为 `out/validation/mpm_cell_transfer_20260910/transfer_prepare_frozen`。RTX 5080，驱动 610.88，CUDA 13.3。每模式、版本各五个独立进程，交错先后顺序；GPU 完成时间覆盖 World step，上传、下载、质量扫描和渲染另计。

| 执行/区间 | 分母中位 ms/步 | 最终版中位 ms/步 | 耗时变化 | 逐对变化范围 |
| --- | ---: | ---: | ---: | --- |
| eager 静置 | 24.4073 | 19.3743 | −20.62% | −23.08% 至 −11.17% |
| eager 落水 | 25.2728 | 20.2116 | −20.03% | −22.79% 至 −16.82% |
| graph 静置 | 19.1002 | 10.7855 | −43.53% | −44.52% 至 −39.67% |
| graph 落水 | 20.5661 | 11.6711 | −43.25% | −44.26% 至 −42.18% |

每进程先计算完整区间均值，再取五进程中位数。`transfer_prepare_pairs/summary.json` 保存结果，各进程另保存 host call、同步墙时、分位数、硬件条件和质量/状态身份。二十个运行全部 valid；每个版本自身跨进程、eager/graph 的完整质量字典和末帧状态一致，跨版本轨迹不同。

| 存储 | 分母 B | 最终版 B | 差值 |
| --- | ---: | ---: | ---: |
| Data arena | 94,306,816 | 141,999,616 | +47,692,800（+50.57%） |
| Model arena | 221,952 | 221,952 | 0 |

新增区域就是 cell 质量/动量部分和；arena 预算不等于包含驱动和渲染的设备峰值。大环境复制时需保留该容量代价，不能由单环境收益宣称普遍加速。

## water pool 与 jelly

复用已有完整场景 `MpmJellyBall.DropSquashRecoverNoCollapseNoSpike` 与 `MpmFluidRest.HeavyBodyIntoPoolDeceleratesAndReacts`，包含创建、完整步进、读出及断言，交错运行五对。以下是**进程/场景墙时**，不与 GPU step 延迟混用。

| 范围 | 分母中位 s | 最终版中位 s | 变化 |
| --- | ---: | ---: | ---: |
| 两场景完整进程 | 8.9357 | 8.4591 | −5.33% |
| jelly 场景 | 4.878 | 4.600 | −5.70% |
| 小水池落体场景 | 4.032 | 4.360 | +8.13% |

完整进程四对改善、一对回退；jelly 五对均改善。小水池逐对有改善也有回退，不能用总体中位数隐藏其回退，亦不能据 profiler 宣布根因解决。原始日志、gtest JSON、程序 SHA 和硬件条件在 `water_jelly_pairs/`，分场景统计在 `per_scene_summary.json`。

用户已允许性能足够时转阶段，并暂停压痕 demo。原 bunny 的稳定收益及 jelly 改善支持进入耦合；小水池墙时、host 提交、稀疏网格、多环境及其余热点保留 TODO，未宣称全性能范围完成。

## 物理与完整 pipeline

最终冻结版上轮已完成 34 项场景，另有 2 项既有 disabled 未启用。包括 transfer、材料选择、jelly、静水、刚体/关节支承、MPM/XPBD 共驻；颗粒套件在本轮接手前已自然结束，此后未再启动压痕测试。

只新增一项非零应力与正负动量相消检查，并扩充原恒速 transfer 的加速度观测。两个相邻 cell 各含 37 粒子，覆盖 shared tile 尾块；以 B-spline 零/一/二阶矩检查质量、线动量和含 APIC 内部项的角动量，三个 dt 检验应力冲量缩放。最终版质量相对误差 `2.616e-8`，线/角动量归一化误差最高 `1.879e-8` / `2.230e-8`，低于 `2e-6` 门。

匀加速在相同物理时间、1/2/4 子步的位置误差为 `1.38886637e-4`、`6.9447226e-5`、`3.46977181e-5 m`，符合 symplectic Euler 一阶收敛。原始值见 `transfer_prepare_scenarios.log`。这不证明完整非线性落水的步长收敛已完成。

静水 `deep_J=.98265`、`surf_J=1.01290`，质量相对误差 `1.250e-8`、无 escape。jelly 最大压缩约 21.2%，随后恢复，未坍缩、尖峰或 escape。小水池刚体减速并产生向上反作用，初验 `min_J=.9855`、`max_J=1.3732`。这些是对应输入的验收，不表示全部材料或耦合已正确。

固定 E=16 robot-cloth-fluid 完整 450 步 graph pipeline 通过：最大距离应变 1.838344%、最坏每环境 RMS 0.203045%，reset/replay/replica/读出通过。该 pipeline 流体为 PBF，不能代替上述 MLS-MPM 验收。证据见 `transfer_prepare_pipeline.json` 及命令/日志。

最终版 bunny 初验 `min_J=.936041117`、`max_J=78.626503`、最大总体积误差 1.66798911%；`env_status_union=16`、`coupling_complete=false` 保留。总体积通过不证明局部大 J 和完整耦合已解决。数值比较通过 `transfer_prepare_numerical_gate.json` 绑定最终源码、程序及成功物理日志 SHA，未放宽配置、体积预算和每版本确定性。

## 内存、同步与渲染

针对传输、尾块、无效粒子、jelly、MPM/XPBD 共驻及双环境关节支承，memcheck 和 synccheck 各运行 10 项已有场景，均通过且 `ERROR SUMMARY: 0 errors`。分母与最终版各完成一次原 bunny 仿真及 8 张 headless 渲染：640×360、8 samples、落水步 75 至 390、步距 45。画面对照未发现新增缺失或明显异常；每对平均绝对 RGB 差为 0.057–0.230/255，描述数值轨迹差异，不作为物理真值。证据为 `render_comparison.json`、`render_contact_sheet.png` 及两套原图。CUDA 验证不表示其他后端已具备 MPM 求解能力。

## 保留的中间版本与失败

`cell_frozen` / `cell_reviewed_frozen` 为含重复 Unique 的首版，`active_cells_frozen` 移除重复筛选，`grid_finalize_frozen` 合并节点汇总/速度更新，最终 `transfer_prepare_frozen` 再合并应力/记录准备。各冻结版本保留；本报告主表只使用最终版五对，未用中间版数据冒充。

中间 `active_cells_frozen` 原 bunny 五对落水 eager/graph 减少 14.92%/42.24%，但压痕五对进程墙时 15.9521 → 18.0066 s（+12.88%），未采纳为最终候选。其 profile 中 P2G GPU 累计 2855.067 → 669.508 ms、全部 kernel 5459.232 → 3333.021 ms，不能推翻普通进程回退；另多出 35,610 次 launch。暂停后未再追测，不能声称最终版已消除该回退。

中间版本的小共驻/关节支承五对也有进程波动。节点阶段合并后的轨迹相对早期 cell 版有差异，保留完整物理门，不统称逐字节等价。

## 身份、review 与后续

性能改造前已 review spec 和 Newton `e8974b314b60d2c10cb4d02a4d1d60dadb7a54e1`、MuJoCo `c04c9c726c93852ee3b5b58211ce085f80d98b5d`、Genesis `b3c6c73a7a671fc486df5d69c9f481a91b1d57b6`，见 `transfer_prepare_review.json`。未新增三家引擎的匹配实跑结果；物理依据是解析不变量和生产场景。

| 最终 artifact | SHA256 |
| --- | --- |
| source | `85bae533e68b8ba1a8776a474e6a1dddd45e6a11ac740d0d502983a36504b916` |
| demo | `83d7e6525eb000affad8ac78b631a3f5ff00e832f900187f90e71c6f40cfa09c` |
| scenario | `d340582d1e20fd03456d918cf2e3ae983786f859cad799e838c8a602d7a9852a` |
| library | `39b33102a8db709a7203b51564fccb9a9a98afcf611d101e635c05b641b15794` |

scenario/demo 静态链接求解器，必须运行各自冻结 executable，不能只换 `LD_LIBRARY_PATH`。早先 `baseline_transfer_invariants` 误用候选程序，已排除分母身份；真正分母随后运行 `capacity_frozen/nuka_scenario_test`。冻结失败的 `cell_validated_frozen` 也不是有效证据。

原始证据位于 `out/validation/mpm_cell_transfer_20260910/`，不提交临时日志/脚本。Compute 硬件计数器仍受权限限制，没有实测 bandwidth/stall/实际 occupancy 的结论。

下一阶段按 [共享几何与 owner 细化 spec](../plans/2026-09-10-coupling-surface-owner-detailed-spec-zh.md) 开始完整耦合：无 SDF 几何、真实动力学 owner、多有限质量端点、关节子步反馈及 MPM↔XPBD 直接表面。局部大 J、single owner 和延迟反馈不因性能通过而关闭；耦合后继续原 spec 与其余性能项，总 goal 保持 active，Editor 暂缓。
