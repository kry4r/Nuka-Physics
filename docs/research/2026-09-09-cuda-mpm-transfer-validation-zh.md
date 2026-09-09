# CUDA MLS-MPM 传输优化验收

本批只改变 CUDA 后端的工作组织与读取方式。原 bunny-water 的粒子数、网格、材料、40 子步、完整反作用和执行完成边界保持；共享粒子归属/接触容量的收益另见 [架构验收](2026-09-09-mpm-ownership-validation-zh.md)。

## 五进程结果

每种执行模式运行五对独立进程，交替 A/B、B/A。分母为体积契约修复与粒子归属改造后的 `ownership_frozen`，候选为 `cuda_transfer_v3_frozen`。测完整 `World` step 的 GPU 完成时间，上传、读回、质量扫描与渲染在计时外；创建与 graph capture 单列。运行环境为 RTX 5080、驱动 610.88、CUDA 13.3，清除继承的 `CUDA_SCALE_LAUNCH_QUEUES`，各冻结目录显式绑定自己的动态库。

| 执行 | 阶段 | 分母中位数 ms/步 | 候选中位数 ms/步 | 耗时变化 |
| --- | --- | ---: | ---: | ---: |
| eager | 静置 | 41.2126 | 29.7164 | −27.89% |
| eager | 落水 | 48.2696 | 32.8117 | −32.02% |
| graph | 静置 | 36.9482 | 25.0946 | −32.08% |
| graph | 落水 | 44.6110 | 28.8090 | −35.42% |

十对均改善。落水的逐对变化范围为 eager −32.41% 至 −31.79%，graph −35.47% 至 −35.37%。Data arena 从 71,218,432 B 增至 95,699,456 B，增加 24,481,024 B（34.37%）；Model arena 保持 221,952 B。增量来自显式索引及连续传输记录，不是设备峰值显存测量。节点数量远大于粒子的网格仍会按共同 workspace 容量保留记录，该额外内存成本尚未消除。

## 实现与失败候选

- 稳定 cell sort 后，连续存放原位置、质量、速度、参考体积、三轴 base、APIC 矩阵与应力；P2G 保留粒子顺序及 APIC/应力分项累加，只计算所需的单轴权重。
- 每个占用 cell 的首项标记公共 27 节点 stencil。活跃列表供 P2G、grid update 和 body project 使用，物理网格场仍完整初始化。
- CUDA 私有 workspace 使用命名区域；质量和反作用字段不再充当整数索引。CUB 布局按 device/count 缓存，每个 op 只分区一次，并传播查询、排序、选择与 launch 失败。
- 每个 body 的节点线/角反作用协作分块读取，按原稳定顺序累加。现有自由体与关节反作用时间层未在性能批次中修改。

首版缓存全部权重与 64 位 base，静置 P2G 变慢，已拒绝并保存 profile。第二版连续属性读取明显提速，但按 cell 代替真实 base 丢弃了越界粒子的部分 stencil，内联坐标计算也存在不同的收缩舍入机会，未进入正式测量。最终版本保留真实 int32 base 和裁剪贡献，显式保留权重坐标减法舍入；没有通过降低子步、改变 EOS 或放宽门槛取得收益。相关决策与上游增量见 [review](2026-09-09-mpm-performance-review-zh.md)。

## 质量与内存安全

十对均通过预先定义的有限状态、正 J、无网格逃逸、全程体积比偏差 ≤5%、splash、减速、反作用与淹没门。最小 J 为 0.935678005，最大体积比偏差为 1.6673171%。既有 33 项 transfer、静水、jelly、颗粒介质、刚体/关节及混合 MPM/XPBD 场景通过；未新增单测集合。

完整 660 步质量记录、轨迹和十个末帧字段在全部进程间一致。轨迹 FNV 为 `6cf2c38c57c8f590`，末帧状态 SHA256 为 `80d6a2404b91bd38afd478b7cd3b8ccd92e0105a2392fb12bab2875a04d8f872`。这些身份检查用于发现运算和输入变化，物理验收仍依据上述质量门及已有解析场景。

复用九项既有场景做针对性 memcheck 与 synccheck，覆盖 transfer、非法坐标、snapshot、混合介质、偏心刚体反作用和多环境关节 deposit；两者均为 0 errors。固定 robot-cloth-fluid 的 E=16 graph 完整 450 步 pipeline 通过，包含 reset、重放、副本等价和逐步反作用；env_status 为 0，最大长度误差 1.838344%，最坏每环境 RMS 0.203044919%。

## 剩余热点

候选 Nsight Systems profile 的落水 kernel busy 为 29.926 ms/步；profile 用于归因，不替代上表的普通运行时间。

| kernel | 落水 ms/步 | kernel busy 占比 |
| --- | ---: | ---: |
| P2G gather | 18.7867 | 62.78% |
| 连续传输记录准备 | 3.9674 | 13.26% |
| 应力计算 | 1.4792 | 4.94% |
| radix sort onesweep | 1.2053 | 4.03% |
| G2P | 1.1445 | 3.82% |
| body project | 0.8488 | 2.84% |
| F 更新 | 0.7992 | 2.67% |
| body reaction gather | 0.4186 | 1.40% |

P2G 为 39 registers/thread、无 local spill。Nsight Compute 硬件计数器仍受 `ERR_NVGPUCTRPERM` 限制，实际带宽、stall 与 occupancy 未采集。接下来应依据 P2G/记录准备的成本审查数据布局与合并读取，以及粒子/节点 workspace 容量分离；减少寄存器数量本身不构成收益证据。

## 保留的物理边界

全程局部最大 J 仍为 79.4880753；总量预算不能证明稀疏自由表面的局部水密度正确。当前 bunny 依赖 SDF，静态 plane 由独立网格 floor 承担，状态 16 和 `coupling_complete=false` 如实保留。single owner、有限质量、多 shape 的实际动力学 owner、关节子步反馈、无 SDF 网格及 MPM↔XPBD 直接界面仍须完善；这些能力没有由本次性能验收关闭。

## 重放与身份

证据目录：`out/validation/mpm_performance_20260909/`。正式报告为 `cuda_transfer_v3_paired/summary.json`，每个进程保存命令、GPU 条件、质量、状态及日志。`cuda_transfer_v3_scenarios.json`、`cuda_transfer_v3_memcheck.*`、`cuda_transfer_v3_synccheck.*` 与 `cuda_transfer_v3_profile_attribution.json` 保存辅助证据。

候选 source SHA256 为 `c9e83ed7c4e819d80a45e9369026dbb001c5abaac64240538dd42cec76b296af`，demo 为 `16f11950c868f8415bd6595bc0b77353da0ecb776cec7f8ac1fdf535edcf5dd0`，library 为 `f2c07e3a5900e506a46812eede863348d9dd9e1ceeffa51607c293e9c9d09e11`。完整源码清单、差异与 build 配置保存在冻结目录；该版本未包含后续 CUB 接触 graph 修复。

```text
python .nuka-runs/measure_mpm_pairs.py --baseline out/validation/mpm_performance_20260909/ownership_frozen --candidate out/validation/mpm_performance_20260909/cuda_transfer_v3_frozen --output <新结果目录>
```
