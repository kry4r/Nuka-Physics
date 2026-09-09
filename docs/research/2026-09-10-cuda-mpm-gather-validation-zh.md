# CUDA MLS-MPM 有序 gather 优化验收

连续记录写入优化后，P2G 仍占原 bunny-water 落水 kernel busy 的 69.49%。本批将一个线程负责一个节点的读取和累加，改为按质量及三轴动量四个独立分量组织协作组。改动仅在 `src/phi/backend_cuda/ops/mpm.cu`，属于 CUDA 后端实现；没有改变材料、子步、几何、耦合公式或公共模型契约。

## 通用实现

稳定 cell sort 中，同一 y/z 行的相邻 x cells 对应连续粒子区间。P2G 合并这些区间的读取，仍保留原 cell/particle 次序、环境边界和真实 stencil base 检查。公开 CUB `WarpLoad` 读取完整原记录，每个线程计算独立粒子贡献；质量与各动量分量分别按原顺序逐项累加，动量仍先 APIC 后应力，没有跨粒子的浮点重排。

协作组大小由质量＋三轴动量的四个独立累加量决定，所有节点和材料使用同一代码。组内共享记录缓冲在读完后复用为贡献缓冲，复用前后显式同步；ballot、同步和最终结果交换使用各逻辑组的掩码与宽度。网格大小复用已有设备资源查询，由 grid-stride 消费有效节点数；CUDA 线程组织和 shared memory 没有进入共享核心。

## 五对完整流程结果

分母为已验收的 `coalesced_records_frozen`，候选为 `logical_group_frozen`。eager/graph 各五对独立进程，交替 A/B、B/A；原输入为 180,000 粒子、110,400 节点、40 子步、dt=1/240 s、dx=0.011 m，每个进程完成 240 步静置和 420 步落水。

计时为完整 `World` step 的 GPU 完成时间，上传、下载、质量扫描和渲染在计时外。运行环境为 RTX 5080、驱动 610.88、CUDA 13.3；各进程绑定对应冻结库，清除继承的 `CUDA_SCALE_LAUNCH_QUEUES`，保存设备时钟、温度、功耗及原始命令。下表为五个进程平均步时的中位数。

| 执行 | 阶段 | 分母 ms/步 | 候选 ms/步 | 耗时变化 | 逐对变化范围 |
| --- | --- | ---: | ---: | ---: | ---: |
| eager | 静置 | 26.6927 | 23.7475 | −11.03% | −12.66% 至 −9.93% |
| eager | 落水 | 30.0604 | 24.4804 | −18.56% | −19.36% 至 −18.06% |
| graph | 静置 | 22.3818 | 18.8973 | −15.57% | −15.63% 至 −15.45% |
| graph | 落水 | 26.1234 | 20.3586 | −22.07% | −22.12% 至 −21.88% |

全部配对均改善。落水 p50/p95/p99 的五进程中位数为 eager 29.857/33.874/34.884 → 24.344/26.310/27.661 ms，graph 25.598/29.959/30.845 → 20.521/21.426/21.665 ms。Data arena 保持 95,699,456 B，Model arena 保持 221,952 B；这两项是引擎分配预算，不是设备峰值显存。

## 质量与补充场景

十对完整质量字典、轨迹和末帧状态均相同。最小 J=0.935678005，全程总体积比最大偏差 1.6673171%，保留原 5% 门；有限状态、正 J、无网格逃逸、splash、减速与反作用检查通过。轨迹 FNV 为 `6cf2c38c57c8f590`，末帧 SHA256 为 `80d6a2404b91bd38afd478b7cd3b8ccd92e0105a2392fb12bab2875a04d8f872`。这些身份用于检查读取/运算语义，物理判断仍依靠质量门及已有解析不变量。

复用已有 33 项 MPM 场景全部通过，包括常速度 transfer、网格总质量、静水、jelly、颗粒介质、刚体/关节支承和混合 MPM/XPBD；未新增单测，两个既有禁用诊断保持禁用。针对小尾块、非法坐标、共享存储复用、偏心反作用及多环境关节 deposit 的九项 memcheck/synccheck 均为 0 errors。

固定 E=16 robot-cloth-fluid graph 的完整 450 步通过，env_status=0，reset、重放、副本等价、逐步 wrench 和原长度门保持；最大长度误差 1.838344%，最坏每环境 RMS 0.203044919%。该固定环境使用 PBF，MPM 验收来自原 bunny-water 和上述场景。

较早的独立回归进程曾显示关节支承套件耗时增加 11.42%。针对该迹象复用完整 `ArticLinkOnMpmRest` 套件运行五对：墙时中位数为 2.388 → 2.253 s（−5.65%），逐对变化为 −19.56% 至 +5.82%，没有复现稳定回退；区间较宽，不宣称稳定的小环境加速。匹配同一支承案例的 1,100 次 P2G 调用，profile 均值为 63.620 → 25.749 μs，全部 kernel busy 为 164.386 → 122.567 ms。该数据只用于归因，包含创建/读出/断言的套件墙时不能冒充 GPU step 延迟。

## 性能归因与未采用候选

当前完整 Nsight Systems profile 的落水 kernel busy 为 19.610 ms/步，P2G 为 11.880 ms/步（60.58%），此前对应 profile 为 17.609 ms/步。应力、radix onesweep、G2P 和记录准备分别约为 1.400、1.127、1.080、0.951 ms/步。P2G 仍是主要热点，不能据本批结束 MLS-MPM 性能工作。

P2G 使用 67 registers/thread、14,976 B shared memory，无 local spill；本次设备资源查询产生 504 个 block，每 block 128 个线程。寄存器和 shared memory 增加换来了完整流程收益，不能只凭资源数判断快慢。实际 occupancy、带宽和 stall 计数器仍受 `ERR_NVGPUCTRPERM` 限制，未将静态资源或推测写成硬件实测。

| 中间候选 | graph 静置/落水初验 ms/步 | 处理 |
| --- | ---: | --- |
| 32 线程 warp、逐贡献 shuffle | 30.718 / 31.192 | 明显回退，拒绝；P2G profile 为 22.626 ms/落水步 |
| 32 线程 warp、共享内存单 owner 折叠 | 21.573 / 21.978 | 保留中间证据，继续改善累加线程利用 |
| 32 线程 warp、四个分量独立折叠 | 21.928 / 22.319 | 单次结果未优于前者，不据此断言稳定差异 |

所有候选保持完整状态和质量相同。最终仅对统一四线程协作组进行正式五对与完整补充验收；失败版本、二进制、命令、状态及必要 profile 均保留，没有提交按场景切换的候选分支。

## 边界、上游与重放

局部最大 J=79.4880753 仍待解释。当前场景的无 SDF 地板由独立网格边界处理，`env_status_union=16`、`coupling_complete=false` 保留；single owner、有限质量、复合形状实际 owner、关节子步反馈、无 SDF collider 与 MPM↔XPBD 直接交换仍未完整实现。字节一致不将这些既有物理限制变成正确结果。

改造前 review 的 HEAD 为 Newton `e8974b314b60d2c10cb4d02a4d1d60dadb7a54e1`、MuJoCo `deb60af526451d588958e03833aea0459528ae1e`、Genesis `b3c6c73a7a671fc486df5d69c9f481a91b1d57b6`。相关 MPM/耦合/约束源码未变；采用公共 CUDA 数据移动与资源调度，不将上游的 atomic scatter 或隐式求解换入当前有序显式 MPM。决策链见细化 spec §5.2 与本地各候选 `*_review.json`。

证据目录：`out/validation/mpm_gather_20260910/`。正式结果为 `logical_group_paired/summary.json`；`logical_group_scenarios.json`、`logical_group_pipeline.json`、两项 sanitizer 日志、`logical_group_profile_attribution.json` 和 `logical_group_artic_pairs/summary.json` 保存辅助证据。

冻结 source SHA256 为 `4ca46a532ada1c2c1b2af9f92300bbe4b4f8e0af4d2472b9a248c807d945c3fd`，library 为 `3b3a81d5c80fb067c3c3977157f7a1638e26ed9e301f7c11bfcd843f9cfaa73c`，demo 为 `1589d6525ce76617d9f925fcdb9a7def526403f728df1538379ebee255a179da`。

```text
python .nuka-runs/measure_mpm_pairs.py --baseline out/validation/mpm_layout_20260910/coalesced_records_frozen --candidate out/validation/mpm_gather_20260910/logical_group_frozen --output <新结果目录>
```
