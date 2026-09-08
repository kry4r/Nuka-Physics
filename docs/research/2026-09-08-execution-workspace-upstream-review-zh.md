# 执行、workspace 与接触缓存改造前 review

日期：2026-09-08。基线：`ea807febfe36804dde2c0d6d4ab22cf051a0f3c3`。

## Spec 与边界

已复核模块 spec 第 14 节和细化 spec 第 3、5、6、9 节。本批合并 T09a、T07a、T08a、T10a 及所依赖的 T06 错误/容量契约，补 T01 的完整 pipeline 计时。Editor 暂缓；求解和物理迭代预算保持同一通用路径。T09b refit 策略、T10b 活跃压缩和 T11/T14 求解调度仍按依赖推进。

## 最新上游

2026-09-08 再次以 `git ls-remote <upstream> HEAD` 核对，源码从本地 upstream 缓存按以下 revision 读取。

| 引擎 | Revision | 读取实现与决策 |
| --- | --- | --- |
| Newton | `5ca5f49c348f1a2772d5d6919e02882e044e4576` | `newton/_src/solvers/style3d/collision/bvh/bvh.py:55–122` 分离分配 build 与可捕获 rebuild/refit；采用固定 owner 和运行时 workspace 复用。`kamino/examples/rl/simulation.py:550–575` 分离 step/reset graph；其警告后 eager 回退不适用于 Nuka 显式 graph 请求，后者必须返回错误。 |
| MuJoCo | `0c3b94a6ed6528bb9a175428cd1e883d347ea8ba` | `src/engine/engine_io.c:1058–1138` 先 checked buffer size，再创建 data/arena 并处理分配失败；采用创建时预算与失败清理，不能将 CPU arena 当作 CUDA graph 实现。 |
| Genesis | `0ce793b42c945b6848aad624501b2ca756d3e244` | `genesis/engine/solvers/rigid/collider/collider.py:698–744` 区分 broadphase pair、candidate contact、solver contact 预算，超限停止；采用明确容量单位，不照搬固定启发阈值。`genesis/engine/simulator.py:358–450` 的 input/substep/coupling/force/sensor 边界用于核对完整 pipeline；未发现与 Nuka 同构的公共 CUDA graph 接口。 |

Newton 本次 HEAD 的变更为 “Honor explicit MuJoCo buffer capacities (#4172)”。已比较上一 review 的 `853c4fef9543fae380de59464cb804465c041721`：仅 `nconmax/njmax=None` 自动估算，显式预算保留，无法容纳初态时有警告增长。Nuka 采用显式预算与 overflow 无效契约，不静默增长。

## 实测热点

冻结二进制的 `RobotClothFluidCoResident.*` Nsight 结果在 `out/validation/execution_workspace_20260908/baseline_main.nsys-rep`，工具为 Nsight Systems 2026.1.3，RTX 5080 / driver 610.88 / CUDA 13.3.73。

- `cudaLaunchKernel` 558806 次，`cudaMalloc/cudaFree` 各 2848 次。
- 接触岛求解占 GPU kernel 总时间约 48.4%；XPBD bend/distance 约 11.6%/5.4%。
- cache prepare/mark/rebuild 合计约 19.3%，源码含多处每环境 O(P²) 全扫描。
- gyro/aero 不是当前主要热点。记录新增功能的成本，先处理实际关键路径。

这是包含创建、reset 和读出的 regression hotspot 归因，不能用 API 总时间冒充 physics-only 吞吐或优化收益。完整 benchmark 必须分离创建、capture、warmup、host submission、GPU completion 与质量读出，最终采用五个独立进程对照。

## 实施契约

1. 碰撞和传感器 TLAS 共用显式 workspace 的稳定 LBVH 排序；保留相同 Morton 算法及 local ID tie-break，运行期不分配。先不改变 refit 触发策略。
2. graph 是同一个单步 op 序列；每次动作写入同一稳定地址。成功计划复用，失败计划缓存 status/op/reason；readout demand 等结构变化才失效。reset 不改变 arena 地址。
3. cache 以 `(env,pair,feature,material)` 精确匹配，稳定 unique 和 merge，保留最低索引 owner、正常/切向冲量、normal gate、材质隔离和 age。用 scan/gather 维持原 slot 布局，不能靠散列冲突或缩小接触集合换速度。
4. workspace 和索引乘积采用 checked 容量计算；copy/memset/completion 返回真实状态。缺测项与尚未覆盖的设备状态保持显式，不把局部 T06 验收标成全项完成。
5. 使用同一 robot-cloth-fluid 场景构造作为 pipeline 与 benchmark 入口。计时路径不逐步 D2H；另行重放验证全程质量与状态，不能仅检查最后一帧。

## 本批验收

缓存 workspace 的生命周期：prepare 写入两个来源的稳定 permutation，exact-key group 合并产生 `current_owner/old_keep/matches`，warm-start 最后消费 matches。commit 先 snapshot 旧缓存；此时 permutation 已无消费者，复用为 `{free,old}` scan 输入，matches 复用为按旧索引顺序压缩的 source 表。scan 输出保留到 rebuild；snapshot/公开状态不参与 alias。prepare 与 commit 之间不重新生成接触 key，物理 solve 只改变 lambda。上述约束由固定 pipeline 的 op 顺序保证。

完整 baseline 已冻结到 `out/validation/execution_workspace_20260908/baseline_frozen/`，二进制 SHA256 为 `bb03fcb5d87329b0299f36ee8526196469477d45530baee4d3e422d471b8fdfc`；未修改的物理为 `ea807fe`。1/16/256 环境各五进程的 GPU 整批 step 中位数为 11.565/11.891/16.807 ms，原始范围与 host/wall 时间在 `baseline_five/summary.json`。各组完整 450 步质量重放零环境错误，reset/定时轨迹/重放/跨进程状态一致。不同环境数量的 state digest 不应相互比较；当前尚无 candidate 性能结论。

## 缓存活跃集合的追加 review

首版候选完整 pipeline 27 项通过，单进程 E=1/256 的 graph 轨迹、缓存和冻结基线逐字节一致，但这不是五进程性能验收。`candidate_v1_graph_e256_profile.sqlite` 显示全容量比较排序仍是热点，因此继续细化 T08 的执行成本，保持 owner/slot 和浮点运算不变。

再次核对上游 HEAD：Newton 与 Genesis revision 不变；MuJoCo 更新到 `22e3f0244b3834cda9a2bca00f988b53da843728`，已 fetch 并核对 `engine_forward.c`。Newton `kamino/_src/solvers/warmstart.py` 用 sorted key、映射和 active count 消费缓存组；Genesis `constraint/island.py::_sort_island_contacts` 在有效 contact slice 内维持确定次序；MuJoCo 的 warmstart 使用 `nefc`，其加速度缓存并非 Nuka 的冲量身份缓存，不照搬其匹配语义。

采用设备 scan 稳定压缩每环境的 current/cache 有效来源，再按 material、feature、pair 进行三轮稳定 segmented radix sort；同键维持原来源索引次序。begin/end 均来自设备，空容量不参与比较，不需运行时分配或 D2H。额外空终段使 CUB 声明的 item extent 与最大 end 一致。压缩 flags/prefix 复用尚未生成排序键的两个 key buffer；prepare 完成后排序 permutation 复用为 commit scan 输入，matches 复用为 retained source。新 workspace 与排序成本须量化，不能因更换算法就宣称加速。

纯 segmented radix 候选逐位通过，但 E=1 graph 单进程实测从首版 3.361 ms 变为 11.050 ms。随后发现 Overwatch（PID 21404，启动于 22:30:23）持续占用约 56% 图形 GPU，当前差异存在外部争用，不能全部归因于排序算法；需在空闲条件下重新比较。该实现每次 prepare 产生 27 次 radix kernel launch，保留源码、构建和 Nsight 证据，继续采用同一活跃集合，改用 CUB `DeviceSegmentedSort::StableSortPairs` 的通用稳定分段排序实现来减少小段调度，最终验收仍待测。

`candidate_v2_graph_e1_profile` 使用 10 步预热和 10 步计时，仅作短程调度归因；尚未产生布料/流体接触，benchmark 按质量规则返回 invalid（exit 2）。完整 450 步的 `candidate_v2_graph_e1.json` 质量和逐位状态通过，两者不能混淆。

扩大 C ABI 检查发现两个既有用例失败，冻结库复跑结果相同：混合介质池底低于脚底，最终流体已全部落到底面，末帧不足以证明机器人反作用；cloth cook canary 使用了已记录无法加载的 `go2.nks`。将公共耦合 fixture 的池底设在接触范围，canary 改用已支持的同一 Go2 USD 场景，保持冻结布料 hash。此处仅修复验收输入，benchmark 场景和引擎物理参数保持冻结。

相机跨环境 RGB 逐字节用例在冻结库中也有相同的 384 个不同标量。源码 `BatchedSensorTraceKernel` 通过 `MakeFidelityRng(seed, camera_pixel_id, sample)` 为各相机提供独立随机采样；中心射线的深度/法线/反照率/primitive 才应跨相同环境逐位一致。将该用例改为比较四种几何 AOV 的字节；RGB 保持原始采样与同状态重复渲染检查，不改渲染输入质量或随机序列来满足错误假设。

固定场景 eager/graph、动作变化、readout demand、graph→reset→graph、D1；现有 LBVH brute-force/退化界限与传感器渲染；必要的容量和 backend 注入失败；完整 pipeline memcheck。保存 baseline/candidate 源码与二进制、配置、分项显存、p50/p95/p99、五进程波动和物理质量。达到初始收益门槛后继续处理实测热点。
