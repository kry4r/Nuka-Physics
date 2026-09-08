# 执行、workspace 与接触缓存验收

日期：2026-09-09。状态：本批通用修复、联合正确性、修正基线五进程、容量等价、公共渲染及最终 profile 已验收。依据 [改造前 review](2026-09-08-execution-workspace-upstream-review-zh.md)、[全路径方向 review](2026-09-09-performance-directions-upstream-review-zh.md) 和 [细化 spec](../plans/2026-09-09-performance-directions-detailed-spec-zh.md)。本报告不关闭完整引擎 goal。

## 比较版本与边界

所有证据位于 `out/validation/execution_workspace_20260908/`。两个新目录均采用禁止覆盖的冻结方式，manifest 包含源码逐文件 SHA256、源码总身份、构建配置、二进制、设备与 CUDA 版本。

| 版本 | 源码身份 SHA256 | Benchmark SHA256 |
| --- | --- | --- |
| `baseline_corrected_frozen` | `0e03737df65359d42c41426b49a078308bb749f00123561eedb0a310b22ca4ae` | `0e065deac17313ee0f701b902c9548f893e3b3a1d483d5fa1327c4bb8a8d0169` |
| `candidate_corrected_frozen` | `41433e3e495aeab4073b9c25417c5645f75744d951e5afda088bfeb7739bb941` | `1cbdf661bb0aaa6a75e6c9380e6ec308d7f0e31af85638d79892f147bab81e91` |

修正基线保留 `ea807febfe36804dde2c0d6d4ab22cf051a0f3c3` 的执行架构、LBVH 临时分配及旧接触缓存算法，仅修复 pair snapshot 的索引和分区。新增 benchmark、字段名称辅助和 CMake 目标属于观测工具。源码补丁可直接核对，不以替换动态库假装切换静态链接 benchmark 的实现。

候选增加公共 eager/graph 模式、稳定 LBVH workspace、按有效接触集合稳定排序/合并的缓存、checked 容量和错误传播，并按实际 rigid sort 域分配 pair scratch。接触身份、GS 顺序、迭代预算、精度及随机输入保持原契约；没有场景专用求解路径。

## 基线正确性修复

原默认容量 E=256 中，env=15 的物理状态与其他相同副本不同；扩大空容量后恢复。旧版与候选在同容量下都发生这一错误。最大粒子位置差约 0.199391 m，关节 q 差约 0.373813，不能解释为浮点噪声。

`PairSortFillKernel` 的排序域是 `E * rigid_slot_cap`，但 snapshot 和 permutation 曾采用 `E * full_slot_stride` 索引。E=256、rigid cap=32、完整 stride=1108 时，env=15 的 snapshot 与紧随其后的环境 prefix 区间重叠；旧 World 的保守大 allocation 遮蔽了跨字段覆盖，allocation 级 memcheck 不一定报告越界。

修复统一 snapshot/permutation 的紧凑索引，每排序项分配两个 u32；候选 World 从 pipeline 读取真实排序容量，向算子传递并检查 workspace bytes。contact cache 仍使用完整接触容量，两者不混用。

benchmark 新增创建时相同副本检查和末帧物理副本逐字节比较，将 `replica_state_equal` 纳入 valid。原先只有有限性、env_status、reset/重放及两种介质接触存在，无法排除旧/新版共同错误。原 `baseline_frozen`、`candidate_frozen`、`baseline_five` 和 `candidate_five` 全部保留；其中 E=256 数据失去同质量加速比较资格。

修复后的完整运行通过：256 个物理副本相等，450 步质量重放 env_status 为零，reset 和定时/质量末帧逐字节一致，布料与流体 articulation 接触均存在。物理状态摘要 `9cc8cb92ff4f4325` 与旧版正确的 capacity×2/4 一致。正式新旧版本已直接比较保存的完整 `.state` 字节，E=1/16/256 的物理状态与缓存全部相等；E=256 完整状态为 89008128 bytes，SHA256 `f7f44f586a9334ffece3c3eb58376a67b9b6c1968854a1bb0861d56c3f9c4dc1`。

## 联合验证

| 范围 | 结果与证据 |
| --- | --- |
| 完整生产 pipeline 及必要不变量 | `pipeline_compact_snapshot.json`：33 passed，0 skip；包含机器人/布料/PBF、控制、读出、接触 graph/reset、旋转、风阻、邻域和 MPM/XPBD 同驻留 |
| LBVH 候选集合和退化界限 | `lbvh_compact_snapshot.json`：3 passed |
| 公共耦合接口 | `coupled_compact_snapshot.json`：8 passed |
| 公共多环境/graph/version | `multi_env_compact_snapshot.json`：5 passed，含 4096 环境 |
| 相机几何 AOV 契约 | `camera_compact_snapshot.json`：4 passed |
| 紧凑 workspace 内存检查 | `memcheck_compact_e16.json/log`：完整 16 环境 graph benchmark，0 errors，质量门通过；计时受 instrumentation 影响，不作性能结果 |
| Python/DLPack/控制/reset/渲染 | `public_compact_eager`、`public_compact_graph` 和 `public_compact_comparison.json`：质量与状态摘要相等，8 对 PNG 直接逐字节一致；粗/细步长 graph 各捕获 1 次，重放 72/144 次；已目视检查画面 |

相机不同 camera ID 使用独立随机采样；相同世界跨环境比较几何 AOV，RGB 检查同输入重复渲染确定性。既有 fixture 问题和旧版失败证据见改造前 review。测试数量不代替完整 pipeline 质量；残差、互补性、完整穿透曲线仍属未完成诊断。

## 性能采集与限制

RTX 5080、计算能力 12.0、driver 610.88、CUDA 13.3；Release、g++-10、nvcc、all-major。相同 Go2/自由体/XPBD/PBF 环境，269 粒子/环境，dt=1/240，48 次速度迭代、24 次布料迭代。每进程 250 步预热、200 步定时，另完整重放 450 步检查每步 env_status 和末帧质量。

计时包含完整物理 pipeline 与接触读出；每步 GPU event，批尾等待 GPU 完成，计时内无 D2H。分别保存 host 提交、GPU 完成、同步墙时及步时 p50/p95/p99。五进程是同确定输入的独立进程，seed 没有用于随机化输入，不称为五个随机实验。

依次采集 `baseline_corrected_five`、`candidate_corrected_five`、`candidate_corrected_capacity2`、`candidate_corrected_capacity4`，每组保存首进程完整末帧状态。共 65 个独立进程全部通过质量门，各配置跨进程状态一致。`corrected_comparison.json` 已比较新旧同容量完整状态，以及容量变化后的权威物理状态；缓存空槽长度变化不参与跨容量物理等价判断。

下表为完整批次 GPU 完成时间的五进程中位数，单位 ms/step；括号为五进程最小–最大值。加速比以修正后的旧 eager 为分母，候选 graph 需显式选择执行模式。

| 环境数 | 旧 eager | 新 eager | 新 graph | graph 加速比 / 延迟下降 |
| --- | --- | --- | --- | --- |
| 1 | 12.456（10.628–14.071） | 10.115（6.977–11.867） | 3.073（3.071–3.078） | 4.053× / 75.33% |
| 16 | 11.906（11.085–13.232） | 10.043（7.256–10.982） | 3.367（3.360–3.371） | 3.536× / 71.72% |
| 256 | 15.651（14.790–16.458） | 9.948（9.852–10.319） | 6.499（6.488–6.506） | 2.408× / 58.47% |

graph 三组范围都与基线明显分离。E=1 eager 范围重叠；E=16 eager 的区间仅相隔约 0.10 ms，且进程内/进程间 host 提交有明显波动，不能将中位数差异当作稳定性能上界。E=256 eager 的完整 pipeline 为 1.573×。host 提交时间可能包括驱动队列背压，不等于纯 CPU 指令成本；GPU event 与 host wall 分别保留，不混为一种时钟。

| 环境数 | 旧 data arena bytes | 新 data arena bytes | 增量 |
| --- | --- | --- | --- |
| 1 | 3284736 | 3519488 | 234752（7.15%） |
| 16 | 52760064 | 56246016 | 3485952（6.61%） |
| 256 | 844197888 | 900229376 | 56031488（6.64%） |

E=256 的新 contact cache workspace 为 68104959 bytes、LBVH workspace 为 95743 bytes；pair scratch 从修正旧架构的 12542464 bytes 降至 373248 bytes，净 arena 仍增加约 53.44 MiB。该表为字段分配预算，未冒充含 graph、驱动和全部渲染资源的显存峰值。

capacity scale 放大槽位并保守分配每槽完整刚体行数：slots=1108/2216/4432，rows=3638/26592/53184。这是空容量压力测试，不代表所有字段均匀扩大两倍或四倍，也不替代真实接触密度曲线。

| scale | E=1 graph ms | E=256 graph ms | E=256 data bytes | 权威状态与默认容量 |
| --- | --- | --- | --- | --- |
| 1 | 3.073 | 6.499 | 900229376 | 逐字节相等 |
| 2 | 6.497 | 18.721 | 3987371264 | 逐字节相等 |
| 4 | 10.426 | 35.315 | 7945769472 | 逐字节相等 |

容量等价修复通过，空容量成本仍显著。当前改造没有完成 T10b 的 active row/J 布局，不能把整体吞吐增益称为容量问题已经解决。

Nsight Systems 用 GPU correlation ID 选择定时区间，不把创建/预热/质量重放混入热点占比。Nsight Compute 采样曾返回 `ERR_NVGPUCTRPERM`，当前没有硬件带宽、stall 或实际 occupancy 数据；不以静态寄存器数冒充瓶颈证明。

最终 `candidate_corrected_graph_e256_profile` 从 901 次 graph launch 选取预热后的 200 次，`candidate_corrected_steady_kernels.json` 记录 kernel busy 合计 6.226 ms/step。岛求解 1.644 ms（26.40%）、LinkContactWrench 1.262 ms（20.28%）、XPBD bend/distance 0.659/0.251 ms（合计 14.63%），行生成 0.476 ms（7.64%）。这些是归因数据，不替代上表无 profiler 的正式计时。

下一组优化选择：稳定有效 row/endpoint 索引、确定性 link wrench gather，随后减少无效 J/质量算子工作并优化岛调度。依据是完整有效基线中的 20.28% 读出成本及空容量曲线；必须计入索引构建与额外内存，保持原 row/A→B 顺序。达到首批收益后继续实测热点。

## 公共使用与复现

Python 对已创建的 World 使用 `world.set_execution_mode("graph")`，之后 `step()`/`step_n()` 消费相同单步 pipeline；`world.synchronize()` 等待物理完成，`world.execution_info` 提供模式、捕获/重放次数与错误信息。选择 graph 不推进状态，失败显式返回；默认 eager 保持兼容。C ABI 对应 `nuka_world_set_execution_mode`、`nuka_world_synchronize`、`nuka_world_get_execution_info`，查询前初始化结构大小和版本。

WSL 下构建 `nuka_pipeline_benchmark` 后，可按以下形式运行，输出目录必须不存在：

```bash
/root/nuka-vla/bin/python tools/perf/run_pipeline_sweep.py \
  --binary out/validation/execution_workspace_20260908/candidate_corrected_frozen/nuka_pipeline_benchmark \
  --source-id 41433e3e495aeab4073b9c25417c5645f75744d951e5afda088bfeb7739bb941 \
  --output out/validation/execution_workspace_20260908/replay_candidate \
  --executions eager graph --envs 1 16 256 --save-state
```

实际 baseline/candidate/capacity、memcheck、公共接口和 profiler 命令保存在对应 `_command.json`；原 invalid 输出和失败 profiler 日志均保留。

## 后续范围

继续固定环境的 MLS-MPM/SDF/光追/端到端基线，以及已具备基线的 active row/endpoint、读出和岛求解热点。MPM 缺失 SDF、异构映射、双浮基重叠、T06 完整错误传播、T08 密度、T09b refit、T10b/J 布局和其余物理/API/可微契约仍需完成。Editor 暂缓。
