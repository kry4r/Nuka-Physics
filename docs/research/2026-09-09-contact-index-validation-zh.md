# 有效接触索引与确定性读出验收

日期：2026-09-09。对应 T01/T06/T10b/T24 的接触索引子集；完整引擎 goal 保持 active。改造前已完成 [规格和最新上游 review](2026-09-09-contact-index-upstream-review-zh.md)。

功能与 graph 子集已通过验证；eager 性能尚未通过，保留原始回退结果并继续处理。

## 实现及质量边界

原 link wrench 对每个 link 扫描该环境全部 row capacity。通用索引现在以 stable row ID 为身份，scan 生成有效 row 和 0/1/2 个 articulation 端点，按 `(global_link, original_row_id, A/B)` 排序形成 link 的有效区间。每个 link 按原 row 顺序累加，保留 A 在 B 前的 FP32 表达式。力学求解、接触行身份、迭代预算和公开 tensor 地址保持原契约。

新增字段在 `fields.yaml` 尾部追加，并同步生成视图、字段预算、DataView 绑定、World 创建、pipeline 参数和 reset。索引仅在 readout 有消费者时构建；创建时预留稳定地址。prefix、输入键、输出键互不覆盖。两端数量、索引上限、行/槽布局和 workspace 不足均显式检查。masked reset 仅清所选环境的有效 count、link spans 和 wrench。

当前索引先服务通用 readout；求解与 J 的复用仍需将 producer 移到共同依赖边界。其余 solver/J 的全容量存储和调度尚未改变，不能据此将 T10b/T11/T13 全部标为完成。

## 冻结身份与环境

证据根目录为 `out/validation/contact_index_20260909`，所有冻结目录与原始失败日志保留。

| 身份 | 观测基线 | 索引候选 |
| --- | --- | --- |
| 目录 | `baseline_frozen` | `candidate_v1_frozen` |
| 基础 Git HEAD | `3ad8388af9e2ffee42179d396925b2af4882eb9e` | 同左，加归档 source delta |
| 源码 SHA256 | `17f81f8511a0a33ee1288232b97e2417d3c953f7bbdb673a28c3142ed7fa4378` | `5224ab80d3041638ef904917665d6816f9dff67e296fdbcfb84b41f1a6122339` |
| benchmark SHA256 | `d7ed038b4a321de0fa484e96fb02b95007a59bae244cc98cf8f1e0341641d45f` | `7c6969dd47823f564162be7294382f05fc4388e88427e79cbc66043436c3454d` |
| `libnuka.so.0` SHA256 | `5d96f3a4f689867b6e063f3981f24e20b8cfb727d5afd4aad2be71bb886d9125` | `d88839ba5f05a581c996a44aec0839cefd0551e0def34d73812adfc7eb288dfe` |

硬件为 RTX 5080、driver 610.88、CUDA 13.3，CUB 版本宏为 `300304`。使用 WSL Ubuntu-24.04、`build-linux` Release 的原编译配置。每个进程保存 GPU 时钟/温度、命令、源码和二进制 hash；正式测量期间无并行构建或另一 GPU 验证任务。

`candidate_final_frozen` 保存扩充公共 wrench 验证后的最终源码，SHA256 为 `b2914556bc036c3f8ccd493c8ade598690e9370e4118e273b71785cd21e10493`。相较性能候选只增加验证脚本检查；benchmark、共享库和全部 C++ 验证程序的 hash 与上表候选相同。

基线在改引擎前只扩充 benchmark 的计时外观测：schema 3 保存 450 步全部 LinkContactWrench 原始 float 字节、有限性与 FNV，完整末帧在物理/缓存后追加 wrench。`--save-wrench` 独立于原 `--save-state`，旧 schema/二进制的 sweep 仍可使用。

## 完整 graph 五进程

固定 robot-cloth-fluid 环境，dt=1/240、seed=20260908、warmup=250、计时步数=200；同一控制和完整生产 pipeline。下表为 GPU 完成事件的 ms/step，已包含 scan、压缩、排序、span 构建和 gather 成本。

| 环境数 | 旧版中位数［五进程范围］ | 新版中位数［五进程范围］ | 延迟下降 |
| --- | --- | --- | --- |
| 1 | 3.080793［3.076005, 3.082834］ | 2.550691［2.549209, 2.554378］ | 17.21% |
| 16 | 3.380915［3.378037, 3.385136］ | 2.837360［2.835533, 2.841999］ | 16.08% |
| 256 | 6.516359［6.509417, 6.529342］ | 5.367076［5.365828, 5.378783］ | 17.64% |

三档范围均完全分离。结果来自 `baseline_five/summary.json` 和 `candidate_v1_five/summary.json`。`candidate_v1_five_comparison.json` 同时保存两种执行模式与旧 graph 的逐字节质量比较；其中 eager 对 graph 的耗时字段仅描述执行模式差异，不能用作 eager 优化收益。

### eager 回退：尚未通过性能验收

首次分组五进程的旧 eager 中位数为 6.917/7.027/9.810 ms，新版为 7.208/8.650/10.922 ms，分别回退 4.20%/23.09%/11.34%。范围重叠，但不能因此忽略超过 5% 的回退。

已额外执行两组交错对照：每个环境数各五对独立进程，顺序按 AB/BA 交替；第二组仅将测量进程及其子进程固定到 WSL CPU 2，其他系统设置不变。

| 环境数 | 普通交错旧→新中位数 ms | 普通交错成对变化中位数 | CPU 2 成对变化中位数 |
| --- | --- | --- | --- |
| 1 | 7.172 → 10.825 | +54.87% | +37.66% |
| 16 | 7.252 → 9.259 | -0.67% | -0.39% |
| 256 | 9.910 → 10.578 | +7.71% | +9.06% |

成对变化为每对 `new/old - 1` 的中位数，正值表示更慢，与两组中位数的比值不是同一统计量。数据保留在 `eager_interleaved` 和 `eager_cpu2_interleaved`。固定 CPU 没有消除问题，不能归因于线程迁移后就标为通过。

普通 trace 的 eager GPU kernel busy 从 6.205 降至 5.082 ms，host 提交从 10.168 降至 9.833 ms。完整 API trace 的对应数值为 6.208→5.153 ms、11.006→9.953 ms；额外每步 5 次 launch 和 26 次 `cudaGetLastError`，未新增每步分配/同步。主要 host 成本仍是约 622/627 次 launch；这些 profiler 结果用于归因，不能替换未经 profiler 的五进程分母，也未证明原始回退的根因。eager 的完整性能收口继续留在 T01c/T07，不能把本批称为所有执行模式均加速。

### 容量扩展

| 环境数 | contact capacity scale | slots/env | rows/env | 旧→新 graph ms | 延迟下降 |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 2,216 | 26,592 | 6.504965 → 2.638657 | 59.44% |
| 256 | 2 | 2,216 | 26,592 | 18.730659 → 10.568950 | 43.57% |
| 1 | 4 | 4,432 | 53,184 | 10.450800 → 2.706945 | 74.10% |
| 256 | 4 | 4,432 | 53,184 | 35.314424 → 19.043676 | 46.07% |

每格五进程，范围均完全分离。scale 指 contact slots；benchmark 放大后预留完整 row 布局，因此 rows 并非仅增加 2/4 倍。相同容量的新旧完整状态和 450 步 wrench 文件逐字节相同；不同容量之间的权威物理状态和每步 wrench 也逐字节相同，见 `candidate_v1_capacity2_comparison.json`、`candidate_v1_capacity4_comparison.json` 与 `capacity_identity.json`。剩余全容量工作仍令 E=256 从默认 5.367 增至 19.044 ms，未宣称容量成本已全部消除。

## 内存成本

| 环境数 | 旧 Data arena bytes | 新 Data arena bytes | 增量 bytes |
| --- | --- | --- | --- |
| 1 | 3,519,488 | 3,740,160 | 220,672 |
| 16 | 56,246,016 | 59,741,952 | 3,495,936 |
| 256 | 900,229,376 | 956,140,544 | 55,911,168 |

默认 E=256 的增量为 6.21%。其中 active row IDs 3,725,312 B、输出端点键 14,901,248 B、workspace 37,255,935 B，另有 count、link spans 与对齐。当前 CUB 所需 temp 包含最大排序域，因此实际约为 60 B/row；不能只按不含 CUB temp 的 44 B/row 报告。

这是字段和 arena 的精确预算，尚不等同于包含 graph/driver 的实测峰值显存。没有 readout 消费者时仍预留可按需启用的地址，但不会运行索引构建。

E=256、scale=2 的 Data arena 为 3,987,371,264→4,395,855,872 B；scale=4 为 7,945,769,472→8,762,707,200 B，额外约 10.24%/10.28%。索引与 CUB 的最大容量预留仍需后续生命周期和有效域优化。

## 正确性验收

- 整批构建通过；`pipeline_index_v1.json` 为 34 passed、0 skip，覆盖固定耦合 pipeline、graph/control/readout/reset、动力学、风阻、邻域、错误和容量、MPM/XPBD 共存及接触身份。
- 只新增一个主环境无法充分覆盖的解析读出不变量：同 link 双端点力抵消但力矩叠加、两个 link 的反作用、inactive stale metadata、跨环境 link 隔离、空环境、工作区/索引范围失败和 reset。另在现有 graph/reset 场景补读出字段和稳定地址检查。
- 新版 eager/graph E=1/16/256 各五个进程均质量有效、跨进程末帧与逐步 wrench D1 通过；它们与旧版 graph 的完整末帧及全部 450 步 wrench 文件直接逐字节相同。E=256 的完整末帧为 89,088,000 B、wrench 轨迹为 35,942,400 B。
- 标准五进程矩阵共 100 个独立进程、两组交错诊断另 60 个、初验 3 个，全部质量有效。性能结论按上文区分 graph 收益与未解决的 eager 回退。
- LBVH 3、公共耦合 8、多环境 5、相机 4 和 Python reset 10 项通过。公共 pipeline 现在实际启用 wrench consumer，验证 DLPack 地址、masked/full reset 和重放。新版 eager/graph 与冻结旧库的报告一致，各 8 张 PNG 逐字节相同；已目视创建/推进/重置画面。
- 完整 E=16 graph benchmark 的 CUDA memcheck 为 0 errors；双端点/空环境/非法参数、graph 控制/reset 和 masked reset 三个场景的 memcheck 也为 0 errors。命令与结果见 `memcheck_pipeline_command.json`、`memcheck_readout_command.json`。

## 新热点与执行成本

新 graph profile 只统计 warmup 后 200 个计时 graph launch 的节点，kernel busy 为 5.016661 ms/step。原 link wrench 的 1.262 ms 扫描由下面六个 kernel 替代，总计 0.114715 ms，已计入完整 pipeline 分母。

| kernel | ms/step |
| --- | --- |
| scan init + scan | 0.059108 |
| row/endpoint compact | 0.035969 |
| endpoint segmented sort | 0.005531 |
| link spans | 0.007419 |
| link wrench gather | 0.006688 |

此表不含未改动的 contact geometry/force/link 输出 kernel。新最大热点为岛求解 1.611507 ms（32.12%）；XPBD bend/distance 合计 0.902234 ms（17.98%）；接触行生成 0.475927 ms（9.49%），body-particle narrowphase 0.333677 ms（6.65%）。岛求解静态资源仍为 154 registers/thread。下一批先审核实际岛 launch 上界、有效岛/J 数据流和 host launch 成本，继续保持原浮点行顺序与通用路径。

## 失败记录与剩余工作

`build_index_v1.log` 记录 CUDA 13.3 移除旧 CUB iterator 头文件的编译失败；改为 Thrust counting/transform iterator。`build_index_v2.log` 记录 DataView 的 byte storage 到 NkRow 的不合法 static_cast；修正为与现有行读取一致的 reinterpret_cast。`build_index_v3.log` 为整批成功构建，两个失败均发生在性能采集前。

Compute 硬件计数器沿用已记录的 `ERR_NVGPUCTRPERM` 限制，尚无实测带宽、stall 或实际 occupancy。后续以 Systems 和 GPU 完成时间继续归因，不填估算值冒充计数器结果。

本环境的流体为 PBF；MLS-MPM、SDF、光追和外部引擎端到端对照仍需各自完整基线。索引的显存、剩余全容量扫描、J/质量算子与岛调度继续优化；达到当前收益门槛不关闭完整 goal。

## 重放

从项目根目录使用 `/root/nuka-vla/bin/python tools/perf/run_pipeline_sweep.py`，传入冻结目录的 `nuka_pipeline_benchmark`、对应 source-id、全新 output 目录以及 `--executions graph eager --envs 1 16 256 --save-state --save-wrench`。容量测量另传 `--executions graph --envs 1 256 --capacity-scale 2` 或 `4`；默认配置仍为各五进程、250 warmup、200 steps。

所有具体参数、完整绝对路径和 GPU 条件保存在各组的 `*_command.json`，失败构建日志亦保留。12 份本地诊断脚本的最终快照及 SHA256 存于 `reproduction/manifest.json`；脚本按原 `.nuka-runs` 布局定位项目根。公共验证入口为 `tools/validation/rigid_inputs_pipeline.py`，已经包含实际 wrench consumer、DLPack、控制/reset、渲染和重放。
