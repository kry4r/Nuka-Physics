# 有效接触索引与读出改造前 review

日期：2026-09-09。基线为 `3ad8388af9e2ffee42179d396925b2af4882eb9e`，对应 `candidate_corrected_frozen` 的已验收实现。上一 goal turn 有实际进展：提交执行/workspace 改造、修正错误基线并完成五进程、容量、内存和渲染证据；本批接续完整 goal。

已核对两份总 spec 的 T01/T06/T10b/T24，以及 [性能方向 spec 第 4.1 节](../plans/2026-09-09-performance-directions-detailed-spec-zh.md)。本批首先改善共享有效 row/endpoint 索引与确定性力读出；J/质量算子、岛调度和未覆盖路径继续按依赖推进，不缩小最终目标。

## 最新上游

本批通过三个官方仓库 `git ls-remote ... HEAD` 再次核对。Newton `3f54b4a16c8b7125e6559f2668cce2a6a7d929c5`、Genesis `0ce793b42c945b6848aad624501b2ca756d3e244` 均未改变；复核已归档精确版本的 active rows/cols 与确定接触排序实现。MuJoCo 更新至 `297f5fc6592e89fc41960251e88b6cf94fd7573e`，已 fetch 精确提交并与上次版本比较。

| 实现 | 采用决定与边界 |
| --- | --- |
| Newton Kamino `linalg/sparse_operator.py` | 有效 rows/cols 使用设备 count，与最大分配容量分离；Nuka 保留 stable row identity，单独构建可共享的有效索引。 |
| MuJoCo `engine_core_constraint.c` 的 `nefc`、rowadr/rownnz/colind | 采用有效范围与 CSR 契约；当前 CPU 求解次序和稀疏乘法不直接替换 CUDA。 |
| MuJoCo 最新 `mj_makeImpedance` | 未解析弹簧调整改为保留 authored damping ratio。该变化涉及物理模型，记录到 T12 的后续对照；本批不改变离散接触方程。 |
| MuJoCo Unity `MjScene.cs` | 新增实例 ID 排序，说明输入枚举顺序也影响再现性；Nuka 本批保持原 stable row ID，不按线程 append 次序定义力累加次序。 |
| Genesis `constraint/island.py::_sort_island_contacts` | 只排序有效 slice、使用确定全序；其 insertion sort 与几何位置键不直接照搬。Nuka 端点全序为 global link、原 row ID、A/B。 |
| CCCL `DeviceSegmentedSort::StableSortKeys` | 已核对可使用双缓冲 key 和设备 begin/end。排序只消费有效端点段，预分配 temp storage；额外空终段固定声明的最大 item extent。 |

## 基线与实施契约

修正基线的完整 graph E=1/16/256 为 3.073/3.367/6.499 ms，E=256 容量×2/4 为 18.721/35.315 ms，容量变化不改变权威物理状态。最终 Systems 定位 link wrench 为 1.262 ms/step（20.28%），源码是每 link 扫整个 row capacity，因此优先减少无效访问。

1. 在改变引擎前，扩展同一 benchmark 的计时外质量重放，保存每步完整 link wrench、有限性和逐位轨迹摘要；末帧比较也包含 wrench。冻结仅含观测改动的旧求解实现。原质量检查未覆盖每步输出，不能用物理末帧相同替代读出正确性。
2. schema 增加有效 row ID/count、link endpoint begin/end 和显式 workspace。所有数量从 row/link/env 容量推导并 checked；每行最多两个端点来自通用 row schema，不使用机器人或场景身份。
3. 稳定 scan 得到有效 row 和端点 rank，按环境私有段压缩；排序键含原 row/side ID。prefix、输入、输出分区分别存放，禁止并发读写重叠；只有完成最后消费后才可复用。
4. 每 link 只读取其端点区间，沿原 row 升序、同 row 的 A→B 顺序进行相同 FP32 运算。无贡献的 link 输出零；关节/静态/刚体/粒子端点按同一元数据契约处理。
5. 索引由 readout 的通用 producer 构建，只在存在消费者时执行。当前先服务读出，稳定 row 索引保留给之后的 J/质量算子和求解调度复用；接入新消费者时将 producer 移到共同依赖边界，不新增并行求解路径。
6. 合并 schema/预算/pipeline/内核/观测后一次构建。以完整生产 pipeline、每步 wrench 字节对照、容量×1/2/4、graph/demand/full/masked reset 和针对性 memcheck 验证；只补 pipeline 无法覆盖的双端点同 link 不变量。

净收益计入 scan、压缩、排序和 CSR 构建。记录额外显存、少环境回退与剩余全容量成本；达到初始门槛后继续热点。Compute 计数器仍未获得权限，带宽/stall/实际 occupancy 不填估计值。MLS-MPM、SDF、光追的算法优化仍需自己的有效基线。

## 本批冻结与接口检查

仅增加计时外 wrench 观测的基线已冻结在 `out/validation/contact_index_20260909/baseline_frozen`，源码 SHA256 为 `17f81f8511a0a33ee1288232b97e2417d3c953f7bbdb673a28c3142ed7fa4378`。graph 的 15 个独立进程全部有效，E=1/16/256 的五进程 GPU 中位数为 3.08079346/3.38091461/6.51635864 ms/step；每组完整末帧与 450 步 wrench 轨迹均满足跨进程 D1，结果在 `baseline_five/summary.json`。

CUDA 13.3 随附的 CCCL 已移除旧 `cub/iterator/*_input_iterator.cuh`，实现使用同一 CCCL 的 Thrust counting/transform iterator 驱动 CUB scan。失败构建日志保留。新 DataView 字段需要在手写 `data.cpp` switch 同步绑定；直接 op 参数同时检查两端数量、row/slot 布局和工作区容量，防止仅依赖 World 创建检查。
