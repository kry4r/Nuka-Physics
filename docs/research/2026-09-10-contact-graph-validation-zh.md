# 接触索引的 CUDA graph 验收

日期：2026-09-10。分类：CUDA 后端执行与索引实现。修复大环境的 graph capture 失败；共享物理契约、缓存身份、求解顺序和多后端接口不变。本项不计为 MLS-MPM 吞吐优化，也不代表剩余性能工作完成。

## 原因与实现

CUDA 13.3 的 CUB `DeviceSegmentedSort` 在 segment 数增大后回读分组计数并同步 stream，无法在 graph capture 中执行。旧实现的 E=1024/2048 在 ContactWarmStart 失败；只修缓存排序后又在 endpoint 排序失败。两处均改用公开的 `DeviceSegmentedRadixSort`，不按环境数量切换实现。

- Endpoint 的压缩输入已按 row/side 有序，只稳定排序 upper global-link bits，位宽由总 link 数推导；完整 64 位端点键继续保留。
- Cache 将完整 pair/feature/material 通过共用 `ContactHashWord` 混合、xor-fold 到 u32 桶，只排序桶键。桶内仍比较完整 Valid/env/pair/feature/material；哈希相同不代表身份相同，不丢弃碰撞项。
- 稳定输入顺序保证每个完整键仍选择最低 current source、最低 old retain source 和首个 age 合法的 warm source。后续 rank、淘汰及物理冲量规则不变。
- Benchmark 的 step 失败包含 op、native code 和 message，保留真实失败位置。

改造前 spec、上游 revision 和相关文件 SHA256 保存在 `contact_graph_hash_review.json`：Newton `90c56c3be73e35a34ccb37ab593a529e8a3d18dd`、MuJoCo `319cf22fdf6fabb803205fcafbb9398707736c8d`、Genesis `8325a478f0d52a1c28bb58c95dfe519fcf38fc2b`。此次修复没有采用不等价的上游物理算法。

## 五进程完整管线

RTX 5080、驱动 610.88、CUDA 13.3.73。固定 robot-cloth-fluid 输入、250 步预热、200 步正式计时及完整质量重放，逐步 GPU completion 计时。每组使用五对独立进程，交替前后版本顺序，共 60 次运行。比较对象为 `cuda_transfer_v3_frozen` 与 `contact_graph_hash_v1_frozen`。

| 环境数 | 执行 | 基线中位 ms/步 | 候选中位 ms/步 | 中位数变化 | 配对变化中位数 |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | eager | 5.619213 | 5.507881 | −1.98% | −2.43% |
| 1 | graph | 3.409889 | 3.575523 | +4.86% | +0.78% |
| 16 | eager | 6.154331 | 5.838305 | −5.14% | −2.44% |
| 16 | graph | 3.835410 | 3.835099 | −0.01% | −0.16% |
| 256 | eager | 7.679253 | 7.784286 | +1.37% | +1.37% |
| 256 | graph | 6.530555 | 6.512562 | −0.28% | +0.59% |

E=1 graph 的基线进程范围为 3.379792–3.553117 ms，候选为 3.398141–3.589605 ms；配对变化为 −2.55% 至 +6.04%。保留中位数回退，不据配对中位数将其抹去。结果支持接纳 capture 修复，不支持宣称所有模式显著加速；接触索引更早版本的 eager 回退归因仍未关闭。

Data arena 不变：E=1/16/256/2048 分别为 3,740,160 / 59,741,952 / 956,140,544 / 7,648,902,144 B。这是引擎分配预算，不是设备峰值显存。Nsight Systems trace 已保留用于归因；profiler 和 sanitizer 的时间不作为普通性能。

## 正确性与边界

五对采样的完整末帧、逐步 wrench、岛划分、工作量和既定 XPBD 质量均一致。E=2048 的 eager/graph 都完成完整管线，包含 reset、重放和副本隔离；状态文件为 712,704,000 B，两种模式 SHA256 同为 `1fe8ec912b655c608703ebcb5543c8c75198156f1553e5b0f2c155fd588c0fc7`，逐步 wrench 为 287,539,200 B，SHA256 同为 `7ca5dcd192a795f48b33d262feea0f24c75ef8a1ee8ce0b17dd8c6499e8df100`。

复用五项既有 `PairDrivenNarrowphase` 场景，覆盖接触 ID、摩擦锥、warm start、材料失效与 age/reset，全部通过。固定 E=16 graph 完整管线 memcheck 为 0 errors。状态一致用于检验索引语义与运算顺序；物理质量继续受原约束误差、接触反作用和材料检查约束，旧引擎不是独立物理真值。

原 MLS-MPM bunny-water 的完整 graph probe 通过：660 步轨迹 FNV 为 `6cf2c38c57c8f590`，末帧 SHA256 为 `80d6a2404b91bd38afd478b7cd3b8ccd92e0105a2392fb12bab2875a04d8f872`，与原传输优化基线相同。最小 J 为 0.935678005，最大总体积偏差为 1.6673171%。局部最大 J=79.4880753、无 SDF 地板状态 16 和 `coupling_complete=false` 继续保留，不能据本项关闭完整耦合问题。

最坏情况下，多个不同完整键落入同一哈希桶会触发平方次数的精确比较。本实现没有桶容量截断，不改变物理正确性；极端碰撞的性能成本尚未量化，后续接触密度工作继续覆盖。

## 失败候选与重放

直接三轮全宽 radix 的 `contact_graph_radix_v2_frozen` 虽修复 capture，但 E=1/16/256 eager 回退 6.61%/15.86%/7.38%，没有接受。完整 pair 分桶的 `contact_graph_buckets_frozen` 对应为 +7.41%/+3.03%/−2.58%，继续减小排序范围后才得到本报告版本。失败候选及日志保留。

证据根目录为 `out/validation/mpm_performance_20260909/`。`contact_graph_hash_v1_paired/summary.json` 保存五对结果；`contact_graph_hash_v1_2048_*` 保存大环境结果；`contact_graph_hash_v1_cache_scenarios.*`、`contact_graph_hash_v1_memcheck.*`、`contact_graph_hash_v1_profile.*` 和 `contact_graph_hash_v1_bunny_graph.*` 保存其余验证。命令和退出码均在对应 `_command.json`。

冻结目录为 `contact_graph_hash_v1_frozen`。源码 SHA256 为 `0218f87b9ec7ba54424bccbeda6f46be4f303b056c6b6b26f5b5153b0a601b33`，库为 `3e4d3f4164d64e40ad68106e0e8cf969578560e73c241c01f48a1edf132a781e`，benchmark 为 `70ae465ebf8caff8cc3626d2361fbc106ac406039d78c0d1dada4f6ffe2c8e74`。早期 `contact_graph_hash_frozen` 目录未冻结完整，不用于测量。
