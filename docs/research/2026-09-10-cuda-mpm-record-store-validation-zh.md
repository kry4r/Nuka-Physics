# CUDA MLS-MPM 记录写入优化验收

原连续传输记录由每个线程直接写入一个 116 B 结构，记录准备占既有落水 profile 的 13.26%。本批保留完整记录和消费者布局，使用公开 CUB `BlockStore` 的 warp transpose 合并写入。改动仅在 `src/phi/backend_cuda/ops/mpm.cu`，分类为 CUDA 后端实现；共享架构、物理算法和公共 API 未改变。

每个线程构造原记录，通过 `memcpy` 无损转换为整数 words，长度由类型大小推导。尾块的无效线程仍参与协作存储，实际写入量按有效记录数限制。P2G/G2P、权重坐标、浮点累加次序、材料、边界反作用和 40 子步均保持。

## 五进程完整流程结果

分母为 `contact_graph_hash_v1_frozen`，候选为 `coalesced_records_frozen`。eager/graph 各五对独立进程，交替 A/B、B/A；每个进程完成原 bunny-water 的 240 步静置和 420 步落水。输入为 180,000 粒子、110,400 节点、dx=0.011 m、dt=1/240 s。测量完整 `World` step 的 GPU 完成时间，上传、下载、质量扫描和渲染在计时外。

运行环境为 RTX 5080、驱动 610.88、CUDA 13.3。各进程绑定对应冻结库并清除继承的 `CUDA_SCALE_LAUNCH_QUEUES`；设备时钟、温度、功耗和命令分别留存。表中步时为五个进程平均步时的中位数。

| 执行 | 阶段 | 分母 ms/步 | 候选 ms/步 | 耗时变化 | 逐对变化范围 |
| --- | --- | ---: | ---: | ---: | ---: |
| eager | 静置 | 29.7410 | 26.7592 | −10.03% | −10.61% 至 −9.29% |
| eager | 落水 | 32.6693 | 29.9972 | −8.18% | −8.81% 至 −8.04% |
| graph | 静置 | 25.1457 | 22.3840 | −10.98% | −11.06% 至 −10.89% |
| graph | 落水 | 28.8552 | 26.0676 | −9.66% | −9.76% 至 −9.57% |

全部配对均改善，未以重复采样筛选结果。落水 p95/p99 的五进程中位数分别为 eager 36.536/37.655 → 34.059/34.864 ms，graph 32.689/33.553 → 29.874/30.765 ms。Data arena 保持 95,699,456 B，Model arena 保持 221,952 B；这里是引擎分配预算，不是设备峰值显存。

本批落水收益低于 10%，静置超过 10%；这是在此前约 32%–35% 传输收益之上的增量改进，不据此关闭剩余性能工作。

## 质量与安全

十对完整质量字典及末帧状态均相同。最小 J 为 0.935678005，全程总体积比最大偏差为 1.6673171%，保留原 5% 门；有限状态、正 J、无网格逃逸、splash、减速和反作用检查通过。轨迹 FNV 为 `6cf2c38c57c8f590`，末帧 SHA256 为 `80d6a2404b91bd38afd478b7cd3b8ccd92e0105a2392fb12bab2875a04d8f872`。身份检查只验证存储语义，不能代替物理判断。

复用已有 transfer、静水、jelly、颗粒介质、刚体/关节支承及混合 MPM/XPBD 场景，33 项通过，两个原先禁用的诊断未启用。已有解析检查包含常速度传输、网格总质量、重力与支承反作用、偏心冲量转矩；本批未新增单测。九项针对性 memcheck/synccheck 均为 0 errors，覆盖小于一个 block 的尾部、非有限坐标、混合介质及多环境关节反作用。

固定 robot-cloth-fluid 的 E=16 graph 完整 450 步通过，env_status=0，reset、重放、副本等价、逐步 wrench 和原长度质量门保持。该固定环境使用 PBF；MPM 覆盖来自原 bunny-water 和上述既有场景，不把两者混写。

## 归因与未采用方案

当前完整 Nsight Systems profile 的落水 kernel busy 为 25.342 ms/步。记录准备为 0.9566 ms/步，占 3.77%；此前对应 profile 为 3.9674 ms/步。新 kernel 为 40 registers/thread、14,864 B shared memory、无 local spill。P2G 仍为 17.6094 ms/步，占 69.49%，39 registers/thread、无 shared memory/local spill。profile 用于归因，正式收益以上述普通五进程测量为准。

硬件带宽、stall 与实际 occupancy 仍受 `ERR_NVGPUCTRPERM` 限制，未声称测得这些指标。记录准备减少后，后续重点继续是 P2G 的有序读取与计算，以及粒子/节点工作区容量分离。

| 中间候选 | graph 落水初验 ms/步 | Data arena B | 处理 |
| --- | ---: | ---: | --- |
| 128 B 对齐记录与偏移缓存 | 46.1020 | 97,859,328 | 明显回退，拒绝 |
| SoA 分量平面与偏移缓存 | 54.9659 | 97,863,424 | 明显回退，拒绝 |
| 原记录协作写入与独立偏移缓存 | 28.4234 | 97,859,584 | 收益较小，移除偏移缓存 |

这些候选的完整质量和状态相同，但均未进入正式接纳。对齐版本的 profile 显示准备降时、P2G 反而升至 37.170 ms；没有证据将具体原因断言为 cache 或 spill。源码、二进制、完整初验和该 profile 全部保留，没有覆盖失败记录。

## 物理边界与重放

局部最大 J=79.4880753 仍待解释；总量预算不证明稀疏自由表面局部密度正确。当前场景的无 SDF 地板由独立网格边界处理，`env_status_union=16`、`coupling_complete=false` 保留。single owner、有限质量、复合形状实际 owner、关节子步反馈、无 SDF collider 以及 MPM↔XPBD 直接交换均未由本批解决。

改造前 review 见细化 spec §5.1 和 `store_review.json`，采用 Newton `0c50f937`、MuJoCo `319cf22f`、Genesis `8325a478` 的相关实现对照；不把其 atomic scatter 或隐式求解直接换入有序显式 P2G。

证据目录为 `out/validation/mpm_layout_20260910/`：`coalesced_records_paired/summary.json` 包含全部二十个进程，`coalesced_records_scenarios.json`、`coalesced_records_pipeline.json`、两项 sanitizer 日志和 `coalesced_records_profile_attribution.json` 保存补充验收。冻结 source SHA256 为 `b4d575e1edb0dead9bd283c09338728cfb23a737e422dacc88b91a2e210521d4`，library 为 `a65c38f40a38fda8562a8b09f8289793c146694f13231ed9b32e0dca9aa2c7cc`，demo 为 `6ccc3123374b6993f46bd403df36276d97c64c6d996a9db13b3b774e4781f70c`。

```text
python .nuka-runs/measure_mpm_pairs.py --baseline out/validation/mpm_performance_20260909/contact_graph_hash_v1_frozen --candidate out/validation/mpm_layout_20260910/coalesced_records_frozen --output <新结果目录>
```
