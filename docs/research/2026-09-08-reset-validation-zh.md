# 多关节体 reset 验收与成本

日期：2026-09-08。对应 [模块 spec M04/T03](../plans/2026-09-07-physics-module-specs-newton-port-zh.md)；改造前的规格和三引擎对照见 [reset review](2026-09-08-reset-upstream-review-zh.md)。

## 结果与范围

支持的同构多树、纯自由刚体和粒子环境完成 reset 验收。全量和 masked reset 共用环境恢复核，根覆盖 E×K；非法 ID 在写入前失败，重复 ID 去重，未选环境逐位保留。恢复包括权威状态、自由体世界逆惯量、选中 FK/proxy，并清接触历史和读出。控制 target 和公开 view 地址保留。

Python 增加缩窄前的 ID 范围检查，修复 `reset_envs([2**32])` 回绕到环境 0。内部空集合 `nk::World::Reset({})` 表示全量；C ABI/Python 的 `reset_envs([])` 仍为 no-op，`reset()` 为全量。

接触图捕获仍因 LBVH 临时 workspace 返回 Unsupported，未验收为通过；无接触关节图的真实 capture → reset → replay 已通过。异构 DOF 属于 T04，未纳入本项完成范围。按用户最新指示暂停 Editor，后续先完善引擎功能。

## 构建身份与原始证据

结果目录：`out/perf/reset_contract_20260908/`。WSL Ubuntu 24.04，RTX 5080 / 驱动 610.88，CUDA 13.3.73，g++ 10.5.0，Release，CUDA architectures `all-major`。

| 对象 | SHA-256 |
| --- | --- |
| 冻结 baseline 场景二进制 | `3548ad4289308780ea8518d13edaa78aaa7c0382258dc1770143eef41b649de7` |
| 候选场景二进制 | `e7bf558d90c474b9d765ef91e80c4a2580f8e9596428317c65f0babbcd6c52b0` |
| 候选源码 manifest（含 Python） | `4889e4301c7ce68e7b68a58db524be4ac02ea797a4b4072de711fd2dc52a35d2` |
| 候选 libnuka.so.0 | `5cd12f73279d6f75eb060e9cabbd478c630c645edfee74a405bb050d1fccfe8f` |
| 候选 Python binding | `b68bd673199fec75a3cdd94f4145704f22bd8301548f72acba7260d2e758aee6` |

候选二进制、库、binding、源码差异和新增 reset 场景源码已冻结；baseline 未覆盖。`candidate_manifest.json`、`candidate_hardware.json`、`candidate_CMakeCache.txt` 记录具体身份。场景二进制静态链接实现，不能用替换 `LD_LIBRARY_PATH` 冒充旧版本。

| 验证 | 结果 | 文件 |
| --- | --- | --- |
| 主 pipeline + reset 行为 | 11 passed，1 skipped（接触 graph） | `acceptance.log/.json` |
| 主 pipeline CUDA memcheck | 1 passed，0 errors | `memcheck_pipeline.log/.json` |
| reset CUDA memcheck | 10 passed，0 errors；排除已知不支持接触图 | `memcheck_reset.log/.json` |
| Python 非法 ID 修复前 | 2 failed，2 passed，实际复现大整数回绕 | `python_id_before.log/.xml` |
| Python reset 修复后 | 10 passed | `python_reset.log/.xml` |
| C ABI coupled / multi-env | 2 + 3 passed | `coupled_cabi.*`、`multi_env_cabi.*` |
| 接触/MPM 相关回归 | 30 passed，1 个旧失败 | `related_regressions.*` |

主环境复用 `RobotClothFluidCoResident.Go2StanceCouplesClothAndFluidOnOnePipeline`：Go2 + 169 个布料粒子 + 100 个 PBF 粒子，240 Hz，250 步 settle + 200 步 hold。验证 cook、创建、控制、每步状态、接触/耦合、读出、full/masked reset 与各 8 步重放；布料/流体接触行数为 3819/600，介质对机器人的反作用仍存在。未修改渲染输入或流程，本次没有追加图像验收。

解析补充覆盖 E=3/K=2、ID 集合、缓存污染、K=0 MPM 六类状态和首步新 World 对照、偏转非等方自由体惯量及多形状代理。它们补足主环境无法测量的环境隔离与矩阵状态，不以测试数量替代 pipeline 验收。

## reset 成本

每种 E/K、full/masked 均为 10 次预热 + 200 次 reset；计时包含 API 调用并在末尾等待 GPU 完成。5 个独立进程，下表是每进程均值的中位数和范围，单位 µs/reset；不是 physics step 吞吐，也不是批内 p95。

| E/K | 模式 | baseline 中位数 [min,max] | 候选中位数 [min,max] |
| --- | --- | --- | --- |
| 1/1 | full | 170.93 [165.96,489.58] | 41.21 [36.26,43.22] |
| 1/1 | masked | 77.82 [54.09,113.99] | 113.65 [91.96,197.24] |
| 1/2 | full | 303.11 [183.90,483.42] | 42.12 [41.02,70.07] |
| 1/2 | masked | 85.03 [52.55,140.91] | 105.86 [97.66,152.00] |
| 3/1 | full | 191.42 [174.90,406.78] | 43.77 [41.23,79.57] |
| 3/1 | masked | 70.44 [61.27,78.08] | 110.86 [83.14,154.97] |
| 3/2 | full | 203.93 [174.05,485.97] | 66.64 [57.54,109.24] |
| 3/2 | masked | 56.68 [51.46,145.31] | 132.99 [114.71,187.93] |
| 16/1 | full | 414.63 [166.27,487.90] | 66.14 [41.41,72.68] |
| 16/1 | masked | 75.06 [51.85,155.32] | 115.27 [112.88,122.17] |
| 16/2 | full | 301.46 [169.43,409.01] | 60.95 [59.78,64.52] |
| 16/2 | masked | 76.78 [62.61,103.45] | 151.41 [103.74,183.56] |

全量恢复减少了分散的复制/清零 dispatch，成本下降；masked 恢复增加完整历史/读出清理及选中 FK/proxy 同步，成本上升 24.5%–134.7%。未采集各新增工作的独立耗时，不能把全部差额归因于某一核。baseline 已知状态恢复不完整，且计时噪声较大，因此这只是正确性修复的成本比较，不宣称等质量性能优化。

三个 arena 的 bytes 在全部六组 E/K 上完全相同。snapshot 原已按 E×K 分配；此次修复复制长度，没有增加分配，root 复制增加 `E*(K-1)*28` bytes。原始五次结果为 `baseline_perf_1..5.*`、`candidate_perf_1..5.*`，完整样本与内存比较见 `reset_cost_comparison.json`。性能用例未先扰动 roots，其 `roots_valid=true` 只作 sanity；正确性由上述行为回归证明。

## 重放命令

在 WSL 仓库根目录执行：

```bash
cmake --build build-linux --target nuka_scenario_test nuka_multi_env_world_test nuka_coupled_world_cabi_test --parallel 6
build-linux/tests/nuka_scenario_test --gtest_filter='RobotClothFluidCoResident.*:MultiArticulationReset.*-MultiArticulationReset.ReportsResetCostAndMemory'
LD_LIBRARY_PATH=build-linux/src:/usr/local/cuda/lib64 PYTHONPATH=python /root/nuka-vla/bin/python -m pytest python/tests/test_nuka_dlpack.py -q -k reset
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 build-linux/tests/nuka_scenario_test --gtest_filter='RobotClothFluidCoResident.*'
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 build-linux/tests/nuka_scenario_test --gtest_filter='MultiArticulationReset.*-MultiArticulationReset.ReportsResetCostAndMemory:MultiArticulationReset.CapturedContactStepsRemainValidAfterReset'
out/perf/reset_contract_20260908/candidate_scenario_test --gtest_filter='MultiArticulationReset.ReportsResetCostAndMemory'
```

## 后续项

- M04/M06：`MultiDogContact.TwoDogsPushApartAndExchangeMomentum` 中两个浮基 Go2 的重叠 trunk 未分离。冻结旧二进制复现同样失败，见 `baseline_contact_followup.*`；原因尚未定位，不能称 fixture 错误，也不记为本次 reset 引入。
- M03/M05/T09：补齐 LBVH workspace 后完成接触 graph → reset → graph。
- M03/M11：masked reset 后续需分项 profile；完整 benchmark 和外部引擎实测仍待执行。
- Editor 暂缓：Go2 `.nks` 含非零 margin 和 `condim=6`，加载在接触参数校验阶段被拒绝。未完成旧 Editor 对照，未改资产或放松校验；按用户要求不再展开 Editor 排查。
