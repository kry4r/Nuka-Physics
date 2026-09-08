# 自由旋转、风阻、邻域与执行契约验收

日期：2026-09-08。对应 T02c/T04a/T05a/T06a 和验证中追加的 T17a 容量池基础部分。[上游 review](2026-09-08-dynamics-pipeline-upstream-review-zh.md) 记录最新 Newton/MuJoCo/Genesis revision、时间层与算法差异。沿用一个通用生产 pipeline，Editor 暂缓。

## 已验证行为

自由体在所有物理冲量之后，以惯性主轴中点动量方程配合 Cayley 姿态漂移；更新姿态后刷新世界逆惯量，并由物理世界角动量重建角速度。外力与接触的完整分裂仍为一阶，无外力漂移验证二阶收敛。pseudo 改变几何时保持真实世界角动量，不宣称同时保能量。非法输入或 Newton 不收敛时保留该 body 的漂移前 pose/omega，输出残差、迭代数和错误位；先前 kick 和其他 body 工作不回滚。

面风阻从同一速度时间层计算独立 impulse，再按 tri ID 有序 CSR 汇总至粒子。共享顶点的总速度增量与动能增量受 incidence 上界约束；固定顶点不写回。沿用静止空气和法向/切向二次阻力，没有新增 wind API。

创建检查实际 articulation spans、parent、joint type、DOF、共享映射上限和必要索引范围。当前异构拓扑明确拒绝。Pipeline Build 保留缺失 required ops 并清空不可运行调用表；Step 遇首个 host/launch 错误停止。新增 readout demand 先建立候选 pipeline，回填成功后才替换和失效 graph，失败保留原 pipeline。完整 buffer/capture/completion 错误传播仍属后续 T06。

公共 Field 末尾追加 `ENV_STATUS=39`、`BODY_GYRO_RESIDUAL=40`、`BODY_GYRO_ITERATIONS=41`、`BODY_GYRO_STATUS=42`、`PARTICLE_NEIGHBOR_ATTEMPTED=43`、`PARTICLE_NEIGHBOR_COUNT=44`，旧编号不变。gyro 状态为 0 正常、1 不收敛、2 非法输入；环境位 `1<<5` 表示 gyro 失败。邻域 attempted/count 是最后一次网格构建的真实/保留数，可相减获得 dropped。设备诊断需同步后查询；普通 step 不新增 D2H，host launch 成功不等于设备数值有效。主验收逐步检查环境状态，避免最终一帧掩盖中途错误。

## 邻域失败与修复

初次整批运行报告 21 passed、1 个已知 graph skip，但新增读出发现主环境 `env_status=2`，不能算有效验收。将固定每粒子 32 条改为真实计数→64 位 scan→每环境容量池；六类消费者共用实际 CSR offset，列表按粒子 ID 排序。默认总预算仍为 `32*P`，`ModelCapacities::neighbor_pool_capacity_per_env` 可显式设置，溢出标无效且不借用其他环境容量。密集 40 粒子 oracle 验证超过 32 邻居、完整集合/双向性和小容量隔离。

容量池首轮仍失败：attempted=30996、retained=8608、max=168，流体失去有效邻域并且耦合断言失败。根因是 `SoftFluidPredictKernel` 未写 soft 分支的共享 `pbf_predicted_pos`，169 个布料粒子被查询为原点的假密集团。修复动态和固定 soft 的预测位置后，保持原容量，最终真实/保留数均为 2882（介质在位）与 2996（分离控制），max=20，每一步 `env_status=0`。这同时恢复了完整邻域下的布料/流体耦合，不通过扩大容量或放宽断言验收。

完整 T17 的投影后邻域刷新/skin、边界密度与公共容量配置仍未完成。当前查询列表保证构建时间层的集合，不宣称覆盖任意后续投影位移。

## 结果和物理质量

结果目录：`out/validation/dynamics_pipeline_20260908/`。

| 验证 | 结果 | 原始文件 |
| --- | --- | --- |
| 主 pipeline、旋转/风阻/邻域、执行契约、reset、MPM+XPBD | 25 passed，1 skipped（接触 graph） | `pipeline_positions.log/.json` |
| CUDA memcheck，含完整主场景与新核 | 12 passed，0 errors | `memcheck_positions.log/.json` |
| Python 公共 pipeline，两个环境、两种 dt、DLPack、reset、渲染 | 通过，环境状态均为零，邻域无截断 | `public_api.json`、`public_positions.log` |
| Python reset | 10 passed | `python_reset_positions.log`、`python_reset.xml` |
| C ABI coupled / multi-env | 2 + 3 passed | `coupled_cabi_positions.*`、`multi_env_cabi_positions.*` |
| 公共创建错误 | Go2+H1、fixed+floating 返回 NOT_SUPPORTED(7)，NaN dt 返回 INVALID_ARG(1) | `public_creation_contracts.json`、`public_contracts_positions.log` |
| schema/codegen | 274 个内部字段，四个生成头与 schema 一致 | `schema.json` |

主环境仍是 Go2、169 个布料粒子、100 个 PBF 粒子和偏心非等方自由体；240 Hz，250 步 settle＋200 步 hold，并验证 reset 后 8 步逐位重放。风阻启用，自由体初始世界 omega 为 `{5,-3,2}`。末步 gyro 残差 `9.39902e-8`、1 次迭代，世界角动量绝对误差 `2.13852e-5`。布料/流体接触行累计 525/600，布料相对基准下压 12.1 mm、流体表面升高 9.1 mm，介质引起关节速度 L1 差 0.575727。粒子间行计数仍为零，不能用本场景证明薄布自碰撞或连续碰撞。

指定库的公共自由旋转反例为无地面非等方 box、world omega `{11,6,-3}`、240 Hz/480 步：

| 指标 | 旧冻结库 `0953516` | 新库 |
| --- | --- | --- |
| 世界角动量相对误差 | 0.0813432263 | 5.1581049e-6 |
| 转动能相对误差 | 1.1738635e-8 | 1.3893688e-6 |
| 最终 omega | `{11,6,-3}`，错误地保持不变 | `{12.065565,3.913863,-1.981771}` |

旧版能量误差小不能说明轨迹正确；其世界角动量和自由进动错误。数据、库路径和哈希在 `gyro_before.json`、`gyro_after.json`。额外 double RK4 oracle 覆盖旋转惯性帧和整体旋转：2 秒用 240/480 步，组合指标 `||omega-omega_ref|| + ||axis-axis_ref||` 从约 0.1292 降至 0.0337，缩小约 3.8 倍。该指标不是单独的姿态角误差。世界角动量相对误差最大 `4.35e-5`，能量相对误差最大 `8.53e-5`，全程残差不超过 `1e-6`。接触 impulse oracle 在 pos_iters=0/4 下验证最终姿态的真实世界角动量。

共享面风阻在三环境、不同质量和固定顶点下，coefficient=0/1/1e6 的动能变化分别为 0/-6.05668/-20.1079；总 dv≤0.5，五次 reset 重放逐位相同。MPM+XPBD 验证最终粒子偏移上的 aero adjacency。

已目视检查 `render_roundtrip.png` 及其三帧：球体推进后下落，reset 恢复创建图像，逐像素一致。这里使用 Go2 测试资产的简化几何；原 π0.5/G1 demo 和冻结库保留。

## 成本、身份与限制

主环境 model packed buffer 为 183296 bytes（含段对齐），data 有效字段字节之和为 3265508，后者不含段对齐和外部分配。新增 gyro 诊断为每 body 12 bytes；aero adjacency 为每粒子 8 bytes＋每面 12 bytes，独立 impulse 为每面 12 bytes；无 aero 时不分配新 adjacency。邻域池默认总容量字节不变，新增 attempted/64 位 prefix 共每粒子 12 bytes，主环境相对初次 aero/gyro 版本增加 3484 data bytes（含 scan workspace 差异）。

风阻由一核变两核，自由体漂移增加迭代和环境错误位清理核。此次未做带宽、occupancy、同质量五进程性能对照，不能宣称更快；T05a 带宽分析及 T01b/c 仍待性能批次。记录的整组原生 pipeline 进程耗时 16.315 s、公共流程 6.231 s、memcheck 428.664 s 均包含验证与读出开销，不能当作 physics-only GPU step 时间。

WSL Ubuntu 24.04，RTX 5080 / 驱动 610.88，CUDA 13.3.73。`frozen/` 保留二进制、构建配置、相对 `0953516` 的源码 patch、新增/修改文件归档及逐文件 manifest。

| 对象 | SHA-256 |
| --- | --- |
| 源码 manifest | `398fafe2e6d398d6417d4b1f8e60bbc23fff97cbb442be3c121b73064692987a` |
| 场景二进制 | `4e1323d2059da86ae67b3c5fd9d55c479c98ec327515d003766ea26d9cbd07cd` |
| libnuka.so.0 | `7620a7318bc36fc823b04b0a9174efea81eace94939aaf2abf122b556413d76e` |
| Python binding | `ffb98a698c742f0d8162b70dd82aeeb79b70e28315f4f874e9d5a36e20b2efd4` |

`pipeline.*`、`pipeline_neighbors.*` 和失败构建日志保留。接触 graph/LBVH workspace、双浮基重叠、异构映射、完整异步错误与容量、薄壁/CCD/高级耦合继续未完成；goal 保持 active。本批没有实跑外部引擎，也没有外部性能领先结论。

## 重放

在 WSL 仓库根目录执行；每个命令的完整参数、退出码和耗时均保存在对应 `*_command.json`。

```bash
cmake --build build-linux --target nuka_scenario_test nuka_coupled_world_cabi_test nuka_multi_env_world_test nuka -j 8
build-linux/tests/nuka_scenario_test --gtest_filter='RobotClothFluidCoResident.*:FreeRigidDynamics.*:ParticleAerodynamics.*:ParticleNeighborhood.*:PipelineContracts.*:MultiArticulationReset.*:NkMpmXpbdCoResidence.*'
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 build-linux/tests/nuka_scenario_test --gtest_filter='RobotClothFluidCoResident.*:FreeRigidDynamics.*:ParticleAerodynamics.*:ParticleNeighborhood.*:MultiArticulationReset.FreeBodyInertiaAndProxiesRestoreOnlySelectedEnvironments:MultiArticulationReset.ParticleStepAfterResetMatchesANewWorld:NkMpmXpbdCoResidence.CoStepNoCrossCorruptionAndD1'
LD_LIBRARY_PATH=build-linux/src:/usr/local/cuda/lib64 PYTHONPATH=python /root/nuka-vla/bin/python tools/validation/rigid_inputs_pipeline.py --output out/validation/dynamics_pipeline_replay
LD_LIBRARY_PATH=build-linux/src:/usr/local/cuda/lib64 PYTHONPATH=python /root/nuka-vla/bin/python tools/validation/rigid_inputs_pipeline.py --output out/validation/dynamics_pipeline_replay --contracts-only
LD_LIBRARY_PATH=build-linux/src:/usr/local/cuda/lib64 PYTHONPATH=python /root/nuka-vla/bin/python -m pytest python/tests/test_nuka_dlpack.py -k reset -q
```

Python 使用与当前库匹配的 binding，构建命令见 `binding_positions_command.json`。源码归档含本地批量 runner、独立库旋转 probe 和 schema 检查入口；使用旧库时连同匹配扩展一起加载，不能仅替换动态库冒充旧的静态链接场景二进制。
