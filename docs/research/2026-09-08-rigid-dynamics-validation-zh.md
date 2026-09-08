# 外力与三轴重力验收

日期：2026-09-08。对应模块 spec M04/T02b，规格及最新 Newton/MuJoCo/Genesis 源码对照见 [review](2026-09-08-rigid-dynamics-upstream-review-zh.md)。Editor 按用户要求暂缓。

## 结果与接口

通用前向 pipeline 已支持完整世界重力向量，以及动态自由刚体 COM 上的世界力和力矩。固定基 ABA 将世界重力变换到根坐标；浮基和自由体使用同一向量。新增公共字段 `BODY_FORCE=37`、`BODY_TORQUE=38`，旧枚举编号不变，复用已分配的 float3/body arena 字段。

力单位 N、力矩单位 N·m，在一次速度积分中消费并清零。`step_n(n)` 仅第一步应用预先写入的输入，持续力须逐步重写；reset 清选中环境的瞬时输入。静态体和关节碰撞代理不会被当作自由体加速，关节广义外力仍走 joint feedforward。Python `World.create_from_scene` 增加尾部 gravity_x/y/z 参数，全部为零仍选择默认地球重力；`set_gravity_z` 仍只配置可微 tape。

扩大完整流程覆盖还修复三个实际缺陷：创建和 reset 共用 GPU FK/proxy 刷新，消除 host 初始位姿与 reset 派生位姿的逐位差异；关闭接触时清空旧 feet 表，使零 contact capacity 与上传校验一致；七个 Python 句柄类的 `__enter__` 统一返回 reference，避免临时 C++ 副本销毁原句柄。

**本项没有实现 gyro。** 当前外力矩更新仍为 `omega += dt * I_world_inv * torque`；非等方自由翻滚、稳定积分及其诊断继续由 T02c 验收。

## 证据与身份

原始目录：`out/validation/rigid_inputs_20260908/`。WSL Ubuntu 24.04，RTX 5080 / 驱动 610.88，CUDA 13.3.73，Release / g++ 10.5.0 / all-major。`frozen/` 保存场景二进制、库、binding、构建配置、源码差异和验证入口；`frozen/manifest.json` 包含逐文件源码及上游身份。

| 对象 | SHA-256 |
| --- | --- |
| 源码 manifest | `2f76fbbd083204db1ce2f31e53ce4a6a00dbfa07ee89817a0953639819b1bb68` |
| 场景二进制 | `23d4a910cfe25aac133137e3b1f76d05161e0ec858926047fa8eefd1c7833ebf` |
| libnuka.so.0 | `5cc0006be2f804e170a412cba21c42b22767e2362c1b650a53b9dac6808e022c` |
| Python binding | `428c1a59cb8f1128e4baa15212ff46af0d25ddd9f6d567f13c610d5e282578f7` |

`before.*` 保存三个失败，其中关节旋转首先失败于 World 创建，不能当作重力数值反例。修复初始化后、修复动力学前的 `initialized_before.*` 仍有三个失败，此时固定/浮基旋转已出现真实数值差异，例如 link 1 的 qdot 差 0.0138242，容差 2e-6。`after.*` 是同组输入修复后的结果。

| 验证 | 结果 | 原始文件 |
| --- | --- | --- |
| 主 pipeline、自由体和 reset | 16 passed，1 skipped（接触 graph） | `after.log/.json` |
| 速度核和 reset memcheck | 15 passed，0 errors | `memcheck_supported.log/.json` |
| Python 公共 pipeline | 两环境，240/480 Hz，各 36/72 步；输入、隔离、重放、渲染均通过 | `public_api.log/.json` |
| Python reset | 10 passed | `python_reset.log/.xml` |
| C ABI coupled / multi-env | 2 + 3 passed | `coupled_cabi.*`、`multi_env_cabi.*` |

首次 memcheck 包含已知不支持的接触 graph，记录 6 个 CUDA capture API 错误，见 `memcheck.log`；不能把那次运行称作 0 errors。随后仅排除该 Unsupported 项以及非行为性能用例，保存独立的 `memcheck_supported.*`。此前 reset 项完整 robot-cloth-fluid memcheck 已通过；本项针对新增向量运算、初始化和恢复路径补查，没有重复整个慢场景。

## 完整流程与物理质量

主验收沿用 `RobotClothFluidCoResident.Go2StanceCouplesClothAndFluidOnOnePipeline`：Go2、169 个布料粒子、100 个 PBF 粒子，并加入有偏心 COM、非等方惯量的自由体；gravity 为 `{0.06,-0.04,-9.81}`，240 Hz，250 步 settle + 200 步 hold。验证首步力/力矩消费、450 步平移、全量及 masked reset、8 步逐位重放。布料/流体接触行计数为 2547/600，流体自由表面相对控制升高 9.1 mm，介质对关节速度的反作用 L1 差为 0.512904。

两个必要的 World 解析补充覆盖偏心/旋转惯性帧的单位与协变性，以及 fixed/floating articulation 的整体旋转重力。它们使用完整生产 pipeline，不直接调用独立测试求解器。

公共 API 流程见 `tools/validation/rigid_inputs_pipeline.py`：两个环境各 14 个 body、129 个粒子，包含同类机器人、布料、PBF 与球体。交叉使用 DLPack 和 host upload 写入力/力矩；与解析积分比较后，验证未选环境、待消费输入、view 地址、reset 和 `step_n` 重放。两种 dt 保持总时长 0.15 s 和外部冲量相同。

| 指标 | 240 Hz / 36 步 | 480 Hz / 72 步 |
| --- | --- | --- |
| 相对离散解析式的最大位置误差 | 1.1921e-7 m | 5.9605e-8 m |
| 相对连续轨迹的最大位置误差 | 3.065586 mm | 1.532793 mm |
| reset / replay | 逐位相同 | 逐位相同 |
| 创建与 reset 后图像 | 逐像素相同 | 逐像素相同 |

连续轨迹误差随 dt 减半缩小 2 倍，符合当前半隐式 Euler 平移的一阶精度；不能把相对离散解析式的亚微米误差称作连续物理误差。文件创建接口中默认重力与显式地球重力逐位一致，XY 重力引起的浮基平移与 `g_xy * dt²` 一致。

已目视检查 `render_roundtrip.png` 的创建、推进和 reset 三帧：自由体下落可见，reset 恢复原位置。这里使用测试资产的简化几何，图像用于位姿和生命周期验收；原 π0.5/G1 demo 视频及冻结引擎保持独立证据。

## 成本与剩余工作

没有新增 model/data arena 字段或每步 dispatch：力缓冲原已存在，本次补读取、清零与速度更新。创建时增加一次与 reset 共用的 FK/proxy 刷新；关闭接触时移除不应存在的 feet 表。本项没有做等质量的 step 吞吐对照，测试进程耗时也不作为 GPU step 性能。完整 benchmark 和分项成本仍属于 T01。

接触 graph 的 LBVH workspace、双浮基 Go2 重叠不分离、异构拓扑和 T24 的 stream/view 生命周期其余契约保持未完成；此次通过不覆盖这些能力。

## 重放命令

在 WSL 仓库根目录执行：

```bash
cmake --build build-linux --target nuka_scenario_test nuka nuka_coupled_world_cabi_test nuka_multi_env_world_test --parallel 6
build-linux/tests/nuka_scenario_test --gtest_filter='RobotClothFluidCoResident.*:FreeRigidDynamics.*:MultiArticulationReset.*-MultiArticulationReset.ReportsResetCostAndMemory'
/usr/local/cuda/bin/compute-sanitizer --tool memcheck --error-exitcode 99 build-linux/tests/nuka_scenario_test --gtest_filter='FreeRigidDynamics.*:MultiArticulationReset.*-MultiArticulationReset.ReportsResetCostAndMemory:MultiArticulationReset.CapturedContactStepsRemainValidAfterReset'
LD_LIBRARY_PATH=build-linux/src:/usr/local/cuda/lib64 PYTHONPATH=python /root/nuka-vla/bin/python tools/validation/rigid_inputs_pipeline.py --output out/validation/rigid_inputs_replay
LD_LIBRARY_PATH=build-linux/src:/usr/local/cuda/lib64 PYTHONPATH=python /root/nuka-vla/bin/python -m pytest python/tests/test_nuka_dlpack.py -q -k reset
```

Python 重放需先构建与当前库一致的 binding，或在独立包目录使用 `frozen/` 中匹配的库与扩展；不能只替换枚举声明而加载旧扩展。具体 C ABI 命令保存在 `public_commands.json`。
