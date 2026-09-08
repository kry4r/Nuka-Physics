# 外力、重力与自由旋转：规格及上游 review

日期：2026-09-08。依据模块 spec M04/T02 和细化 spec 第 5 节；reset 已在 `5511b77` 提交，Editor 按用户要求暂缓。

## 最新源码身份

开始本项前重新查询三个官方仓库的 `refs/heads/main`：

| 引擎 | revision | 核对位置 |
| --- | --- | --- |
| Newton | `853c4fef9543fae380de59464cb804465c041721` | `newton/_src/solvers/solver.py:66`、`newton/_src/sim/state.py:147` |
| MuJoCo | `3d4447e1044ac99a4604653f74e97e79fd4ffd83` | `src/engine/engine_forward.c:1017`、`:1726` |
| Genesis | `0ce793b42c945b6848aad624501b2ca756d3e244` | `genesis/engine/solvers/rigid/abd/forward_dynamics.py:2000`、`genesis/engine/simulator.py:363` |

MuJoCo 相比 reset review 的 revision 有更新；已 fetch/checkout 并核对相关动力学文件没有变化。最新提交时间为 2026-09-08 06:47:43 UTC。以上是源码对照，没有运行外部引擎性能实验。

## 采用与差异

Newton 的通用刚体积分消费世界坐标 COM wrench 与三维 gravity，以惯性坐标计算 `tau - omega × I omega`，再更新 COM 和 authored pose。采用其坐标和单位契约；其显式 gyro 不直接作为稳定性方案。

MuJoCo 在 `mj_fwdAcceleration` 汇总被动、bias、用户、执行器及投影后的笛卡尔外力；隐式积分与约束有效质量的关系明确。Nuka 先补已分配但未消费的自由体 force/torque 和三轴重力，复用当前世界逆惯量。不能把这项当成完整 MuJoCo 隐式约束 metric 或 articulation 笛卡尔外力接口已完成。

Genesis 使用惯性帧隐式 midpoint、Newton/backtracking 处理自由翻滚，并使用 midpoint 速度推进姿态，再还原真实速度。采用稳定隐式求解、数值残差和守恒检查的要求；不复制它按独立 free joint 特判的调度。其外层 step 完成后清 external force，为 Nuka 的瞬时力生命周期提供参考。Newton/MuJoCo 的输入保留方式不能覆盖 Nuka 已选定的消费后清零契约。

## 本地核对与实施决策

- `BodyIntegrateVelocityKernel` 仅做 `v.z += gravity_z*dt`；`body_force/body_torque` 已分配、初始化为零且 reset 会清零，但正常 step 未消费。
- `AbaForwardParams` 虽有 `gravity[3]`，实际只取 Z；浮基积分也只传 Z。固定基根的世界重力还必须经过 root motion transform，才能在旋转场景中保持协变。
- 布料/PBF/MPM 已接收 vec3 gravity；沿现有参数传递补齐刚体/关节体，不另建物理路径。
- 新公开 `BODY_FORCE/BODY_TORQUE` 使用既有 body 布局及 COM 世界坐标，C ABI 只追加字段编号。它们用于自由刚体 owner，静态/关节碰撞代理不会被当成自由体；articulation 的广义外力仍使用既有 joint feedforward。
- 力在一次 IntegrateVelocity 中消费并清零，`step_n(n)` 仅第一步应用此前写入的力；持续力每步重新提供。reset 清瞬时力，drive/task target 保留。
- Python 文件创建入口补创建时的三轴 gravity 参数，沿用 C ABI 的全零表示默认重力约定；不改变现有仅配置可微 tape 的 `set_gravity_z` 语义。

扩展 pipeline 的实际运行还暴露两个前置问题：创建时 host-seeded body proxy 未经过与 reset 相同的 GPU FK/sync，因此增加 body pose 的逐位检查后不一致；`enable_contacts=false` 仍留下非空 `feet` 表，与零 contact capacity 冲突，World 在 UploadTo 被拒绝。修复采用创建/reset 共用派生位姿刷新，以及关闭接触时清该旧接触表，不放宽容量校验。`before` 的三个失败中，关节旋转对照首先失败于创建，不能作为旋转重力数值反例；解除创建阻碍后另存反例。

## 自由旋转的后续契约

稳定 gyro 仍是独立验收项：在惯性帧求解欧拉方程，必须同时定义速度 kick、接触冲量、姿态 drift 和 pseudo rotation 的时间层。若采用一阶隐式 Euler，应明确耗散和 dt 收敛；若采用 midpoint，必须配套 midpoint 姿态积分，不能只替换角速度公式就宣称守恒。数值失败与残差必须可读，不静默换成显式更新。

主回归继续使用机器人＋布料＋流体 pipeline，加入同场自由刚体以覆盖完整力输入/积分/reset；少量解析对照补充旋转协变、质量/惯量单位和力消费。完成三轴重力/外力后再记录自由翻滚结果，不能将前者完成扩大为 gyro 已完成。

## Python 生命周期补充核对

公共 pipeline 发现 `World.__enter__()` 返回了拥有同一句柄的 C++ 副本；在 `with world:` 中，临时返回值释放立即销毁底层 World。最小复现已区分 builder 销毁与 enter 副本：builder 销毁后仍可读取，enter 返回对象与原对象不同，释放 enter 返回对象后读取报 null handle。

修改前复核 M10/T24 的 owner/lifetime 契约，并于 2026-09-08 08:39 UTC 重新查询官方 main。Newton 与 Genesis revision 未变；MuJoCo 更新至 `dbe1e2e2c336684cf60674a19ba96ec62b839c16`，新增 URDF mimic 警告和文档变更，动力学文件及 Python ownership 实现无变化。Newton `coupled/model_view.py` 的上下文管理借用现有 view；Genesis `utils/misc.py` 的上下文管理返回现有 self；MuJoCo `python/mujoco/structs.cc` 对内部对象明确使用 reference/reference_internal。这些不是相同的句柄 API，但都不通过隐式复制持有者转移资源所有权。

Nuka 对全部七个句柄类的 `__enter__` 统一指定 nanobind reference 返回策略，保留创建接口的 take_ownership。用真实 World 的进入、读写、step、reset、渲染与离开流程验证，不扩展一批仅断言返回策略的单测。该修复不代表 T24 的 stream/event 与销毁后 view 契约已全部完成。
