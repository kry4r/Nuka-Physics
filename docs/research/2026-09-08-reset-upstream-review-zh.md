# 多关节体 reset：规格与上游实现 review

日期：2026-09-08。对应 [模块 spec 的 M04/T03](../plans/2026-09-07-physics-module-specs-newton-port-zh.md) 和 [细化规格第 4 节](../plans/2026-09-08-physics-optimization-detailed-spec-zh.md)。本次 review 用于完成已有 reset 候选；功能、内存安全和性能验收结果另行记录，源码对照不表示外部引擎实测。

## 规格核对

M04 要求 root 使用 `a=e*K+k`，并恢复 q/qdot、link velocity、自由体和粒子 F/C/plastic；M02/M10 要求 reset 不使 graph 或公开 view 地址失效；M03/M11 要求记录真实构建身份、质量和成本。当前候选将 full/masked restore 接入同一个恢复核，并给现有 FK/proxy 同步增加环境选择参数，没有新增物理求解路径。

本次只验收支持的同构多树布局。异构 DOF 映射属于 M01/M04 的 T04；有接触的 graph 当前被 LBVH 临时 workspace 阻塞，属于 M03/M05 的 T09。不得将本次 reset 成功扩展为这两项已完成。

## 最新主干定位

2026-09-08 03:48–03:49 UTC 通过官方 Git 主干查询并下载，精确身份保存在 `.nuka_cache/engine-review/manifest.json`。原模块 spec 的 Newton v1.5.0 仍作为固定比较基线；下表用于本次设计 review。

| 项目 | 官方仓库 / main revision | 源码提交时间 |
| --- | --- | --- |
| Newton | `newton-physics/newton` / `853c4fef9543fae380de59464cb804465c041721` | 2026-09-07 15:06:03 UTC |
| MuJoCo | `google-deepmind/mujoco` / `a7e365917a447dc44e280ff9567064228657d266` | 2026-09-07 20:39:29 UTC |
| Genesis | `Genesis-Embodied-AI/Genesis` / `0ce793b42c945b6848aad624501b2ca756d3e244` | 2026-09-06 18:42:06 UTC |

## 采用与差异

| 上游实现 | 观察 | Nuka 的处理 |
| --- | --- | --- |
| [Newton SolverMuJoCo.reset](https://github.com/newton-physics/newton/blob/853c4fef9543fae380de59464cb804465c041721/newton/_src/solvers/mujoco/solver_mujoco.py#L4175) | 分别恢复关节权威状态并清 `qacc_warmstart/qfrc_applied/xfrc_applied/act/ctrl`；通常下一步 FK 才刷新 body，sleeping 时即时刷新 | 采用状态与历史分离；Nuka 在 reset 返回后就要能读正确 FK/proxy，因此即时刷新；drive/task target 按 Nuka 契约保留 |
| [Newton State.assign / clear_forces](https://github.com/newton-physics/newton/blob/853c4fef9543fae380de59464cb804465c041721/newton/_src/sim/state.py#L189) | 原地数组复制保持图引用地址稳定；显式清力 | 复用 Data arena，不重新分配 snapshot 或输出 view；瞬时累加量仅清选中环境 |
| [MuJoCo _resetData / mj_resetData](https://github.com/google-deepmind/mujoco/blob/a7e365917a447dc44e280ff9567064228657d266/src/engine/engine_io.c#L1289) | 除 qpos/qvel，还重置接触/约束/island 计数、诊断、lazy flags、warm-start 和外部输入 | 采用完整生命周期清单；单个 CPU mjData 布局不作为 Nuka GPU 批处理实现或速度依据 |
| [Genesis rigid set_state](https://github.com/Genesis-Embodied-AI/Genesis/blob/0ce793b42c945b6848aad624501b2ca756d3e244/genesis/engine/solvers/rigid/rigid_solver.py#L1758) 与 [constraint reset](https://github.com/Genesis-Embodied-AI/Genesis/blob/0ce793b42c945b6848aad624501b2ca756d3e244/genesis/engine/solvers/rigid/constraint/solver.py#L210) | 按环境写 rigid 状态，清 collider/constraint 缓冲并恢复 awake lists | 采用环境隔离和失效缓存清理；不引入 Nuka 尚未实现的 sleeping 状态 |
| [Genesis Simulator.reset](https://github.com/Genesis-Embodied-AI/Genesis/blob/0ce793b42c945b6848aad624501b2ca756d3e244/genesis/engine/simulator.py#L237) | solver、coupler、sensor 各自负责恢复 | 明确粒子/MPM 与接触状态的覆盖范围；不借由整批刷新破坏未选环境 |
| [Genesis MPM set_state](https://github.com/Genesis-Embodied-AI/Genesis/blob/0ce793b42c945b6848aad624501b2ca756d3e244/genesis/engine/solvers/mpm_solver.py#L823) | 该 revision 的入口接受 `envs_idx`，却未传入 `_kernel_set_state`；核遍历全部 `_B` | 不能把这个入口当作 masked MPM reset 已正确的依据；Nuka 用 E=3、K=0 的 x/prev_x/v/F/C/plastic 独立隔离用例验证。这里只是源码观察，未运行 Genesis |

Newton 的 mask 包含 global world -1，长度为 `world_count+1`；Nuka 接口是环境 ID 集合，不能机械复制该布局。Nuka 的重复 ID 去重、非法 ID 在写设备前失败、control target 保留行为以本地 spec 为准。

## 状态生命周期核对

| 类别 | reset 行为 | 下一消费者 / 验证 |
| --- | --- | --- |
| roots、q/qdot、link velocity | 恢复选中环境 snapshot | E=3/K=2、独特 root、full/masked、首步新 World 对照 |
| body pose/v、world inertia、proxy | 恢复权威状态，计算世界逆惯量，FK 后同步选中 proxy | K=0、旋转非等方惯量、多 collidable；未选环境逐位不变 |
| particle x/prev_x/v/F/C/plastic | 按粒子环境 span 完整恢复 | K=0/E=3；各字段使用不同非零值，并检查 snapshot 没被改写 |
| lambda/cache/pseudo/readout、row coupling | 清选中环境，owner index 置 invalid | 污染历史后 reset；清零和未选环境保留分别断言 |
| drive/task target | 保留 | reset 后原值不变 |
| MPM grid、排序等 scratch | 不恢复创建时内容 | `MpmGridPrepareKernel` 清质量/owner/cell start；P2G 与 grid update 在下一消费前覆写 momentum/velocity；用首步新 World 对照验证 |
| arena/graph/view | 地址不变 | 真正成功捕获的无接触关节图重放；接触图单列 unsupported |

## 实施与验收步骤

下列步骤已执行，结果与限制见 [验收报告](2026-09-08-reset-validation-zh.md)。接触 graph 保持 Unsupported；Editor 按用户要求暂缓。

1. 在 `tests/scenario/multi_articulation_reset.cpp` 增加纯 MPM full/masked 状态 roundtrip 与首步一致性；增加纯自由刚体惯量和代理恢复检查。复用真实 World 路径，避免用被测核计算期望值。
2. 运行 `cmake --build build-linux --target nuka_scenario_test --parallel 6`，再运行 `build-linux/tests/nuka_scenario_test --gtest_filter='MultiArticulationReset.*-MultiArticulationReset.ReportsResetCostAndMemory'`。预期除已记录的接触 graph unsupported 外，功能用例全部通过。
3. 运行相关接触、MPM、C ABI/Python reset 和 CUDA memcheck 回归；结果保存到 `out/perf/reset_contract_20260908/`，memcheck 必须为零错误。
4. 固定候选二进制和源码 manifest，独立进程运行 `MultiArticulationReset.ReportsResetCostAndMemory` 五次。与已冻结 baseline 比较 full/masked reset 微秒数和 arena bytes；这是 reset 成本，不能当作 step 吞吐。
5. 把验收与限制回写模块 spec、细化 spec 和 TODO。完成后按 M04/T02 继续，改造前重新 review spec 和三引擎最新主干。
