# CUDA 岛调度与 XPBD 提交验收

当前 CUDA 批次通过既有物理与同步验收。五个独立进程交错对照中，eager 完整步耗时下降 39.02%–55.21%；graph 中位数增加 0.22%–2.38%，没有 graph 加速结论。共享模型布局、迭代预算、材料公式和材料/接触交替顺序未改变。

## 比较范围与身份

性能分母是粒子物理修复 `b0c2816` 的 `particle_contract_closure_frozen`，不是之前失真的布料轨迹。两版均运行默认 robot-cloth-fluid，E=1/16/256，预热 250 步、计时 200 步，再逐步检查完整 450 步质量和重放。每组五对进程交错，默认 CUDA 环境；GPU 完成时间包含完整物理与接触读出，不含质量下载。

原始证据位于 `out/validation/island_scheduling_20260909/`。

| 版本 | source SHA256 | benchmark SHA256 | library SHA256 |
| --- | --- | --- | --- |
| 物理基线 | `818337451b329cd25f166a3a8c7c17a69e2fd72ee574864d2d7af1aba88d58f4` | `54141e539ba920caf7649784bea6fd799934d7a5e25aec58ef3bdcd8bb2ca5f5` | `7e3de5e12e5a87e63db63008ea42140c082bdf653a8cbd30792934372454034f` |
| CUDA 候选 | `77b60b334a3b45ceaa1bedc9579dacd80add57836691c544513d5df3ccfdde07` | `85ca55df6e06ced6abba6d28d318aa0c22b27d917cd806fbd56e425966336cc7` | `77b123b1398eb14c4019a54ad573b38a0d3cd848042fff0f41fabfc50b34b3e8` |

## 完整步耗时

单位 ms；变化为候选相对基线的五进程中位数变化。完整进程范围、逐对变化、状态文件和命令见 `cuda_execution_v1_paired/summary.json`，不将 profiler 数字作为性能分母。

| 环境数 | 执行 | 基线中位数 | 候选中位数 | 变化 |
| --- | --- | ---: | ---: | ---: |
| 1 | graph | 3.40950 | 3.49074 | +2.38% |
| 16 | graph | 4.13453 | 4.18980 | +1.34% |
| 256 | graph | 6.97383 | 6.98945 | +0.22% |
| 1 | eager | 12.56333 | 5.62700 | −55.21% |
| 16 | eager | 11.30448 | 5.81943 | −48.52% |
| 256 | eager | 12.78841 | 7.79793 | −39.02% |

eager 的全部十五对均改善。graph E=1 有一对 +15.29%，其余四对为 −2.30% 到 +5.33%；五对中位变化 +3.89%，须保留抖动与小幅代价，不能称所有单次运行都没有回退。E=16/256 的五对中位变化分别 +1.65%/+0.16%。模型与数据 bytes 在两版间完全相同。

## 物理与执行证据

- 六组全部进程通过原长度预算：最大长度应变 1.838344%，最坏每环境 RMS 0.203045%，限值仍为 5%/1%；固定点不动。完整状态、逐步 wrench、canonical island 和 XPBD 工作量一致。身份用于检查运算顺序，物理质量由上述独立长度门及既有解析场景判断。
- 既有联合 pipeline 26/27 通过，distance/bend/volume/shape-match 现有材料验收 12/12 通过；公开 eager/graph、reset 与渲染链路通过。唯一 soft-tet 失败见下，未删除或改容差。
- E=16 graph 完整 450 步的 memcheck 和 synccheck 均为 0 errors，命令与完整日志为 `cuda_execution_v1_memcheck.*`、`cuda_execution_v1_synccheck.*`。
- E=2048 eager 的完整 450 步对照通过同一物理质量、逐步 wrench 和状态身份；九次调度观测均为 2048 个 live islands，超过实际 1008-block 驻留网格，验证跨岛循环复用。该组只有一对，42.836→36.412 ms 仅作边界运行记录，不充当五进程性能结论。数据区 7,648,919,296 B、模型区 310,965,760 B。

## Profile 与下一热点

每次 XPBD op 的 distance/bend 颜色序列各改为一次 cooperative launch，两者合计从每步 480 次提交降为 48 次；不跨接触边界合并 24 轮。E=1 eager 的 host API 从 9.7272 降至 5.1860 ms/步，host 提交区间从 10.3864 降至 5.8127 ms/步。新普通 launch 为 3.9014 ms，cooperative launch 为 0.6500 ms。

E=256 graph 的 kernel busy 从 6.0191 到 6.1216 ms/步；岛求解 1.9667→1.9689 ms，bend 0.6951→0.7536 ms，distance 0.2587→0.2940 ms。新同步并未降低这些 GPU 热点的净成本。岛 kernel 静态寄存器为 166/thread，grid=1008、block=32、shared=72 B；这是静态资源数据，不是实测 occupancy。仍需继续有效行/J/质量算子和设备端访存优化。

## 保留的失败与限制

`SoftTetPresserTwoWay.SoftCubeLandsDeformsAndRecovers` 在基线和候选上得到相同失败：最小 extent 0.179912567（要求 <0.175000012），恢复体积 0.00025918262（要求 >0.000259953376）。输入 distance compliance 为 0，却要求明显压缩；材料/验收契约仍待归因。旧版相同不能证明物理正确。

E=1024/2048 的旧版 graph 首步均失败。新诊断确定候选 E=2048 在 `ContactWarmStart` 的 graph capture 返回 Failed，native code 未保留；相同容量的 eager 可完成，不能称为 OOM。具体 CUB/capture 原因继续处理，此问题不是新岛循环的边界失败。

旧较大布料、地形、双浮基重叠失败仍保留；本批不关闭完整 goal。按用户指定优先级，接续 MLS-MPM bunny-water 的独立基线与通用优化，之后完善无 SDF、有限质量、多 owner 和子步多体反馈。
