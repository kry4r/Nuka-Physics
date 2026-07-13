# BDX training-coarse media performance

## 结论

状态：`DONE_WITH_CONCERNS`。

独立的 `training-coarse` authoring profile 将原电影场景的 MLS-MPM 网格从
165,095 降到 16,170 nodes/env，并在 RTX 4000 Ada 的物理 GPU1 上将 N=1
从 135.008 ms/step 降到 29.826 ms/step。`<=80 ms @ N=1` 已达到；实测大 N
显存斜率约 42.39 MiB/env，`<=8 MiB/env` 未达到。本工作没有修改通用 grid
公式、没有场景专用 engine 分支，也没有进入 P2G gather 重写。

## Authoring 配置与选择

训练显式选择：

```bash
python examples/demo/bdx_oneshot_author.py --profile training-coarse
```

默认输出为 `examples/scenes/bdx_oneshot_training_coarse.nks/.nka`。不传
`--profile` 时仍选择 `fine` 和 `examples/scenes/bdx_oneshot.nks`。

| 配置 | fine | training-coarse | 变化 |
|---|---:|---:|---:|
| Zone B spacing | 0.013 m | 0.026 m | 2x 粗化 |
| Zone C spacing | 0.018 m | 0.026 m | 1.44x 粗化 |
| MPM substeps | 14 | 7 | -50% |
| MPM dx | 0.026 m | 0.052 m | 2x |
| loft headroom | 1.00 m | 0.15 m | -85% |
| Zone B particles/env | 9,600 | 1,200 | -87.5% |
| Zone C particles/env | 1,860 | 546 | -70.6% |
| MPM particles/env | 11,460 | 1,746 | -84.8% |
| all particle rows/env | 12,211 | 2,497 | -79.6% |
| grid dims | 89×35×53 | 49×22×15 | |
| grid nodes/env | 165,095 | 16,170 | -90.2% |

通用 cook 仍使用 geometry union AABB、`margin=4*dx` 和 authored loft 计算 grid。
两场景的 union 为 `lo=(0.970,-0.330,0.022)`、
`hi=(3.030,0.330,0.082)`。coarse authoring 得到 `base_z=-0.188`、
authored `top_z=0.500`；向上取整后的最后网格节点为 z=0.540。600-step
稳定探针中的真实 MPM z 范围仅 `[0.02843,0.07121]`，因此 0.15 m loft
由运动包围盒支持，并非为了性能硬压到粒子附近。

粗粒代价是 Zone B 表面采样密度降为 1/8、Zone C 降约 71%，小于 2.6 cm
的局部颗粒/足印细节无法与电影 fine scene 等价。训练保留相同床体 AABB、
Drucker-Prager 材料、地面与刚体耦合语义；视觉电影必须继续选择 fine scene。

## Fine scene 不变证明

实现前后哈希均为：

| 文件 | SHA-256 |
|---|---|
| `bdx_oneshot.nks` | `a3bf4ee2740d7c3ffd6d63438e4a915a8cf2abdb58948cd6c84e39dd55e03d63` |
| `bdx_oneshot.nka` | `5d7a271add6b798be3df83f0b67682ba738f58524cb129ba2812315b10906812` |

`bdx_oneshot_profile_test.py` 每次运行固定检查这两个哈希和 fine profile 参数。
coarse `.nka` 与 fine `.nka` 也具有相同 SHA-256；它复用相同电影几何资产，
差异位于独立 `.nks` 的 granular authoring 参数。最终验证还运行：

```bash
git diff --exit-code 6ea559d -- examples/scenes/bdx_oneshot.nks examples/scenes/bdx_oneshot.nka
```

## 稳定性

GPU1 命令：

```bash
CUDA_VISIBLE_DEVICES=1 python examples/demo/bdx_oneshot_granular_stability.py \
  examples/scenes/bdx_oneshot_training_coarse.nks --steps 600
```

结果为 exit 0：Zone B/C 分别 1,200/546 粒子，NaN=0、grid 外=0、lane
escape=0；z mean=0.03973 m，末 10 步速度 mean/p95/max=
0.000294/0.001962/0.006315 m/s，450→600 步位移 mean/p95=
0.000193/0.001282 m。原始日志：
`/data/xtzhang25/_work/activate/t2_coarse_granular_stability600.log`。

现有 `bdx_oneshot_gate.py --stage settle` 也被执行并以 exit 1 保留在
`t2_coarse_settle600.log`。它使用 `p[-728:]` 当 cloth、`p[:-728]` 当 MPM，
但真实布局是 `[1746 MPM | 728 cloth | 23 cable/slab]`，因此把 cable 粒子
误计为 cloth/gravel；`links[-1]` 也不是粒子 slab。这个失败不能用于判定
granular 稳定性，不能隐藏或改写成通过。

通用材料稳定门 fresh 运行 3/3 passed、exit 0：

- sand35：`h=0.0395 r=0.0740 angle=28.3 esc=0`；
- CapsuleFootprint：`depth0=0.0214`，3 s 后 `depth1=0.0215`；
- coh15：`h=0.1195 r=0.0390 esc=0`。

命令与日志：

```bash
CUDA_VISIBLE_DEVICES=1 build-t2-media/tests/nuka_scenario_test \
  --gtest_filter=MpmGranular.ConeReposeTracksFriction:MpmGranular.CapsuleFootprintPersists:MpmGranular.CohesionBindsTheColumn
```

`/data/xtzhang25/_work/activate/t2_gate_granular.log`。

## 性能口径与完整矩阵

所有正式对比使用物理 GPU1（RTX 4000 Ada 20,475 MiB）、Release
`build-t2-media`、`bake_link_sdf=True`、dt=1/240、无 wrench demand。每个 N
在独立进程中构建 world，warmup 20 steps，然后采 10 组、每组 10 steps；
每组前后同步，表中 mean/median/p95 是 10 个组内平均 step time 的统计量。
所有 6 个正式命令 exit 0，没有 OOM 或超时。

命令模板：

```bash
CUDA_VISIBLE_DEVICES=1 python examples/demo/bdx_oneshot_perf.py SCENE \
  --envs N --warmup 20 --samples 10 --steps-per-sample 10
```

| 场景/代码 | N | mean ms | median ms | p95 ms | peak GPU1 MiB | CUDA delta MiB/env |
|---|---:|---:|---:|---:|---:|---:|
| fine / 6ea559d | 1 | 135.008 | 134.696 | 142.358 | 774 | 610.000 |
| fine / 6ea559d | 64 | 826.486 | 828.814 | 838.991 | 4,226 | 63.469 |
| fine / 6ea559d | 256 | 2,854.459 | 2,871.101 | 2,884.784 | 14,744 | 56.953 |
| coarse / 6ea559d | 1 | 29.826 | 29.509 | 37.172 | 760 | 596.000 |
| coarse / 6ea559d | 64 | 98.836 | 98.660 | 106.139 | 3,432 | 51.063 |
| coarse / 6ea559d | 256 | 312.895 | 311.861 | 323.938 | 11,570 | 44.555 |

T2.3 的同 build 增量为：N=1 快 4.53x（-77.9%），N=64 快 8.36x
（-88.0%），N=256 快 9.12x（-89.0%）。正式 timing 日志为
`t2_perf_fine_6ea_n{1,64,256}.log` 和 `t2_perf_coarse_n{1,64,256}.log`；
peak 由外层每 100 ms 的 `nvidia-smi -i 1` 采样，coarse 另以独立进程短
复测确认 760/3432/11570 MiB。

历史 pre-6ea N=1=419 ms/step 仅来自 controller 给定基线，没有 N=64/256
同构建产物或分位数，故不伪造扩展行。当前原 fine authoring 已在 N=1/64/256
全部 fresh 实测，作为 coarse 的严格同口径 before。

## T2.1 / T2.2 / T2.3 增量边界

T2.1 在同一 6ea binary、fine N=1 上用旧 readout 行为等价的 `--wrench`
直接隔离。20 warmup + 30 连续 measured steps 的 mean 为：wrench on
141.083 ms，off 131.749 ms，即 demand gate 省 9.334 ms/step（6.6%）。日志：
`t2_6ea_fine_n1_wrench_fresh.log` 与 `t2_6ea_fine_n1_fresh.log`，均 exit 0。

T2.2 与 T2.1 同在 6ea 提交，本轮没有构建 parent binary，因此不能把历史
419 到 131.749 的其余 277.917 ms 全部冒充 T2.2。可精确确认的容量变化是：
fine 的 11,460 MPM 粒子过去会额外预留 `11,460*4=45,840` contact slots/env；
6ea 后 MpmXpbd 的 MPM slice 预留为零，仅 XPBD slice 参与 row reserve。
419→131.749 的 combined 改善为 287.251 ms（68.6%）；扣除可隔离 T2.1 后的
277.917 ms 是 T2.2 加历史测量/构建漂移的上界混合项，不作为独立点估计。

T2.3 使用上表正式同口径 fine/coarse 数据，贡献可独立归因。

## 显存残余与目标判定

以 N=64→256 的 `cudaMemGetInfo` delta 斜率消除固定项：

- fine：54.781 MiB/env，截距约 556 MiB；
- coarse：42.385 MiB/env，截距约 555 MiB；
- authoring 粗化实际减少 12.396 MiB/env，但 residual 仍比 8 MiB/env 高 5.3x。

cooker 对两场景都报告已有 link-SDF 估算 `1,148,946,648 bytes`
（893 unique / 1402 pieces）。它是 build/cook 内容与 CUDA context/model upload
的共享固定成本，不应除以 N 作为线性 per-env；约 555 MiB 的实测截距已把
固定项从斜率中分离。warning 的 host cook 估算也不等于 `nvidia-smi` peak。

根据 `arena_layout.hpp`，MPM grid 每节点至少含 mass(4)、momentum(12)、
velocity(12)、force(12)、body_dp(12)、body_owner(4)，共 56 bytes/node：fine
约 8.817 MiB/env，coarse 约 0.864 MiB/env，单 grid 节省约 7.953 MiB/env。
剩余粒子 F/C/snapshot/sort 字段解释 coarse/fine 斜率差的另一部分。

约 42.4 MiB/env 的共同 residual 不随 coarse MPM node/particle 大幅下降，边界为：
row-making XPBD 的 contact/row/Jacobian arenas、77 rigid bodies 与 articulation
state、broadphase/island scratch、CUDA graph/allocator workspace。现有外部
`cudaMemGetInfo`/`nvidia-smi` 口径不能再无歧义地把这些项逐字段拆分。因此
N=1 时间目标已达，per-env 显存目标未达；本轮在这里停止，不进入 P2G 或
allocator 重写。

## 冻结门与单系统字节一致性

本轮 fresh 证据：

| 门 | 实际值/结果 | exit |
|---|---|---:|
| Go2 owner canary | `actual: 0.923080623` | 1（owner 定义为值通过、测试既有失败） |
| cloth FNV | orig/loaded 均 `777503423208024307` | 0，2/2 passed |
| ConeRepose sand35 | `h=.0395 r=.0740 angle=28.3 esc=0` | 0 |
| CapsuleFootprint | `depth0=.0214 depth1=.0215 esc=0` | 0 |
| Cohesion coh15 | `h=.1195 r=.0390 esc=0` | 0 |

Go2 命令：

```bash
CUDA_VISIBLE_DEVICES=1 build-t2-media/tests/nuka_regression_test \
  --gtest_filter=Go2Stand.OwnerGoldenTrajectoryMatchesWithinTolerance
```

日志 `t2_gate_go2.log`。owner 规则明确该 test 的 tolerance failure 是既有状态，
判断条件是实际值逐位保持 `0.923080623`。

cloth 命令：

```bash
CUDA_VISIBLE_DEVICES=1 build-t2-media/tests/nuka_nks_media_roundtrip_test \
  --gtest_filter=NksMediaRoundtrip.*
```

日志 `t2_gate_cloth.log` 明确打印两个精确 FNV。

补充 `t2_gate_byte_identity.log` fresh 6/6 passed：MpmXpbd superset/slice layout、
body-particle budget、MPM jelly/fluid/granular byte identity；
`t2_gate_oplist_particle_paths.log` 的 op-list 9/9 passed；正确 executable 的
`t2_gate_particle_subsystems.log` soft/fluid subsystem 1/1 passed。这些覆盖
particle-free、XPBD、PBF/soft-fluid、pure MPM 与 co-resident cook 的不变量。

## SDF 存储与预烘焙 follow-up（本轮不实现）

`1,148,946,648 bytes, 893 unique/1402 pieces` 表明下一轮应把 SDF 当作通用、
版本化内容资产处理，而不是 BDX 特化：

1. 以 collision mesh content hash、voxel size、band、decomposition 参数、尺度/
   变换约定、cooker/format 版本组成 cache key；预烘焙写入 `.nka` chunk 或
   content-addressed cache，并记录 manifest、校验和和失败回退。
2. 同 mesh/同参数跨 instance、跨 env 只上传一份 device SDF；shape header
   引用共享 atlas/page table，验证生命周期与多 world 并存。
3. 评估 GPU-friendly 窄带 sparse brick atlas（例如 Morton 排列 8³/16³ brick）
   与 page table；比较 dense、现 sparse、brick sparse 的随机访问合并度和
   miss 分支成本。
4. FP16 distance/gradient 或以 voxel 为尺度的定点量化必须单独量化误差。
   测最大/分位 SDF、normal、contact depth/force 误差，并原样运行 Go2/cloth/
   MPM 冻结门；在 byte-identity 约束未批准前默认 FP32，禁止 regen golden。
5. 只依据通用 collision filter、shape kind 和 media-coupling provider 需求烘焙
   真正需要 SDF 的 collidable；解析 primitive 与未参与 provider 的 visual
   mesh 不烘焙。规则不得检查 scene/name/BDX。
6. benchmark 必须同时报告 cook/cache hit/miss/load 时间、disk/host/device
   bytes、N=1/64/256 显存斜率、随机 SDF sample throughput 与 p50/p95 latency、
   contact step time、cache 冷/热启动，以及所有冻结门精确值。

该 follow-up 与 P2G gather 重写均不属于本提交。
