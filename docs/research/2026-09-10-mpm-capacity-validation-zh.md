# MLS-MPM 粒子与节点工作区容量验收

MPM 工作区此前按 `max(P,N)` 同时分配粒子记录和节点索引，粒子密集时多分配节点数组，网格稀疏时多分配传输记录。本批把实际 MPM 粒子数 P 与网格节点数 N 分别传入容量查询，回收这些多分配。原 bunny-water 的 Data arena 减少 1,392,640 B，完整步时基本持平；本批作为容量优化验收，不计作求解加速。

## 实现与分类

后端无关的改动位于 `src/nk/pipeline/world.cpp` 和 `src/phi/op_schema.hpp`：工作区查询分别接受 P、N，保留既有整型容量保护和零 MPM 行为。核心没有新增 CUB 类型、CUDA launch 参数或设备布局要求。

CUDA 的具体划分位于 `src/phi/backend_cuda/ops/mpm.cu`：传输记录按 P，五个节点数组按 N；排序 key/idx 输出在粒子与节点阶段顺序复用，仍按 `max(P,N)`。缓存以 device/P/N 为键，分别查询 sort(P)、sort(N)、select(N) 的临时空间并取最大值，不假设大数量查询可以替代小数量查询。P2G 内核、记录格式、稳定排序位宽、浮点运算、材料、子步和调度保持已验收版本。

公共 CUB 查询沿用完整 key 位宽作为各自数量的 radix 上界。已核对 CUDA 13.3 `dispatch_radix_sort.cuh` 的单 tile、onesweep 和 multipass 分配逻辑；没有调用私有策略 API。全局 workspace 查询的后端分派仍属架构债，两数量契约不代表完整跨后端内存规划或 CPU MPM 求解器已完成。

## 容量结果

原 bunny 为 P=180,000、N=110,400。十对进程均得到 Data arena 95,699,456 → 94,306,816 B（−1.46%），Model arena 保持 221,952 B；其中 MPM 工作区从 27,387,648 降为 25,995,008 B（−5.08%）。这包含各区段的实际对齐差额。

另在独立进程中调用两份冻结生产库的容量查询，确认数量比例改变时的请求量：

| P | N | 原工作区 B | 新工作区 B |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 |
| 1 | 216 | 33,536 | 8,704 |
| 180,000 | 110,400 | 27,387,648 | 25,995,008 |
| 1,000 | 1,000,000 | 152,121,600 | 36,237,824 |
| 1,000,000 | 1,000 | 152,121,600 | 132,142,080 |
| 110,400 | 110,400 | 16,798,976 | 16,798,976 |

1,000 粒子、1,000,000 节点的工作区请求量减少 76.18%。该表只量测生产查询的字节数，没有据此宣称这些配置完成仿真、实际分配或显存峰值验收；原 bunny 的 Data arena 才来自完整运行。记录见 `workspace_baseline.json` 和 `workspace_candidate.json`。

## 五对完整步时

分母为 `out/validation/mpm_gather_20260910/logical_group_frozen`，候选为 `out/validation/mpm_capacity_20260910/capacity_frozen`。eager/graph 各五对独立进程，交替 A/B、B/A；每个进程完成原 240 步静置和 420 步落水，180,000 粒子、110,400 节点、40 子步、dt=1/240 s、dx=0.011 m。

运行环境为 RTX 5080、驱动 610.88、CUDA 13.3。计时为完整 World step 的 GPU 完成时间，上传、下载和质量扫描在计时外；每个进程绑定冻结库并保存设备时钟、温度、功耗和完整命令。下表是五个进程平均步时的中位数。

| 执行 | 阶段 | 原 ms/步 | 新 ms/步 | 耗时变化 | 逐对变化范围 |
| --- | --- | ---: | ---: | ---: | ---: |
| eager | 静置 | 23.6556 | 23.5921 | −0.27% | −1.81% 至 +1.10% |
| eager | 落水 | 24.4418 | 24.5103 | +0.28% | −0.96% 至 +1.30% |
| graph | 静置 | 18.8965 | 18.9235 | +0.14% | −0.09% 至 +0.14% |
| graph | 落水 | 20.3471 | 20.3432 | −0.02% | −0.24% 至 +0.15% |

落水 p50/p95/p99 的五进程中位数为 eager 24.377/26.395/27.941 → 24.419/26.455/27.995 ms，graph 20.514/21.380/21.627 → 20.495/21.356/21.693 ms。没有显著步时收益，也没有超过 5% 的完整步时回退。

创建耗时中位数为 eager 25.243 → 26.514 ms、graph 26.541 → 26.232 ms；graph capture 为 69.425 → 68.052 ms。新契约在每种 device/P/N 的首次初始化增加一次公共 sort 容量查询，后续步进复用缓存。这些模式间不同方向的小差异不支持创建加速结论，初始化成本随结果保留。

## 物理与管线验收

十对完整状态和质量字典全部一致。轨迹 FNV 保持 `6cf2c38c57c8f590`，末帧 SHA256 保持 `80d6a2404b91bd38afd478b7cd3b8ccd92e0105a2392fb12bab2875a04d8f872`。最小 J=0.935678005，总体积最大偏差为 1.6673171%，原 5% 预算没有放宽。

复用的 33 项 MPM 场景全部通过，两个原禁用诊断保持禁用，没有新增单测。覆盖静水、jelly、颗粒介质、刚体/关节支承及 MPM/XPBD 共驻；既有静水网格质量与粒子质量同为 5.832000，相对误差为 `3.285e-8`，独立检查了传输的质量守恒。

针对性 memcheck 的十项场景均为 0 errors。除尾块、非法坐标、混合介质、多环境和偏心反作用外，加入已有 `GridMassMomentumDeterministicAndConserved` 场景覆盖 P>N 的节点数组收缩；现有 transfer 和共驻场景覆盖 N>P 的记录收缩。同步算法未改动，没有重复上一批的 synccheck 矩阵。

固定 E=16 robot-cloth-fluid graph 的全部 450 步通过，env_status=0，reset、重放、副本等价和逐步 wrench 检查通过；最大长度误差 1.838344%，最坏每环境 RMS 0.203044919%。该主环境使用 PBF，真实 MPM 验收来自上述 bunny 和 MPM 场景。无 CUDA 的 `nuka_nk`、`nuka_phi2` 核心库构建通过，只证明依赖隔离。本批没有重新进行渲染验收。

既有场景套件首次墙时为 76.725 s，高于历史 72.817 s；补充相邻旧/新进程得到 73.664/71.770 s，没有复现稳定的整套回退。对曾出现差异的 MPM/XPBD 共驻案例再做五对，进程墙时中位数为 1.544 → 1.449 s，逐对 −15.62% 至 +3.22%。该时间包含初始化、创建、步进、读出和断言，波动明显，不能据此宣称稀疏环境的 GPU step 加速。原始结果及 `co_step_pairs/summary.json` 保留。

## 未采用的读取候选

容量改造前，按相同物理输入验证了三个 CUDA 读取候选。完整状态和质量均相同，但没有形成足够的净性能依据，生产 P2G 保留原协作读取。

| 候选 | graph 静置/落水初验 ms/步 | Data arena B | 决定 |
| --- | ---: | ---: | --- |
| 逐粒子直接读，保留共享贡献 | 45.208 / 45.714 | 95,699,456 | 明显回退，拒绝 |
| 每线程直接计算一个守恒分量 | 18.610 / 20.069 | 95,699,456 | 单次差异约 1.4%，未确认稳定收益，未采用 |
| 分量直接计算与完整权重缓存 | 23.414 / 24.478 | 102,179,328 | 步时和容量回退，拒绝 |

前两项完整 profile 中，P2G 为 37.162/11.481 ms 每落水步，分别使用 48/40 registers 与 3,584/0 B shared memory，均无 local spill。已验收版本对应为 11.880 ms、67 registers、14,976 B shared。减少静态资源本身不保证完整性能；实际带宽、stall、occupancy 的硬件计数器仍不可用，未填入推测值。

三份冻结源码/二进制、完整 probe、状态及前两份 profile 位于 `out/validation/mpm_direct_read_20260910/`。未对明确失败候选扩展五对或全套回归，也未提交候选切换分支。

## 边界与重放

改造前再次 review spec，并查询 Newton `e8974b314b60d2c10cb4d02a4d1d60dadb7a54e1`、MuJoCo `c04c9c726c93852ee3b5b58211ce085f80d98b5d`、Genesis `b3c6c73a7a671fc486df5d69c9f481a91b1d57b6`。MuJoCo 增量只扩展 Python rollout 的整数参数类型，相关 MPM/耦合/约束实现未变。本批不把参考引擎的 atomic scatter 或隐式求解替换进当前有序显式 MPM。

局部 max J=79.4880753 仍待解释，`env_status_union=16`、`coupling_complete=false` 保留。无 SDF 地板、single owner、有限质量、复合形状实际 owner、关节子步反馈及 MPM↔XPBD 直接交换尚未完整实现。存储身份和局部验收不将这些限制变成正确结果。已有 P2G 热点、稀疏网格及其他剩余性能问题继续优先推进，再处理完整耦合和原 spec 后续项。

最终证据目录为 `out/validation/mpm_capacity_20260910/`，包含 `review.json`、`capacity_paired/summary.json`、场景与补充对照、完整 pipeline、memcheck、核心构建和容量查询。冻结 source SHA256 为 `0256f3f5bbc131692113a91acff2742031d75d470b2392ea359faff69881910d`，library 为 `24ec87403ac50e354f04361f02be2bd3f62355315ec60d138aa54284b6602e96`，demo 为 `fcd560c4918cafb9f1f9835f9068d95b0d4d5ff9050859c7a03dd485514c1dc8`。

```text
python .nuka-runs/measure_mpm_pairs.py --baseline out/validation/mpm_gather_20260910/logical_group_frozen --candidate out/validation/mpm_capacity_20260910/capacity_frozen --output <新结果目录>
```
