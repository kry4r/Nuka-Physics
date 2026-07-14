# BDX 感知重训 — 后续任务清单与记忆索引

更新:2026-07-10(全停快照)。状态:**PAUSED-ALL**(owner 指令),双 GPU 空闲,等恢复指令。

---

## 0. 当前停点(恢复时先读)

| 项 | 状态 |
|---|---|
| 任务书(已批准) | `docs/plans/2026-07-09-bdx-perception-retrain-plan.md`,已推 GitHub |
| windows-editor | == origin == `016137f`(尾注已清洗,任务书在 tip) |
| 场景分支 | 本地 `worktree-agent-a79c434fcb3e6c705` == 远端 **`oneshot-scene-v2`** == `c540db4`(19 commits:Phase B=73f6286、地面接触修复=d6d15c3、demo WIP=c540db4) |
| **T1 BDX 建模(Fable,被暂停)** | worktree `/data/xtzhang25/_work/activate/wt-t1-bdx`(分支 t1-bdx-modeling):**修复已写、未提交(2 个 dirty 文件,勿 reset)**;Go2 金丝雀已验逐位不动;cloth FNV + MPM 门未跑;质量审计 T1.2 未起 |
| **T2 media 提速(Fable,被暂停)** | worktree `/data/xtzhang25/_work/activate/wt-t2-media`(分支 t2-media-speedup):**T2.1+T2.2 已完成并提交 `6ea559d`**(按需 gate 接触读出 + MpmXpbd 不占接触槽);T2.3 粗粒训练变体未做;实测表未出 |
| 恢复方式 | SendMessage 续跑原 agent(transcript 不丢),或派新 agent 接上述 worktree(T1 先 commit dirty 文件) |
| GPU 约定 | T1→GPU0,T2→GPU1;训练→GPU1;禁同卡并行抢显存 |

---

## 1. 恢复后立即做(等 owner 下令)

1. **T1 续跑收尾**:跑完 cloth FNV + MPM 三门金丝雀 → 提交修复 → T1.2 质量审计(2.107kg/头 0.4066kg,对源核实或修 importer)→ 工程报告。
2. **T2 续跑收尾**:T2.3 粗粒训练场景变体(spacing 0.026 / substeps 7 / 紧 loft,验证 granular 稳定)→ 实测表(N=1/64/256,media-on,对照基线 419ms)→ 工程报告。
3. **⚠️ owner 硬 gate:T1/T2 验收核对后再次暂停**,不自动进入下面任何一步。

## 2. T1/T2 之后的主线(按序,每步默认等放行)

4. **④ 对抗审查**(REFUTE 式,4.8-high ≤20 agent 惯例):范围 = `oneshot-scene-v2` 19 commits + t1-bdx-modeling + t2-media-speedup。
5. **合并 windows-editor + 推送**;随后按站规删除已合并 worktree(wt-t1-bdx / wt-t2-media / agent-a79c434)+ 清各自 build 目录。
6. **T3 传感器(depth+RGB 共同设计;opus 按 controller 规格,media-BLAS 可上 Fable)**:
   - depth-only 快路径(primary hit 早退,通用 flag);
   - **MPM/media 进传感 BVH**(每介质粒子球 BLAS,受 4096 instance/env 硬帽约束;深度和 RGB/成片同一机制);
   - BDX 头部相机(真机机位):64×48 深度训练 + 640×480 RGB 评估/出片/蒸馏;frame-skip 旋钮;
   - obs 接线(44+patch 拼 flat MLP,冷启动)+ RL media 入口改走 SceneBuilder.build;
   - 开工日前置核查:真场景 sensor 吞吐复测(7.6ms 是合成场景数)、走廊 instance 数审计、`corridor_nomedia.nks` 补 `.nka`。
7. **T4-A rigid 走廊深度直训**:N=4096(实测 36.5k env-steps/s,≈6h/轮,GPU1);imitation 权重离台阶 gate(W_JOINT_POS=15 平地先验在对抗下台阶=必修);gate = realsim 真闭环贯穿两个 5cm 台阶不摔、均速 ≥0.12m/s、抵 x≥3.8m。
8. **T4-B media-on 微调**:N=64-256 warm-start(需 T2 提速 + T3 media 可见);碎石段加深床(**碎石没脚走**)+ **留痕**(MPM 塑性本有);gate = 稳步穿越整段碎石、痕迹保留、rigid 段无回退。
9. **T4-C RGB 蒸馏学生**:特权/深度教师 → DAgger 学生(640×480 渲染 → resize 224² → 冻结 SigLIP-B/ViT-S embedding,N=128-256);管线直接复用到 π0.5 zero-shot + IHI 轨道;gate = 学生 realsim 贯穿等同教师。
10. **T5 场景视觉提升 + 终版出片**:美术债清单 = 碎石不规则粒形、绳渲成光滑管(非珠链)、吊板换好看石板、提亮远端暗部、鸭子去塑料感/去噪、封远墙天带;然后连续一镜到底真物理渲染(sweep2 机位)→ 控制器逐帧 Read 审 → owner 终审。

## 3. 具名债 / 背景项(不阻塞主线)

- **rolling-friction 通用材质属性**(task#15,动全局求解器 → owner 级决策)。
- **heightfield 双重烘焙债**(.nks TerrainRecord + contact_family=1 各烘一份,debug assert 会崩;哪天走 heightfield 路时治)。
- `fields.yaml:782` 注释 `*6`→`*9`(纯文档,零行为)。
- XPBD body-particle 对带 silhouette-SDF 的体仍走 primitive(视觉头可能轻微扎门帘;渲染质量债)。
- render 债:studio 地板无条件注入(studio_beauty.cpp:171)、cook visual-twin shape 的 body_row 串线待审。
- CUDA 并行光追演进(task#6,最后)。
- R5 报告里标"推断"的数字(π0 token 数/Helix 编码器细节)进正式对外材料前抽查原文。
- 预存红(不修,owner 已裁):2× CoupledWorldCAbi、FeatherstoneOracle.NkWorldBatchedContactStepPlannedByteExact、V01FoundationE2E.Phase6。

## 4. 关键路径速查

| 用途 | 路径 |
|---|---|
| 产物/构建总目录 | `/data/xtzhang25/_work/activate/`(build-scene2 + build-scene2-pyext = LIVE;tmp/、out/) |
| v9 checkpoint | `/data/xtzhang25/_work/activate/out/bdx_walk_v9/nn/last_bdx_walk_v9_ep_2000_rew_447.30893.pth` |
| 走廊场景 | 场景 worktree `examples/scenes/bdx_oneshot.nks`(5cm 正典,粗碎石 0.013) |
| 真闭环 harness(未跟踪) | 场景 worktree `examples/demo/bdx_oneshot_realsim.py`、`bdx_oneshot_floortest.py` |
| R4 性能探针 | `/data/xtzhang25/_work/activate/tmp/bdx_*probe.py` |
| 运行器 | `runpy_gpu.sh <gpu> <script>`(绑 agent-a79c434 worktree + build-scene2;其他 worktree 需改副本) |
| BDX 源模型 | `/root/third_party/Open_Duck_Playground/playground/open_duck_mini_v2/xmls/open_duck_mini_v2.xml` |
| 分镜/场景规格 | `docs/storyboard/nv-BDX/`(bdx-sb.md + oneshot-scene-plan-v2.md) |

## 5. 重要记忆文件夹索引

位置:`/root/.claude/projects/-root-Nuka-Physics/memory/`(`MEMORY.md` = 总索引,每次会话自动加载)。与本项目后续直接相关的:

| 文件 | 内容 |
|---|---|
| **bdx-oneshot-demo.md** | ★★ 本项目主 LIVING 记忆,**读我优先**:分镜 D01-D04、训练 v2→v9 全史、引擎修复链(FK 轴/EPA/地面接触/混合 world)、五路调研结论(R1-R5)、owner 全部指令与 gate、当前停点(续11) |
| activate-artifacts-dir.md | 产物目录规约(/data/.../activate)、build 目录现状、worktree 合并后即删的生命周期 |
| subagent-model-opus48-max.md | 模型分层:简单调研→Sonnet5 / 训练+照方改脚本→Opus / 建模/提速精细活→Fable |
| subagent-350k-relay-rule.md | subagent 上下文 >400k 必须返回接力 |
| v03-git-push-procedure.md | git-lfs 必须在 PATH、代理三备选(121.4.45.119:31157 → 192.168.2.185:7897 → 192.168.2.5:7897)、[skip ci] 在 body、禁 Co-Authored-By |
| go2-golden-canary-semantics.md | Go2 0.923080623 = 冻结字节恒等金丝雀,EXPECT"失败"~0.92 是设计如此,只有数值移动才算回归 |
| per-milestone-ultracode-adversarial-review.md | 每里程碑必须 REFUTE 式对抗审查后才 commit/合并 |
| render-verify-view-images.md | 渲染验收必须控制器逐张 Read 看图逐缺陷分析,探针绿≠没有可见缺陷 |
| unified-world-no-special-grasp-binding.md | ★★★ 最高指令:ONE 通用物理求解路径,禁 case 特惠 |
| offline-rhi-render-plan.md | 批量 GPU 传感核 D2.2 全史(get_sensor_view / 零拷贝 / 已知 follow-on) |
| vla-inference-engine-plan.md | π0.5 VLA zero-shot + IHI C++ 推理引擎轨道(T4-C 管线的下游) |
| mpm-direction-decided.md / mpm-fluid-constitutive-design.md | MPM 混合路线拍板 + fluid 本构设计 |
| windows-editor-box.md | Windows 编辑器盒子(ImGui/Vulkan)状态与验证方式 |
| go2-stairs-demo.md / go2-stairs-climb-gap.md | go2 台阶训练/height-scan/课程修复先例(v10 感知训练的参照) |
| v07-build-run-env.md / v03-go2-training-single-gpu.md | 老构建环境与单 GPU 训练纪律(背景) |
| v05/v07/v08-progress.md 等 pointer | 各版本全史指针(背景,不常用) |

---

## 6. 新会话开启指令(直接粘贴,不含任何 skill/斜杠命令)

```
继续 BDX 感知重训项目(全停后恢复)。动手前先完整读三份文件:
1. /root/.claude/projects/-root-Nuka-Physics/memory/bdx-oneshot-demo.md(主记忆,重点 续8-续11:五路调研结论、任务书、停点、我的硬 gate)
2. /root/Nuka-Physics/NEXT-TASKS.md(任务清单+停点快照+路径速查)
3. /root/Nuka-Physics/docs/plans/2026-07-09-bdx-perception-retrain-plan.md(已批准任务书)

读完后恢复两个被暂停的 Fable 任务收尾(旧会话的 agent 续不上,派新 Fable 接各自 worktree):
- T1 BDX 建模:worktree /data/xtzhang25/_work/activate/wt-t1-bdx(分支 t1-bdx-modeling),里面有已写完但未提交的修复(2 个 dirty 文件,先读 diff 理解再继续):跑完 cloth FNV + MPM 金丝雀门(Go2 已验过)→提交→做 T1.2 质量审计→报告。
- T2 media 提速:worktree /data/xtzhang25/_work/activate/wt-t2-media(分支 t2-media-speedup),T2.1+T2.2 已提交(6ea559d):做 T2.3 粗粒训练场景变体+实测表(N=1/64/256,基线 419ms)→报告。

硬规则:T1/T2 完成并验收核对后立即暂停等我指令(不自动进对抗审查/合并/T3);冻结金丝雀逐位不动;subagent 分层=简单调研 sonnet5/照方改脚本 opus/建模提速精细活 fable;所有编译与临时产物放 /data/xtzhang25/_work/activate/;所有 git 操作先 export PATH=/root/.nuka-toolchain-gcc14/bin:$PATH(git-lfs);commit 带 [skip ci] 于 body、禁 Co-Authored-By、不推送;T1 用 GPU0、T2 用 GPU1;用中文跟我交流。
```

补充硬规则(2026-07-14):只用 GPU1 且 GPU 任务串行;不调用 skill;不再新建 worktree;求解器/MPM/SDF 只能走通用 ONE-path,禁止按 BDX/MpmXpbd 特化;SDF 要规划预烘焙、稀疏/分块和缓存复用;不再新增单元测试,后续盘点删除无效单测,只写可重复运行的 pipeline/scenario/regression gate;subagent 仅 5.6 sol max,分析可 ultra,模型不可保证时不派。

*本文件为工作清单快照,不自动更新;权威动态状态在 memory/bdx-oneshot-demo.md 与会话任务板。*
