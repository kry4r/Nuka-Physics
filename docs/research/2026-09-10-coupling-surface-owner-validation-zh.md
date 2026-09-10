# 全量耦合：共享表面与真实 owner 验收

更新：2026-09-10。状态：共享几何/owner 基础修复已验收；真实 bunny 渲染与公共接口通过。冲击瞬时穿透、有限质量、时间层及连续表面仍未满足全量耦合验收，总 goal 保持 active。

## 目标与本批边界

按用户最新要求，当前目标是 ABA 机器人、刚体、柔体（布料/体积软体）、XPBD/PBF 和 MLS-MPM 的全量耦合。最终新增机器人提起装有多个动态球的塑料袋、双夹爪拧毛巾两个真实操作 demo，验收抓持、滑移、释放、连续表面与自碰撞、负载/扭矩反馈和渲染。固定顶点、动画或中间刚体不代替这些物理链路。

本批修复各系统共用的几何与真实动力学归属，不能据此宣布全量耦合完成。总矩阵及后续实施见 [全量 spec](../plans/2026-09-10-full-coupling-detailed-spec-zh.md)，当前实现按 [共享几何/owner spec](../plans/2026-09-10-coupling-surface-owner-detailed-spec-zh.md) 验收。主介质输入继续使用 water pool/原 bunny-water 与 jelly，未启动压痕 demo。

## 改造前参考 review

已复核原模块 spec 与细化 spec 的接触、连续介质、provider 部分；已读上游 revision：

| 引擎 | revision | 采用的依据 |
| --- | --- | --- |
| Newton | `31f585713a631d9a87acb5bb332b6a8ba61e2410` | `coupling.rst` 的状态 ownership、有效质量与增量反馈；implicit MPM rasterized collision 的真实表面速度/几何 |
| MuJoCo | `c04c9c726c93852ee3b5b58211ce085f80d98b5d` | `mj_contactJacobian` 由真实 body/flex 插值端点构造同一 Jacobian |
| Genesis | `b3c6c73a7a671fc486df5d69c9f481a91b1d57b6` | legacy coupler 的 link 表面速度与等量反作用；延迟外力不能当作有限质量同时求解 |

另读 Newton v1.5.0 `cca3bb8a17a3620a1343df3cf12c625e4161b317` 的 `example_cloth_franka.py` 和 `example_cloth_twist.py`：前者写入机器人目标速度并临时禁用机器人重力，后者移动非 ACTIVE 边界顶点并求布料自接触。参考其几何/大变形负载，不以这些控制方式验证 Nuka 的 ABA 动态抓持反馈。尚未定位用户描述的塑料袋内小球的准确公开脚本/资产，demo 行为按用户描述独立定义。本批没有新增外部引擎匹配实跑轨迹。

## 生产代码变化与分类

**后端无关架构：** 新增 `src/collision/primitive_surface.hpp` 与 `src/nk/solve/collidable_owner.hpp`。表面查询返回外正内负的距离、单位外法线、表面点和 feature；owner 查询将 collidable 解析为静态、自由刚体或 articulation/link，环境优先排列的映射表使用全局 body 行读取，映射值仍是局部索引。proxy 不另有一份质量或速度。公共 schema 增加可用 SDF/样本数量和状态位，没有 CUDA 类型或 launch 要求。pipeline 依据真实几何能力调度采样算子，缺失数据的 mesh 仍进入诊断。

cooking 只给选中的碰撞形状绑定对应样本/SDF，代理行保留各自几何并更新 hull/sample 容量，不再用同一 body 的其他形状覆盖它。无 SDF 的 mesh/hull 也保留表面样本。visual SDF 不覆盖 authored primitive 或已经绑定的 collision SDF；可视 SDF 给未绑定 mesh 的旧兼容入口仍保留，完整网格身份/拓扑改造另行继续。

**物理算法修复：** SphereBox 的内部球心不再 clamp 到自身；按最近真实面计算法线、表面点和深度。CapsuleSphere 复用同一查询，轴线上的球得到径向分离方向。MPM 对 Sphere/Capsule/Box/Plane 使用精确解析几何，其他已有表示仍可读取 cooked SDF；不再因解析体没有 SDF 而跳过其耦合。解析体优先于其离散 SDF，属于物理/几何变化，不能以旧版字节一致为验收门。

采样 narrowphase 固定遍历两个方向，用实际样本查询对侧解析/SDF 表面，写同一 manifold 和 ContactBlock，保持规范化 a/b、法线和 feature。修正表面点为 `x_surface = x_query - phi*n`，margin 只改变激活深度。真实 bunny 声明为 SdfMesh，包围盒只用于宽相；35,947 个实际顶点支持网格与地面接触。没有把有限采样声称为一般非凸碰撞或薄层 CCD。

**CUDA 实现适配：** row assembly 和 MPM 使用同一 owner 解析。MPM 读取自由体 COM 速度，或将 link-local 的角/线速度旋转到世界，再计算网格节点处的表面速度。几何姿态来自 collidable，质量、惯量与反作用属于真实 owner。网格冲量与对端力矩使用同一个网格节点；link gather 与 generalized deposit 同时使用真实 link 原点作为力矩参考点。

MPM 将代理产生的冲量收集到实际自由刚体，将 link 冲量送入对应 articulation；静态 collidable 的线/角冲量也保存在已有 reaction 字段中，作为外部约束反作用。独立网格 floor/domain walls 的全部外部收支仍未统一，不能将这些读出当作闭系统守恒的完整账本。MPM 和 body-particle 接触跳过 `contype|conaffinity==0` 的禁用形状。

无效 owner、越界端点或未知端点类别会设置 `INVALID_ENDPOINT`（bit 6），保持到 world reset；row assembly 正常清理该 slot 的旧 row，不提前返回留下旧约束。C ABI 和 Python 已公开同名状态，缺失 articulation 映射不再默认为 0。`MPM_ONE_WAY_BODY`（bit 4）为兼容保留名称，其实际含义是“MPM 不支持的几何表示被跳过”；它与 grid escape 每次 MPM op 重置、跨子步累积。

新增 `CONTACT_GEOMETRY_UNAVAILABLE`（bit 7），用于缺失/越界的采样几何、粒子接触不可用的 SDF/heightfield 等，保持到 reset；C/Python 同步导出。粒子 SDF 不再越界读取描述符或以任意方向代替退化梯度。bunny 报告直接量测旋转后的真实网格最低点，输出几何、SDF 分辨率、质量及既有 AABB 箱体近似惯量，并要求全部环境状态为 0。

没有改变 P2G/G2P、本构、迭代预算或新增 workspace；没有新增 CUDA 性能特调。本批不发布加速比，后续性能比较需冻结通过物理检查的新分母。

## 已完成的验证

证据目录：`out/validation/coupling_surface_owner_20260910/`。所有运行有完整命令、退出码和日志，CUDA 环境为 RTX 5080、驱动 610.88、CUDA 13.3；GPU 作业串行运行。

`scenarios_v3.json` 的生产场景合批通过 **62 项、0 skip**，包括 water pool、jelly、MPM transfer 不变量、刚体/关节支承、body-particle、两刚体碰撞与释放、robot-cloth-fluid、MPM/XPBD 共存和多环境 reset。共存场景的通过只证明状态不互相覆盖，不证明存在直接介质界面。

只扩充已有水池落体场景：相同的盒子/流体，在两环境中分别使用有 SDF、无 SDF、无 SDF 且具有局部偏移的 proxy 表示。proxy 版本同时偏移 owner 的 body frame 和 inertial frame，保持真实 COM 与几何相同；每步独立检查 `m*(v_next-v_prev-g*dt)` 等于真实 owner 的 MPM 冲量，且 proxy 不误收反作用。

| 表示 | 峰值向上冲量 N·s | J 范围 | 最大动量记账误差 kg·m/s | 副本速度差 | 状态 |
| --- | ---: | --- | ---: | ---: | --- |
| 有 SDF 的解析盒子 | 0.2607903 | 0.9886–1.4180 | 6.891787e-7 | 0 | 0 |
| 无 SDF 盒子 | 0.2607903 | 0.9886–1.4180 | 6.891787e-7 | 0 | 0 |
| 无 SDF、偏移 proxy | 0.2607881 | 0.9886–1.4179 | 5.550683e-7 | 0 | 0 |

三种表示均在入水后减速，最低竖直速度约 −0.5632 m/s；proxy 自身反作用为 0。全部保持原有 `0.9 < J < 2` 质量门。已有两 link 支承的平均反作用为 0.04013349 N·s，重力冲量为 0.040875 N·s，差约 1.81%；这不是多端点有限质量收敛已完成的证明。

固定 E=16 graph pipeline 完整 450 步通过：最大布料长度应变 `0.01838344`、最坏每环境 RMS `0.00203044919`，原 5%/1% 门通过；`env_status=0`、reset/replay/replica 和逐步 wrench 有效。Data arena 为 `59,741,952 B`，Model arena 为 `2,475,264 B`。该环境流体为 PBF，真实 MLS-MPM 验收来自上述水池/jelly。

已有解析 manifold 套件 20 项通过，补充内部/旋转 Box、Capsule 轴线的独立几何断言。第一次运行的旋转断言错误地用 xyzw 初始化本项目的 wxyz quaternion，已改用 `FromAxisAngle`；`geometry.json` 保留该失败，`geometry_v2.json` 是修正后的结果，没有放宽误差门。

扩充现有落地场景而未新建框架：真实半高 0.10 m 的网格使用 0.40 m 保守 AABB、没有 SDF，两个 a/b 排列都与解析盒子落在 `z=0.0997 m`，末态竖直速度为 0、状态为 0。已有 SDF oracle 5 项通过，增加表面点位于球面/盒面且不受 margin 偏移的独立断言，见 `geometry_v3.json`。

公共 C ABI 耦合 8 项、多环境 5 项（含 4096 环境创建/步进）、相机 4 项通过；Python 耦合/reset 12 项通过、0 skip，新状态导出值为 64/128。既有 `rigid_inputs_pipeline.py` 的 graph 创建、外力、step、读出、reset/replay 和 headless 渲染通过，dt 减半后的自由体位置误差比为 2.0。

基础 memcheck 完整运行水池三种表示、双环境 link 支承、body-particle 和 graph 控制/读出/reset 四项场景，全部通过，`ERROR SUMMARY: 0 errors`。补充 `memcheck_v3.log` 中新网格落地和 graph 控制/reset 均通过、CUDA 0 errors；其总体进程失败来自下述过期 host cook 断言。无 CUDA 的 `nuka_nk`/`nuka_phi2` 最终构建通过，证明共享契约没有引入 CUDA 依赖，不代表其他后端求解能力完备。

保留并处理两组已有元数据断言不一致：`provider_v3.json` 仍要求纯粒子模式发出空操作、PPC 在 finalize 后执行，且要求没有采样几何的盒子发出 SDF 算子；改为当前投影→检测/求解→提交的有效依赖，`provider_v4.json` 9 项通过。`memcheck_v3` 的 MultiGeomCook 要求没有 authored ground 的输入凭空包含尾部地板，改为检查原碰撞形状的几何/owner 均保留，`cook_v4.json` 3 项通过。单独套 sanitizer 跑这 3 个 host case 时，断言全过但工具因未调用 CUDA API 返回 255，保留 `cook_memcheck_v4`，不把它宣称为 CUDA 验收。纯 XPBD 自碰撞和更多粒子组合仍属后续功能，并未因调整算子清单断言标记完成。

## 真实 bunny 渲染与保留的物理问题

首次 `render*` 把兔子 SDF 标为 Box，原 MPM 使用 SDF、刚体使用 Box。解析优先暴露了两种几何不一致，产生 `max_J=9.5222` 的盒子落水轨迹；`candidate_frozen/` 和该运行保留为失败，不将较小数值或时间当作改善。

最终 `render_final.json` 使用真实 SdfMesh：180,000 粒子、110,400 网格节点，dt=1/240 s、40 个 MPM 子步、dx=0.011 m，240 步静置+420 步落水。质量 2.5 kg，SDF 体素 0.0066 m，惯量保留已声明的 AABB 箱体近似。8 帧 640×360 光追输出已检查，诊断量测修正前后的完整末态和全部帧逐字节相同。

| 量测 | 最终结果 | 解释 |
| --- | ---: | --- |
| 环境状态 | 0 | 没有已报告的几何跳过、越界或 grid escape |
| J 范围 | 0.922408581–74.4319077 | 飞溅局部大 J 保留，不能据平均体积门推断全部局部物理正确 |
| 最大平均体积比误差 | 1.80738928% | 通过既有 5% 门，未放宽预算 |
| 真实网格最低点 | −0.00952065364 m | 冲击瞬间仍有约 9.5 mm 穿透；薄层/CCD 验收未通过 |
| 末态真实网格最低点 | −0.000156413764 m | 约 0.156 mm 支承误差；独立从保存的姿态/OBJ 顶点重算一致 |
| 峰值 MPM 向上反作用 | 0.3885551 N·s | 存在双向反作用，仍非有限质量共同解证明 |
| Data / Model arena | 141,999,616 / 653,568 B | 表面样本增加 Model 数据，不发布性能加速比 |

报告保留 `coupling_complete=false`。既有落水范围的 `status.valid=true` 只对应该场景的稳定/体积/飞溅/反作用门，不覆盖 CCD、薄壳、强耦合收敛和完整外部冲量账本。机器人提袋/拧毛巾的验收必须要求更完整的表面、时间层和穿透控制。

## 冻结身份与后续工作

最终冻结目录为 `final_verified/`，包含 source delta、变更源文件归档、完整源码身份和各可执行文件/库；scenario/demo 静态链接求解器，重放使用相应版本的 executable。`final_artifact_verification.json` 确认完整管线验收后只有 demo 诊断与上述两个既有测试断言变化，engine library 和 pipeline executable 不变；末态与 8 帧渲染不变，因此没有重复整批物理验收。`surface_geometry_frozen/`、`final_frozen/` 保留各运行的准确版本；错误 Box 版 `candidate_frozen/` 不能作为最终几何来源。首次未完成的 `frozen/` 无 manifest，不作为验收来源。

| 项目 | SHA256 |
| --- | --- |
| 源码 | `b44400ef8163eb03a96a877430382bd184fc8bb0bc079056e2e8a1721f1f7d79` |
| scenario | `37da3eae944fc61ed60b223bf259932dfdb49d03204ff6b28b4373a9f0ff616e` |
| pipeline | `b64b017ab46402ffdf779c903ccf4d099e18b6f26267b52f0f520f2c8dcdb9ce` |
| bunny-water demo | `9a913fe39ac27beef09cc8ca8303f45deafaec3fb3213aad0579245a7313f717` |
| libnuka.so.0 | `780d049ab2f7ae372477e731f9beafa7f02b2fe289f6c85fc0359a9ce3f2b56e` |

当前仍未完成：无 SDF 凸体/非凸三角网格精确表面、多个同时有效接触端点的有限质量共同解、每真实子步的 articulation 反馈/速度刷新、连续布面/软体表面及点面/边边自碰撞、完整 CCD、PBF↔MPM 和柔体↔PBF/MPM 的直接界面。共享单速度 MPM 网格不能视为所有多材料接触已支持，全局 `ParticleMode` 的组合布局也未证明全介质同时共存。

后续按全量矩阵继续共同端点/质量/时间层及真实表面拓扑，验收上述两项正式操作 demo，随后接续原 spec 的其余工作。历史性能回退和热点保留；不据本批通过关闭整体耦合、性能或总 goal。
