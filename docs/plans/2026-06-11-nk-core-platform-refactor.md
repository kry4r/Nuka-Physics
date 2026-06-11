# Nuka-Physics 重构总计划 v3（终版，无回退方案，文件级施工图）

> 状态：owner 已批准方向；本 v3 为交叉评审（ggml / Genesis / Newton / Isaac Lab）后的终版。**写盘后暂停，等 owner 开工信号。** 开工后第一件事：本文件已入库为 `docs/plans/2026-06-11-nk-core-platform-refactor.md`，以它为多 session 执行基准。
> 纪律：禁止 TDD、禁止新增单测；每个里程碑 = 交付物 + 验收门（oracle/场景/性能），门绿才前进；**本计划不含任何回退方案，每个设计点只有一条承诺路径**。
> **修订 2026-06-11（owner，开工后）**：§3.6 场景树化——场景以**节点树（SceneGraph）**形式编辑与管理，原生承载机器人层级；`name` 只是单段名字、`path` 由树派生专管；基底参考 owner 的 Mangifera `core/manager/scene-graph.{hpp,cpp}` 并在其上扩充。波及 §2 / §3.7 / §3.9 / M2（已就地改写）。

---

## 0. 交叉评审结论（v2 → v3 的变更与依据）

对照 ggml/llama.cpp、Genesis、Newton、Isaac Lab 逐项评审后，v2 计划做出以下修订：

| # | v2 设计 | v3 终版设计 | 依据 |
|---|---|---|---|
| 1 | 后端缝 = `nk::ops` 函数集 + **link-time** 选择后端库 | **ggml 式运行时后端**：`phi::BackendI` 函数表 + `BufferTypeI/BufferI` 缓冲抽象 + 设备/后端注册表 + `supports_op` 能力查询 + `Plan`（CUDA Graph 编译执行路径）。后端可独立编译、运行时注册（结构上支持 dlopen，本期静态注册） | owner 指令（"类似 llama.cpp ggml 多后端"）；ggml 八原则 P1-P8（单一 op IR、buffer 归后端所有、能力先查询、host 侧调度、单后端零开销、plan 快路径、registry 解耦、device≠backend 实例） |
| 2 | 物理 kernel 放 `src/nk/kernels/` | **全部物理 kernel 是后端的 op 实现**，放 `src/phi/backend_cuda/ops/`（ggml 把所有 op kernel 放在 ggml-cuda 内，同构）。`src/nk/` 引擎侧零 CUDA token | ggml P1/P7；这才是"one PHI multi backend"的字面兑现：换后端 = 换 op 表实现 |
| 3 | Stage 虚类每步 `Run()` | **Pipeline = 构建期生成的 `OpCall` 线性列表**（物理步是固定管线不是任意 DAG，不复制 ggml 的 5-pass 图切分调度器）；运行 = 逐条 dispatch 或 `Plan` 整体重放 | ggml 报告 §7c "What NOT to copy"；Newton `wp.ScopedCapture` 整步捕获同型 |
| 4 | M2 解算含回退方案（grid-sync 失败→color 扇出） | **唯一承诺路径：block-per-island 融合解算 kernel**（见 §3.4 论证），删除 cooperative-launch/grid-sync 选项 | owner 指令（禁止回退）；论证：180 row/env 在单 block 内 color-串行/row-并行是 MuJoCo 单核可解规模，现 58–1400ms 是实现病灶不是架构上限；批量下 island 数=N×env 内分量数→天然多 block 占满 GPU |
| 5 | GJK/EPA "推迟"（默认决策可否决） | **承诺：mesh 接触 = 预烘焙 SDF 采样，解析对 = 原语专核；GJK/EPA 全删**（cook 期凸分解保留用于 hull 资产与 SDF 烘焙源） | Genesis（非凸 mesh 全走预烘焙 SDF 栅格，`sdf_cell_size/min_res/max_res` 挂在 Rigid 材质上）+ Newton（SDF 碰撞一等公民）+ PhysX5 SDF mesh 碰撞先例 |
| 6 | 资产组件未含视觉网格（"v1 用碰撞几何渲染"） | **VisualMesh + RenderMaterial 从 M2 起即一等组件**；实体五组件模型（Transform / VisualMesh+RenderMaterial / CollisionShape+PhysicsMaterial / RigidBody / Joint+Actuator），物理材质与渲染材质**正交、双绑定** | owner 指令（渲染产物要 visual、双材质、1:1）；Isaac Lab `RigidBodyMaterialCfg` vs `PreviewSurfaceCfg` 正交 + UsdShade purpose="physics"/"render" 双绑定模型；Genesis morph/material/surface 三元组 |
| 7 | 场景文件 = `.scene.json` sidecar | **自有资产格式 `.nks`（场景 config-graph JSON）+ `.nka`（二进制资产容器）**，导入(MJCF/URDF/USD)→Registry→可导出 .nks/.nka，加载与运行时无源格式依赖；带 override layer 机制 | owner 指令（自有资产保存产物、场景导入导出解析、物理参数经物理材质附加）；Isaac T12（自有格式应为 config-graph 而非 USD 依赖）+ T10（转换器懒缓存）+ USD layering 思想 |
| 8 | 域随机化未设计 | **材质桶**：设备常驻 `(num_buckets, k)` 物理材质表 + 每 env/body 桶索引；接触 row 携带 material_id 查表；随机化 = 一次 scatter op | Isaac T5（bucketing），与现 `CookedContactParamTable` 同构升级 |
| 9 | 死代码 M7 才删 | **M0 大清扫**：无生产消费者的 5 条路径 + 依赖测试 + apps 立即删（~12k LOC） | owner 指令（能删就删，大刀阔斧） |
| 10 | （隐含）可能引入 Newton 式 solver 插件多求解器 | **不采用** solver-plugin 多求解器与 Newton 双缓冲 State：统一 row solve 是 owner 主张（耦合在 row 层），PGS 顺序冲量天然 in-place，单 Data 省一半显存；Control 作为 Data 内独立段 | Genesis coupler 证明"共享显存 + 行级耦合"可行；Newton 双缓冲服务于其插件可替换性，与我们的统一解算冲突 |

其余 v2 决策维持：固定容量+水位掩码、arena=persistent/scratch/tape 三块、codegen 拥有字段注册表、SceneIR 保留 facade、覆盖清单制删单测、settle 数据化、RL 冻结至回归里程碑。

---

## 1. Context 摘要（审计结论，详证据见 8-agent 扫描）

src 73.5k / tests 70.7k / python 13k / tools 26k LOC。①PHI 只兑现分配收口（cudaMalloc 全部在 `phi/backend_cuda/buffer.cu`），136 裸 `<<<>>>`、52 裸 memcpyAsync、46 裸 memset、graph capture 散落全仓。②无 arena，`ArticulationDeviceBuffers` 28 个独立 Buffer，6+ 种 world↔solver 配对各写各的，`UnifiedSolve` 每步全量 host↔device 往返。③`SceneIR→Compose→Cook` 管线存在但被 H1 场景绕开（1,026 行工厂 + ~120 魔法数 + `BatchedSceneTemplate` 平行路径），cook 后名字丢失。④15 条 stepping 路径，生产脊柱 = `BatchedArticulatedWorld`(快, 0.95µs/env-step@4096, 设备常驻+CUDA graph) + `BatchedUnifiedWorld`(慢, union 场景 665ms/step@N=32, 97.6% 在 row_solver：每步 CPU 重着色 140-200ms + 全量 PCIe 往返 + N=1 单 block + CPU narrowphase + 每步 ~35MB host 堆分配)。⑤187 测试文件 / 116 可执行（各自重新初始化 CUDA），109 文件 ~40k LOC 为 TDD 单测；高价值 = 49 oracle/golden + 9 场景。⑥渲染只有离屏 compute-shader 2D debug 线框，无 viewer、无互操作，批量 world 从不推位姿给渲染。

**Owner 拍板**：PHI 本期只实现 CUDA、契约按多后端（ggml 式）；并行新核心，单块显存原生支持 机器人/刚体/柔体/流体 多耦合；SG spec 冻结；可视化 = Vulkan 本地(imgui)+离屏，先离屏；渲染要 visual 网格 + 双材质 + 1:1；要自有资产格式。

---

## 2. 终态架构总览

```
src/phi/                        ★ PHI v2 = ggml 式多后端层（"one PHI multi backend"的本体）
  ├─ backend.hpp                BackendI/DeviceI/RegistryI 函数表 + 句柄类型
  ├─ buffer.hpp                 BufferTypeI/BufferI（取代现 phi::Buffer 的角色）
  ├─ op_schema.hpp              NkOp 枚举 + 每 op 参数 POD 结构（op IR，后端无关）
  ├─ plan.hpp                   Plan 句柄（编译执行路径 = CUDA Graph）
  ├─ event.hpp  registry.cpp    事件、全局注册表
  └─ backend_cuda/              CUDA 后端（本期唯一实现）
      ├─ cuda_backend.cu        vtable 实现：dispatch/plan/event/sync
      ├─ cuda_buffer.cu         buffer_type + buffer 实现（现 buffer.cu 吸收于此）
      └─ ops/                   ★ 全部物理 kernel（每文件 = 一组 op 实现）
src/nk/                         新统一核心（零 CUDA token，只 include phi/）
  ├─ model/                     nk::Model + fields.yaml + codegen 产物
  ├─ data/                      nk::Data（arena：persistent/scratch/tape 三块）
  ├─ pipeline/                  OpCall 列表构建 + World（Step/StepPlanned/Reset）
  └─ solve/                     SolveSchedule（构建期着色/island 划分缓存）
src/scene/                      场景树 + ECS Registry + 五组件 + 双材质 + SceneIR facade
  ├─ graph/                     SceneGraph 场景树（first-child/next-sibling + path 寻址，Mangifera 基底）
  ├─ ecs/                       entity/components/registry
  ├─ asset/                     .nka 容器读写 + AssetCache（内容寻址）
  ├─ format/                    .nks 场景文件读写（serialize/deserialize/override-layer）
  ├─ cook/                      CookToModel + SceneMap + settle + placement
  └─ （scene_ir/compose/cooker 既有文件改造保留）
src/render/                     RenderWorld + raster/（Vulkan 离屏 3D）+ rt 适配
src/runtime/app/                Simulation 帧循环 + 系统 + CommandQueue（viewer/ 后期）
src/import/                     MJCF/URDF/USD importers（升级：读视觉几何+材质）
src/diffsim/                    保留，指针源改 arena DiffVisible 字段
src/c_abi/ python/              单一 nuka.Scene/World/Recorder(/Viewer)
tests/{oracle,scenario,perf,fixtures}/   5 个可执行
```

---

## 3. 核心设计规格（实现必须逐字遵循的契约）

### 3.1 PHI v2：ggml 式后端层

**`src/phi/backend.hpp`**（纯 C++ 头，无任何 CUDA 类型）：

```cpp
namespace nuka::phi {

struct Backend; struct Device; struct RegistryEntry; struct Plan; struct Event;
struct ModelView; struct DataView;            // 由 nk codegen 生成，phi 只前向声明

enum class Status : uint8_t { Ok, Unsupported, Failed, OutOfMemory };

struct OpCall {                                // op IR 的执行单元
    NkOp        op;                            // src/phi/op_schema.hpp
    const void* params;                        // 指向该 op 的 POD 参数结构（生命周期=Pipeline）
};

struct BackendI {                              // ggml_backend_i 对应物
    const char* (*get_name)(Backend*);
    void        (*free)(Backend*);
    // 单 op 派发：读取 ModelView/DataView（设备指针视图）+ params，发射 kernel
    Status      (*dispatch)(Backend*, const ModelView&, const DataView&, const OpCall&);
    void        (*synchronize)(Backend*);
    // 编译执行路径（= CUDA Graph）：capture 整条 OpCall 序列
    Plan*       (*plan_create)(Backend*, const ModelView&, const DataView&,
                               const OpCall* calls, int n_calls);
    Status      (*plan_execute)(Backend*, Plan*);
    void        (*plan_free)(Backend*, Plan*);
    // 事件（跨流序）
    Event*      (*event_new)(Backend*);
    void        (*event_record)(Backend*, Event*);
    void        (*event_wait)(Backend*, Event*);
    void        (*event_free)(Backend*, Event*);
};

struct DeviceI {                               // ggml_backend_device_i 对应物
    const char* (*get_name)(Device*);
    void        (*get_memory)(Device*, size_t* free_b, size_t* total_b);
    Backend*    (*init_backend)(Device*, const char* params_json);
    BufferType* (*get_buffer_type)(Device*);
    BufferType* (*get_host_buffer_type)(Device*);
    bool        (*supports_op)(Device*, NkOp);            // 构建期能力查询
};

struct RegistryEntryI {                        // ggml_backend_reg_i 对应物
    const char* (*get_name)(RegistryEntry*);
    size_t      (*device_count)(RegistryEntry*);
    Device*     (*get_device)(RegistryEntry*, size_t i);
    void*       (*get_proc_address)(RegistryEntry*, const char* name); // 可选扩展点
};

// 全局注册表（src/phi/registry.cpp）：
void          RegisterBackend(RegistryEntry*);   // CUDA 后端经静态初始化注册；结构兼容未来 dlopen
size_t        BackendCount();  RegistryEntry* GetBackend(size_t);
Device*       InitBestDevice();                  // 本期 = 第一个 CUDA device
}
```

**`src/phi/buffer.hpp`**：

```cpp
struct BufferTypeI {
    const char* (*get_name)(BufferType*);
    Buffer*     (*alloc)(BufferType*, size_t bytes);     // 后端拥有显存
    size_t      (*alignment)(BufferType*);               // CUDA 后端返回 256
    bool        (*is_host)(BufferType*);
};
struct BufferI {
    void  (*free)(Buffer*);
    void* (*base)(Buffer*);                              // 设备基址
    void  (*upload)  (Buffer*, const void* src, size_t off, size_t n);   // 异步，后端流
    void  (*download)(Buffer*, void* dst,       size_t off, size_t n);
    void  (*memset)  (Buffer*, uint8_t v,       size_t off, size_t n);
    void  (*copy_from)(Buffer* dst, Buffer* src, size_t doff, size_t soff, size_t n); // D2D
};
```

迁移规则：现 `phi::Buffer/OwnedStream/StreamView/DeviceContext/UploadVector/DownloadVector` 在迁移期保留供遗留路径使用，M9 随遗留路径删除；新核心一律走 BufferI。CUDA 后端内部持有 1 条主流 + 1 条捕获流（吸收现 `owned_stream.cu`），52 处裸 memcpyAsync/46 处 memset 的职能由 `BufferI.upload/download/memset/copy_from` 与 op 实现内部接管。

**lint 红线**（`tools/lint/banned_patterns.yaml` 新增，scope `nk_engine` = `src/nk/**`、`src/scene/**`、`src/render/**`、`src/runtime/app/**` 的全部 `.hpp/.cpp`）：禁 `<<<`、禁 `\bcuda[A-Z]\w*\s*\(`、禁 `#include <cuda_runtime`、禁 `#include "phi/backend_cuda`；`hot_path_cuda_malloc` 扩到 `src/phi/backend_cuda/**`（op 实现内禁分配）。

### 3.2 op 模型与 Plan

**`src/phi/op_schema.hpp`**——op 枚举（初始集 28 个，新增物理能力 = 加 op + 参数结构 + 后端实现 + supports_op）：

```cpp
enum class NkOp : uint16_t {
  // 控制/动力学
  ApplyDrives, AbaForward, IntegrateVelocity, FkWorldPoses, IntegratePosition,
  CrbaComputeM, CrbaFactorM,
  // 碰撞
  BuildAabbs, LbvhBuild, LbvhQueryPairs, ParticleGridBuild,
  NarrowphasePrimitives,        // sphere/capsule/box/plane 解析对全集（一个分发 kernel 族）
  NarrowphaseSdf,               // 顶点/特征点 × 预烘焙 SDF（mesh 接触唯一路径）
  ContactTangentBasis,
  // 行装配与统一解算
  AssembleRows, SolveRowsBlockIsland,
  // 粒子（XPBD/PBF）
  ParticlePredict, XpbdProject, PbfDensityLambda, PbfApplyDelta, ParticleFinalize,
  // 读出/重置/随机化
  ReadoutContactWrench, ExportObs, ResetEnvs, SnapshotState, RestoreState,
  RandomizeMaterialBuckets, RandomizeBodyParams,
  Count
};
// 每 op 一个 POD 参数结构，命名 <Op>Params，集中定义在本头；例：
struct SolveRowsBlockIslandParams { float dt; uint16_t vel_iters; uint16_t pos_iters; };
struct NarrowphaseSdfParams      { float contact_margin; uint8_t max_contacts_per_pair; };
```

**Pipeline 与 Plan**：`nk::Pipeline::Build(Model)` 按 Model 含有的系统产出 `std::vector<OpCall>`（固定顺序：ApplyDrives→AbaForward→IntegrateVelocity(+ParticlePredict)→FkWorldPoses→BuildAabbs→LbvhBuild→LbvhQueryPairs(+ParticleGridBuild)→NarrowphasePrimitives→NarrowphaseSdf→ContactTangentBasis→CrbaComputeM→CrbaFactorM→AssembleRows→SolveRowsBlockIsland(+XpbdProject/PbfDensityLambda 内联为统一解算前置 op)→IntegratePosition(+ParticleFinalize)→ReadoutContactWrench）。无该系统的 op 不进列表。`World::Step()` = 逐条 `dispatch`；`World::StepPlanned()` = 首次 `plan_create`（CUDA 后端内 `cudaStreamBeginCapture` 逐条 dispatch 到捕获流后 `cudaGraphInstantiate`），之后 `plan_execute`（`cudaGraphLaunch`）。拓扑不变永不重捕获（固定容量+水位保证）；`Reset/RandomizeXxx` 等非每步 op 不入 Plan，单独 dispatch。

### 3.3 nk::Model / nk::Data / 字段注册表

- **Model**（不可变，cook 产物，整块上传一次）：关节/连杆拓扑（parent_link/joint_type/joint_axis/link_body/art offsets）、link 惯量与 local pose、形状表（原语参数 + hull 顶点 + SDF 栅格）、XPBD 约束模板（索引/restlen/compliance）、PBF 参数、**物理材质桶表初值**、过滤策略（含 env 间碰撞过滤 flag）、各 `max_*` 容量、字段注册表 schema。
- **Data**（每 World 一次分配）：`Arena` 持 3 个 `phi::Buffer`（class：`Persistent`（状态+λ+RNG+材质桶表+可随机化 body 参数）/`Scratch`（contacts/rows/AABB/grid/CRBA 暂存）/`Tape`（diffsim checkpoint））；段表 256B 对齐、确定性布局；env-major。
- **`src/nk/model/fields.yaml`**（唯一信源；`tools/codegen/fields/gen_fields.py` 生成 `field_ids.hpp / views.hpp / arena_layout.hpp / dlpack_table.hpp`，并入 `tools/codegen/regen.py`）。schema 与初始字段表（节选语法示例，完整表按现 `ArticulationDeviceBuffers/BodyState/XpbdWorld/PbfWorld` 字段 1:1 搬迁，实现时照此格式补全全部 ~70 字段）：

```yaml
schema: {name, dtype: [f32,u32,u64,vec3,quat,transform,spatial6,mat36],
         per: [env, dof, link, body, contact_slot, row_slot, particle, dist_con, env_dof2, scalar],
         arena: [persistent, scratch, tape], flags: [diff, param, readout]}
fields:
  - {name: q,            dtype: f32,       per: dof,          arena: persistent, flags: [diff]}
  - {name: qdot,         dtype: f32,       per: dof,          arena: persistent, flags: [diff]}
  - {name: link_pose,    dtype: transform, per: link,         arena: persistent}
  - {name: base_pose,    dtype: transform, per: env,          arena: persistent, flags: [diff]}
  - {name: body_pose,    dtype: transform, per: body,         arena: persistent, flags: [diff]}
  - {name: body_inv_mass,dtype: f32,       per: body,         arena: persistent, flags: [param]}
  - {name: contact_count,dtype: u32,       per: env,          arena: scratch}     # 水位
  - {name: rows,         dtype: f32,       per: row_slot,     arena: scratch, elem: 16}
  - {name: lambda,       dtype: f32,       per: row_slot,     arena: persistent}  # warm start
  - {name: m_inv,        dtype: f32,       per: env_dof2,     arena: scratch}
  - {name: particle_pos, dtype: vec3,      per: particle,     arena: persistent, flags: [diff]}
  - {name: mat_buckets,  dtype: f32,       per: scalar, count: num_buckets*8, arena: persistent, flags: [param]}
  - {name: mat_index,    dtype: u32,       per: body,         arena: persistent, flags: [param]}
  - {name: link_contact_wrench, dtype: spatial6, per: link,   arena: scratch, flags: [readout]}
```

- **DLPack/diffsim**：`flags: diff` 字段集 = tape 可见集；`dlpack_table.hpp` 全字段查表 → `c_abi/buffer.cpp` 不再手写分支。容量策略：每 env 固定槽位 + `*_count` 水位（u32 原子 append），kernel 最大 grid 发射 + 早退；溢出置 env 状态位（经 ExportObs 可查），容量是 Model 属性。

### 3.4 统一解算（唯一承诺路径）

**`SolveRowsBlockIsland`**（`src/phi/backend_cuda/ops/solve_rows.cu`）：
- **island = env 内 row 连通分量**（共享 body/articulation/particle 的 row 归同 island）。`nk::SolveSchedule`（`src/nk/solve/schedule.hpp/.cpp`）在 World 构建/Reset 时按**最大容量槽位**用现 `BuildRowColorPartitions/BuildRowComponentPartitions`（host，`src/solver/gpu/row_scheduler.*` 算法整体迁入 `src/nk/solve/`）算 worst-case 划分一次，上传设备常驻：`island_row_offsets / island_color_segments / row_order` 三个 Persistent 字段。运行期水位掩码屏蔽空槽，**每步零 host 参与、零重着色**。
- kernel：`grid = (total_islands)`（= N_env × env 内 island 数），`block = 256`。block 内循环 `for it in vel_iters: for c in colors(island): 并行处理该 color 的 rows（thread/row）→ __syncthreads()`。row 处理读 ModelView/DataView 设备指针（rows/sides/art_refs/chain_jacobians/m_inv/qdot/body vel/particle vel 全 arena 常驻），λ warm-start in-place，side 分发（RigidInvMass/ArticulationChainJ/ParticleInvMass/StaticNull）沿用现 `row_solver.cu:302/346` 逻辑。51-DOF M⁻¹·Jᵀ 乘积按 thread 内寄存器循环（51 floats），chain_jacobian 段 coalesced 读取。
- **为什么这是唯一路径且必达标**：union 场景 ~180 rows/env、~10-170 color；一次 color-sweep = O(rows) 行更新，64 迭代 ≈ 1.2 万次行更新 ≈ 数十万 FLOP + 数 MB 读写——MuJoCo 单 CPU 核毫秒内完成同量计算，单 SM block 无理由更慢。现 58–1400ms 是「每行重复全量上传/重算 + host 同步」的实现病灶。批量时 N×islands 个 block 自然占满 GPU（与现 `SolveArticulatedContactRows` 的 block-per-articulation 已验证模式同构）。验收门 §4-M4 直接以 ≤5ms@N=1 与 ≥11k eps 检验，不达标 = 实现 bug，修 kernel，不换架构。

### 3.5 碰撞（唯一承诺路径）

- 原语×原语（sphere/capsule/box/plane 两两组合）：`NarrowphasePrimitives` 解析专核（移植现 foot-ground/sphere×box/box×plane + 补全缺口对）。
- **任意 mesh 接触：`NarrowphaseSdf`**。cook 期每个碰撞 mesh 按其 **PhysicsMaterial 的烘焙指令**（`sdf_cell_size/sdf_min_res/sdf_max_res`，Genesis 同款字段）经现 `sparse_sdf_cooker` 生成稀疏窄带 SDF（AssetCache 内容寻址缓存，存入 .nka）；运行期对方形状的采样点集（hull 顶点 + 边中点，cook 期生成存 Model）查 SDF 值/梯度出接触点。凸分解（V-HACD）保留用于生成 hull 资产（broadphase AABB 细化 + 采样点源），**CPU GJK/EPA（`src/collision/convex_narrowphase.*`、`cvx::SphereHull` 特例）在 M9 删除**；`test_gjk_epa_convex.cpp` 的解析口径改造为 `NarrowphaseSdf` 的精度 oracle（球×球/球×盒解析真值，SDF 路径容差 ≤ 烘焙 cell 尺寸）。
- broadphase：`BuildAabbs`→`LbvhBuild`→`LbvhQueryPairs`（现 `broadphase_lbvh.cu/candidate_pair.cu` 算法迁入 ops，thrust 调用保留在 op 实现内部）+ `ParticleGridBuild`（现 `particle_uniform_grid.cu`）。env 间过滤 = pair 生成期 env_id 门控（Model flag `filter_cross_env`）。

### 3.6 场景树 + ECS 组件模型（SceneGraph 一等公民 + 五组件 + 双材质；Mangifera/Isaac/Genesis 合成）

> **Owner 修订 2026-06-11**：场景以**场景树**形式编辑与管理（对象管理/机器人层级统一为节点树）；`name` 只是单段名字，`path` 专由节点树管理派生；基底 = owner 的 Mangifera `core/manager/scene-graph.{hpp,cpp}`（first-child/next-sibling 节点、`path_of`/`node_of` 寻址、entity↔node 映射、选中态），在其上扩充。

**`src/scene/graph/scene_graph.hpp/.cpp`**（基底 Mangifera，扩充点标 ★）：

```cpp
struct SceneNode {                          // first-child/next-sibling（Mangifera Scene_Node 同构）
    uint64_t    id;                         // 图内唯一自增
    std::string name;                       // 单段名（不含 '/'）；★同级唯一（插入校验，冲突自动后缀 _1/_2）
    EntityId    entity;                     // ★ 节点↔实体 1:1（root 持哨兵实体）
    std::weak_ptr<SceneNode>   parent;      // ★ parent 用 weak_ptr 防引用环（Mangifera 用 shared）
    std::shared_ptr<SceneNode> first_child, next_sibling;
};

class SceneGraph {                          // ★ 非 Singleton：每 Scene 一实例（compose 需多场景共存）
  public:
    std::shared_ptr<SceneNode> Root();      // root->name = "Scene"，不入 path
    std::shared_ptr<SceneNode> Selected();  void SetSelected(std::shared_ptr<SceneNode>);  // 编辑选中态（M11 viewer）
    // —— Mangifera 原 API（语义不变，命名仓内化）——
    std::string                PathOf(const std::shared_ptr<SceneNode>&) const;   // 向上拼 "h1/right_hand_link"（不含 root 段）
    std::shared_ptr<SceneNode> NodeOf(const std::string& path) const;             // 按段下行；同级唯一名 ⇒ 确定解析
    std::shared_ptr<SceneNode> ParentOf/FirstChildOf/NextSiblingOf(node) const;
    std::shared_ptr<SceneNode> AddEntity(EntityId, parent, const std::string& name); // 尾插 ⇒ 兄弟序确定（序列化/D1 之锚）
    // —— ★ 扩充 ——
    void AttachSubtree(std::shared_ptr<SceneNode> subtree, parent);               // compose 嫁接（跨图迁移 + 实体 remap 由 compose 层做）
    std::shared_ptr<SceneNode> Detach(node);                                       // 摘下子树（不毁实体）
    void DestroyRecursive(node, Registry&);                                        // 子树连实体一并销毁
    template<class F> void Traverse(node, F&&) const;                              // 先序遍历（确定序，序列化/diff 用）
};
```

**机器人层级原生入树**：importer 把 articulation 按运动学树建子树（robot 根节点 → link 节点按 parent_link 逐层嫁接；h1 即 `h1/pelvis/left_hip_yaw_link/...`），`JointComponent` 挂在 child-link 实体上（parent_body/child_body 即父子节点实体）。树承载层级与命名（编辑视图），组件承载数据；`TransformComponent.local` 相对父**节点**，world = 沿树合成；cook 后运行期权威位姿在 `nk::Data`，TransformSync 经 SceneMap 回写（§3.8）。

**`src/scene/ecs/components.hpp`** 完整字段契约（层级不再是组件——唯一权威 = SceneGraph）：

```cpp
struct NameComponent      { std::string name; };                  // ★单段名，与节点 name 同步（节点为权威）；path 一律 SceneGraph::PathOf 派生
struct TransformComponent { math::Transform local; math::Vec3 scale{1,1,1}; };   // local 相对父节点
struct PhysicsMaterial {   // 独立资产，实体经 id 引用（材质桶的 cook 源）
    float static_friction=0.5f, dynamic_friction=0.5f, restitution=0.f;
    enum class Combine : uint8_t {Average,Min,Multiply,Max} friction_combine=Combine::Average,
                                                            restitution_combine=Combine::Average;
    float compliant_stiffness=0.f, compliant_damping=0.f;        // ↔ 现 solref/solimp 映射
    float density=1000.f;
    float sdf_cell_size=0.005f; uint16_t sdf_min_res=32, sdf_max_res=128;  // 烘焙指令
};
struct RenderMaterial {    // 独立资产
    float base_color[4]{0.8f,0.8f,0.8f,1}; float metallic=0.f, roughness=0.5f;
    float emissive[3]{0,0,0}; float opacity=1.f;
    std::string tex_albedo, tex_normal, tex_metallic_roughness;  // .nka 内 TEXB 引用
};
struct VisualMeshComponent   { AssetRef mesh; uint32_t render_material_id; };
struct CollisionShapeComponent {
    enum class Kind : uint8_t {Sphere,Capsule,Box,Plane,ConvexHull,SdfMesh} kind;
    float params[4]; AssetRef cooked;            // hull/SDF 在 .nka 的引用
    uint32_t physics_material_id;
    uint32_t group, mask;                        // 替代魔法 handle 7000/8500/9000+
};
struct RigidBodyComponent { float mass; math::Vec3 inertia_diag; math::Transform inertial_frame;
                            bool kinematic=false; float linear_damping=0, angular_damping=0; };
struct JointComponent     { enum Kind{Revolute,Prismatic,Fixed,Spherical,Free} kind;
                            EntityId parent_body, child_body; math::Vec3 axis;
                            float limit_lo, limit_hi, damping, armature, friction; };
struct ActuatorComponent  { enum Mode{PD,Torque,Velocity} mode; float stiffness, damping;
                            float effort_limit, effort_limit_sim, velocity_limit; }; // Isaac T9 双限幅
struct SystemKindComponent{ enum K{Rigid,Articulated,Soft,Cloth,Fluid} kind; };       // morph→solver 路由
struct SoftBodyComponent  { AssetRef tet_or_cloth_mesh; float young, poisson, xpbd_compliance; };
struct FluidComponent     { float particle_size, rho0, viscosity, surface_tension; };
struct InitialStateComponent { std::vector<float> qpos; math::Transform root; };       // settle 产物
struct CameraComponent / LightComponent { ...现 Record 字段原样... };
```

`Registry`（`registry.hpp/.cpp`）：每组件一个 `std::vector<T>` 池 + `EntityId{u32 index,u32 gen}`→slot 稠密映射 + ★entity→node 反查表（O(1) 找回节点）；**路径寻址一律走 `SceneGraph::NodeOf`，Registry 不再提供 `Find(path)`**。**1:1 物理↔渲染**：物理世界与渲染世界是同一 Registry 上的两组系统视图，TransformSyncSystem 每帧把 Data 位姿写回 RenderWorld 实例（§3.8），实体身份经 SceneMap 双向。

### 3.7 自有资产格式（.nks / .nka）

**`.nka` 二进制容器**（`src/scene/asset/nka.hpp/.cpp`，magic `"NKA1"`，LE）：
头 `{magic, version=1, chunk_count}` + TOC `{fourcc, u64 offset, u64 size, u64 content_hash}`。chunk 类型：`MESH`（视觉网格：pos/normal/uv/index，f32/u32）、`CMSH`（碰撞网格源）、`HULL`（V-HACD 凸件组）、`SDF0`（稀疏窄带 SDF：cell 表 + 值 + 梯度，现 `CookedSdfTable` 序列化）、`SAMP`（SDF 接触采样点集）、`TEXB`（纹理原始字节）。`AssetRef = {nka 路径, fourcc, index}`，文本形式 `"cup.nka#SDF0/0"`。

**`.nks` 场景文件**（`src/scene/format/nks.hpp/.cpp`，JSON，`"nks_version":1`）：

```json
{ "nks_version": 1,
  "physics_materials": {"mat_cup": {"static_friction":0.8, "dynamic_friction":0.8,
      "restitution":0.0, "compliant_stiffness":1.8, "density":250,
      "sdf_cell_size":0.004, "sdf_min_res":32, "sdf_max_res":128}},
  "render_materials":  {"rm_cup": {"base_color":[0.9,0.9,0.95,1], "metallic":0.0,
      "roughness":0.3, "tex_albedo":"cup.nka#TEXB/0"}},
  "tree": [
    {"name":"cup", "children":[
      {"name":"body", "transform":{"pos":[0.42,0,0.86],"quat":[1,0,0,0]},
       "rigid_body":{"mass":0.2},
       "collision_shape":{"kind":"sdf_mesh","cooked":"cup.nka#SDF0/0","physics_material":"mat_cup"},
       "visual_mesh":{"mesh":"cup.nka#MESH/0","render_material":"rm_cup"}}]}],
  "imports": [{"file":"h1_with_hand.xml","attach_at":"h1","transform":{...}}],
  "initial_state": {"h1":{"joint_pos":{"left_knee":0.70}}},
  "settle": {"steps":200, "dt":0.004167, "holds":[{"dofs":"h1/.*","mode":"pd"}]},
  "env": {"replicate":1, "spacing":2.0, "filter_cross_env_collisions":true},
  "solver": {"dt":0.004167, "gravity":[0,0,-9.81], "vel_iters":32,
             "max_contacts_per_env":64, "max_rows_per_env":320}}
```

API：`scene::format::Save(const Scene&, path)` / `Load(path) → Scene`（Scene = SceneGraph + Registry 对）。**`tree` 节 = 场景树的嵌套序列化**：Save = `SceneGraph::Traverse` 先序遍历输出嵌套 `{name, <components...>, children:[]}`（兄弟序 = 树内尾插序 ⇒ 二次 Save byte 相同）；Load = 自顶向下重建节点+实体；`imports` 节经 importer 展开为 `attach_at` 节点下的子树后内联。override layer：`Load(base, overlay)`，overlay 以**派生 path** 为键、仅含被覆盖键（域随机化/实验配置不改基准文件）。**导入转换懒缓存**（Isaac T10）：`LoadMjcf/Usd/Urdf` 结果按 `(源文件内容哈希+导入配置哈希)` 缓存到 `.nuka_cache/<hash>.{nks,nka}`，命中直加载。importers 升级：MJCF 读 visual geoms（contype=0 几何 → VisualMeshComponent，h1_with_hand 视觉手指因此自然入渲染）与 rgba/material → RenderMaterial；URDF visual/collision 标签分流；USD 按 purpose 分流 + UsdPreviewSurface→RenderMaterial、UsdPhysics material→PhysicsMaterial。

### 3.8 渲染 1:1 模型

`render::RenderWorld`：`RenderInstance{entity, mesh_id, world_xform, render_material_id}[]` + `MeshLibrary`（.nka MESH 去重加载）+ 材质/相机/灯表。`BuildRenderWorld(Registry, SceneMap)` 一次构建；每帧 `TransformSyncSystem` 把 Data 的 body/link 位姿经 SceneMap 写入 instance.world_xform（M8 走 host 下载选定 env；互操作是 M11 的 `CudaVulkanInteropPublisher`）。**1:1 验收口径**：每帧对每个有双侧组件的实体断言 `render_instance.world == data.body_pose ∘ visual_local`（场景门 `render_physics_parity`）。光栅：真三角形前向管线 + PBR 参数/纹理（RenderMaterial 全字段生效）；debug 线框降级为 overlay。RT 路径追踪器经 `RenderWorldToTwoLevelScene` 吃同一 RenderWorld（M11）。

### 3.9 Python API（唯一表面）

```python
dev   = nuka.Device.create(0)
scene = nuka.Scene.load("h1_with_hand.xml")            # mjcf/urdf/usd/nks 同一入口
scene.compose(nuka.Scene.load("cup.usda"), pose, attach_at="cup")   # 子树嫁接到 "cup" 节点
node  = scene.find("cup/body")                          # → SceneNode 句柄（树寻址）
node.set_local(pos=..., quat=...)
scene.root / node.name / node.path / node.parent / node.children()  # 场景树遍历/编辑一等 API
scene.set_physics_material("cup/.*", static_friction=0.8)
scene.settle(steps=200)
scene.save("h1_cup.nks")                               # 自有格式导出（含 .nka 烘焙产物）
world = nuka.World.create(dev, scene, env_count=4096)  # cook→Model→Data→Pipeline
q = torch.from_dlpack(world.buffer_view(nuka.Field.JOINT_POSITION))  # 零拷贝不变
world.step()
rec = nuka.Recorder(world, camera=...); rec.capture("out/frames"); rec.to_video("out/run.mp4")
```

命名约定（Isaac T4）：`set_*` 只写 host 缓冲，`write_*_to_sim` 落盘设备；`World/GraspWorld/UnionWorld` 三族在 parity 窗口后删除。

---

## 4. 里程碑（每个 = 新建/修改/删除 + 验收门命令）

### M0 — 大清扫 + 计划入库

**删除（立即，连同 CMake 注册）：**
| 文件 | 说明 |
|---|---|
| `src/runtime/gpu/batched_device_world.{cu,hpp}` | 死代码 ~3k LOC，无生产消费者 |
| `src/runtime/gpu/device_world.{cu,hpp}`、`cuda_world_stepper.{cu,hpp}` | 测试专用 GPU 单 env 路径 |
| `src/runtime/gpu/cuda_particle_world.{cu,hpp}` | 实验性 ~2.1k LOC |
| `src/runtime/world_stepper.{cpp,hpp}`、`src/runtime/world_instance.hpp` | CPU 旧路径 |
| `src/solver/rigid_solver.{cpp,hpp}` | 仅 world_stepper 消费 |
| `src/apps/` 整目录（debug_shell + cuda_particle_demo） | 旧 demo，M8 由 Simulation+Recorder 取代 |
| `src/sensor/gpu/` 中绑定 BatchedDeviceWorld 的 lidar/camera kernels（保留 contact_wrench 与 noise） | 死耦合 |
| 测试（依赖上述者全删，~25 文件）：`test_world_stepper、test_cuda_world_stepper、test_cuda_device_world、test_cuda_batched_world、test_cuda_contacts、test_step_timing、test_cuda_step_timing、test_cuda_batch_timing、test_cuda_batch_contact_timing、test_cuda_batch_joint_drive_timing、test_cuda_batch_sensor_timing、test_camera_render、test_cuda_sensors、test_cuda_particle_world、test_cuda_particle_coupling_timing、test_v01_foundation_pipeline`（含 owner 已知 RED 的 v01 e2e，随 CPU 路径退役）、apps 测试 2 个 | 能删就删 |

**修改：** `tests/collision/test_lbvh_vs_sap_pair_set.cpp`、`tests/runtime/test_sdf_tier_wired.cpp`（oracle，剥离 DeviceWorld 依赖改直驱 collision 组件）；`tests/CMakeLists.txt` 清注册。
**新建：** `docs/plans/2026-06-11-nk-core-platform-refactor.md`（= 本文件入库）。
**门：** `cmake --build build-cuda128 -j` 全绿；`ctest --test-dir build-cuda128 -L ''`（剩余测试）全绿（除既有 owner 裁定 RED 外无新红——该 e2e RED 已随删除消失）；`git grep -l BatchedDeviceWorld src/ tests/` 零命中。

### M1 — PHI v2 后端层（ggml 式）

**新建：**
| 文件 | 内容（签名见 §3.1/§3.2） |
|---|---|
| `src/phi/backend.hpp` | BackendI/DeviceI/RegistryEntryI/Status/OpCall + 注册表函数 |
| `src/phi/buffer.hpp`（v2 重写，旧类暂存为 `buffer_legacy.hpp` 供遗留路径） | BufferTypeI/BufferI |
| `src/phi/op_schema.hpp` | NkOp 28 枚举 + 全部 `<Op>Params` POD |
| `src/phi/plan.hpp`、`src/phi/event.hpp` | Plan/Event 不透明句柄 |
| `src/phi/registry.cpp` | 全局注册表 + `InitBestDevice` |
| `src/phi/backend_cuda/cuda_backend.cu` | vtable：dispatch（查 op 函数表 `OpFn g_ops[NkOp::Count]`）、plan_create（BeginCapture→逐条 dispatch→EndCapture→Instantiate，缓存 exec）、plan_execute、event_*、synchronize |
| `src/phi/backend_cuda/cuda_buffer.cu` | buffer_type/buffer 实现（cudaMalloc/Async copy/memset 唯一所在地，吸收现 buffer.cu） |
| `src/phi/backend_cuda/ops/registry.cu` | `g_ops[]` 表 + `REGISTER_NK_OP(op, fn)` 宏；本期所有 op 槽位先填 `Unsupported` 占位 |
| `src/phi/backend_cuda/launch.cuh` | `LaunchCuda<K>(k, dims, stream, args...)` —— `<<<>>>` 唯一封装点 |

**修改：** `tools/lint/banned_patterns.yaml`（§3.1 红线）+ `tools/lint/physics_smell.py`（scope exclude）；根/`src` CMake 新增 `nuka_phi2` 目标。
**门：** 新建场景门 `tests/scenario/phi2_smoke.cpp`：注册表枚举到 CUDA device → alloc buffer → upload/download 字节回环 → 注册一个测试 op（向量加）→ dispatch 与 plan_execute 结果一致且两次 plan 重放 byte 相同（D1）。`ctest -R phi2_smoke` 绿 + lint 全绿。

### M2 — ECS + 双材质 + 导入升级 + 自有格式

**新建：**
| 文件 | 内容 |
|---|---|
| `src/scene/ecs/entity.hpp` | `EntityId{u32 index, u32 gen}` + 哨兵 |
| `src/scene/graph/scene_graph.hpp/.cpp` | §3.6 SceneGraph 场景树（Mangifera 基底 first-child/next-sibling + PathOf/NodeOf；扩充 AttachSubtree/Detach/DestroyRecursive/Traverse/同级唯一名/EntityId 关联；★per-Scene 实例非 Singleton） |
| `src/scene/ecs/components.hpp` | §3.6 全部组件（字段照抄；NameComponent=单段 name，无 HierarchyComponent——层级唯一权威是 SceneGraph） |
| `src/scene/ecs/registry.hpp/.cpp` | 组件池 + Create/Destroy/Get/Add + entity→node 反查 + 遍历器（路径寻址走 SceneGraph::NodeOf） |
| `src/scene/scene_map.hpp/.cpp` | EntityId ↔ {body_row, joint_row, dof_index, shape_row, link_index, bp_group} 双向 + `<pair>` 展开 |
| `src/scene/asset/nka.hpp/.cpp` | §3.7 容器读写（WriteChunks/ReadToc/LoadChunk） |
| `src/scene/asset/asset_cache.hpp/.cpp` | 内容哈希寻址：mesh→MeshGeometry、(mesh,材质烘焙指令)→HULL/SDF0/SAMP、纹理→TEXB；`.nuka_cache/` 落盘 |
| `src/scene/format/nks.hpp/.cpp` | §3.7 Save/Load/override-layer（`tree` 嵌套序列化 = SceneGraph 先序；JSON 用仓内既有 json 依赖或自写极简解析器——与现 importer 同栈） |
| `tests/scenario/scene_roundtrip.cpp` | 门：LoadMjcf(h1)+LoadUsd(cup)→Compose→Save(.nks/.nka)→Load→**场景树等价**（先序遍历：节点名/派生 path/兄弟序/组件字段逐项）+ SceneMap 名↔行回环 + 二次 Save byte 相同 |

**修改：**
| 文件 | 改动 |
|---|---|
| `src/scene/scene_ir.hpp/.cpp` | 改为 Scene(SceneGraph+Registry) facade（读 API 签名不变，oracle 不动） |
| `src/scene/scene_compose.hpp/.cpp` | 子树嫁接 compose：AttachSubtree + EntityId-gen 安全 remap + 前缀**节点**命名空间（原路径前缀改为挂载节点） |
| `src/import/mjcf_importer.cpp` | 读 visual geoms（contype=0→VisualMeshComponent）、rgba/material→RenderMaterial、friction/solref→PhysicsMaterial；接 AssetCache |
| `src/import/urdf_importer.cpp` | visual/collision 标签分流 + material 色→RenderMaterial |
| `src/import/usd_importer.cpp`/`usd_stage_adapter` | purpose 分流、UsdPreviewSurface→RenderMaterial、UsdPhysics material→PhysicsMaterial |
| `src/scene/cooker.cpp` | 烘焙入口改走 AssetCache（V-HACD/SDF 结果可复用、可入 .nka） |

**门：** `scene_roundtrip` 绿；`tests/scene/` 既有 facade 等价测试绿；`tests/import/` importer 往返绿（新增视觉几何/材质计数断言并入既有文件）。

### M3 — Model/Data + 字段 codegen + articulation 管线移植

**新建：**
| 文件 | 内容 |
|---|---|
| `src/nk/model/fields.yaml` + `tools/codegen/fields/gen_fields.py` | §3.3（生成 4 个头入库：`src/nk/model/generated/{field_ids,views,arena_layout,dlpack_table}.hpp`） |
| `src/nk/model/model.hpp/.cpp` | `nk::Model`（§3.3 内容 + 整块设备上传 + max 容量） |
| `src/nk/data/arena.hpp/.cpp`、`src/nk/data/data.hpp/.cpp` | Arena(3 buffer)/段表/`Ptr<T>(FieldId)`/`View()`/快照恢复（SnapshotState/RestoreState op） |
| `src/nk/pipeline/pipeline.hpp/.cpp` | `Build(Model)→vector<OpCall>`（§3.2 顺序）+ params 结构存储 |
| `src/nk/pipeline/world.hpp/.cpp` | `nk::World{Model,Data,Pipeline,SolveSchedule,Backend*}`：`Step/StepPlanned/Reset(env_mask)/FieldPtr` |
| `src/scene/cook/cook_to_model.hpp/.cpp` | `CookToModel(Registry, env_count) → {Model, SceneMap}`（含 replicate 与材质桶表生成） |
| `src/phi/backend_cuda/ops/articulation.cu` | op 实现：ApplyDrives/AbaForward/IntegrateVelocity/FkWorldPoses/IntegratePosition —— 从 `src/runtime/articulation/featherstone_aba.cu` **kernel 体逐行移植**（归约顺序不动，保 D1），入参改 View |
| `src/phi/backend_cuda/ops/crba.cu` | CrbaComputeM/CrbaFactorM（自 `articulation_contacts.cu` CRBA 段移植） |
| `src/phi/backend_cuda/ops/contacts_foot.cu` | NarrowphasePrimitives 首批（sphere×plane foot + ContactTangentBasis，自 `batched_articulated_world.cu` 移植） |
| `src/phi/backend_cuda/ops/solve_articulated.cu` | 过渡：现 fused `SolveArticulatedContactRows` 移植为临时 op（M4 被 SolveRowsBlockIsland 取代后删） |
| `src/phi/backend_cuda/ops/readout.cu` | ReadoutContactWrench/ExportObs/ResetEnvs/Snapshot/Restore |

**修改：** `src/c_abi/diffsim.cpp` + `src/diffsim/{step_backward,tape,backward_runner}.cu`（指针源 = arena diff 字段，算法不动）；`tests/oracle/featherstone_oracle_harness.cpp`（加 nk::World 驱动入口）；`tests/CMakeLists.txt`。
**门：** 全部 Featherstone golden（go2/go2_floating/h1/stand_5s/foot_contact/foot_box）对 nk::World **byte-exact**；`ctest -R Go2_4096`：≤~1µs/env-step 无回退；`StepPlanned` 1000 步与 `Step` 轨迹一致；`tests/diffsim/test_aba_reverse_fd` + 9 个 adjoint FD 绿；`go2_system_id.py` 收敛不变。

### M4 — 设备常驻统一解算

**新建：** `src/nk/solve/schedule.hpp/.cpp`（§3.4 SolveSchedule：迁入 row_scheduler 算法 + 设备常驻三字段）；`src/phi/backend_cuda/ops/assemble_rows.cu`（AssembleRows：contacts→CSR rows + 链 Jacobian（`ComputeContactChainJacobians` 移植）+ meff，全 arena 零 host）；`src/phi/backend_cuda/ops/solve_rows.cu`（SolveRowsBlockIsland，§3.4 规格）；`tests/scenario/h1_union_parity.cpp`（对照旧 BatchedUnifiedWorld 同初值同步数轨迹，容差 = 现 G1c/G1d）；`tests/perf/nk_union_n1.cpp`（N=1 union 全接触 1000 步计时）。
**修改：** `fields.yaml` 增 rows/sides/art_refs/chain_jacobians/meff/lambda/island 调度字段；Pipeline 接入 AssembleRows/SolveRowsBlockIsland（替换 M3 过渡 op 并删除 `ops/solve_articulated.cu`）。
**删除（门绿后）：** `src/solver/gpu/row_solver.{cu,cuh}` 全部（kernel 逻辑已移植）、`src/solver/unified_solve.{cpp,hpp}`、`src/solver/gpu/row_scheduler.*`（算法已迁 nk/solve）。
**门：** `h1_union_parity` 绿；`nk_union_n1` **≤5ms/step**；grasp 场景 parity（`test_batched_h1_hand_grasp` nk 入口）；`test_solref_solimp/test_foot_ground_mjx_parity/test_unified_solve`（oracle 口径）重指绿；含解算整步 plan 重放一致。

### M5 — 全 GPU 碰撞 + SDF 主路径

**新建：** `src/phi/backend_cuda/ops/broadphase.cu`（BuildAabbs/LbvhBuild/LbvhQueryPairs/ParticleGridBuild——`broadphase_lbvh/candidate_pair/rigid_candidate_pairs/particle_uniform_grid` 算法迁入）；`src/phi/backend_cuda/ops/narrowphase_prims.cu`（原语对全集，含 grasp sphere×hull 自 `narrowphase_grasp.cu` 移植、box×plane、sphere×box）；`src/phi/backend_cuda/ops/narrowphase_sdf.cu`（§3.5：采样点×SDF0，吃 `SAMP`+`SDF0` Model 资产）。
**修改：** `cook_to_model.cpp`（SAMP 采样点 cook：hull 顶点+边中点）；`fields.yaml`（aabb/pair/grid 字段）；`tests/collision/test_gjk_epa_convex.cpp` 改造为 SDF 精度 oracle（解析真值容差 ≤ cell）。
**删除（门绿后）：** `batched_unified_world.cpp` 内 CPU 逐 env foot/table narrowphase 段及 host pair 下载路径（文件整体 M9 删）。
**门：** `test_lbvh_vs_sap_pair_set`/`test_particle_grid_correctness`/`test_sdf_tier_wired`/改造后 SDF 精度 oracle 绿；grasp 聚合 **≥11k env-steps/s**；`nk_union_n1` 复测 ≤5ms。

### M6 — 多物理耦合共步

**新建：** `src/phi/backend_cuda/ops/particles.cu`（ParticlePredict/XpbdProject/PbfDensityLambda/PbfApplyDelta/ParticleFinalize——自 `xpbd_world.cu`/`pbf_world.cu` kernel 移植，PBF 网格缓冲改 arena 常驻）；`tests/scenario/coupled_grasp_soft.cpp`（机器人手×软体/流体共步：耦合 row 经 SolveRowsBlockIsland，力平衡+体积/密度守恒，容差取现 XPBD/PBF oracle）。
**修改：** `fields.yaml`（粒子/约束 λ/密度/网格字段）；`solve_rows.cu`（ParticleInvMass 臂读 arena 粒子字段——机制已在，换指针源）；`cook_to_model.cpp` + `components.hpp`（SoftBody/Fluid cook，复用 `xpbd_cooker/fluid_cooker`）。
**删除（门绿后）：** `src/runtime/coupling/unified_costep.{cpp,hpp}`。
**门：** XPBD 4 件套 + PBF 5 件套 + `test_pbd_costep_unified/test_coupling_framework` oracle 口径重指绿；`coupled_grasp_soft` 绿。

### M7 — 场景授权 + 工厂之死

**新建：** `src/scene/cook/settle.hpp/.cpp`（`SettleSpec{steps,dt,holds[]}`，用 nk::World 生产步进跑 settle→SnapshotState→InitialStateComponent，确定性）；`src/scene/cook/placement.hpp/.cpp`（`FindRestPlacement`）；`examples/scenes/h1_cup_table.nks`（+配套 .nka：H1 抓取场景资产化，工厂 ~120 经验常数入库一次）；`python/nuka/tasks/h1_grasp_choreo.py`（编排表迁 python，按 SceneMap 名查 dof）；`tests/scenario/h1_grasp_lift.cpp`（核心门：纯 .nks 路径复现 G1c/G1d 力平衡/lift 冲量三角/BITE 断言；吸收 4 个 dev-spike 关键断言后删原文件）。
**删除（门绿后）：** `src/runtime/coresident/h1_union_scene_factory.cpp`、`grasp_scene_factory.cpp`、`BatchedSceneTemplate` 全部引用；`tests/coresident/{test_h1_dense_grasp,test_h1_power_grasp_lift,test_h1_scaled_cup_grasp,test_h1_grasp_feasibility_probe}.cpp`。
**门：** `h1_grasp_lift` 绿；settle 两次运行 byte 相同；`test_scene_compose_h1_cup_table` 升级为 .nks 全链路。

### M8 — 帧循环 + 离屏 3D 渲染 + Recorder

**新建：** `src/runtime/app/command_queue.hpp`（Command 枚举 + MPSC 队列）、`pose_publisher.hpp/.cpp`（接口 + HostDownloadPublisher）、`systems.hpp/.cpp`（Input/Sim/TransformSync/Render 四系统）、`simulation.hpp/.cpp`（Frame 循环）；`src/render/render_world.hpp/.cpp`（§3.8）；`src/render/raster/vulkan_raster_renderer.hpp/.cpp` + `shaders/{mesh.vert,mesh_pbr.frag}`（离屏真 3D + PBR 材质/纹理生效，复用现 headless Vulkan 初始化）；`src/c_abi/recorder.cpp` + `src/include/nuka/nuka_recorder.h`；`python/nuka/recorder.py`；`tests/scenario/render_physics_parity.cpp`（§3.8 1:1 断言 + 两次渲染 byte 相同 + 非背景像素>0）。
**修改：** `src/render/vulkan_renderer.cpp`（debug 线框降为 overlay）；`python/src/nuka_ext.cpp`（绑 Recorder）；`examples/demo/` 新增 `render_rollout.py`。
**删除：** `examples/demo/go2_demo_render.py`（火柴人）。
**门：** `render_physics_parity` 绿；python 一条命令出 go2 与 h1 rollout mp4（**带视觉网格与材质**）；渲染关闭时步进吞吐零差异。

### M9 — 切换 + 删除全部遗留 + 测试重组

**新建：** `src/c_abi/scene.cpp` + `src/include/nuka/nuka_scene.h`（`nuka_scene_{load,compose,find,set_local,set_physics_material,settle,save,destroy}`）；`python/nuka/scene.py`。
**修改：** `src/c_abi/world.cpp`（create 走 Scene→CookToModel→nk::World；删 StepWorldGpu 内联路径）；`src/c_abi/buffer.cpp`（查 `dlpack_table.hpp`，字段枚举二进制语义不变 = RL 硬契约）；`diffsim.cpp/noise.cpp/handle_table.hpp/internal.hpp`（句柄指 nk::World）；`python/src/nuka_ext.cpp`、`python/nuka/{__init__.py,gym/env.py,rl_games/vecenv.py,tasks/*}`、`examples/training/*.py`（入口换新 API）。
**删除：** `src/c_abi/{grasp_world,union_world}.cpp` + `nuka_grasp.h/nuka_union.h`；`src/runtime/gpu/batched_articulated_world.{cu,hpp}`；`src/runtime/coresident/` 整目录；`src/runtime/{soft,fluid}` 包装类；`src/runtime/sdf/sdf_device_world.*`（上传职能入 Model）；`src/collision/` 中已迁移的 gpu kernel 原文件与 CPU `convex_narrowphase.*`；`src/scene/scene_pipeline.{cpp,hpp}`、`src/render/render_scene.{hpp,cpp}`；`src/phi/buffer_legacy.hpp` + 旧 `phi::{OwnedStream,DeviceContext,UploadVector...}`；剩余全部 UNIT/TDD 测试文件（覆盖清单制：逐文件断言→映射 oracle/场景门→孤儿先折入再删）。
**测试重组：** `tests/{oracle,scenario,perf,fixtures}` 四目录五可执行（`nuka_oracle_test/nuka_scenario_test/nuka_perf_test/nuka_import_test/nuka_render_test`），label `fast/full/perf`，每可执行 gtest Environment 单次 CUDA init；`tests/CMakeLists.txt` 重写。
**门：** `ctest -L full && ctest -L perf` 只跑新核心全绿；lint 全仓绿；`git grep -lE "BatchedUnifiedWorld|BatchedArticulatedWorld|BatchedSceneTemplate|UnifiedCoResidentStepper" src/ python/ tests/` 零命中。

### M10 — RL 回归（SG spec 解冻）

`examples/training/train_go2_ppo.py` 短跑 reward 轨迹达既往；`train_h1_grasp_ppo.py` 冒烟可学（catch-eval 上升）；union 吞吐满足 100M env-steps ≤24h；`docs/specs/2026-06-10-h1-whole-body-rl-grasp-spec.md` 改述为 nk 术语（目标不变）。

### M11 —（紧随期）imgui viewer + 互操作 + RT beauty

`src/runtime/app/viewer/{viewer_main,imgui_layer,camera_controller}.{cpp,hpp}`（GLFW+swapchain+imgui：场景树/播放/env 选择/相机/驱动滑块/**拖动实体（MoveEntity 命令→写 Data）**）；`src/runtime/app/cuda_vulkan_interop.cpp`（CudaVulkanInteropPublisher：Vulkan 变换 buffer 注册为 CUDA external memory，设备 kernel 直散射）；`src/render/rt_adapter.hpp/.cpp`（RenderWorldToTwoLevelScene，BLAS 一次/TLAS 每帧，自写不用 OptiX）。门：有显示器机器交互运行；RT still D1；离屏路径仍是 CI 门。

---

## 5. 性能红线（每里程碑复测）与风险

红线：union(H1+cup+table) N=1 全接触 ≤5ms/step；grasp 批量 ≥11k env-steps/s；Go2 4096 ≤~1µs/env-step；整步 plan(graph) 可重放且 D1；热路径零分配（lint）。

风险（均为"修实现，不换架构"）：① SolveRowsBlockIsland 的 51-DOF 行更新带宽——靠 chain_jacobian coalesced 布局与 color 内并行兑现，门 = ≤5ms 硬指标；② D1 byte-exact——kernel 移植逐行保序，golden 红 = 退化（永不再生成 golden）；③ plan 有效性——一切数据依赖控制流 = 最大 grid + 水位早退；④ SDF 接触精度——容差以 cell 尺寸为界写入 oracle，材质烘焙指令给 owner 留每资产调节旋钮。

---

## 6. 约束衔接

构建 `build-cuda128`（CUDA 12.8 `/opt/cuda-12.8-root` + g++-10，Makefiles）；git 全程 git-lfs PATH + 代理 push + `[skip ci]`；`tests/oracle/golden/**` owner 保护、agent 不可写；python 环境 conda `nuka-v03`；SG spec 冻结至 M10；本文件与 `docs/plans/2026-06-11-nk-core-platform-refactor.md` 同步维护（以仓内版本为执行基准）。
