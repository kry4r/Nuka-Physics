# 统一求解器、柔体/流体与可微：架构与公式推导

本文按六个主题分开整理：① 整体统一（Uniform）物理求解器的流程设计与大块内存布局；② README 布料与 Go2 接触 Demo 的布料实现（XPBD）及公式推导、刚体耦合；③ 主页 Bunny-Water 与 Jelly 球的 MLS-MPM 实现、公式推导、刚体耦合与沃辛顿射流效应；④ XPBD/PBD 柔体约束补充；⑤ 可微（diffsim）实现；⑥ 场景 IR 与 nks 场景语言、场景编译设计。

文中英文标识符均为源码真实名称，公式与源文件 `src/phi/backend_cuda/ops/{particles,mpm,solve_rows}.cu`、`src/runtime/soft/cloth_topology.cpp`、`src/diffsim/*` 及论文原文（Macklin 2016 / Hu et al. 2018 / Stomakhin 2012 / Klar et al. 2016）复核一致，正文为中文。

---

## 1. 整体 Uniform 物理求解器的流程设计

### 1.1 设计总纲：一个求解器，一种行，一个解

Nuka 遵循"**一个通用物理求解路径**"的架构约束：机器人+地面、机器人+抓取物、刚体+布料、刚体+流体、粒子+粒子，全部收敛到**同一条接触行边界**（`RowKind::Contact` 行 → 统一 PGS 解）与**同一个 body 反作用汇**（`ReactionProviderKind` 分派）。没有任何按场景特化的求解捷径。

与 MuJoCo 的对照（设计意图上的类比，非代码复制）：

| 概念 | MuJoCo | Nuka |
|---|---|---|
| 模型 | `mjModel`（不可变、cook 一次） | `nk::Model`（不可变 cook 表） |
| 状态 | `mjData`（大块连续内存，固定容量） | `nk::Data`（Arena 三缓冲，固定容量） |
| 每步执行 | `mj_step` 固定函数序 | `nk::Pipeline` 固定 op 列表 |
| 接触 | `mj_contact` + 解算行 | `NkRow` 槽位 + block-island PGS |
| 求解 | `mj_solve`（PGS/PGD） | `SolveRowsBlockIsland`（PGS） |

两者共同的工程哲学：**一次性分配的大块连续内存 + 固定容量的槽位布局 + 固定的执行顺序**，使每步无主机参与、可 CUDA-graph 捕获、可逐位复现。

### 1.2 三层对象与内存布局

```text
SceneIR ──CookToModel──▶ nk::Model（不可变 cook 表）
                              │  Model::UploadTo（env-major 上传设备）
                              ▼
                      nk::Data（可变设备状态）
                              ▲
              nk::World（生命周期边界：模型+数据+管线+步进+读回）
```

**`nk::Model`**：全部由 cook 决定的不可变表（body 行、关节、形状、材料、SDF 网格、XPBD 约束、MPM 材料、能力值 `ModelCapacities`、调度三元组）。字段表由 `src/nk/model/fields.yaml` 生成（`generated/field_ids.hpp`、`views.hpp`、`arena_layout.hpp`），保证字段 id 与布局单一来源。

**`nk::Data` / `Arena`**（`src/nk/data/arena.hpp`）——大块内存布局的核心：

- 三个 phi `Buffer`：**Persistent（持久）/ Scratch（每步临时）/ Tape（记录回放）**，按 `ModelCapacities × env_count` 一次分配；
- 每个数据字段在所属 arena 内分得一个 **256B 对齐段**；布局 **env-major**（第 0 环境的全部字段段，第 1 环境……），相同 Model + env_count 得到**逐字节相同**的段表（`ComputeSegments` 是纯函数，确定性门测试比较两份段表）；
- `Ptr(FieldId)` 返回字段段基址，op 通过生成视图（`ModelView/DataView`）零开销访问；
- 全部零初始化；Reset 走 `ZeroAll`。

**`nk::Pipeline`**：cook 期把 Model 的维度解析成固定 `OpCall` 列表，设备能力不支持的 op 在 Build 期拒绝（而非运行期）。op 列表在拓扑不变时**永不重捕获**（CUDA-graph 友好）。

### 1.3 固定操作调度（每步完整顺序）

`pipeline.cpp` 中注释给出的规格固定顺序：

```text
ApplyDrives → AbaForward → IntegrateVelocity(+ParticlePredict)
→ FkWorldPoses → BuildAabbs → LbvhBuild → LbvhQueryPairs(+ParticleGridBuild)
→ NarrowphasePrimitives → NarrowphaseSdf → ContactTangentBasis
→ CrbaComputeM → CrbaFactorM → AssembleRows → SolveRowsBlockIsland
   (+XpbdProject / PbfDensityLambda 内联于解缝)
→ IntegratePosition(+ParticleFinalize) → ReadoutContactWrench
```

各段职责：

| 段 | op | 职责 |
|---|---|---|
| 驱动 | `ApplyDrives` | $\boldsymbol\tau = \mathrm{clamp}(K_p(\mathbf{q}_{\mathrm{target}}-\mathbf{q}) - K_d\dot{\mathbf{q}},\ \pm\tau_{\max})$ |
| 动力学 | `AbaForward` | 3 遍 Featherstone ABA：$\ddot{\mathbf{q}}$ / $\mathbf{a}_{\mathrm{base}}$ |
| 预积分 | `IntegrateVelocity` | $\dot{\mathbf{q}} \mathrel{+}= \ddot{\mathbf{q}}\,dt$；刚体重力踢；浮基速度 |
| 预测 | `ParticlePredict` | 粒子/布料位置预测（见 §2） |
| 宽相位 | `BuildAabbs → LbvhBuild → LbvhQueryPairs` | 每形状世界 AABB → LBVH → 候选对 |
| 窄相位 | `Narrowphase*` | 图元/凸包/SDF/高度场/粒子-刚体流形 |
| 质量 | `CrbaComputeM / CrbaFactorM` | 关节空间质量阵 CRBA + LDLT 分解 |
| 装配 | `AssembleRows` | 候选流形 → `NkRow` 槽位（含预计算 $\mathbf{w} = \mathbf{M}^{-1}\mathbf{J}^\top$、$m_{\mathrm{eff}}$） |
| 解算 | `SolveRowsBlockIsland` | 岛级颜色化 PGS（+ XPBD/PBF 内联投影） |
| 后积分 | `IntegratePosition` | $\mathbf{q} \mathrel{+}= \dot{\mathbf{q}}'\,dt$；浮基位姿；分裂冲量推挤 |
| 读回 | `ReadoutContactWrench` | 接触力/力矩读回 |

### 1.4 统一行求解：SolveSchedule + NkRow + block-island PGS

**`NkRow`**（`src/nk/solve/nk_row.hpp`）：一个约束行的槽位记录，**32 个 f32 = 128B**（与 `urows` 字段元素大小 static_assert 绑定）。字段：`rhs`（=aref 参考加速度）、`compliance_alpha`（=R 对偶正则化）、`lower/upper`（法向 $[0, \infty)$；摩擦 spoke 用内核内 $[0, \mu\cdot\text{TotalNormalLambda}]$ 锥界）、`mu`、`group_first/group_normal_count`（摩擦锥的分组锚）、双侧 `NkRowSide{kind, index, jlin, jang}`（kind ∈ {Rigid, Artic, Particle, Static}，$\mathbf{j}_{\mathrm{ang}} = \mathbf{r}\times\mathbf{j}_{\mathrm{lin}}$ 刚性增强）。每候选槽位固定扩展为 4 触点 ×（1 法向 + 4 spoke）= **20 行**（粒子槽位 5 行）——布局由 cook 与装配共享。

**`SolveSchedule`**（`src/nk/solve/schedule.*`）：主机在 World 构造时**一次**构建的"最坏情况"调度，上传设备后常驻：

- 每个最坏情况行槽位描述冲突键（≤2 个共享可变状态键：稠密 body/关节 tile 键 + λ 组锚 + 关节索引）；
- **贪心最低可用色**按行索引序着色（冲突图边着色）→ 稠密槽位重映射 → union-find 连通分量（岛）→ $(color, \text{intra-color})$ 段发射；
- **最坏情况槽位有效性不变量**：调度按"全部槽位活跃"计算，运行时任意子集活跃仍然有效——着色是超图的合法着色（去激活只删边）；岛是最坏分量界的细化（活跃子集的真分量是最坏岛的并的子集，块内串行化只多不少）；λ 组锚保证摩擦和与紧凑发射相等（非活跃槽位 λ=0 不贡献）；
- 运行时 `SolveRowsBlockIsland` 内核**零主机参与**遍历 $\{\text{island\_row\_offsets},\ \text{island\_color\_segments},\ \text{row\_order}\}$ 三元组，逐块逐颜色 `__syncthreads` 并行行更新。

**PGS 更新公式**（每行每迭代，保持旧解算器数值逐位一致）。先合成两侧投影速度：

$$jv = jv_A + jv_B,\qquad
jv = \begin{cases}
\mathbf{j}_{\mathrm{lin}}\cdot\mathbf{v} + \mathbf{j}_{\mathrm{ang}}\cdot\boldsymbol\omega & \text{rigid}\\[2pt]
\sum_r \mathbf{J}[r]\,\dot{\mathbf{q}}[r] & \text{artic（升序 } r\text{）}\\[2pt]
\mathbf{j}_{\mathrm{lin}}\cdot\mathbf{v} & \text{particle}\\
0 & \text{static}
\end{cases}$$

再更新乘子并应用冲量（$\Delta\lambda = \lambda' - \lambda$，$R$ = compliance_alpha）：

$$\lambda' = \mathrm{clamp}\big(\lambda + m_{\mathrm{eff}}\cdot(\mathrm{rhs}\cdot dt - jv - R\,\lambda),\ \ \mathrm{lower},\ \mathrm{upper}\big)$$

$$\text{rigid:}\ \ \mathbf{v} \mathrel{+}= \mathbf{j}_{\mathrm{lin}}(m^{-1}\Delta\lambda),\quad \boldsymbol\omega \mathrel{+}= \mathbf{j}_{\mathrm{ang}}(\mathbf{I}^{-1}\Delta\lambda)$$
$$\text{artic:}\ \ \dot{\mathbf{q}}[r] \mathrel{+}= \mathbf{w}[r]\,\Delta\lambda,\quad \mathbf{w} = \mathbf{M}^{-1}\mathbf{J}^\top\ (\text{装配期预计算})$$
$$\text{particle:}\ \ \mathbf{v} \mathrel{+}= \mathbf{j}_{\mathrm{lin}}(m^{-1}\Delta\lambda)$$

- $m_{\mathrm{eff}}$ 是装配期预计算的 Compliant 有效质量（迭代间为常量，避免每迭代重算 51-DOF 的 $\mathbf{M}^{-1}\mathbf{J}^\top$ 内积）；
- 摩擦 spoke 的边界 $[0,\ \mu\cdot\text{TotalNormalLambda}(group)]$ 是单侧耦合金字塔锥（`IsCompliantFrictionRow` 语义）；
- 关节 $\dot{\mathbf{q}}$ tile 在岛的**共享内存**中加载/回写（块内串行）；无接触环境回写自身原值（空操作）；
- 非活跃槽位（水位标志 bit0 清除）立即早退——最大网格 + 早退使内核保持可图形捕获；
- `pos_iters` 仅在 PairDriven 路径生效：分裂冲量位置推挤（pseudo velocity），旧 Union 路径保持纯速度（Baumgarte 位置投影为 0）。

### 1.5 单一路径的落实

- 刚体-刚体、粒子-刚体、关节-刚体全部经 `ucontact_*` 缓冲 → `AssembleRows` → 同一 `SolveRowsBlockIsland`；
- XPBD 约束投影（`XpbdProject`）与 PBF 密度投影在解缝内联——同一时间步内先解接触行再投影内部约束，位置修正与速度解通过 $v_{\mathrm{pre}}$/$dv$ 合成（§2.4）；
- MPM 作为一个"网格传递"耦合提供者，`Couple` 在预解缝发出**一个** `MpmStep` op（§3.6）。

---

## 2. README 布料与 Go2 接触 Demo：XPBD 布料实现

README 的 *Cloth × Go2*（`go2_cloth_drape`）是一块 XPBD 布料披在 Go2 四足机器人上，双向 body-particle 耦合。布料本体是 **XPBD（扩展位置动力学）**，约束在 `src/phi/backend_cuda/ops/particles.cu` 的 `XpbdProject` op 中求解，拓扑在 cook 期由 `src/runtime/soft/cloth_topology.hpp` 构建。

### 2.1 布料拓扑（cook 期）

`BuildClothConstraints` 从三角网格构建两条约束族：

- **distance（拉伸）**：每条**唯一边**一条，静止长度 = 静止边长度；`compliance_alpha = 0` 时不可延展（刚性拉伸）；
- **bend（弯曲）**：每个**内部边**（恰被两个三角形共享）一条，采用 Bergou et al. 2006 **等距弯曲**模板。

**Bergou 等距弯曲模板推导**。flap 是 4 顶点 $\{a, b, c_1, c_2\}$（共享边端点 $a,b$ + 两侧三角顶点）。等距标量模板 $k = [k_0, k_1, k_2, k_3]$ 是"线性精确"权集，满足

$$\sum_i k_i = 0 \qquad\text{（平移不变）}$$

$$\sum_i k_i\,\mathbf{x}_i^{\mathrm{rest}} = \mathbf{0} \qquad\text{（旋转不变——平面静止 flap 上线性场梯度为零）}$$

4 个未知数 3 个条件 → 解空间 1 维 → 对非退化（共面）flap，$k$ 被唯一确定到标量（== 余切等距模板的标量倍数；标量吸收进 compliance，符号无关紧要——投影把 $C$ 推向 0）。**代码的实际解法**（`ComputeIsometricBendStencil`）：把 flap 顶点投影到面内正交基 $(u_j, v_j)$（由共享边与 flap 法向构造），$k$ 是 $3\times4$ 矩阵 $\big[(1, u_j, v_j)\big]$ 的零向量，直接按**广义叉积/余子式恒等式**解出

$$k_j = (-1)^j \det(M \setminus j),\qquad \sum_j k_j\,(1,\, u_j,\, v_j) = 0$$

归一化 $\max_j |k_j| = 1$（标量进 alpha）。(u, v) 是 3D 点的等距（刚性）像，故同一 $k$ 在 3D 也满足 $\sum_j k_j \mathbf{x}_j = \mathbf{0}$。存储的每粒子梯度是**常量向量**

$$\mathbf{K}_i = k_i\,\hat{\mathbf{n}}_{\mathrm{rest}} \qquad\text{（}\hat{\mathbf{n}}_{\mathrm{rest}} = \text{单位静止 flap 法向）}$$

于是行的标量约束为

$$C = \sum_i \mathbf{K}_i \cdot \mathbf{p}_i \qquad\text{（平坦静止时 } \equiv 0\text{）}$$

$$\nabla_{\mathbf{p}_i} C = \mathbf{K}_i \qquad\text{（常量，无需每步重算）}$$

退化的 flap（共线/重合/零面积三角）跳过 bend（模板无定义），distance 照发。

### 2.2 XPBD 公式推导（从 PBD 到 XPBD）

**PBD 位置投影**：对约束 $C(\mathbf{x}) = 0$，一步线性化投影

$$\Delta \mathbf{x}_i = w_i\,\nabla C_i\,\Delta\lambda, \qquad \Delta\lambda = -\frac{C}{\sum_j w_j \|\nabla C_j\|^2}, \qquad w_i = \frac{1}{m_i}$$

PBD 把刚度和迭代次数混在一起、无动量守恒的收敛意义。

**XPBD（Macklin 2016）** 引入**柔度（compliance）$\alpha$** 与**乘子 $\lambda$**，把约束写成含势能的正则化形式：

$$\tilde\alpha = \frac{\alpha}{dt^2}$$

$$\Delta\lambda = \frac{-\,C \;-\; \tilde\alpha\,\lambda}{\displaystyle\sum_i w_i \|\nabla C_i\|^2 \;+\; \tilde\alpha}, \qquad \Delta\mathbf{x}_i = w_i\,\nabla C_i\,\Delta\lambda, \qquad \lambda \leftarrow \lambda + \Delta\lambda$$

- $\alpha = 0$ 恢复刚性 PBD；$\alpha > 0$ 允许约束软性偏离；
- $\lambda$ 跨迭代持久（步骤开始时清零 `XpbdLambdaResetKernel`），使多迭代收敛到与迭代数无关的定点——$\alpha = 1/k$ 时对应隐式欧拉力学意义上的弹簧柔度；
- 确定性：**颜色化并行 Gauss-Seidel**——一个颜色内的约束不共享粒子，同一颜色内线程并行无写竞争（无原子，D1）；op 按固定顺序发射颜色。

### 2.3 四种约束的梯度推导（`particles.cu`）

**① distance（拉伸/弹簧）**：

$$C = \|\mathbf{p}_a - \mathbf{p}_b\| - L_0, \qquad \hat{\mathbf{n}} = \frac{\mathbf{p}_a - \mathbf{p}_b}{\|\mathbf{p}_a - \mathbf{p}_b\|}$$

$$\nabla C_a = \hat{\mathbf{n}}, \qquad \nabla C_b = -\hat{\mathbf{n}}, \qquad \sum w\|\nabla C\|^2 = w_a + w_b$$

$$\Delta\lambda = \frac{-\,C - \tilde\alpha\,\lambda}{w_a + w_b + \tilde\alpha}, \qquad \mathbf{p}_a \mathrel{+}= w_a\,\hat{\mathbf{n}}\,\Delta\lambda, \qquad \mathbf{p}_b \mathrel{-}= w_b\,\hat{\mathbf{n}}\,\Delta\lambda$$

**② bend（Bergou，§2.1 模板）**：4 粒子、预计算常量梯度 $\mathbf{K}_j$：

$$C = \sum_j \mathbf{K}_j \cdot \mathbf{p}_j, \qquad \nabla C_j = \mathbf{K}_j\ (\text{常量})$$

$$\Delta\lambda = \frac{-\,C - \tilde\alpha\,\lambda}{\displaystyle\sum_j w_j \|\mathbf{K}_j\|^2 + \tilde\alpha}, \qquad \mathbf{p}_j \mathrel{+}= w_j\,\mathbf{K}_j\,\Delta\lambda$$

**③ volume（四面体体积）**：

$$\mathbf{e}_1 = \mathbf{p}_1 - \mathbf{p}_0, \quad \mathbf{e}_2 = \mathbf{p}_2 - \mathbf{p}_0, \quad \mathbf{e}_3 = \mathbf{p}_3 - \mathbf{p}_0$$

$$C = \det(\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3) - 6V_0 \qquad\text{（rest\_times6 = } 6\cdot\text{静止体积）}$$

$$\nabla C_0 = -(\mathbf{g}_1 + \mathbf{g}_2 + \mathbf{g}_3), \quad \nabla C_1 = \mathbf{e}_2 \times \mathbf{e}_3, \quad \nabla C_2 = \mathbf{e}_3 \times \mathbf{e}_1, \quad \nabla C_3 = \mathbf{e}_1 \times \mathbf{e}_2$$

$$\Delta\lambda = \frac{-\,C - \tilde\alpha\,\lambda}{\displaystyle\sum_i w_i \|\nabla C_i\|^2 + \tilde\alpha}$$

**④ shape-match（Mueller et al. 2005 形状匹配）**：聚类内**刚性目标**，无 $\lambda$：

$$\mathbf{c} = \frac{\sum_i m_i\,\mathbf{p}_i}{\sum_i m_i}, \qquad \mathbf{A} = \sum_i m_i\,(\mathbf{p}_i - \mathbf{c})\,\mathbf{q}_i^\top, \qquad \mathbf{q}_i = \text{静止相对向量}$$

$$\mathbf{R} = \mathrm{polar}(\mathbf{A}) \qquad \text{（Higham 牛顿极分解 24 迭代 + 行列式修正）}$$

$$\mathbf{g}_i = \mathbf{c} + \mathbf{R}\,\mathbf{q}_i, \qquad \mathbf{p}_i \leftarrow \mathbf{p}_i + s\,(\mathbf{g}_i - \mathbf{p}_i) \qquad (s = \text{刚度})$$

极分解迭代：$\mathbf{R} \leftarrow \tfrac{1}{2}(\mathbf{R} + \mathbf{R}^{-\top})$（det 修正保证真旋转），固定 24 次、固定顺序 → D1。

### 2.4 单步完整流程（XPBD 布料）

1. `XpbdPredictKernel`：$\mathbf{p}_{\mathrm{prev}} = \mathbf{p}$；$\mathbf{p} \mathrel{+}= \mathbf{v}\,dt + \mathbf{g}\,dt^2$；$v_{\mathrm{pre}} = \mathbf{v}$（速度不动）。若有气动阻力，`ParticleAeroDrag` 在 predict 前修改 $\mathbf{v}$——布料受风。
2. `XpbdProject`（iters 次）：distance / bend / volume / shape-match 各按颜色扫描 iters 次。
3. `XpbdCorrectKernel`：

$$\mathbf{v}_{\mathrm{pbd}} = \frac{\mathbf{p}_{\mathrm{proj}} - \mathbf{p}_{\mathrm{prev}}}{dt} \qquad\text{（投影速度）}$$

$$\mathbf{v} = \mathbf{v}_{\mathrm{pbd}} + (\mathbf{v}_{\mathrm{now}} - \mathbf{v}_{\mathrm{pre}}) \qquad\text{（合成接触修正 } dv\text{）}$$

$$\mathbf{p} \mathrel{+}= (dv + \mathbf{v}_{\mathrm{pseudo}})\,dt \qquad\text{（分裂冲量位置推挤，不注入能量）}$$

- `iters` 由场景配置（Go2 布料 `iters=30`）；
- 分裂冲量：位置推挤只推位置不动速度，避免能量注入；pseudo velocity 来自位置级接触行（PairDriven `pos_iters`）。

### 2.5 与刚体的耦合（双向）

**关键设计**：粒子被建模为**一个半径为 `particle_radius` 的球**，走**同一条**接触路径（`narrowphase_body_particle.cu`）：

1. 每个粒子构建球查询 AABB，遍历环境 LBVH（`data.lbvh_nodes`）收集重叠刚体候选（私有插入排序列表，上限 `kCrossSystemMaxCandidates`，超限 OR `kEnvStatusPairOverflow`）；
2. 对每个候选刚体跑 **sphere-vs-shape** 流形：图元对用闭式；凸包用 `cvx::SphereHull`（EPA 旁路最近点查询，规避 EPA 浅穿透死区）；高度场用面处理；
3. 流形写入粒子的**保留槽位子区间**——基址是粒子索引的**固定函数**（非原子），body-particle 槽位流天然 D1，且与刚性-刚性槽位有确定性的子区间排序；
4. 行侧：A = 粒子（全局 id + `kUContactSideParticle` 标签），B = 刚体 collidable（`kUContactSideBody`）；流形法向 = 粒子的分离方向（把粒子推出刚体）；
5. 同一 `AssembleRows` 扩展成 `NkRow`，同一 `SolveRowsBlockIsland` 求解。

**反向**：刚体在解中通过行 Jacobian 收到粒子冲量（$\mathbf{v} \mathrel{+}= \mathbf{j}_{\mathrm{lin}}\,m^{-1}\Delta\lambda$）——布料压在机器人上，机器人被布料拖拽，双向耦合自然成立。

### 2.6 Go2 场景配置（`bdx_oneshot*.nks`）

```text
cloth_grid: 26×28 格, spacing 0.02, pin 4（4 个固定点挂起）
xpbd: particle_mass 0.003, friction 1.0, distance_alpha 0.0（不可延展）,
      bend_alpha 0.02（可弯曲）, iters 30
      aero_drag_normal 0.8, aero_drag_tangent 0.2（布料风阻）
```

---

## 3. 主页 Demo：MLS-MPM（Bunny-Water 与 Jelly）

`mpm_water_drop_demo.cpp`（重刚体兔坠入 MLS-MPM 水池，双向耦合）与 `mpm_jelly_demo.cpp`（弹性果冻球落地反弹）共用 `src/phi/backend_cuda/ops/mpm.cu` 的 **MLS-MPM（APIC；Hu et al. 2018）** 步进 op `MpmStep`。

### 3.1 先看整体：一个子步里的五件事

MLS-MPM 每步做的事情一句话：**把粒子上的材料状态搬到网格，在网格上解动量方程与边界条件，再把更新后的速度搬回粒子，最后按速度梯度更新每个粒子的变形**。每次 `MpmStep` 内含 `substeps` 次子步（水 demo 40 次、jelly 20 次，为显式时间步满足 CFL）：

```text
┌───────────────────────────────────────────────────────────────┐
│  子步循环（每帧 substeps 次）                                   │
│                                                                │
│  ① P2G        粒子 → 网格：质量 / 动量(APIC) / 本构力 外插到节点 │
│  ② 网格更新   重力踢 + 动量归一化 + 静态平面 BC + 分离域壁 BC     │
│  ③ 刚体耦合   （可选）SDF 速度投影 + 反作用冲量归还             │
│  ④ G2P        网格 → 粒子：v_p 插值、C_p 拟合、位置对流          │
│  ⑤ F 更新     变形梯度（弹性 F、流体 J、颗粒 Drucker-Prager 回映）│
└───────────────────────────────────────────────────────────────┘
```

为什么"粒子存状态、网格解方程"（§3.2 选型）？每一步"为什么这么写、公式从哪来"（§3.4–3.8 逐步展开）？三个本构模型单独讲透（§3.9）？§3.12 给出面试口述版。

### 3.2 为什么是 MPM：方法与选型

- **纯拉格朗日网格法**（FEM）：形变网格在大变形/断裂/飞溅下纠缠、翻转，需要昂贵的 remesh；
- **纯无网格粒子法**（SPH）：不用网格，但自由表面、密度场与近不可压约束难精确控制，邻居搜索昂贵；
- **MPM**：**粒子携带全部材料状态**（位置、速度、变形梯度、质量、初始体积），**网格只临时承载动量方程的求解**。粒子随材料流动（拉格朗日），网格每步重建（欧拉计算舞台）——自由表面、大变形、拓扑变化（splash、飞溅）天然无压力。

**MLS-MPM（Hu et al. 2018）** 相对经典 MPM 的两个关键改进：

1. **APIC 仿射速度场**（Jiang et al. 2015/2017）：粒子除速度 $\mathbf{v}_p$ 外再携带一个 $3\times3$ 仿射矩阵 $\mathbf{C}_p$。P2G 用 $\mathbf{v}_p + \mathbf{C}_p(\mathbf{x}_i-\mathbf{x}_p)$ 而不是纯 $\mathbf{v}_p$ 传动量——PIC 纯插值每步把高频抹平（耗散），FLIP 用全程差分引入噪声，APIC 在两者之间：**守恒线动量与角动量**，同时保留亚格分辨率的梯度（涡量/角速度）不丢失；
2. **应力散度新离散**：本构力写成 $-\sum_p w_{ip} V_p^0 \frac{4}{dx^2}\boldsymbol\tau_p(\mathbf{x}_i-\mathbf{x}_p)$ 的紧凑形式，单步计算量约减半。

### 3.3 粒子与网格：各存什么

| 量 | 符号 | 含义 |
|---|---|---|
| 位置 / 速度 | $\mathbf{x}_p,\ \mathbf{v}_p$ | 粒子位置与速度 |
| 变形梯度 | $\mathbf{F}_p$ | 参考构型 → 当前构型的形变张量（初始 $\mathbf{I}$） |
| 仿射矩阵 | $\mathbf{C}_p$ | APIC 仿射项，粒子邻域速度梯度的最小二乘代理（§3.4/3.7） |
| 质量 / 初始体积 | $m_p,\ V_p^0$ | 拉格朗日守恒量，不随时间变 |
| 节点质量 / 动量 | $m_i,\ (m\mathbf{v})_i$ | 网格临时量，每子步清零重建 |
| 核函数 | $w_{ip}$ | 三次 B 样条乘积，支撑 $3^3$ 个节点，粒子-网格传递的桥 |

网格节点坐标 $\mathbf{x}_i$ 落在**固定笛卡尔格**上；粒子与节点通过 $w_{ip} = N(\mathbf{x}_i - \mathbf{x}_p)$ 相互传递（三次 B 样条，$C^2$ 光滑、可微）。

### 3.4 第 1 步 P2G：粒子 → 网格（`MpmP2GGatherKernel`）

**为什么需要这一步**：动量方程是 PDE，梯度/散度算子定义在网格上，离散求解最方便。所以先把粒子的质量、动量"外插"到节点。

**① 质量**（节点质量 = 粒子质量 × 核权重之和）：

$$m_i = \sum_p w_{ip}\, m_p$$

**② 动量（APIC）**：每个粒子贡献的动量不是 $m_p\mathbf{v}_p$，而是带仿射修正：

$$(m\mathbf{v})_i = \sum_p w_{ip}\, m_p\,\big(\mathbf{v}_p + \mathbf{C}_p(\mathbf{x}_i - \mathbf{x}_p)\big)$$

$\mathbf{C}_p(\mathbf{x}_i-\mathbf{x}_p)$ 的意义：粒子周围的真实速度场不是常数，而是"粒子速度 + 线性修正"。$\mathbf{C}_p$ 记录了邻域速度场的线性部分（旋转/剪切/伸缩），外插时把相对位移乘上它，**把亚格分辨率的梯度信息一并带到网格**——这是 APIC 不耗散的关键。

**③ 本构力**（把连续体内部应力"翻译"成节点力，折叠为动量增量 $dt\,\mathbf{f}_i$）：

$$\mathbf{f}_i = -\sum_p w_{ip}\, V_p^0\, \frac{4}{dx^2}\, \boldsymbol\tau_p\,(\mathbf{x}_i - \mathbf{x}_p), \qquad \boldsymbol\tau_p = \mathbf{P}(\mathbf{F}_p)\,\mathbf{F}_p^\top$$

**这个公式怎么来**（连续 → 离散三步）：

1. 连续动量方程含应力散度项 $\nabla\cdot\boldsymbol\sigma$（Cauchy 应力）；
2. 用网格核做测试函数取弱形式，分部积分把应力散度变成"应力 × 测试函数梯度"的积分 $\int \boldsymbol\sigma : \nabla w_{ip}\, dV$；
3. 离散：积分 → 粒子求和；$\nabla w_{ip} \to \frac{4}{dx^2} w_{ip}(\mathbf{x}_i-\mathbf{x}_p)$（三次 B 样条质量矩阵 $\mathbf{W} = \frac{dx^2}{4}\mathbf{I}$ 的逆给出系数 $\frac{4}{dx^2}$）；体积元 $dV \to V_p^0$；Cauchy 应力换成与参考体积元配套的 **Kirchhoff 应力** $\boldsymbol\tau = J\boldsymbol\sigma = \mathbf{P}\mathbf{F}^\top$——即得 $\mathbf{f}_i$。

**代码落地**：`MpmPrecomputeStressKernel` 先算好每个粒子的 $\boldsymbol\tau_p$（弹性一次 $\mathbf{P}\mathbf{F}^\top$；流体/颗粒直接产出 $\boldsymbol\tau$），P2G 免去每目标节点重复的 ~27 次 SVD/polar。粒子先按 env-offset 格键 **radix sort**，节点按排序序累加、`__fadd_rn` 钉住浮点加法序 → 两次运行逐位一致（D1）。

### 3.5 第 2 步 网格更新与边界条件（`MpmGridUpdateKernel`）

动量方程在网格上解：节点动量已含本构力冲量 $dt\,\mathbf{f}_i$，加重力踢后归一化得节点速度：

$$\mathbf{v}_i = \frac{(m\mathbf{v})_i + dt\, m_i\, \mathbf{g}}{m_i}$$

（节点质量 $m_i$ 过小则置零速度，防除噪。）然后按需施加三类边界条件：

**① 静态平面 BC**（地板）。节点在平面下且接近（$\mathbf{v}\cdot\hat{\mathbf{n}} < 0$）时，切法向 + 库仑摩擦：

$$\mathbf{v}_t = \mathbf{v} - \hat{\mathbf{n}}(\mathbf{v}\cdot\hat{\mathbf{n}}), \qquad \text{预算} = \mu\,(-\mathbf{v}\cdot\hat{\mathbf{n}})$$

$$\|\mathbf{v}_t\| \le \text{预算} \Rightarrow \mathbf{v} = \mathbf{0}\ \text{（静摩擦粘住）}; \qquad \|\mathbf{v}_t\| > \text{预算} \Rightarrow \mathbf{v} = \mathbf{v}_t\Big(1 - \frac{\text{预算}}{\|\mathbf{v}_t\|}\Big)\ \text{（动摩擦拖拽）}$$

**② 分离域壁 BC**（水箱 x/y 壁）：最外层节点环上**只切掉向外**的速度法向分量——水被限制在网格盒内，但不会被"吸住"。

**③ 刚体 SDF BC**：见 §3.6（耦合）。

### 3.6 第 3 步 刚体耦合：双向（SDF 网格 BC）

MPM 与刚体/关节的耦合**没有专门的流固耦合器**，全部通过网格 BC 涌现，三个内核三件事：

**① 选 owner**（`MpmGridBodyProjectKernel`）：每个网格节点在其 $3^3$ 邻域内的刚体中，挑 **SDF 采样最深**（$\varphi$ 最负，即"插进刚体最深"）的作唯一 owner（确定性单一归属，避免多刚体拉扯）。刚体表面速度：

$$\mathbf{v}_{\mathrm{surf}} = \mathbf{v}_b + \boldsymbol\omega_b \times (\mathbf{x}_i - \mathbf{x}_b)$$

**② 速度投影**：节点相对速度 $\mathbf{v}_{\mathrm{rel}} = \mathbf{v} - \mathbf{v}_{\mathrm{surf}}$，若 $\mathbf{v}_{\mathrm{rel}}\cdot\hat{\mathbf{n}} < 0$（在接近表面）→ 切法向 + 库仑摩擦得 $\mathbf{v}_{\mathrm{after}}$。记录该节点因刚体产生的动量变化：

$$\Delta\mathbf{p} = (\mathbf{v}_{\mathrm{after}} - \mathbf{v}_{\mathrm{before}})\,m_i$$

**③ 反作用**（`MpmGridBodyReactKernel` + `MpmArticReactDepositKernel`）：$\Delta\mathbf{p}$ 按 stable sort 归集到每个刚体：

$$\mathbf{J}_{\mathrm{lin}} = -\sum \Delta\mathbf{p}, \qquad \mathbf{J}_{\mathrm{ang}} = -\sum (\mathbf{x}_i - \mathbf{x}_b)\times\Delta\mathbf{p}$$

- **自由刚体**：直接应用 $\mathbf{v} \mathrel{+}= \mathbf{J}_{\mathrm{lin}}/m$，$\boldsymbol\omega \mathrel{+}= \mathbf{I}^{-1}\mathbf{J}_{\mathrm{ang}}$；
- **关节体**（cooked inv_mass=0）：力螺旋存进 `body_reaction`，随后每关节一线程沿链式 Jacobian 累加进广义力 $\mathbf{g}$，再 $\dot{\mathbf{q}} \mathrel{+}= \mathbf{M}^{-1}\mathbf{g}$——即通过 **$\mathbf{M}^{-1}\mathbf{J}^\top$ 冲量沉积** 受流体反作用。

**这就是"双向"**：bunny 坠入时，靠近它的节点被 bunny 表面速度"推着走"（水被排开），同时节点动量变化以冲量还给 bunny（减速/浮起）。splash 与刹车是同一个网格 BC 的两面，无特例代码。

### 3.7 第 4 步 G2P：网格 → 粒子（`MpmG2PGatherKernel`）

网格速度是"场"，粒子拿回自己的速度，同时**拟合**新的 $\mathbf{C}_p$：

$$\mathbf{v}_p = \sum_i w_{ip}\,\mathbf{v}_i$$

$$\mathbf{C}_p = \frac{4}{dx^2}\sum_i w_{ip}\,\mathbf{v}_i\,(\mathbf{x}_i - \mathbf{x}_p)^\top \qquad\text{（MLS 最小二乘拟合）}$$

$$\mathbf{x}_p \mathrel{+}= dt\,\mathbf{v}_p \qquad\text{（对流；粒子钉在材料上）}$$

**$\mathbf{C}_p$ 公式的来源**：在粒子邻域内把网格速度场近似成仿射场 $\mathbf{v}(\mathbf{x}) \approx \mathbf{v}_p + \mathbf{C}_p(\mathbf{x}-\mathbf{x}_p)$，按权重 $w_{ip}$ 最小化残差 $\sum_i w_{ip}\big\|\mathbf{v}_i - \mathbf{v}_p - \mathbf{C}_p(\mathbf{x}_i-\mathbf{x}_p)\big\|^2$，一阶最优条件就是上面的式子；系数 $\frac{4}{dx^2}$ 同样来自三次 B 样条质量矩阵的逆。**所以 $\mathbf{C}_p$ 就是该粒子处速度梯度的最小二乘代理**——第 5 步的变形梯度更新直接用到它。

### 3.8 第 5 步 变形梯度更新（`MpmUpdateFKernel`）

变形梯度描述"这一小块材料从参考状态变成了什么样"，速度梯度负责拉伸/旋转它：

$$\mathbf{F}^{n+1} = (\mathbf{I} + dt\,\nabla\mathbf{v})\,\mathbf{F}^n \approx (\mathbf{I} + dt\,\mathbf{C}_p)\,\mathbf{F}^n \qquad\text{（弹性）}$$

（$\nabla\mathbf{v}$ 用 §3.7 的 $\mathbf{C}_p$ 代理——这就是 $\mathbf{C}_p$ 在循环里的位置。）

**流体**（`model_kind=3`）只关心体积、不关心剪切：$\det(\mathbf{I} + dt\,\mathbf{C}_p) \approx 1 + dt\,\mathrm{tr}\,\mathbf{C}_p$（一阶），

$$J \leftarrow J\,(1 + dt\,\mathrm{tr}\,\mathbf{C}_p), \qquad \mathbf{F} = J^{1/3}\,\mathbf{I}$$

流体无剪切模量，$\mathbf{F}$ 直接各向同性化为 $J^{1/3}\mathbf{I}$——剪切不产生应力、不引起倒置，体积由 $J^{-\gamma}$ 压力（§3.9.4）负责拉回 1。

**颗粒**（`model_kind=4`）：先弹性预测 $\mathbf{F}^{\mathrm{trial}} = (\mathbf{I} + dt\,\mathbf{C}_p)\mathbf{F}^n$，再做 Drucker-Prager 塑性回映（§3.9.5）。

### 3.9 本构模型详解

#### 3.9.1 预备：应力与变形度量

- **变形梯度** $\mathbf{F} = \partial \mathbf{x}/\partial \mathbf{X}$：参考构型（初始）里的微小线段 $d\mathbf{X}$ 被映射成当前构型里的 $\mathbf{F}\,d\mathbf{X}$。行列式 $J = \det\mathbf{F}$ = 体积比（$J=1$ 不可压，$J>1$ 膨胀，$J<1$ 压缩）。
- **第一 Piola-Kirchhoff 应力** $\mathbf{P}(\mathbf{F})$：作用在**参考构型**上的应力度量——参考面元 $d\mathbf{A}$（法向 $\mathbf{N}$）上的力 $= \mathbf{P}\,\mathbf{N}\,d\mathbf{A}$。超弹性本构最自然写成 $P = \partial\psi/\partial F$（能量密度对变形求导）。
- **Kirchhoff 应力** $\boldsymbol\tau = \mathbf{P}\mathbf{F}^\top = J\boldsymbol\sigma$（$\boldsymbol\sigma$ 为 Cauchy 应力）：代码里粒子存的就是 $\boldsymbol\tau$，因为 MPM 弱形式（§3.4 ③）里与参考体积元 $V_p^0$ 配套出现的是它。

#### 3.9.2 固定共旋转弹性（Stomakhin 2012，jelly 默认）

**为什么需要"共旋转"**：最简单的线性弹性 $\psi = \frac{\mu}{2}\|\mathbf{F} - \mathbf{I}\|_F^2$ 只对**小变形**成立——物体整体旋转（$\mathbf{F} = \mathbf{R}$，纯旋转）时 $\|\mathbf{F}-\mathbf{I}\| \ne 0$，能量会说"物体变形了"，产生伪应力（物体转个角度自己会崩）。真实材料**纯旋转不产生任何应力**。

**修法：先拧掉旋转**。极分解 $\mathbf{F} = \mathbf{R}\,\mathbf{U}$（$\mathbf{R}$ 旋转、$\mathbf{U}$ 对称正定），能量只惩罚"偏离纯旋转的部分"：

$$\psi(\mathbf{F}) = \mu\,\|\mathbf{F} - \mathbf{R}\|_F^2 + \frac{\lambda}{2}(J - 1)^2$$

- 第一项：剪切/拉伸形变（相对刚性旋转）的能量，$\mu$ = 剪切模量；
- 第二项：体积变化的能量，$\lambda$ = Lamé 第一参数；$J=1$ 时为零（体积不变无能量）。

**推导应力**：$P = \partial\psi/\partial F$。第一项对 $F$ 求导得 $2\mu(\mathbf{F}-\mathbf{R})$（$\mathbf{R}$ 是 $F$ 的最近旋转，其一阶变分对能量的贡献为 0——这是"最近旋转"的定义）；第二项用 Jacobi 公式 $\partial J/\partial\mathbf{F} = J\mathbf{F}^{-\top}$：

$$\mathbf{P}(\mathbf{F}) = 2\mu(\mathbf{F} - \mathbf{R}) + \lambda\,(J-1)\,J\,\mathbf{F}^{-\top}$$

**直觉**：$\mathbf{F} = \mathbf{R}$（纯旋转）时第一项为零、第二项为零（$J=1$）→ 零应力 ✓；球被压扁（$\mathbf{U} \ne \mathbf{I}$）→ 第一项把形变弹回；被压缩（$J<1$）→ 第二项按 $\lambda$ 推回。代码里 $\mathbf{R}$ 由自写确定性 3×3 SVD（固定迭代、无数据相关分支）取 $\mathbf{R} = \mathbf{U}\mathbf{V}^\top$。

#### 3.9.3 Neo-Hookean（`model_kind=2` 备选）

Neo-Hookean 是最常用的大变形超弹性模型之一（对数体积项版本，Stomakhin 2012 同文）：

$$\psi(\mathbf{F}) = \frac{\mu}{2}\big(\mathrm{tr}(\mathbf{F}^\top\mathbf{F}) - 3\big) - \mu\ln J + \frac{\lambda}{2}(\ln J)^2$$

- 第一项：$\mathrm{tr}(\mathbf{F}^\top\mathbf{F}) = \|\mathbf{F}\|_F^2 = I_1$（第一主不变量），纯旋转时 $\|\mathbf{R}\|_F^2 = 3$ 该项为零——**内禀旋转不变，不需要显式极分解**；
- 第二、三项：对数体积项，$J \to 0$ 时 $-\mu\ln J \to +\infty$——**自动抵抗压缩到零体积**（体积排斥墙），是大变形下比二次惩罚更物理的性质。

**推导应力**：$\partial\|\mathbf{F}\|_F^2/\partial\mathbf{F} = 2\mathbf{F}$，$\partial\ln J/\partial\mathbf{F} = \mathbf{F}^{-\top}$：

$$\mathbf{P}(\mathbf{F}) = \mu(\mathbf{F} - \mathbf{F}^{-\top}) + \lambda\,\ln J\,\mathbf{F}^{-\top}$$

**与固定共旋转的对比**：

| | 固定共旋转 | Neo-Hookean |
|---|---|---|
| 能量 | $\mu\|\mathbf{F}-\mathbf{R}\|^2 + \frac{\lambda}{2}(J-1)^2$ | $\frac{\mu}{2}(I_1-3) - \mu\ln J + \frac{\lambda}{2}(\ln J)^2$ |
| 旋转处理 | 显式极分解 $\mathbf{F}=\mathbf{R}\mathbf{U}$ | 内禀各向同性（$I_1$ 旋转不变） |
| 压缩行为 | 二次惩罚（线性刚度） | $\ln J$ 对数排斥（$J\to0$ 应力 $\to\infty$） |
| 实现成本 | 每粒子一次 SVD | 一次 $\mathbf{F}^{-\top}$（转置逆） |
| 典型用途 | 实时图形学（本项目 jelly） | 大压缩/极端变形（软组织等） |

两者都满足"纯旋转零应力"（物理必需）；大压缩下 Neo-Hookean 更硬、更稳定。

#### 3.9.4 弱可压缩流体（`model_kind=3`，Tait EOS，water drop 默认）

**为什么"弱可压缩"而不是严格不可压缩**：严格不可压缩需要每步解压力泊松方程（投影法），在 MPM 里昂贵且难保持粒子自由度。**弱可压缩**用一个大体积模量 $K$ 允许 $J$ 在 1 附近有微小偏差，把压力**显式**写出来——用显式时间步的代价换掉隐式压力求解。

**Tait 状态方程**：把密度比映射成压力：

$$p(\rho) = K\Big(\big(\tfrac{\rho}{\rho_0}\big)^{\gamma} - 1\Big)$$

密度与变形的关系 $\rho/\rho_0 = V_0/V = 1/J$，代入即代码形式（`MpmPrecomputeStressKernel`）：

$$p = \max\!\big(K\,(J^{-\gamma} - 1),\ 0\big), \qquad K = 2\times10^5\ \mathrm{Pa},\ \gamma = 7$$

- $J^{-1}$ 即密度比：$J<1$（压缩）→ 正压；$J>1$（膨胀）→ 负压（吸力）；
- $\gamma = 7$ 是水的经验指数（Tait 方程拟合实验压缩数据，小体积变化内极硬）；
- **钳制 $\ge 0$**：水不能承受拉力（会空化），膨胀区压力直接取 0——自由表面"不拉裂"由此而来；
- 声速 $c = \sqrt{K/\rho_0} \approx 14\ \mathrm{m/s}$（代码注释口径）$\gg$ 撞击速度 3.4 m/s——体积模量把内部 $J$ 钉在 1 附近（近不可压缩），CFL 由 $c$ 决定（§3.11）。

**应力**：压力各向同性，Cauchy 应力 $\boldsymbol\sigma = -p\,\mathbf{I}$，转 Kirchhoff：

$$\boldsymbol\tau = -pJ\,\mathbf{I}$$

**粘度**（牛顿流体 $\boldsymbol\sigma_{\mathrm{visc}} = 2\nu\,\mathbf{D}$，$\mathbf{D} = \tfrac{1}{2}(\nabla\mathbf{v} + \nabla\mathbf{v}^\top)$ 为应变率张量）：

$$\boldsymbol\tau \mathrel{+}= J\,\nu\,(\mathbf{C} + \mathbf{C}^\top)$$

应变率用 APIC 仿射 $\mathbf{C}_p$ 作 $\nabla\mathbf{v}$ 的代理（§3.7 已说明 $\mathbf{C}_p$ 即速度梯度最小二乘拟合）：对角 $2C_{kk}$ = 拉伸/压缩率，非对角 $C + C^\top$ = 剪切率；$J$ 因子保持 Kirchhoff 语义。低粘度 0.4 只压显式格点振荡，不吞表面波/冠冕。

**为什么流体只更新 $J$ 不更新剪切**：流体没有剪切模量（$\mu = 0$），剪切应变不产生应力；体积由 $J \leftarrow J(1 + dt\,\mathrm{tr}\,\mathbf{C}_p)$ 更新（§3.8），下一子步的 $J^{-\gamma}$ 压力由此而来。

#### 3.9.5 颗粒材料（`model_kind=4`，Klar et al. 2016，沙）

超弹性用 Hencky 对数应变：

$$\boldsymbol\varepsilon = \ln\sigma_i\ (\text{特征值}), \qquad \boldsymbol\tau = \mathbf{U}\,\mathrm{diag}(2\mu\boldsymbol\varepsilon + \lambda\,\mathrm{tr}(\boldsymbol\varepsilon))\,\mathbf{U}^\top$$

塑性用 Drucker-Prager 屈服锥回映（Box 3）：拉过凝聚顶点 → 无应力（丢弃体积塑性应变）；剪切过锥 → 径向回映；锥内 → 弹性。效果：沙有摩擦角、能堆起来、能崩塌（Klar 2016 验证）。

### 3.10 沃辛顿射流（Worthington jet）效应机制

README 视频：bunny 坠入 → **冠冕（crown sheet）** → 下潜形成**空腔（air cavity）** → 空腔坍缩把水向中心汇聚、动能聚焦成高速**中心射流（Worthington jet）** → **辐射波纹（ripple rings）** 外扩。文献确认射流与空腔形成/坍缩密切相关（crown → cavity growth → jet generation）。在 MLS-MPM 中这是**自然涌现**，无需任何射流/表面张力专用代码，四个要素叠加：

1. **近不可压缩**：Tait EOS $p = K(J^{-\gamma}-1)$ 在 $K = 2\times10^5$ Pa 下把被排开的水体积"逼"向唯一出口——物体边缘向上 → 冠冕；下潜留下的空腔在静水压/浮力下坍缩时，动量守恒 + 体积守恒把动能聚焦成中心射流；
2. **双向网格 BC**：bunny 的 SDF 表面速度驱动节点（撞击点水被推离），反作用冲量同时减速 bunny——排开的水获得向上动量；
3. **低粘度**：$\nu = 0.4$ 只压显式格点振荡（自由表面落平），不吞冠冕/射流/波纹——"水，不是果冻"；
4. **多子步 + 域约束**：`substeps = 40` 满足 CFL（$c\,dt_{\mathrm{sub}} \approx 0.09\,dx$，4 倍余量，代码注释口径）；网格 xy 分离壁 BC 当水箱壁、地板平面 BC 兜底，能量限制在池内让波纹多次反射可见。

数值层面：MLS 仿射项 $\mathbf{C}_p$ 提供亚格角动量，避免数值耗散提前把冠冕"溶掉"——射流是动量守恒 + 体积守恒 + 自由表面在 MPM 弱形式下的自然解。

### 3.11 Demo 参数（与代码常量一致）

| 参数 | bunny-water | jelly |
|---|---|---|
| dx | 0.011 | 0.025 |
| substeps / dt | 40 / $\tfrac{1}{240}$ s，CFL $dt_{\mathrm{sub}} < 0.4\,dx/c$（实取 $0.09\,dx$） | 20（显式 CFL 余量） |
| 本构 | 流体（$K{=}2{\times}10^5$, $\gamma{=}7$, $\nu{=}0.4$） | 固定共旋转（$Y{=}3{\times}10^4$, $\nu{=}0.3$） |
| 采样 | dx/2 格（8 粒/格），$\rho_0 = 1000$ | dx/2 格 |
| 边界 | 盒壁分离 BC + 平面 BC（$\mu{=}0$ 滑移） | 静态平面 BC（Coulomb） |
| 刚体 | bunny 2.5 kg（石密），释放高 0.6 m，双向 SDF BC | 无动态体，单向 |
| 渲染 | 等值面 marching + 折射（ior 1.33） | 皮肤绑定最近 MPM 粒子 + Taubin 平滑 |

### 3.12 面试口述版（约 2 分钟）

> **一句话**：MLS-MPM 是 MPM 的一个变体——粒子携带全部材料状态，网格只当临时计算舞台解动量方程；两个核心改进是 APIC 仿射速度场（保动量、不耗散）和更紧凑的应力散度离散（快约两倍）。
>
> **为什么用 MPM**：要模拟水花四溅、自由表面、大变形和拓扑变化，纯网格法（FEM）要 remesh，纯粒子法（SPH）难控制密度和近不可压；MPM 让粒子跟着材料走（拉格朗日），网格每步重建（欧拉舞台），两边的好处都拿到。
>
> **一个子步五件事**：
> 1. **P2G 粒子→网格**：把质量、动量外插到节点。动量不是简单的 $m_p\mathbf{v}_p$，而是 $m_p(\mathbf{v}_p + \mathbf{C}_p(\mathbf{x}_i-\mathbf{x}_p))$——$\mathbf{C}_p$ 是 APIC 仿射矩阵、粒子邻域速度梯度的代理，把角动量/涡量带到网格，避免 PIC 的耗散和 FLIP 的噪声。再把本构力 $\mathbf{f}_i = -\sum_p w_{ip} V_p^0 \frac{4}{dx^2}\boldsymbol\tau_p(\mathbf{x}_i-\mathbf{x}_p)$ 折进动量，其中 $\boldsymbol\tau = \mathbf{P}\mathbf{F}^\top$ 是 Kirchhoff 应力，$\frac{4}{dx^2}$ 是三次 B 样条质量矩阵的逆。
> 2. **网格更新**：$\mathbf{v}_i = ((m\mathbf{v})_i + dt\,m_i\mathbf{g})/m_i$ 重力踢，然后施加边界条件——静态平面库仑摩擦、水箱分离壁、刚体 SDF 速度投影。
> 3. **刚体反作用**（有刚体时）：节点因刚体产生的动量变化 $\Delta\mathbf{p}$ 归集为线/角冲量还给刚体；关节体经 $\mathbf{M}^{-1}\mathbf{J}^\top$ 沉积到广义坐标——这就是双向耦合。
> 4. **G2P 网格→粒子**：$\mathbf{v}_p = \sum_i w_{ip}\mathbf{v}_i$；$\mathbf{C}_p = \frac{4}{dx^2}\sum_i w_{ip}\mathbf{v}_i(\mathbf{x}_i-\mathbf{x}_p)^\top$ 是 MLS 最小二乘的最优仿射项；$\mathbf{x}_p \mathrel{+}= dt\,\mathbf{v}_p$。
> 5. **F 更新**：$\mathbf{F}^{n+1} = (\mathbf{I} + dt\,\mathbf{C}_p)\mathbf{F}^n$（$\mathbf{C}_p$ 就是速度梯度代理）；弹性直接用；流体只更新体积 $J \leftarrow J(1+dt\,\mathrm{tr}\,\mathbf{C}_p)$、$\mathbf{F} = J^{1/3}\mathbf{I}$；颗粒做 Drucker-Prager 回映。
>
> **本构怎么选**：弹性用固定共旋转 $\mathbf{P} = 2\mu(\mathbf{F}-\mathbf{R}) + \lambda(J-1)J\mathbf{F}^{-\top}$（极分解 $\mathbf{F}=\mathbf{R}\mathbf{U}$ 拧掉旋转，只惩罚偏离刚性旋转的形变和体积变化，纯旋转零应力）；Neo-Hookean 是内禀旋转不变的大变形模型 $\mathbf{P} = \mu(\mathbf{F}-\mathbf{F}^{-\top}) + \lambda\ln J\,\mathbf{F}^{-\top}$（对数体积项天然抗压缩到零）；水用 Tait EOS $p = \max(K(J^{-\gamma}-1), 0)$——弱可压缩（$K$ 大把 $J$ 钉在 1 附近）避免每步解压力泊松方程，$\gamma=7$ 是水的经验指数，钳制 $\ge 0$ 因为水不抗拉。
>
> **加分点（确定性）**：粒子按格键排序、固定序浮点累加（`__fadd_rn`）、自写固定迭代 SVD，两次运行逐位一致——这是 RL 训练可复现的工程关键。

**常见追问 Q&A**：

| 追问 | 回答 |
|---|---|
| 为什么 APIC 而不是 PIC/FLIP？ | PIC 纯插值每步把高频信息抹平（耗散）；FLIP 用全程差分引入噪声、不保能量；APIC 带仿射项，守恒线动量+角动量，介于两者之间。 |
| $\frac{4}{dx^2}$ 哪来的？ | 三次 B 样条质量矩阵 $\mathbf{W} = \frac{dx^2}{4}\mathbf{I}$ 的逆——P2G 的应力散度项和 G2P 的 $\mathbf{C}_p$ 拟合都要"除以质量矩阵"。 |
| $\mathbf{C}_p$ 的物理意义？ | 粒子邻域速度场的线性部分（速度梯度代理）：P2G 用它把梯度带上网，G2P 用它更新 $\mathbf{F}$。 |
| 流体为什么弱可压缩？ | 严格不可压缩要每步解压力泊松方程（隐式、贵）；Tait EOS 把压力显式写出来，$K$ 大就近似不可压，代价是声速决定 CFL，所以要 substeps。 |
| 刚体耦合是双向的吗？ | 是。节点速度被刚体 SDF 表面速度投影（单向施加），节点动量变化以冲量还给刚体（反作用）；关节体经 $\mathbf{M}^{-1}\mathbf{J}^\top$ 沉积。 |
| 为什么不直接用 SPH 做水？ | MPM 无密度演化误差、自由表面自然、与刚体/布料共用统一接触行边界；本工程还要求 GPU 确定性，MPM 的网格排序累加天然 D1。 |

## 4. 柔体补充：XPBD/PBD 约束深入

### 4.1 PBD 与 XPBD 的数学关系

PBD 把约束作为**位置层投影**处理：预测位置 $\mathbf{p}^{*} = \mathbf{p} + \mathbf{v}\,dt + \mathbf{g}\,dt^2$，迭代投影 $\Delta\mathbf{x} = w\nabla C\,(-C / \sum w\|\nabla C\|^2)$，最后合成速度 $\mathbf{v} = (\mathbf{p}' - \mathbf{p})/dt$。问题是：刚度由迭代次数决定、无乘子、与真实弹性的对应关系模糊。

XPBD 从**隐式时间离散的能量最小化**出发：约束 $C(\mathbf{x}) = 0$ 带拉格朗日乘子 $\lambda$ 与正则化柔度 $\alpha$（Macklin 2016），离散后得到

$$\tilde\alpha = \frac{\alpha}{dt^2}, \qquad \Delta\lambda = \frac{-\,C \;-\; \tilde\alpha\,\lambda}{\displaystyle\sum_i w_i \|\nabla C_i\|^2 \;+\; \tilde\alpha}$$

物理意义：

- $\alpha = 0$：刚性约束（PBD 极限）；
- $\alpha = 1/k$：弹簧柔度；$\Delta\lambda$ 分母的 $+\tilde\alpha$ 是对偶正则化，等价于隐式欧拉的 $\mathbf{M} + dt^2\mathbf{K}$ 有效质量；
- $\lambda$ 跨迭代持久（步首清零）→ 迭代收敛到与迭代数无关的定点——XPBD 可以每步 1-2 次迭代就得到"解算器级"的软约束行为。

### 4.2 各约束的梯度与实现要点汇总

| 约束 | 粒子数 | $C$ | $\nabla C$ | 备注 |
|---|---|---|---|---|
| distance | 2 | $\|\mathbf{p}_a - \mathbf{p}_b\| - L_0$ | $\pm\hat{\mathbf{n}}$ | 布料边/绳索/cable |
| bend（Bergou） | 4 | $\sum_i \mathbf{K}_i\cdot\mathbf{p}_i$ | $\mathbf{K}_i$（常量） | 等距弯曲，$\mathbf{K}_i = k_i\hat{\mathbf{n}}_{\mathrm{rest}}$ |
| volume | 4 | $\mathbf{e}_1{\cdot}(\mathbf{e}_2{\times}\mathbf{e}_3) - 6V_0$ | 叉积组合 | 四面体体积保持 |
| shape-match | n | —（无显式 $C$） | — | 质心+协方差+polar 刚性目标 |

### 4.3 颜色化并行 Gauss-Seidel

`XpbdColoring`（`src/nk/solve/xpbd_coloring.cpp`）在 cook 期给约束着色：**同一颜色内的约束不共享粒子**。`XpbdProject` 每颜色发射一个内核，颜色内约束线程并行（无竞争、无原子）；颜色按固定顺序发射——等价于串行 Gauss-Seidel 的固定顺序，两次运行逐位一致。

### 4.4 速度合成与分裂冲量

`XpbdCorrectKernel` 的关键公式：

$$\mathbf{v}_{\mathrm{pbd}} = \frac{\mathbf{p}_{\mathrm{proj}} - \mathbf{p}_{\mathrm{prev}}}{dt} \qquad\text{（投影产生的速度）}$$

$$dv = \mathbf{v}_{\mathrm{now}} - \mathbf{v}_{\mathrm{pre}} \qquad\text{（接触解造成的速度增量）}$$

$$\mathbf{v} = \mathbf{v}_{\mathrm{pbd}} + dv, \qquad \mathbf{p} \mathrel{+}= (dv + \mathbf{v}_{\mathrm{pseudo}})\,dt$$

- $v_{\mathrm{pre}}$ 在 predict 时保存（接触前速度）；$v_{\mathrm{now}}$ 是接触行求解后写回的速度——XPBD 与刚体接触行解在**同一时间步**内通过该合成耦合；
- 分裂冲量（Split Impulse）消除穿透只推位置，不向系统注入能量；
- 未接触（$dv = 0$）时退化为纯 PBD 速度合成，与旧路径逐位一致。

### 4.5 与其他粒子介质的共处

`ParticleMode` 枚举：`Xpbd`（全柔体）、`Pbf`（全流体）、`Coupled`（柔体+流体内部子型）、`SoftFluid`（每 env [soft\|fluid] 显式分片）、`Mpm`、`MpmXpbd`（MPM 前片 + XPBD 后片）。XPBD op 用 `mpm_per_env` 跳过 MPM 切片；PBF 密度投影中软粒子不贡献流体密度（`SfIsSoft` 门）。所有介质共享**同一** body-particle 接触行边界。

---

## 5. 可微（diffsim）实现

`src/diffsim/` 是自写反向模式自动微分的多层实现，不依赖任何外部自动微分库。

### 5.1 总架构（三层）

```text
p02-A  StepBackward          单步手写反向伴随（无接触 PD 路径）
p02-B  Tape + Checkpoint     多步 rollout 记录 + 梯度检查点
p02-C  RecomputeOrchestrator 窗口重计算编排 + BackwardRunner 反传
p03    KktBuilder + IftRunner 接触路径：Delassus 矩阵 + IFT-at-convergence
p08-C  SdfContactIft         SDF 接触的 d/dM、d/dJ 通道（系统级 IFT）
```

### 5.2 p02-A：单步反向伴随

对一步无接触 PD 步 `drive → ABA → integrate`，手写反向。正向：

$$\boldsymbol\tau = \mathrm{clamp}(K_p(\mathbf{q}_{\mathrm{target}} - \mathbf{q}) - K_d\,\dot{\mathbf{q}}, \pm \tau_{\max})$$

$$\ddot{\mathbf{q}},\, \mathbf{a}_{\mathrm{base}} = \mathrm{ABA}(\mathbf{q}, \dot{\mathbf{q}}, \boldsymbol\tau, \mathbf{M}, \mathbf{g})$$

$$\dot{\mathbf{q}}' = \dot{\mathbf{q}} + \ddot{\mathbf{q}}\,dt, \qquad \mathbf{v}_{\mathrm{root}}' = \mathbf{v}_{\mathrm{root}} + (\mathbf{a}_{\mathrm{base}} - \mathbf{a}_{\mathrm{grav}})\,dt, \qquad \mathbf{q}' = \mathbf{q} + \dot{\mathbf{q}}'\,dt$$

反向（mirror 正向，固定序累加，无浮点原子）：

- $\bar{\mathbf{q}}$ 沿 ABA 三遍反推（含关节树伴随固定序 `+=`）；
- 浮基：根陀螺偏置 $\mathbf{p}_{\mathrm{root}} = \mathrm{ForceCross}(\mathbf{v}, \mathbf{I}\mathbf{v})$ 是 $\mathbf{v}$ 的**二次型**，其伴随在**积分前** $\mathbf{v}_{\mathrm{root,pre}}$ 处线性化（大 dt 双变体 FD 测试守护）；
- 姿态：$\mathbf{a}_{\mathrm{grav}} = \mathbf{R}(\mathbf{q}_{\mathrm{base}})^\top \mathbf{g}$ 与四元数位姿积分器的伴随，全部在**步前** $\mathbf{q}_{\mathrm{base,pre}}$ 处线性化。

覆盖（FD 验证，`test_aba_reverse_fd.cpp`）：固定基铰链链 d/d(τ, q̇, q, mass)（M⁻¹ 精确交叉检验 ~1e-7）、浮基速度通道、姿态通道；支持损失种子 $\mathbf{q}' / \dot{\mathbf{q}}' / \mathbf{v}_{\mathrm{root}}' / \mathbf{q}_{\mathrm{base}}'$。

### 5.3 p02-B/C：Tape + 梯度检查点 + 重计算

- **Tape** 每步只记**廉价**的每 env 动作切片（`total_link_count` 浮点）+ 标量记录（含为接触预留的 λ/事件槽，ABI 稳定前向兼容 p02-D）；昂贵的 ABA 中间量**不存**；
- **CheckpointManager** 每 $K$ 步（默认 16）快照权威状态；`recompute_on_backward=0` 时每步一个检查点（全磁带单遍反传，作为重计算路径的 CI oracle）；
- **BackwardRunner** 统一"反传一个窗口"内环：

1. D2D 复制 $\mathbf{q}, \dot{\mathbf{q}}$ → $\mathbf{q}_{\mathrm{pre}}, \dot{\mathbf{q}}_{\mathrm{pre}}$（StepBackward 需要步前值）；
2. `StepOnce(action[j])` 重生成步 $j$ 的 ABA 中间量 + 推进状态；
3. 清零 $\mathrm{grad\_tau}$；
4. `StepBackward(...)` 种子携带 + 原地覆盖为步 $j-1$ 的种子；
5. 写 $\mathrm{grad\_target}[j]$ 到动作梯度切片 $j$；
6. $\mathrm{grad\_mass}$ 累加（流序固定序 `+=`，无原子）。

- 检查点模式：先恢复窗口基检查点，正向重填窗口内每步步前状态缓存，再**降序**反传窗口；前向约跑两次，总时间 ~2 倍、与 $K$ 无关；
- 两种模式的 grad_actions/grad_mass **逐位相同**——该相等性本身就是重放确定性门（R6）。

### 5.4 p03：接触 KKT（Delassus）+ IFT-at-convergence

接触路径不 tape 固定迭代 PGS 内环，而是用**隐函数定理**在收敛活动集上**一次稀疏求解**完成反传。正向（一关节，收敛活动集 $\mathbf{A}\boldsymbol\lambda = \mathbf{r}$）：

$$\mathbf{A} = \mathbf{J}\,\mathbf{M}^{-1}\mathbf{J}^\top \qquad\text{（SPD Delassus，接触空间 Schur 补）}$$

$$\dot{\mathbf{q}}_{\mathrm{post}} = \dot{\mathbf{q}}_{\mathrm{free}} + \mathbf{M}^{-1}\mathbf{J}^\top \boldsymbol\lambda$$

$$\mathbf{b}_c = \text{目标分离速度} = \begin{cases} \dfrac{\beta}{dt}\,\max(\text{depth} - \text{slop},\ 0) & \text{法向行}\\[4pt] 0 & \text{摩擦行}\end{cases}$$

IFT 反向（下游余切 $\mathbf{g} = \partial L/\partial\dot{\mathbf{q}}_{\mathrm{post}}$，一次 $\mathbf{A}^{-1}$ 应用 $\mathbf{z} = \mathbf{A}^{-1}(\mathbf{J}\mathbf{M}^{-1}\mathbf{g})$）：

$$\frac{\partial L}{\partial \dot{\mathbf{q}}_{\mathrm{free}}} = \mathbf{g} - \mathbf{J}^\top \mathbf{z}, \qquad \frac{\partial L}{\partial \mathbf{b}_c} = \mathbf{z}$$

推导：

$$\dot{\mathbf{q}}_{\mathrm{post}} = \dot{\mathbf{q}}_{\mathrm{free}} + \mathbf{M}^{-1}\mathbf{J}^\top \mathbf{A}^{-1}(\mathbf{b}_c - \mathbf{J}\,\dot{\mathbf{q}}_{\mathrm{free}})$$

$$\frac{d\dot{\mathbf{q}}_{\mathrm{post}}}{d\dot{\mathbf{q}}_{\mathrm{free}}} = \mathbf{I} - \mathbf{M}^{-1}\mathbf{J}^\top \mathbf{A}^{-1}\mathbf{J} \quad\Rightarrow\quad \frac{\partial L}{\partial \dot{\mathbf{q}}_{\mathrm{free}}} = \mathbf{g} - \mathbf{J}^\top \underbrace{\mathbf{A}^{-1}(\mathbf{J}\mathbf{M}^{-1}\mathbf{g})}_{\mathbf{z}}$$

$$\frac{d\dot{\mathbf{q}}_{\mathrm{post}}}{d\mathbf{b}_c} = \mathbf{M}^{-1}\mathbf{J}^\top \mathbf{A}^{-1} \quad\Rightarrow\quad \frac{\partial L}{\partial \mathbf{b}_c} = \mathbf{A}^{-1}(\mathbf{J}\mathbf{M}^{-1}\mathbf{g}) = \mathbf{z}$$

（$\mathbf{A}, \mathbf{M}$ 对称 ⇒ $\mathbf{A}^{-\top} = \mathbf{A}^{-1}$, $\mathbf{M}^{-\top} = \mathbf{M}^{-1}$。）

- 为什么 CG 足够：$\mathbf{A}$ 是 **SPD**（不是鞍点 $[\mathbf{M}\ \mathbf{J}^\top; \mathbf{J}\ 0]$），`sparse_solver_cg.cu` 是自写确定性 CG（与 Eigen-oracle 对比验证）；
- **活动集判据**：接触槽位装配为法向行**且 depth > 0** 即接合（与正向装配器同一活动标志），产生粘滞等式行（切向速度 ~0 时粘滞 Jacobian 是正确局部约束集）；滑动/锥边界行是 v0.7+ 精化；
- D1：每关节一个 warp、固定序 fp64 累加、$\mathbf{A}$ 显式"上三角再镜像"使 $\mathbf{A}$ 逐位对称、零浮点原子；
- 主验证：`test_ift_vs_tape_backward.cpp` 走真实引擎接触步的有限差分；稠密 Eigen::LDLT 作为第二 oracle。

### 5.5 p08-C：SDF 接触 IFT（d/dM、d/dJ 通道）

v0.5 的 `ift_runner` 显式延迟的 d/dM、d/dJ 通道在此实现。系统级推导（在收敛 $\boldsymbol\lambda$ 处对 $\mathbf{A}\boldsymbol\lambda = \mathbf{r}$ 求导）：

$$\mathbf{d}\mathbf{A}\,\boldsymbol\lambda + \mathbf{A}\,\mathbf{d}\boldsymbol\lambda = \mathbf{d}\mathbf{r} \quad\Rightarrow\quad \mathbf{d}\boldsymbol\lambda = \mathbf{A}^{-1}(\mathbf{d}\mathbf{r} - \mathbf{d}\mathbf{A}\,\boldsymbol\lambda) \qquad\text{（IFT：一次 } \mathbf{A}^{-1}\ \text{应用）}$$

$$\text{d/dM 通道:}\quad \mathbf{d}\mathbf{A} = \mathbf{J}\,\mathbf{d}\mathbf{M}^{-1}\mathbf{J}^\top,\ \ \mathbf{d}\mathbf{r} = \mathbf{0}$$

$$\text{d/dJ 通道:}\quad \mathbf{d}\mathbf{A} = \mathbf{d}\mathbf{J}\,\mathbf{M}^{-1}\mathbf{J}^\top + \mathbf{J}\,\mathbf{M}^{-1}\,\mathbf{d}\mathbf{J}^\top,\ \ \mathbf{d}\mathbf{r} = \mathbf{d}\mathbf{b}_c - \mathbf{d}\mathbf{J}\,\dot{\mathbf{q}}_{\mathrm{free}}$$

- SDF 接触行（id 4/5，`sdf_contact`）通过同一接触求解喂入 $\mathbf{J}, \mathbf{M}^{-1}, \mathbf{b}_c$，因此其 d/dM、d/dJ **就是**该 $\mathbf{A}\boldsymbol\lambda = \mathbf{r}$ 的系统 IFT 导数；
- SDF 接触的 Newton 见证点本身是隐函数（$\nabla(\varphi_a + \varphi_b) = 0$），`SdfContactResult` 携带收敛点、两梯度、两 $\varphi$、残差与曲率代理 `step_curvature`，使伴随层无需重解几何即可应用 $dr/dp \approx \kappa\,I$ 的局部二次模型；
- 验证：`test_sdf_contact_adjoint_fd.cpp` / `test_sdf_contact_ift_fd.cpp` 通过**扰动 M/J 的实际条目并重解约束系统至收敛**与 FD 对比。

### 5.6 验证纪律

每层都有独立 oracle：手写伴随 vs FD（`*_fd`）；CG vs Eigen LDLT；检查点路径 vs 全磁带路径（逐位）；IFT vs 真实引擎接触步 FD。所有反向核零浮点原子、固定序累加（R6 两次运行逐位）。

---

## 6. 场景 IR 与 nks 场景语言、场景编译

### 6.1 SceneIR：双表示门面

`scene::SceneIR` 是场景的**规范中间表示**，采用门面（facade）双表示：

- **记录向量**（`RigidBodyRecord / CollisionShapeRecord / JointRecord / MediaRecord / SensorRecord / ...`）：存储与遗留读 API（cooker/render/compose/oracle 零扰动）。记录保持**全遗留保真**：接触元数据（contype/conaffinity/solref/solimp/condim/priority/solmix/margin/gap）、内联网格几何、分解模式、逐形状接触参数（MuJoCo 逐 geom 语义）、视觉网格资产引用；
- **结构化世界**（M2b 起）：每个 `Add*` 变更器写穿构建 ECS 实体 + 组件（`scene/ecs`）+ 场景树（`scene/graph`）——树与 Registry 是 `.nks` Save/Load（M2c）消费的结构权威。

复制语义：`SceneIR` 可拷贝；因为树是 `shared_ptr` 节点图，拷贝构造/赋值从记录向量**重建**树与 ECS（同一写穿路径）——`Compose` 需要这个深拷贝保持纯性。

### 6.2 nks 格式：自描述场景容器

`.nks` 是自有场景文件格式（`src/scene/format/nks.*`）：

```text
<nome>.nks    JSON 边车：结构场景（材质 + 前序节点树，每节点全遗留保真记录字段）
<nome>.nka    兄弟二进制容器：重 mesh/hull/SDF/纹理字节，JSON 用 AssetRef 引用
              （"foo.nka#MESH/0"）
```

- **Save** 按前序序列化树（imports 不重发），从节点映射的记录读取字段值 → cook 保真；**Load** 按 record-id 保真相位回放（materials → bodies → shapes → joints → cameras/lights/actuators → sensors/filters），从 .nka 读回网格块；同一 SceneIR ⇒ 逐字节相同 .nks+.nka（M2c roundtrip 门）；
- **确定性保证**：树前序、兄弟序 = 尾部追加序、插入键序 JSON 对象；浮点以 `"%.9g"` 渲染（binary32 无损）；.nka 块序 = Save 首次触达序，按内容哈希去重；
- **Overlay**：`Load(base, overlay)` 应用 `{"overrides": {"<derived/path>": {<部分组件对象>}}}` 补丁——只改命中节点记录（位姿、质量、材质字段…），其余不动（RL 参数扫描的轻量入口）；
- `imports` 在 Load 期按扩展名解析（MJCF/USD/URDF）再 `Compose`（attach_at 前缀）。

示例（`examples/scenes/soft_ball.nks`）：`physics_materials`（摩擦）+ `render_materials`（PBR 参数）+ `tree`（前序节点）+ `sensors/exclude_pairs/contact_pairs` + `media`（软体：`kind=soft_tet, method=xpbd`，`tet_sphere` 球网格 + `xpbd` 参数（particle_mass/distance_alpha/volume_alpha/iters）+ `render_skin` 皮肤）。`bdx_oneshot*.nks` 的 `media` 含 `cloth`（§2.6 参数）、`granular`（mlsmpm：`fluid_box` 填充 + `mpm` 材料 + `mpm_fills` 多区填充）、`cable`（`cable_line` 线段 + bend）。

### 6.3 导入与合成

```text
NKS / MJCF / URDF / text USD
        │  importers（LoadMjcf/LoadUsd/LoadUrdf/LoadNks）
        ▼
    SceneIR（记录 + 树 + ECS）
        │  Compose(base, addon, placement, prefix)    格式无关合并
        ▼
    SceneIR（合并后）
```

- `Compose` 纯且确定：复制 base，按原序追加 addon 全部记录，id 偏移 base 对应计数，addon 内交叉引用按偏移重映射，`kInvalid*` 哨兵永不变；每个 addon **根** body 重挂到 base 帧（`placement ∘ local`）；可选名称前缀防重名；
- 导入器输出同一 SceneIR，使 "USD ↔ MJCF 共存"（USD 杯合成到 MJCF 厨房台面、MJCF 机器人合成进 USD 场景）归结为纯 SceneIR 合并；
- 二进制 `.usdc/.usdz` 不支持（文本 USD 可入）。

### 6.4 场景编译管线

```text
SceneIR
   │ CookScene(scene, options)     场景级 cook（重物阶段）
   ▼
CookedBlob（SoA 表：形状/V-HACD 片/SDF/接触参数表/过滤器/关节树）
   │ CookToModel(scene, env_count) 转录 + 环境复制
   ▼
nk::Model（不可变 cook 表，env-major 容量）  +  SceneMap（实体↔行映射）
   │ Model::UploadTo
   ▼
nk::World（GPU-resident；Pipeline 由 cook 维度推导）
```

- **`CookScene`**（`scene/cooker.*`）把 SceneIR 摊平成 SoA 表。`CookSceneOptions` 门控两个重阶段：
  - `bake_sdf`：每个唯一凸片烘焙窄带 SDF（一般 PairDriven 路径不读 SDF，置 false 省时）；
  - `general_single_hull`：每个 mesh 碰撞形状视为单凸包（支持函数要一个 hull/mesh），除非作者显式 `DecomposeMode::Force` 强制 V-HACD；`Skip/Auto` 在开启时都坍缩为单 hull；
  - `bake_link_sdf`：从 body 的**视觉** trimesh 烘焙 SDF 绑定到该 body 碰撞行（粒子/MPM/刚体接触走真实轮廓而非内缩碰撞图元）；
- **`CookToModel`**（`src/scene/cook/cook_to_model.*`）驱动既有 cook + `CookArticulations`（运动学树），转录为 nk::Model 字段表，环境模板复制 `env_count` 份（env-major）；`SceneMap` 把每个实体绑定到其 cook 行（body/关节按记录序；形状按 V-HACD 片展开）；物理材质桶表由 cook 的逐形状摩擦/接触参数构建；
- **接触族**（`CookContactFamily`）：`PairDriven`（一般 LBVH → cvx 窄相位 → 混合岛求解）为默认；`enable_contacts=false` cook 出接触关闭世界（行预算归零 → 管线不发射宽/窄/解 op，纯关节+刚体动力学 oracle）；
- 纯 C++、零 CUDA token（设备上传在 World 期）；确定性：同 SceneIR + env_count ⇒ 逐字节相同 Model 与调度。

### 6.5 确定性贯穿编译期

导入→合成→cook→转录→上传每一步都保持"同输入同字节"：前序树、固定记录序、内容哈希去重的资产、`%.9g` 无损浮点、固定着色/排序。这使黄金场景文件与 golden 轨迹（nk trajectory == legacy 共居世界在 FP 地板上一致）成为可能。

---

## 附：主题与源文件索引

| 主题 | 主要源文件 |
|---|---|
| 统一求解器 | `src/nk/solve/{schedule,nk_row,xpbd_coloring}.*`, `src/phi/backend_cuda/ops/solve_rows.cu`, `src/nk/pipeline/pipeline.cpp`, `src/nk/data/arena.*` |
| XPBD 布料 | `src/phi/backend_cuda/ops/particles.cu`, `src/runtime/soft/cloth_topology.*`, `src/nk/solve/xpbd_coloring.*`, `src/phi/backend_cuda/ops/narrowphase_body_particle.cu` |
| MLS-MPM | `src/phi/backend_cuda/ops/mpm.cu`, `examples/demo/mpm_water_drop_demo.cpp`, `examples/demo/mpm_jelly_demo.cpp` |
| 可微 | `src/diffsim/{step_backward,tape,checkpoint,recompute_orchestrator,backward_runner,kkt_builder,ift_runner,sdf_contact_ift}.*` |
| SceneIR/nks | `src/scene/{scene_ir,scene_compose,cooker,cooked_blob}.*`, `src/scene/format/{nks,json}.*`, `src/scene/cook/cook_to_model.*`, `src/import/*` |
