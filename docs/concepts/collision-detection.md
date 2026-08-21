# 碰撞检测架构与算法：LBVH 宽相位、GJK/EPA 与 SDF 窄相位

本文梳理 Nuka Physics 碰撞系统的两条主路径：以 **LBVH（层次包围盒树，Linear Bounding Volume Hierarchy）** 为核心的宽相位（broadphase），以及以 **GJK/EPA**、**解析流形**、**SDF（符号距离场）** 为核心的窄相位（narrowphase）。文档覆盖架构设计（数据流、确定性约定、模块职责）与基本算法原理的公式推导（Morton 码、Karras 并行构建、GJK 相交判定、EPA 穿透深度、SDF 烘焙与 Newton 接触求解）。

文中出现的英文标识符均为源码中的真实名称，正文说明使用中文。

---

## 1. 总览：从场景到接触流

### 1.1 完整数据流

```text
状态（刚体位姿/速度）
   │
   ▼
① 宽相位：AABB 生成 → LBVH 构建/重建 → 重叠对查询
   │          （broadphase_lbvh / lbvh_batched / lbvh_refit）
   ▼
② 候选对过滤：bitmask（contype/conaffinity）+ excluded + explicit
   │          （rigid_candidate_pairs / BuildCandidatePairsTagged）
   ▼
③ 窄相位：GJK/EPA（凸包 vs 凸包/图元）、解析流形（图元 vs 图元）、
   │        SDF 双场 Newton（SDF vs SDF / 高度场）
   │          （convex_narrowphase / analytical_manifold / sdf_contact）
   ▼
④ 接触流形 → 约束行（法向行 + 摩擦行）→ 求解器 → 积分
   │          （contact_stream / row_builder / solver）
   ▼
传感器、张量视图、渲染
```

### 1.2 文件地图

| 阶段 | 文件 | 职责 |
|---|---|---|
| Morton 码 | `src/collision/morton_codes.cuh` | 10 位/轴 → 30 位 Morton 码（`ExpandBits10` / `Morton3D30`） |
| 节点结构 | `src/collision/lbvh_node.cuh` | `LbvhNode` 扁平数组、`LbvhMerge`（顺序无关合并）、`LbvhDelta`（最长公共前缀） |
| 构建 | `src/collision/broadphase_lbvh.cu` | 单环境 LBVH 全流程（build + 查询 + 排序压缩） |
| 批构建 | `src/collision/lbvh_batched.cuh` | 多环境一次启动的 Karras 构建（`EnvBuildInternalKernel` 等） |
| 重建 | `src/collision/lbvh_refit.cu/.cuh` | 复用拓扑的自底向上 AABB 刷新 |
| 遍历 | `src/collision/lbvh_traversal.cuh` | 重叠对查询（显式栈、规范化发射） |
| 过滤流 | `src/collision/rigid_candidate_pairs.hpp/.cu` | LBVH 候选对 → 过滤 → `CandidatePairStream` |
| 跨系统 | `src/collision/cross_system_query.cu` | 粒子-刚体 LBVH 查询（CSR 压缩） |
| 窄相位 | `src/collision/convex_narrowphase.hpp` | 自写 GJK → EPA → face-clip 流形 |
| 解析流形 | `src/collision/analytical_manifold.hpp` | 图元 vs 图元闭式多触点（SAT + Sutherland–Hodgman） |
| SDF 烘焙 | `src/import/cooker/sparse_sdf_cooker.cpp/.hpp`、`sdf_bake_backend.hpp` | 主机 CPU 窄带 SDF 生成（确定性 golden） |
| SDF 查询 | `src/runtime/sdf/sparse_sdf_query.cuh` | 单元键编解码、二分查找、三线性插值采样 |
| SDF 接触 | `src/collision/sdf_contact.hpp/.cu/.adjoint` | 双 SDF 场 Newton 求解接触见证点 |

### 1.3 贯穿性设计原则

- **单一路径，拒绝特例**：机器人与地面、机器人与抓取物体走同一条接触求解路径；GJK/EPA 通过 `SupportProxy` 标签联合体统一处理 hull vs hull 与 hull vs 图元；不存在场景特化的求解捷径。
- **D1 确定性**：两条铁律——（1）内核中**禁用浮点原子**（`float_atomic_add` 被 lint 在 `src/collision/**` 全局禁用），只有 `uint32` 整数原子；（2）一切依赖线程调度顺序的产物（发射顺序、扫描顺序）在收尾处用**确定性排序**还原。目标：相同输入逐位（byte-for-byte）复现。
- **验证而非强制**：新路径（LBVH、GJK/EPA 管线）以"与参考实现字节级一致"的测试为契约，但不强行替换生产路径；生产默认仍由经过验证的路径承担。
- **黄金参考（golden）**：烘焙类产物（如 SDF）必须是字节可复现的，主机单线程 + 固定遍历顺序 + 顺序无关归约是免费获得确定性的手段。

---

## 2. 宽相位：LBVH

### 2.1 问题与设计目标

宽相位的任务：在一组物体中快速找出所有 AABB 重叠的物体对（候选对），使窄相位只需测试这些候选对。

- 朴素做法为 O(n²) 两两测试；SAP（扫描与裁剪，sweep-and-prune）为 O(n log n + k) 但依赖排序序贯更新；LBVH 在 GPU 上一次性构建平衡树，查询为每叶子一次显式栈下降。
- **契约**（`tests/collision/test_lbvh_vs_sap_pair_set.cpp`）：LBVH 输出的候选对经压缩后，其规范化的 `{min(a,b), max(a,b)}` 集合必须与 SAP 重叠集合**逐字节相等**。
- 确定性：最终输出是排序后的紧凑候选对列表（键 = `(a<<32)|b` 的 `uint64`），由一次稳定的基数排序恢复 D1。

### 2.2 Morton 码：空间填充曲线的编码

Morton 码（Z 序曲线）把 3D 位置映射到一个整数，使空间上邻近的点在整数轴上大致保持邻近（局部保持性）。LBVH 先对 Morton 码排序，再按排序结果构建树，从而把"空间聚类"问题转化为"排序"问题。

**编码流程**（`morton_codes.cuh`）：

1. 归一化：场景 AABB 由各 AABB 中心点的归约得到，位置 `p` 按每轴线性映射到 `[0,1]`；
2. 量化：每轴取 10 位 → `xi = clamp(p.x * 1024, 0, 1023)`，以此类推；
3. 展开：`ExpandBits10` 把每个 10 位整数的第 `i` 位搬到第 `3i` 位（中间插入两个 0）；
4. 交织：`code = (xx << 2) | (yy << 1) | zz`，即 x 位在 `3k+2`、y 位在 `3k+1`、z 位在 `3k`。

**`ExpandBits10` 的逐步推导**。目标是：给定 10 位 `v = Σᵢ vᵢ 2^i`，输出 30 位 `w = Σᵢ vᵢ 2^{3i}`。采用"复制—掩码"倍增间隔的方法，每次把相邻位的间距翻倍：

```text
v = (v * 0x00010001) & 0xFF0000FF;   ① 把低 8 位复制一份到高 8 位，再保留两组副本
v = (v * 0x00000101) & 0x0F00F00F;   ② 每个 8 位组拆成两个间隔 4 的 4 位组
v = (v * 0x00000011) & 0xC30C30C3;   ③ 每个 4 位组拆成两个间隔 2 的 2 位组
v = (v * 0x00000005) & 0x49249249;   ④ 每个 2 位组拆成两个单比特，间隔 3
```

最终掩码 `0x49249249 = 0100 1001 0010 0100 1001 0010 0100 1001`（二进制从低位起），其置位位置恰好是 `{0,3,6,…,27}`——每 3 位一个 1，验证了"第 i 位落在第 3i 位"。例如 `v = 0x3FF`（10 个 1）经四步得到 `0x09249249`，即位 `0,3,6,9,12,15,18,21,24,27` 全为 1。

每一步的"乘法 + 掩码"等价于"复制一份到更高位 + 用掩码只保留需要的位"：乘法把现有比特组复制一份到更高位置（×0x10001 即 `v + v<<16`，×0x101 即 `v + v<<8`，×0x11 即 `v + v<<4`，×0x5 即 `v + v<<2`），掩码决定每组保留哪个副本，从而实现间距翻倍。该函数是输入（`v`）的纯函数，无原子、无线程序依赖，天然满足 D1。

**性质**：`Morton3D30` 是确定性的——相同浮点输入必得相同码（量化先钳制到 `[0,1023]`）。Z 序的缺点是在某些方向产生较大"跳跃"（对角线折叠），但 30 位码（每轴 1024 档）在典型刚体规模下足够，且 Karras 树对码序只要求"单调不降"，不要求完美邻近。

### 2.3 节点布局与顺序无关合并

`LbvhNode` 是扁平数组 `nodes[2N-1]`：`[0, N-1)` 为内部节点，`[N-1, 2N-1)` 为叶子。约定：

- 内部节点：`left/right` 是子节点在**同一扁平数组**中的索引（≥ N-1 即为叶子）；`parent` 供自底向上传播。
- 叶子：`left` 存放**原始物体索引**（不是叶子序号），使遍历端能按真实 body id 规范化配对。
- 根节点固定为节点 0（N>1 时）。

`LbvhMerge` 用 `fminf/fmaxf`（而非 `std::min/max`）做包围盒合并。`fminf/fmaxf` 是**顺序无关**的（IEEE 规定对 NaN 与 ±0 有确定行为，且合并操作满足交换律），因此无论哪个子线程先到达父节点，合并结果都相同——这是 D1 的关键支撑。

### 2.4 Karras 并行构建：最长公共前缀驱动的二叉树

**核心思想**（Karras 2012, "Maximizing Parallelism in the Construction of BVHs"）：排序后的 Morton 码数组具有"任意前缀共享"结构——`δ(i,j)`（两码的最长公共前缀长度，LCP）随距离 `|i-j|` 单调不增。因此每个内部节点可以由**一个叶子线程独立确定**其覆盖区间与分割点，无需串行递归，天然适合 GPU。

**LCP 定义**（`LbvhDelta`）：

```text
δ(i, j) = clz(code[i] ^ code[j])            // 两码异或后前导零个数 = 共享前缀位长
δ(i, i)  （码相等时）: 32 + clz(i ^ j)      // Karras 平局处理：索引也参与比较
越界哨兵: δ(·, -1) = δ(·, N) = -1           // 使边界上的方向判定成立
```

码相等时按索引回退比较，保证重复 Morton 码（同一位置的多个物体）也能形成合法树。

**内部节点 i 的构建**（`BuildInternalNodesKernel`，每内部节点一线程）：

1. **方向判定**：
   ```text
   d = +1 若 δ(i, i+1) ≥ δ(i, i-1)，否则 d = -1
   ```
   直觉：`i` 与哪一侧邻居共享更长的前缀，它的区间就向哪一侧延伸。（哨兵保证两端叶子的方向唯一。）

2. **区间长度上界**（倍增）：
   ```text
   delta_min = δ(i, i - d)                    // 与前一组（i 的对侧邻居）的 LCP
   l_max 从 2 开始，while δ(i, i + l_max·d) > delta_min 则 l_max *= 2
   ```
   区间 `[min(i,j), max(i,j)]` 内的所有叶子共享前缀**严格大于** `delta_min`（否则它们属于前一组），故 `l_max` 是满足该性质的 2 的幂上界。

3. **区间端点**（二分搜索）：
   ```text
   l = 0；对 t = l_max>>1, l_max>>2, …, 1：
       若 δ(i, i + (l+t)·d) > delta_min 则 l += t
   j = i + l·d
   ```
   在 `[0, l_max]` 中找满足性质的最大 `l`。`δ` 的单调性保证二分正确。

4. **分割点 γ**（把区间一分为二）：
   ```text
   delta_node = δ(i, j)                      // 区间两端共享前缀 = 节点自身前缀
   s 由同样的二分模板在 [0, l] 内找最大的 s 使 δ(i, i + s·d) > delta_node
   γ = i + s·d + min(d, 0)
   ```
   左边叶子（与 i 同侧）共享前缀 > `delta_node`（它们在同一更小组内），右边界处恰为 `delta_node`。分割在 `γ` 与 `γ+1` 之间：左子覆盖 `[first, γ]`，右子覆盖 `[γ+1, last]`。

5. **挂接子节点**：若 `first == γ`，左子是叶子（节点 `internal_count + γ`）；否则左子是内部节点 `γ`。右子同理（`last == γ+1` 时是叶子）。写入 `parent` 链接。

**正确性要点**：有序码的 δ 函数满足"前缀三角"性质（若 `i<j<k` 则 `δ(i,k) = min(δ(i,j), δ(j,k))`），保证二分搜索与分割点搜索的结果与真实层次结构一致；一次启动、每内部节点一个线程、互不依赖，每线程 O(log N) 步（倍增 + 两次二分），并行总工作量 O(N log N)、构建深度 O(log N)。

### 2.5 AABB 自底向上传播：整数原子 + 内存屏障

内部节点建立后只剩 `left/right/parent`，还需为每个节点计算包围盒。方法（`PropagateAabbsKernel`）：**每个叶子一个线程**，从自己的父节点向根走：

```text
while (node >= 0):
    __threadfence()                          // 确保本线程的子包围盒写入全局可见
    prev = atomicAdd(&visit[node], 1)        // uint32 整数原子
    if prev == 0:  return                    // 第一个到达的子线程停下
    第二个到达者：nodes[node].aabb = Merge(左子 aabb, 右子 aabb)
    node = parent
```

- 为什么需要 `__threadfence()`：若不设屏障，第二个到达的线程可能看到计数已加而子包围盒写尚未落盘，从而把陈旧/垃圾包围盒合并进父节点。在 N=2/3 时每条路径单线程所以不显，规模一大就出现运行间变化的配对数量与漏检——这是自底向上传播的经典竞态。
- 合并本身（`fminf/fmaxf`）顺序无关，所以无论哪个子线程先到，结果一致（D1）。
- 访问计数是 `uint32` 原子（允许），全程**无浮点原子**。

### 2.6 重叠对查询：显式栈 + 规范化发射

`LbvhPairQueryKernel`（`lbvh_traversal.cuh`）：**每个叶子一个线程**，用深度 64 的显式栈从根下降，测试该叶子的 AABB 与内部节点包围盒是否重叠，重叠则入栈继续下降；到达叶子时若两叶子 AABB 重叠则发射候选对。

两个确定性关键：

1. **规范化发射去重**：仅当 `my_body < other_body`（按**原始 body id**，而非叶子序号）才发射 `{my_body, other_body}`。这样 (i,j)/(j,i) 的双重访问只产出一条，且集合与 SAP 的 `i<j` 规范集合一致。
2. **发射序无关 → 收尾排序**：`slot = atomicAdd(&pair_count, 1)` 的分配顺序依赖线程调度，因此发射结果顺序不确定；调用方随后用 `PackKeysKernel` 把每对打包成 `uint64` 键 `(a<<32)|b`，做一次 `thrust::stable_sort` 恢复确定性排序。容量 `pair_capacity` 按 `kDefaultPairFanout=32 × N` 估计并钳制在 `[1024, 64M]`，溢出时置 `truncated` 标志并输出警告——绝不静默。

### 2.7 多环境批处理构建

`lbvh_batched.cuh` 实现"一次启动构建所有环境"的批处理版本，供生产管线与渲染 TLAS 共用：

- `EnvMortonKernel`：每线程计算所属环境的中心界 + Morton 码，打包成**复合键 `(env << 32) | morton30`**；
- **一次**跨环境的 `thrust::stable_sort_by_key`：复合键排序使各环境的叶子保持互不相交且按环境分组，与逐环境独立稳定排序逐位等价（每环境内部仍是原序）；
- `EnvBuildInternalKernel`：每环境内部执行与 2.4 相同的 Karras 构建（范围裁剪在环境内）；
- `EnvPropagateKernel`：2.5 的自底向上传播（访问计数按环境索引分配）。
- AABB 来源模板化（`AabbArraySource` 交织存储 vs `AabbSplitSource` 分离的 lo/hi 数组），一条代码处理两种布局。

### 2.8 增量重建：refit

`lbvh_refit.cu/.cuh` 的 `RefitLbvh` 复用既有拓扑（child/parent 链接、叶子→物体映射），只重载叶子 AABB（`RefitLeavesKernel`）并自底向上重新传播（`RefitPropagateKernel`，与 2.5 同款整数原子 + 屏障）。这是"每 N 帧重建一次、其余帧 refit"的调用方策略的支撑——Morton 序只在重建时刷新，refit 期间树质量单调退化，因此约 10–50 帧后必须重建。注意：refit 不改变配对集合契约，只是逼近；配对集合与 SAP 等价的严格契约由完整重建路径的测试守护。

### 2.9 候选对过滤：从形状对到 body 对

`BuildRigidCandidatePairs` / `BuildCandidatePairsTagged`（`rigid_candidate_pairs.hpp`）在 LBVH 输出的形状对上施加 MuJoCo 风格过滤，产出最终 `CandidatePairStream`：

1. **bitmask**：`PassesContactBitmask(contype[a], conaff[a], contype[b], conaff[b])`——两个物体的接触类型/亲和位掩码匹配才保留；
2. **exclude**：body 对命中策略的 `excluded_body_pairs`（已由 `C1c` 并集了作者 `<exclude>` 与父-子自动排除，按 `(min,max)` 升序）则丢弃；设备端对该有序列表做二分查找；
3. **explicit `<pair>`**：强制包含（MuJoCo 语义：`<pair>` 总是生成接触）。实现为**主机侧并集**——内核输出 bitmask/exclude 幸存者，主机再并入 bitmask 曾丢弃的显式对；
4. **压缩而非原子追加**：过滤用 count→scan→compact 结构，`compact` 阶段用排他扫描后的偏移直接写槽位，**无 append 原子**；
5. **`BuildCandidatePairStream`（C2a）**：为每对盖上 `stable_key`（排序稳定键）并 `thrust::stable_sort`；
6. **相邻去重**：多形状 body 在 C2a 排序后相邻重复的 body 对合并为一条（v0.8 每 body 一形状，此步为无操作）。

**标记化核心**：`BuildCandidatePairsTagged` 使每条 AABB 携带自己的标签（`types/reacts/handles/body_ids/contypes/conaffinities`），一次 LBVH+过滤+压缩即可同时产出同型（刚体-刚体）与跨型（关节链接-刚体）候选对。其中 `handle`（刚体时=body id，链接时=link 索引）写入 `CollidableRef.handle` 并参与打包键；为防 handle 与类型位混叠，入口处对 `handle < 2^28` 做守卫（超出即抛）。

### 2.10 跨系统查询：粒子-刚体

`cross_system_query.cu` 的 `QueryParticlesAgainstRigidLbvh` 让每个粒子（球 AABB）遍历刚体 LBVH：

- **两遍结构**：第一遍统计每个粒子的在盒刚体候选数（截断计数，`uint32` 截断原子）→ `thrust::exclusive_scan` 得到 CSR 偏移；第二遍把各粒子的候选写入**私有 CSR 切片**，用**每线程插入排序**保持按 body id 升序（受容量上限截断，保留最小 body id）。
- 确定性：唯一的排序是每线程私有切片的插入排序（无跨线程序依赖）；偏移扫描用确定性的 `exclusive_scan`；唯一的原子是 `uint32` 截断计数器。
- AABB 重叠谓词与 CPU 基准逐位一致（同样的 `</>`，无 MAD），故未超上限时候选集合与暴力测试精确一致。
- 栈深度 64；极端退化的 Morton 碰撞链若溢出栈，通过溢出标志（`TruncatedParticleCount`）显式上报，绝不静默。

---

## 3. 窄相位：GJK/EPA（凸体 vs 凸体）

`convex_narrowphase.hpp` 实现**自写** GJK → EPA → face-clip 单条路径：GJK 判定是否相交，EPA 求出穿透深度与法向（MTV，最小平移向量），face-clip（Sutherland–Hodgman）把流形展开为 ≤4 个接触点。不用 MPR（其原点射线选取依赖数据），不依赖任何外部碰撞库。

### 3.1 支撑函数与 SupportProxy

窄相位的一切建立在**支撑函数** `support(d) = argmax_{x ∈ K} x·d` 之上：凸体 K 沿方向 d 最远的点。

- `SupportHull`：在顶点列表上做 4 路独立链展开 + `(value, lowest-index)` 平局裁决——**最低索引**赢，保证相同几何输入得到相同支撑点（D1：argmax 有平局时以索引裁决，天然确定）。
- `SupportBox / SupportSphere / SupportCapsule`：图元的解析支撑。
- `SupportTrianglePrism`：高度场每格的三角棱柱（沿局部 -Z 挤出），复刻牛顿 `support_function.py:167-174` 的语义——**一般高度场**按格单元也走同一条 GJK/EPA 路径。
- `SupportProxy`：标签联合体 `{Hull, Box, Sphere, Capsule, TrianglePrism}`，使**一条** GJK/EPA 代码同时处理 hull-vs-hull 与 hull-vs-图元，不需要每条路径的特判。

### 3.2 数学基础：闵可夫斯基差与 GJK 判定

**闵可夫斯基差**：

```text
C = A ⊖ B = { a - b : a ∈ A, b ∈ B }
```

核心判定：

```text
A ∩ B ≠ ∅  ⟺  0 ∈ C
```

即两个凸体相交当且仅当原点落在差集 C 中（存在 a = b 的点对 ⟺ a ∈ A∩B）。C 仍是凸体，且其支撑函数可直接从 A、B 的支撑函数合成：

```text
h_C(d) = max_{x∈C} x·d = max_{a∈A,b∈B} (a - b)·d
       = max_{a∈A} a·d + max_{b∈B} b·(-d) = h_A(d) + h_B(-d)
```

即 `SupportMink(d) = support_A(d) - support_B(-d)`（代码中的 `MinkPoint{v, a, b}` 同时记录 C 上的顶点与两侧的世界支撑点，供后续流形重构）。这条合成律是 GJK 单路径处理任意凸体的根基：引擎只需提供两体的支撑函数，不必构造显式差集。

### 3.3 GJK 迭代与 simplex 演化

GJK 维护一个位于 C 上的 simplex（单纯形，1–4 个顶点），反复执行"沿搜索方向取支撑点 → 判断原点与 simplex 的位置关系 → 收缩 simplex"：

1. 初始方向 `d = center_A - center_B`（若为退化向量回退 `UnitX`）；
2. 取 `p = SupportMink(d)`；若 `p·d < 0`：C 的所有点在 d 方向投影为负（支撑点都在过原点的支撑平面内侧）⟹ **原点不在 C**，判定不相交；
3. 否则把 p 加入 simplex，`GjkDoSimplex` 判断原点是否被 simplex 包围：
   - **线（n=2）**：取更靠近原点的端点或线段，向最近子特征收缩；
   - **三角形（n=3）**：用边 `ab/ac` 的外侧判定 + 面 `abc` 相对原点的上下判定（`abc·ao` 符号）确定原点所在区域，收缩到对应边或保持面；
   - **四面体（n=4）**：按**固定顺序**检查四个面的外侧（`abc/acd/adb`，第一个"原点在外侧"的面胜出），若原点在所有面的内侧 ⟹ 被包围 ⟹ **相交**。
4. 迭代上限 `kGjkMaxIter = 32`，容差 `kGjkEps = 1e-9`。达到上限而未干净判定视为"接触"（尽力而为），由 EPA 提升补齐。

**算法性质**：每次迭代 simplex 都更靠近原点，距离单调下降；顶点取自 C 的支撑点，保证 simplex ⊆ C，故"包围原点"是相交的充分条件，"支撑点越过支撑平面"是分离的充分条件。确定性：固定迭代上限、固定判定顺序、`argmax` 平局按最低索引。

### 3.4 EPA：穿透深度与法向

EPA（`EpaExpand`）从 GJK 留下的四面体出发，把多面体向 C 的边界扩张，收敛时**离原点最近的存活面**给出最小平移向量（MTV）：

```text
深度 δ = min_{存活面 F} dist(0, 平面 F)
法向 n̂ = 该面的外向单位法向
```

**种子四面体**：GJK simplex（不足 4 点则沿固定基方向 `±X, ±Y, ±Z` 依序补点，拒绝退化重复点），构造 4 个面。

**关键设计——面朝向用多面体质心而非原点**：`MakeFace` 以 `n̂ · (v_a - interior) > 0` 判定外向，`interior` 是当前多面体的**质心**（非退化种子四面体的质心必为严格内点）。旧姿势按原点符号定向，在"原点落在边界上"的种子（如两盒面对面齐平，原点在 C 的 +Z 面上）会把法向翻成内向，后续支撑查询在同平面空转（已确认的假阴性根因）。质心定向保证原点在任何位置时面法向都正确。

**包围预循环**（`kEpaMaxEnclIter = 12`）：对任一存活面，若其有符号原点距 `dist ≤ kEpaEnclEps`（原点在面上或面外），沿该面法向取支撑点 p：

- 若 `p·n̂ - dist > eps`：C 在该方向越过原点 → 插入 p 扩张多面体，使原点进入严格内部；
- 若 `p·n̂ - dist ≤ eps`：C 的表面恰好经过原点（该方向真实深度为 0）→ 这是**真实接触**，直接以 `ok=false` 返回空流形（正确的"无穿透"，绝非误报——否则度量循环会在盒子其他深侧面上报一个错误深度的假接触）。

**度量循环**（`kEpaMaxIter = 64`）：

```text
1. 固定顺序数组扫描最近存活面：键 = (dist 升序, 面索引 升序)，严格 < 才更新
   （跳过 dist ≤ eps 的"过原点"退化条状面——box-box 的 CSO 常生成与原点共面的
     三角薄片，选中它们会让多面体非单调空转直到面数溢出 → 假阴性）
2. 沿最近面法向取支撑点 p，若 p·n̂ - dist < kEpaTol → 收敛，δ = dist, n̂ = 法向
3. 否则 EpaInsertSupport 插入 p 并重建地平线
```

**地平线重建**（`EpaInsertSupport`）：删除所有"从 p 可见"的面（p 严格在其外侧），用**边切换**（edge-toggle，按面索引序 + 每面固定 `(a-b, b-c, c-a)` 边序）提取轮廓边，再用新质心定向，为每条轮廓边 + p 生成扇面。固定顺序 + 固定容量上限（`kEpaMaxVerts=64 / Faces=128 / Edges=64`）保证 D1；任何容量溢出置 `overflowed` 标志（设备端无法打日志，用标志显式上报，绝不静默丢弃真实重叠）。

**见证点重构**：收敛后把原点向最近面做重心投影，得到两侧世界坐标 `witness_a / witness_b`（由 `MinkPoint` 中缓存的支撑点重建），供流形落点。

**ok 语义**：只有度量循环**真正收敛**（`progressed`）才返回成功；迭代/容量耗尽而未收敛而支撑又证明确有更深面，则返回 `ok=false`（真实重叠绝不输出深度 0 的假流形）。

### 3.5 流形构造：face-clip

接触点法向约定贯穿全引擎：**`point.normal` 是 A 侧（流形的第一个物体）的分离方向**——A 必须沿该方向移动才能解除重叠。GJK/EPA 直接产出的法向是 CSO 面外向（即"从 B 指向 A 的穿透方向"），配合侧序包装按需翻转。多点流形用 Sutherland–Hodgman 把接触多边形裁剪到 ≤4 点，接触点按确定性特征键排序（见第 4 节解析流形的同一约定）。

---

## 4. 窄相位：解析流形（图元 vs 图元）

对基本图元对（box-box、box-plane、sphere-box、capsule-plane 等），`analytical_manifold.hpp` 提供**闭式、多点**流形，比迭代 GJK/EPA 快且输出结构稳定（"Q2 混合窄相位"：图元对走解析，凸包对走 GJK/EPA，SDF 对走 Newton）。

### 4.1 Box-Box：15 轴 SAT + 面裁剪

**分离轴定理（SAT）**：两个凸体不相交当且仅当存在一条轴 L，使两者在该轴上的投影区间分离：

```text
pen(L) = r_A(L) + r_B(L) - |t·L|          // r_X(L) = X 在 L 上的投影半径
A、B 分离  ⟺  ∃L: pen(L) ≤ 0
最小穿透轴 L* = argmin_L pen(L)（所有 pen > 0 时的最小正穿透）
```

OBB 对只需检查 15 条轴：A 的 3 个面轴、B 的 3 个面轴、9 条边-边叉积轴（平行边剔除）。投影半径用相对旋转矩阵 `R[i][j] = A_i · B_j` 紧凑计算：

```text
A 面轴 i:   r_A = e_A[i],              r_B = Σ_j e_B[j]·|R[i][j]|
B 面轴 j:   r_A = Σ_i e_A[i]·|R[i][j]|, r_B = e_B[j]
边叉积 L = A_i × B_j: r = Σ_k e[·]·|axis_k · L̂|（归一化后）
```

**确定性要点**：
- 15 轴按**固定顺序**测试，只有严格改进才替换胜者——对称堆叠时每次都选同一根轴；
- 面轴对边轴有 `kFaceBias = 1e-5` 的偏好（btBoxBox 技巧）：平行面静止堆叠用面-面裁剪出 4 点，而不是退化的 1 点边接触；
- `AbsR` 加 `1e-6` 防平行轴除零抖动。

**面-面胜者**：确定参考盒（胜出面所在的盒）与入射盒（法向最反平行的面），取入射面 4 角（固定顺序），依次对参考面的 4 个侧平面做 **Sutherland–Hodgman** 裁剪（固定裁剪序），保留低于参考面的穿透点，投影到参考面取接触位置，穿透深度 = 低于面的距离。**边-边胜者**：两无限线最近点（标准 2×2 线性方程组，钳制到边区间），单点接触。

**D1 陷阱防护**：裁剪后若 >4 点，`ReduceAndEmitSpread` 以**整数特征键**为主序排序保留（box-box 用裁剪输出索引，box-plane 用角位掩码），浮点穿透只作次级去优先——对称静止情形每个点穿透都相等，特征键是唯一决定因素。同一特征键写入 `ContactPoint.stable_key`（热身身份免费获得），发射槽位按升序特征键排列，两次运行流形逐字节一致。

### 4.2 其他图元对

- **Sphere-Sphere / Sphere-Box / Sphere-Plane**：单点闭式解（球心连线 / 最近 OBB 点 / 平面投影）。
- **Capsule-Plane**：≤2 端点（段-平面最近点）；**Capsule-Sphere**：1 点。
- 全部 handler 是 `__host__ __device__` 纯函数，用**烘焙旋转帧**（`PrimFrame`，列为世界轴）做 OBB 运算，避免非 HD 的 `Quat` 辅助函数——同一几何同时被主机测试与设备派发编译。

---

## 5. 窄相位：SDF（符号距离场）

SDF 路线面向网格/高度场：烘焙阶段把网格转成窄带有符号距离场（cooked golden），运行时用双场 Newton 在 GPU 上求接触见证点。

### 5.1 烘焙：主机 CPU 窄带生成

`CookSparseSdf`（`sparse_sdf_cooker.cpp`）把**一个三角网格**（一个 V-HACD 凸片）烘焙为 `SparseSdfData`。为什么选主机 CPU 而不是规格里的 CUDA：烘焙 SDF 是黄金数据，必须**逐字节复现**；主机单线程 + 固定格子遍历序 + 顺序无关归约免费获得确定性，避开 GPU 浮点重排的不确定性（规格中的"烹饪期可用 GPU"例外被有意放弃，因为确定性压倒一切）。

**算法流程**：

1. **网格 AABB**，按 `(band + padding)` 体素向外扩；原点取扩边后的最小角，使所有带内索引 ≥ 0（打包键单调）。
2. **体素尺寸**：显式 `voxel_size`，缺省按最长边 / `auto_resolution`（默认 64）。
3. **逐三角形光栅化**：每个三角形只访问其 `±(band+1)` 体素邻域（三角形 AABB 扩张后重叠的体素），**成本正比于表面积而非体体积**。`band+1` 的 +1 是为边界梯度中心差分保留的守卫环（存储前丢弃）。
4. **精确距离 + 凸体符号**：每体素对每个重叠三角形计算**精确的三角形-点距离**（Ericson 区域法，双精度）与最近点；保留运行最小 `dist²` 与**最近三角形的面平面符号** `sdot = (p - cp)·n̂`：
   ```text
   φ(p) = sign(sdot) · sqrt(min dist²)      // 凸体内侧符号为负
   ```
   为什么对凸体成立：V-HACD 片是凸包，内部点最近特征必为面内部（`sdot < 0`），外部点（即使最近特征是边）位于两个邻面的正半空间，平局也取正。用**最近点**而非面平面做符号判定，使高接合度顶点/极点（球面极点三角片）也稳健。该 O(1) 符号判定避免了广义绕数（每体素 O(三角形 × atan2)）——烘焙瓶颈。
   - **退化三角剔除**：零面积/细长片（UV 球极点四边形的退化三角）面积平方 ≤ eps²·|ab|²·|ac|² 时剔除——其法向是数值噪声，符号判定不可靠；顶点由同四边形的健康三角保住，|φ| 不变。
5. **符号取整**：固定排序键序处理（先收集键、排序、再顺序写入），使结果与哈希表迭代序无关。
6. **solid 填充**（可选）：从网格边界做**外洪水**（φ<0 的表面单元阻挡洪水），未达单元即封闭内部；内部带外区域用**向内快扫** `φ = φ_neighbor - h` 填充（保留精确带单元不被覆盖）。
7. **梯度**：对带外一层做**中心差分**（缺邻域回退单侧差分），归一化为单位梯度（真 SDF 的 |∇φ|=1 性质）。
8. **窄带提取**：只保留 `|φ| ≤ band·h`（solid 模式额外保留全部内部 φ<0），按键升序，紧凑写入 `SparseSdfData`。

**缓存键**：`ComputeSdfCacheKey` 对（域标签 `"nuka.sparse_sdf.v1"` ∥ 顶点字节 ∥ 索引字节 ∥ 逐字段参数）做 SHA-256，作为按网格去重的键；参数逐字段序列化，避开结构体填充字节。

### 5.2 存储编码：稀疏格子键

窄带稀疏存储（`sparse_sdf_query.cuh`）：

```text
key = (i << 42) | (j << 21) | k          // 每轴 21 位，共 63 位
```

- 21 位/轴覆盖每轴 2M 体素跨度；原点在带下角 ⟹ 索引非负 ⟹ 键在 (i, j, k) 字典序下**单调**，可二分查找；
- `SparseSdfDevice` 是扁平原始指针视图（主机指向 `CookedSdfTable` 向量，设备指向上传缓冲），同一结构同一采样器，主机测试与设备内核共用一份代码；
- 采样器 `sparse_sdf_sample`：二分查找 8 个包围角（缺失角 → 返回 `kOutsideBand = 1e30`，p08 据此判定"无接触"），**固定字典序角序**做三线性插值——无原子、无线程序依赖，两次运行逐位一致。

### 5.3 运行时接触：双场 Newton

`find_sdf_contact_newton`（`sdf_contact.hpp`）对两个 SDF 物体求接触，**最小化两个符号距离场之和**：

```text
f(p) = φ_a(p) + φ_b(p)
见证点 p* = argmin f(p)          （最"同时深入两体"的点）
```

**为什么取和最小化**：接触时两表面相切（或重叠），最深入两体的点即穿透见证点；收敛时

```text
∇f(p*) = ∇φ_a + ∇φ_b = 0  ⟹  ∇φ_a = -∇φ_b
```

两梯度（即两表面外向法向）反平行——面对面的几何事实被编码为驻点条件。

**迭代**（固定 32 次上限、固定容差 `grad_tolerance=1e-5`）：

```text
r = ∇φ_a(p) + ∇φ_b(p)                    // 驻点残差
若 ‖r‖ ≤ tol: 收敛
步长: step = -r/κ，κ = 2 / min(voxel_a, voxel_b)   // 曲率代理
步长钳制: ‖step‖ ≤ min(voxel_a, voxel_b)            // 防薄壳隧穿、防跳出窄带
p ← p + step
```

- `κ = 2/min_voxel` 的直觉：SDF 约单位梯度，和场的曲率沿下降方向量级 O(1/voxel)；此 κ 给出约一个体素的自然步长，与钳制线吻合；
- **出带守卫**：任一侧采样返回 `kOutsideBand` 时，该梯度是垃圾（1e30），绝不喂入步长——向最后一次"两带内"迭代中点回退半步（确定性线搜索）；初始点就在带外 → `valid=false`；
- 步长钳制同时防两种失效：快速/薄体越过亚体素面板（薄壳隧穿），以及单步跳出窄带。

**法向（分离方向约定）**：

```text
n̂ = normalize(∇φ_b - ∇φ_a)
```

推导：`∇φ_a` 是 A 表面**外向**法向，见证点处它从 A 内部指向 B（与分离方向相反），故 A 的分离方向为 `-∇φ_a`；由驻点条件 `∇φ_a ≈ -∇φ_b`，`∇φ_b - ∇φ_a ≈ 2∇φ_b ≈ -2∇φ_a`，即同向归一化。退化情形（二面角 ≥ 90° 的共面接触使两梯度近同向、差向量坍缩）回退 `-∇φ_a`；再退化则固定轴 `+Y`。该式与 `SolveUnitGroundContact` 交叉验证：盒 A 在平面 B 上，`∇φ_a = -Y, ∇φ_b = +Y`，差向量 `+Y` = 盒上移分离方向 ✓。

**穿透深度**：

```text
δ = max(-(φ_a + φ_b), 0)
```

**p08-C 交接**：结果结构携带收敛见证点、两世界梯度、两 φ 值、残差与曲率代理 `step_curvature`——隐函数定理（IFT）伴随层无需重解几何即可对姿态/网格求导（`dr/dp ≈ κI` 的局部二次模型）；伴随正确性由 `test_sdf_contact_adjoint_fd.cpp` 对有限差分验证。

---

## 6. 确定性（D1）体系总表

| 环节 | 确定性手段 |
|---|---|
| Morton 码 | 纯函数，无原子，钳制+量化固定 |
| Karras 构建 | 每内部节点一线程独立求解，无竞争写 |
| AABB 传播 | `fminf/fmaxf` 顺序无关合并；`uint32` 整数原子；`__threadfence` 防读写竞态 |
| 对查询发射 | 规范 `(a<b)` 发射 + 最终 `uint64` 稳定排序还原序 |
| 过滤压缩 | count→scan→compact，无 append 原子 |
| GJK | 固定迭代上限、固定判定顺序、argmax 最低索引平局 |
| EPA | 固定面扫描序、固定地平线边序、质心定向、容量上限+溢出标志 |
| 解析流形 | 固定 15 轴序、固定裁剪序、整数特征键主序排序 |
| SDF 烘焙 | 主机单线程、固定体素遍历序、固定排序键序归约、双精度中间量 |
| SDF 采样 | 二分 + 固定角序三线性插值 |
| SDF Newton | 固定迭代上限、固定步长公式、确定性线搜索 |

全局铁律：**浮点原子在 `src/collision/**` 被 lint 禁用**；一切可能因线程调度而乱序的产物在收尾用确定性排序/扫描还原。

---

## 7. 验证与测试

| 测试 | 守护的契约 |
|---|---|
| `test_lbvh_vs_sap_pair_set.cpp` | LBVH 候选对集合 ≡ SAP 重叠集合（逐字节） |
| `test_lbvh_batched.cu` | 批处理构建与单环境构建一致 |
| `test_lbvh_filtered_pairs.cpp` | 过滤管线输出流（bitmask/exclude/explicit）正确 |
| `test_gjk_epa_convex.cpp` | GJK/EPA 相交判定、深度、法向与流形 |
| `test_analytical_manifold.cpp` | 解析流形闭式结果与 D1 复现 |
| `test_sdf_contact_adjoint_fd.cpp` | SDF Newton 接触及其 IFT 伴随 vs 有限差分 |
| `test_cross_system_query.cpp` | 粒子-刚体查询 vs 暴力基准 |
| `test_candidate_stream.cpp` / `test_contact_stream_driver.cpp` | 候选对流与接触流端到端 |

每类新路径都以"与参考实现逐位一致"的测试为门槛，验证通过才允许进入生产管线（与 p04 LBVH、v0.5 CG/Eigen-oracle 同一纪律）。
