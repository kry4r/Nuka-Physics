# Spec 04：Nuka Physics Core 与 nuka-editor 外部扩展

状态：需求已收敛，等待实现
顺序：独立于 Go2 技能训练；可与既有 Spec 01/02/03 分阶段交付
输入：现有 Vulkan editor、Nuka runtime/inference、CUDA↔Vulkan interop、离线 RT 实现，以及 Isaac Sim/Isaac Lab 工作流对标
输出：核心物理引擎不依赖 editor 的 runtime 分层、双模式 `nuka-editor`、训练观察 IPC、无卡顿 Vulkan 实时显示、增强 gizmo/UI、实时渲染效果和高质量离线 RT

> 本 spec 重新定义 viewer/editor 的产品边界。既有 [Spec 03：渲染补强](2026-08-12-spec-03-rendering-enhancement-zh.md) 中关于离线 RT provider、持久场景资源和 AOV 的内容继续有效；其中把 viewer 视为主要运行时、以及不承诺实时 viewport 帧率的部分由本 spec 修订。

## 1. 结论摘要

### 1.1 需求合理性

需求整体合理，但必须按层次拆开。物理求解世界是 Nuka 的核心；Vulkan editor、ImGui、gizmo、训练观察、IPC 和离线电影渲染都是使用 Nuka 的外部扩展。不能因为 editor 需要实时显示，就把 Vulkan、ImGui、IPC、Python UI 生命周期或 policy-specific 分支塞进 `nk::World`、solver、contact pipeline 或核心 CUDA ABI。

当前卡顿也不是单纯的 Vulkan draw-call 问题，而是执行边界问题：现有 `nuka_editor` 主循环同步执行 physics、controller、`HostDownloadPublisher` 的 GPU→CPU pose 下载、UI 记录和 present。现有 `CudaVulkanInteropPublisher` 已经提供 zero-copy 方向，但 editor 主路径尚未接入，而且只覆盖单个 selected env 的 transform SSBO。第一优先级应是把 physics owner、snapshot publisher 和 Vulkan consumer 解耦，再在实时管线增加效果。

### 1.2 Isaac Sim / Isaac Lab 可借鉴的模式

Isaac Lab 官方训练工作流的关键点不是“把训练塞进 viewer”，而是：

- 训练脚本拥有环境、批量 world、policy 和训练生命周期；
- 训练默认 headless，以保持吞吐；
- 需要观察时按需挂接 visualizer；
- 训练完成后使用较少环境运行 play/eval 并观察策略；
- 观察器展示环境状态，但不是训练 solver 的唯一入口。

参考：

- [Isaac Lab Training Environments](https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/training-environments.html)
- [Isaac Lab 官方文档](https://isaac-sim.github.io/IsaacLab/)
- [Isaac Lab SkillGen / trajectory visualization](https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/skillgen.html)

Nuka 对标后的产品关系如下：

```text
Nuka Physics Core
  nk::World / solver / contact / articulation / CUDA backend
  SceneIR / Model / physics ABI / headless Step()
             ↑
Nuka Runtime Adapter
  PhysicsWorker / fixed tick / command queue / snapshots
  SceneController / inference / reset / metrics / IPC server
             ↑                         ↑
Interactive nuka-editor             External trainer
  Vulkan + ImGui + scene editing     Python/C++ batch envs
  local physics observation          RL / imitation / checkpoint
             ↑
Train Observe nuka-editor
  IPC client / static scene / read-only display / local recording
```

`nuka-editor` 仍然是唯一的 Nuka GUI 入口，名称不改为 viewer；但它不是 Nuka Physics core，也不是训练框架。`nuka-editor` 是一个使用 Nuka runtime 的可选外部应用。

## 2. 产品与运行模式

### 2.1 一个产品入口，两个显式模式

保留可执行文件名 `nuka_editor`，提供显式子模式：

```bash
nuka-editor interactive --scene examples/scenes/go2.nks
nuka-editor interactive --scene examples/scenes/go2.nks --serve 127.0.0.1:8765
nuka-editor observe --connect 127.0.0.1:8765
```

现有旧参数可以在过渡期兼容：

```bash
nuka-editor --scene examples/scenes/go2.nks
```

但新文档、测试和启动脚本使用 `interactive` / `observe` 子模式。CMake target 可以暂时保留内部名 `nuka_editor_exe`，对外 binary 仍为 `nuka_editor`；target 是否重命名属于清理项，不得阻塞架构实现。

### 2.2 Interactive

Interactive 是单环境或小规模本地交互仿真模式：

- `nuka-editor` 通过 runtime adapter 创建并拥有一个 `PhysicsWorker`；
- worker 独占 `nk::World`、controller、policy state 和 physics device stream；
- Vulkan/ImGui 线程只消费 snapshot，不直接调用 `nk::World::Step()`；
- 支持场景加载、Play/Pause/Step/Reset、gizmo、Inspector、Drive、teleop、脚本和 Go2 inference；
- Play 或 inference 期间编辑控件只读；
- Pause 后可以修改 pose、material、drive target 或 scene structure；
- 修改提交后必须 reset controller/inference 的动作历史、decimation clock 和 episode state，再 Resume；
- Reset 是明确的 physics/runtime 操作，不是 UI 线程直接写 device buffer。

Interactive 的目标是“可观察、可交互、可实时控制的单环境 Nuka 仿真”，不是批量训练环境。

### 2.3 Train Observe

Train Observe 是外部训练进程的只读观察模式：

- trainer 是 batch physics、policy、reward、episode 和 checkpoint 的唯一权威；
- `nuka-editor` 不创建用于训练的 `nk::World`，不执行 trainer physics，不参与碰撞求解；
- editor 本地加载与 trainer 严格匹配的静态 `.nks/.nka` scene 和真实 Go2 mesh；
- trainer 通过 IPC 发送最多 32 个环境的动态 pose、接触和指标；
- editor 在同一个大 viewport 中渲染这些环境的真实 world pose；
- 不添加 editor origin、间距、朝向或“可读性偏移”；多个环境在真实 world 中重叠时，画面也原样重叠；
- 环境之间无碰撞由 trainer 的 env isolation 保证，观察端只有 display-only render instances；
- editor 允许选择 env、调整 camera、开启 overlay、开始/停止本地 recording，但不能 pause/reset/command trainer；
- gizmo、Inspector、Drive、scene edit 和脚本写入在 Train Observe 中全部禁用。

Train Observe 的目标是对标 Isaac Lab 的可选 visualizer，而不是把训练循环变成 editor 的一个按钮。

## 3. 严格分层与依赖边界

### 3.1 Nuka Physics Core

以下内容属于 core，可在无 Vulkan、无 ImGui、无 IPC、无 Python、无 editor 的 headless 进程中构建和运行：

- `nk::World`、`Step()`、`StepPlanned()`、reset、checkpoint/replay；
- contact、constraint row/block、articulation、joint limit、effort limit；
- CUDA/CPU physics backend、device buffers、field ABI、DLPack/C ABI；
- SceneIR、cook/model、canonical physics profile；
- deterministic ordering、telemetry field、state hash；
- 不绑定任何具体 policy、Go2 task、Vulkan renderer 或 UI。

Core 禁止依赖：

- Vulkan、GLFW、xcb、ImGui、ImGuizmo；
- `nuka_editor`、IPC、shared-memory protocol；
- embedded Python interpreter 和 UI lifecycle；
- `Go2PolicyController` 的 scene-specific 分支；
- render frame pacing、window event、camera、gizmo。

### 3.2 Runtime Adapter

runtime adapter 是 core 之上的可复用主机层，负责把 core 接入应用或训练脚本：

- `PhysicsWorker`：固定 dt、tick scheduling、controller 调用和 command queue；
- `SceneController`：policy、teleop、scripted controller 的通用接口；
- `FrameSnapshot`：不可变动态状态快照；
- `CommandEnvelope`：tick 边界消费的命令；
- reset、episode、control decimation、policy latency 和 physics telemetry；
- local IPC server/client 和 Python API 的协议适配；
- host fallback pose publish 与 CUDA/Vulkan zero-copy publish 的选择。

runtime adapter 可以依赖 Nuka core，但不能让 core 反向依赖 runtime adapter。现有 `Simulation`、`PosePublisher`、`SceneController` 是迁移起点；应把其中的 frame scheduling、snapshot 和 render-specific 路径拆成可复用 runtime 组件。

### 3.3 nuka-editor Extension

`nuka-editor` extension 负责：

- Vulkan swapchain/present、实时 raster 和 ImGui overlay；
- scene tree、Inspector、gizmo、camera、material UI；
- Interactive 的 local runtime client；
- Train Observe 的 IPC client；
- debug overlays、timeline、本地录制和视频导出；
- UI 视觉层级和 shadcn 风格对标。

editor 可以链接 runtime、render 和 inference adapter，但不得把 editor headers 或 Vulkan symbols 下沉到 core solver target。

### 3.4 外部 Trainer 与脚本

外部 trainer 直接调用 Nuka core/runtime API，自己管理：

- batch env 数量；
- observation/action/reward；
- RL 或 imitation framework；
- checkpoint、episode、训练指标；
- headless 或按需观察输出。

外部脚本通过 `nuka.viewer` Python client 访问 editor/runtime IPC；不通过 Python 直接持有或修改 editor 线程的 live `nk::World`。

## 4. PhysicsWorker 与快照模型

### 4.1 Worker 所有权

两种模式都采用 physics worker 独占物理 world：

```text
UI / external script / teleop
            ↓ command queue
PhysicsWorker
  own nk::World
  own controller / policy state
  fixed physics dt
  apply commands at tick boundary
            ↓ publish fence
FrameSnapshot ring
            ↓
Vulkan / ImGui / recorder
```

要求：

- UI/render 线程不得调用 `world.Step()`、`Data::UploadField()` 或 `Data::DownloadField()`；
- worker 是 `nk::World` 的唯一 owner；
- 所有 pose/drive/scene edit/teleop 命令先进入 queue，在 tick 边界按序消费；
- worker 每个 tick 只发布一次 generation-stamped snapshot；
- render 可以丢弃旧 snapshot，但不能让 UI 等待 physics catch-up；
- physics 不能因为 render 线程慢而改变固定 dt 或求解迭代数；
- Interactive 保留 `--single-threaded` 仅用于 debug/regression，不作为默认路径。

### 4.2 Snapshot ring

使用固定容量的双槽或三槽 ring，避免每帧分配和锁竞争：

```text
SnapshotHeader {
  magic
  schema_version
  byte_size
  generation
  physics_tick
  sim_time
  wall_time_ns
  env_count
  links_per_env
  bodies_per_env
  contacts_per_env
  flags
}

SnapshotPayload {
  base_pose[env]
  link_pose[env, link]
  body_pose[env, body]
  link_velocity[env, link]
  joint_position[env, dof]
  joint_velocity[env, dof]
  contact_points / contact_wrenches (optional)
  command / action / observation groups (optional)
  per-env metrics (optional)
}
```

- 内部 local snapshot 使用 typed C++/device buffer；跨进程 snapshot 使用明确的小端 binary layout；
- `generation` 使用 release/acquire 或 seqlock，读者永远读取完整 slot；
- snapshot age、dropped snapshot count、producer tick 和 consumer frame 都进入 UI telemetry；
- Interactive 默认只发布 env 0；Train Observe 发布由 trainer 选择的最多 32 个 env；
- 大型 batch 的完整 observation 不默认发送，发送可选分组或 selected env，避免训练吞吐被 UI 拖慢。

### 4.3 物理与显示频率

默认基线：

- physics fixed dt：沿用 scene/runtime 配置，Go2 典型为 240 Hz；
- policy/inference：由 controller decimation 决定，不由显示帧率决定；
- present：目标 30 FPS；
- render 消费最近完成 snapshot，不等待当前 physics tick；
- worker catch-up 不阻塞 UI；render 线程只在 swapchain/present 自身的有限 frames-in-flight 上等待；
- 物理 backlog、worker tick duration、snapshot age 在 Stats 中可见。

## 5. IPC 与 Python API

### 5.1 传输层

采用 loopback TCP + shared-memory ring：

- TCP：handshake、控制请求、错误、指标摘要、录制控制和 lifecycle；
- shared memory：高频 pose/contact/action snapshot；
- TCP 消息使用 length-prefixed JSON 或 versioned JSON frame，便于 Python/C++ 调试；
- shared-memory header 明确 magic、schema version、slot count、slot bytes、producer sequence、payload hash；
- IPC 默认只监听 loopback，远程监听必须显式开启；
- malformed frame、版本不匹配、半写 slot、producer crash 必须被识别，不得让 editor 崩溃。

### 5.2 Handshake

Train Observe 连接必须先完成：

```json
{
  "protocol": "nuka.observe.v1",
  "scene_hash": "...",
  "model_hash": "...",
  "asset_hash": "...",
  "physics_profile_hash": "...",
  "pose_schema": "go2.link_pose.wxyz.zup.v1",
  "env_count": 16,
  "links_per_env": 13,
  "dofs_per_env": 12,
  "coordinate_system": "right-handed-z-up",
  "units": "meters-radians-newtons-seconds"
}
```

editor 本地 scene 必须与 trainer 的 `scene_hash`、`model_hash`、`asset_hash`、physics profile 和 pose schema 严格匹配。任一不匹配：

- 拒绝进入 live render；
- UI 显示具体 mismatch；
- 不允许 best-effort 绑定 link，避免真实 mesh 与 pose 顺序错位。

### 5.3 控制消息

Interactive 的 `nuka.viewer` client 支持：

```python
from nuka.viewer import connect

session = connect("127.0.0.1:8765")
session.snapshot()
session.pause()
session.select_env(0)
session.set_command([1.0, 0.0, 0.0])
session.reset()
session.start_recording("out/episode.nktr")
```

Train Observe client 只支持：

- connect/disconnect；
- select env；
- request snapshot；
- camera/overlay preference；
- start/stop editor-local recording。

Train Observe 不发送 pause、reset、drive、pose edit、policy command 或 scene mutation。

### 5.4 脚本面板

现有 `ScriptHost` 不再直接在 editor UI 线程执行 live-world Python。改为：

- Script panel 是一个 IPC command client；
- 外部 Python client 使用相同 command schema；
- script command 在 runtime worker 的 tick 边界执行；
- core 不嵌入 Python；Python 解释器属于可选 runtime/script adapter；
- 脚本只能通过公开 command/state API 访问 world，不暴露裸 device pointer；
- 默认限制为 loopback、超时、输出上限和异常隔离；
- Train Observe Script panel 只读或隐藏。

## 6. Train Observe 的真实多环境显示

### 6.1 渲染语义

Train Observe 默认是一个大 viewport，不是 thumbnail grid。trainer 发送最多 32 个 env 的真实 world pose，editor 把同一份静态 mesh/scene binding 映射到最多 32 个 display-only environment：

```text
env 0 pose ─┐
env 1 pose ─┼─> exact world coordinates
...         │
env 31 pose ┘

no editor spacing
no editor origin offset
no editor yaw correction
no editor collision
```

- 机器人初始位置完全由 trainer physics 决定；
- 机器人朝向完全由 physics pose 决定；
- 多 env 重叠时必须原样显示；
- display instances 不进入 Nuka contact broadphase；
- camera reset 使用所有 display instances 的实际 world AABB；
- selected env 可高亮、显示 overlay 和详情，但不能改动 pose；
- 最多 32 是 editor display 上限，不限制 trainer 的 batch env 数量。

### 6.2 Train Observe UI

第一阶段必须提供：

- 全局训练指标：step、episode、reward mean/min/max、success、termination、trainer FPS；
- 每个显示 env：env id、episode step、reward、success/termination、action、command；
- 物理状态：contact points/wrench、base pose/velocity、joint state、COM；
- 性能与观测：obs 分组、policy latency、physics step time、render FPS、snapshot age、dropped snapshot；
- timeline：editor 本地 snapshot ring / disk recording 的 scrub、play、pause、selected env 切换；
- 导出：snapshot trace、trajectory、Vulkan video 或离线 RT frame request。

UI 中明确显示：`TRAIN READ ONLY`、`POSE SOURCE`、`SNAPSHOT AGE`、`SCENE HASH` 和连接状态。

## 7. 物理开启后的 Vulkan 卡顿修复

### 7.1 当前问题

现有路径存在以下阻塞点：

1. `viewer_main` 在 render/UI 主循环中调用 `Simulation::FramePublish()`；
2. `FramePublish()` 同步 step 后调用 `TransformSyncSystem`；
3. `EditorScene` 默认持有 `HostDownloadPublisher`；
4. `HostDownloadPublisher` 每帧下载 LinkPose/BodyPose/BasePose 到 host；
5. present draw 与 UI 记录依赖同一主循环；
6. per-frame 不应发生的 `WaitIdle`、资源同步或重复 AABB/geometry 工作会放大 hitch。

### 7.2 目标路径

```text
PhysicsWorker
  Step / policy / contact / controller
        ↓
  CUDA event or host snapshot publish
        ↓
RenderConsumer
  latest snapshot / imported SSBO
        ↓
Vulkan command buffer
        ↓
Present
```

硬约束：

- UI 线程零 `DownloadField`；
- UI 线程零 physics step；
- UI 线程不等待 physics catch-up；
- zero-copy 可用时，Interactive 默认使用 CUDA↔Vulkan external memory + semaphore；
- zero-copy 不可用时，worker 使用 HostDownloadPublisher 或 host snapshot fallback，UI 仍不被 D2H 阻塞；
- UI 明确显示 `ZERO-COPY` 或 `HOST FALLBACK`、fallback reason、copy bytes 和 copy time；
- per-frame 只更新 instance transforms/material dynamic state；mesh/BLAS/descriptor 不重复创建；
- `vkQueueWaitIdle` 只允许出现在初始化、swapchain recreate、资源销毁、明确 capture synchronization 路径，不得出现在正常 present frame；
- snapshot 更新失败时保留上一帧并显示 stale age，不阻塞或崩溃。

### 7.3 Zero-copy 扩展

现有 `CudaVulkanInteropPublisher` 是实现起点，但必须扩展：

- Interactive：至少支持一个 scene 的全部 dynamic instances；
- Train Observe：支持最多 32 个 display instances 的 transform table；
- external memory/semaphore import 失败时可靠 fallback；
- instance binding 包含 `PoseSource`、cached visual local、environment index 和 instance row；
- CUDA scatter 在 physics stream 的 FK 完成后运行；Vulkan 消费由 external semaphore 或明确 fence 保证可见性；
- shader 的 `gl_InstanceIndex` 与 transform row 一一对应；
- zero-copy path 与 host path 使用同一 pose schema 和 hash；
- 使用同一份 static mesh cache，不能为 interop 另建一套 scene semantics。

### 7.4 性能验收

基准：NVIDIA CUDA + Vulkan 机器、真实 Go2 13-link scene、物理开启、一个 Interactive env；Train Observe 另测 16/32 display instances。

Interactive 最低目标：

- present 目标 30 FPS；
- end-to-end frame time p95 <= 33 ms；
- UI input 在 physics catch-up 时仍可操作；
- worker physics tick 不被 render fence 反向阻塞；
- report physics、controller/inference、snapshot publish、render record、GPU submit、present 分项耗时；
- zero-copy path 与 host fallback 各有独立结果，不能用 fallback 结果冒充 zero-copy；
- 结果记录 GPU、driver、scene hash、model hash、resolution、env count 和 build flags。

## 8. Gizmo 修复

### 8.1 问题定义

本需求只修 gizmo 手柄随相机缩放/viewport 变化的显示大小，不增加对象 Scale 能力。Nuka 的 `math::Transform`、SceneIR、物理 collider 和 `.nks` 继续只支持 position + rotation 的当前合同。

现有问题包括：

- 没有调用 `ImGuizmo::SetGizmoSizeClipSpace()`；
- `SetRect()` 使用整个 window，而不是 central viewport 的实际 screen rect；
- gizmo projection 与 renderer camera 的 viewport/FOV/near/far 可能不一致；
- `TransformFromModel()` 忽略 scale 是当前明确设计，不应借修手柄大小引入 scale。

### 8.2 修复合同

- 从 ImGui central viewport 取得 origin、width、height；
- `SetRect(viewport_x, viewport_y, viewport_w, viewport_h)`；
- view/projection 与当前 `CameraController` resolved camera 完全一致；
- 每帧设置固定 clip-space gizmo size，目标为稳定的屏幕可操作尺寸；
- size 在 viewport 高度过小时 clamp，不能随 DPI/窗口 resize 消失；
- 选中实体距离变化时，translate/rotate handles 保持近似恒定像素尺寸；
- camera orbit/zoom 与 gizmo hover/drag 不互相抢输入；
- gizmo 仅在 Interactive 且 paused 时可用；Train Observe 禁用；
- 保留 world/local、translate/rotate、snap、undo/redo；
- 不加入 `ImGuizmo::SCALE`，不修改 physics/model transform ABI。

### 8.3 Gizmo 验收

固定 scene + 固定 entity，至少覆盖：

- camera distance 最小/默认/最大；
- FOV 20/45/90；
- window 1280x720、1920x1080、HiDPI；
- world/local、translate/rotate、snap on/off；
- drag 后 entity world pose、SceneIR、physics pose、undo/redo 一致；
- screen-space handle 覆盖区域随距离变化不超过预设容差；
- 截图/交互 smoke 证明 gizmo 与真实 mesh 对齐。

## 9. 原生 ImGui UI 改进

不引入 React、WebView、Chromium 或 Tailwind。shadcn 只作为信息架构、token、间距和交互密度的设计参考，用 Dear ImGui 原生实现。

### 9.1 布局

```text
┌─────────────────────────────────────────────────────────────┐
│ command bar: mode / play / pause / step / reset / status    │
├──────────────┬──────────────────────────────┬───────────────┤
│ Scene Tree    │ Vulkan viewport              │ Inspector     │
│ Assets        │ gizmo / selection / effects  │ Physics       │
│ Selection     │ real-time pose               │ Inference     │
├──────────────┴──────────────────────────────┴───────────────┤
│ Timeline / Metrics / Console / Snapshot / Render profile    │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 交互规范

- command bar 使用图标+tooltip 的 undo/redo、play/pause、step、reset、record、camera fit；
- mode 使用 segmented control：Interactive / Train Observe；
- connection、pose source、physics health、snapshot age 使用 badge/status dot；
- tabs 分离 Scene、Physics、Inference、Render、Metrics、Console；
- inspector 根据 mode 和 paused/running 状态自动锁定；
- 所有数值编辑显示单位、有效范围和 dirty 状态；
- 长任务显示 progress、cancel 和最后一次成功 snapshot；
- 不用嵌套 card 堆叠，中心 viewport 保持主要视觉层级；
- 高 DPI 下字体、按钮、tree row、gizmo 和 tooltip 使用统一 scale；
- UI 颜色 token 与实时/警告/错误状态统一，不让一个青色或暗色覆盖所有语义。

### 9.3 状态显性化

Stats/Performance 面板至少显示：

```text
mode             Interactive / Train Observe
physics          RUNNING / PAUSED / STALE
pose source      ZERO-COPY / HOST FALLBACK
snapshot age     ... ms
dropped snapshot ...
physics tick     ...
physics ms       ...
controller ms    ...
render ms        ...
present fps      ...
scene hash       ...
model hash       ...
```

## 10. Vulkan 实时渲染增强

实时 Vulkan 采用 opt-in realtime profile，不改变 core，不破坏既有 offscreen determinism gate。默认 profile 与旧测试保持兼容；editor 的 Beauty/Realtime profile 可以启用增强效果。

### 10.1 Render graph

```text
snapshot / imported transforms
        ↓
optional sky gradient / fog backdrop
        ↓
directional shadow depth pass
        ↓
opaque PBR forward pass
        ↓
ground / contact decal
        ↓
physics debug overlay
        ↓
ImGui command bar and panels
        ↓
present
```

### 10.2 第一阶段效果

按优先级全部纳入：

1. **实时 directional shadow map + PCF**
   - 替换 present path 当前的 `shadow_dummy`；
   - shadow image/sampler/descriptor 持久化；
   - resize/profile change 才重建资源；
   - sun light、bias、PCF kernel、shadow resolution 可调；
   - 真实 Go2 foot、terrain 和 robot self-shadow 可见；
   - shadow pass 与主 pass 分项计时。

2. **环境与接地**
   - sky gradient、fog、ground plane、contact shadow decal 复用 `RasterOptions`；
   - realtime path 与 offscreen path 共享环境参数，不复制 scene semantics；
   - ground/contact shadow 默认由 profile 控制，debug/offscreen gate 默认不变；
   - 不将 renderer-side floor 偷塞进 physics world。

3. **材质与色彩**
   - tone mapping/exposure；
   - metallic/roughness、emissive、opacity 的稳定解释；
   - authored texture/normal map 能力按资源版本绑定；
   - 统一 realtime/offline material contract，缺少 capability 时显式 fallback；
   - 先完成可验证的 forward PBR，再扩展 IBL/specular reflection。

4. **物理可视化**
   - collider wireframe；
   - contact point、normal、force/wrench vector；
   - joint axes、COM、base frame；
   - selected env trajectory/path；
   - sensor rays/height scan 可选；
   - overlay 使用 snapshot 中的 state，不在 UI 线程重新下载 physics fields；
   - debug overlay 不进入 core solver，不改变 physics。

### 10.3 实时性能约束

- shader/pipeline/descriptor/mesh cache 持久化；
- 正常 frame 禁止 `vkQueueWaitIdle`；
- draw command 录制不扫描所有 mesh vertices，camera override 或缓存 AABB 优先；
- Transform 更新使用 interop SSBO 或 worker snapshot；
- 物理 overlay 的高成本类型可按 profile 限制数量，但 UI 要显示被截断数量；
- 实时效果打开后仍以 30 FPS / p95 33ms 为 Interactive 目标；
- 阴影、PBR、overlay 分别有 on/off profiling，不能只报总帧时间。

## 11. CUDA 离线 RT 增强

离线 RT 质量优先，和实时 Vulkan 是两个 profile，但共享 scene、material、camera、pose snapshot 和 hash。

### 11.1 1080p HQ 基线

第一阶段固定可复现的 HQ profile：

```text
resolution       1920x1080
camera spp       128
soft shadow rays 16
AO rays          8
GI bounces       1
seed             explicit and recorded
AOV              color, depth, normal, albedo, instance/material id, validity
```

目标是 presentation-grade：

- 暗色 Go2 材质仍能读出轮廓和关节层次；
- soft shadow、AO、一次 diffuse GI 稳定；
- 连续动画没有明显 flicker、拖影、随机跳变；
- profile、scene、trajectory、GPU、provider 和 seed 完整记录。

### 11.2 结构性提速

保留既有 self-CUDA/PHI RT provider 抽象，具体实现优先做：

- persistent RT session，不每帧重建静态 mesh BLAS；
- rigid Go2 link 只更新 instance transform；
- deforming surface 仅在 geometry version 变化时 rebuild BLAS；
- TLAS update/rebuild 与 trace 分开计时；
- persistent framebuffer、ray queue、scratch、AOV 和 accumulation buffer；
- GPU-side accumulation，导出时才做有限 host readback；
- static light/HDRI/texture sampling tables 常驻 device；
- sample/frame/pixel/bounce/lobe seeded key 固定；
- 离线动画使用 motion/history AOV 为后续 temporal denoise 保留接口；
- 不能通过偷偷降低分辨率、关闭软阴影/GI 或改变轨迹来宣称提速。

### 11.3 质量扩展

HQ baseline 通过后，可扩展：

- higher spp final profile；
- multi-bounce diffuse/specular；
- transmission/refraction/Beer-Lambert；
- HDRI importance sampling、MIS；
- spatial + temporal denoise；
- raw/AOV/denoised export；
- high-quality video export。

OptiX 或其他 hardware RT provider 作为可选 provider，不能成为 Nuka core 或 nuka-editor 的硬依赖。provider capability/fallback 沿用既有 Spec 03 的 versioned RT ABI 原则。

## 12. 实施顺序

### P0：边界与 profiling

1. 记录当前 `nuka-editor` physics、HostDownloadPublisher、controller、render、present、WaitIdle、AABB 和 UI 各阶段耗时。
2. 把 core/runtime/editor 的依赖方向写入 CMake target 约束和架构测试。
3. 定义 `FrameSnapshot`、`CommandEnvelope`、hash 和 schema version。
4. 明确现有 `Simulation` 哪些字段应迁到 runtime worker，哪些属于 editor。

出口：无功能变化的 profiling 报告，能证明当前卡顿来自哪些同步/拷贝点。

### P1：PhysicsWorker + Interactive snapshot

1. 实现 worker 独占 `nk::World` 和固定 tick。
2. UI 命令通过 queue 在 tick 边界应用。
3. 实现双/三槽 snapshot ring。
4. Interactive 先使用 worker host snapshot fallback，确保 UI 线程不再 step/download。
5. Pause-only edit、edit 后 controller/episode reset。
6. 外部 API 和 editor Script panel 改为 command client。

出口：Interactive 仍可加载 scene、Play/Pause/Step/Reset、Go2 policy、teleop、gizmo、Inspector 和脚本；UI 不调用 physics field readback。

### P2：CUDA↔Vulkan zero-copy

1. 把现有 `CudaVulkanInteropPublisher` 接入 Interactive 默认路径。
2. 完成 external memory/semaphore lifecycle、visibility 和 fallback reason。
3. 扩展 instance table 与 shader binding。
4. 移除正常 frame 的 queue idle 和重复资源工作。
5. 通过 NVIDIA machine benchmark；无 interop 的 lavapipe/CPU Vulkan 仍可 fallback。

出口：zero-copy available 时默认 active；不可用时 editor 明确显示 fallback；30 FPS/p95 目标有分项报告。

### P3：Train Observe IPC

1. 实现 `nuka.observe.v1` TCP handshake。
2. 实现 shared-memory snapshot ring 和 Python client。
3. editor 本地加载严格匹配 scene；hash mismatch 拒绝。
4. 支持最多 32 个真实 pose 的 display-only instances，不做视觉平移。
5. 实现 read-only UI、metrics、physics state、observation stats。
6. 实现 editor-local recording、timeline 和 trace export。

出口：外部 trainer headless 运行时，editor 能实时显示 1/16/32 个 env 的真实 pose，训练端性能不因 editor 阻塞。

### P4：Gizmo 与原生 UI

1. 使用 central viewport rect 和 camera-consistent projection 修 gizmo。
2. 固定 clip-space handle size，覆盖 zoom/FOV/DPI 测试。
3. 完成 Interactive/Train Observe mode gating。
4. 重组 command bar、tabs、Inspector、Metrics、Inference、Render、Timeline 和 Script panel。
5. 添加 pose source、snapshot age、hash、health、fallback status。

出口：gizmo 交互 smoke、UI screenshot smoke、pause-only edit 和 undo/redo 通过。

### P5：Vulkan realtime profile

1. present path 接入真实 shadow map + PCF。
2. 接入 sky/fog/ground/contact effect。
3. 接入 PBR/tone mapping/material capability。
4. 接入 snapshot-driven physics overlays。
5. 完成每个效果的 profiler 和 30 FPS gate。

出口：实时画面明显具备阴影、接地、材质层次和物理可视化，offscreen determinism 默认测试不回归。

### P6：离线 RT HQ

1. 把 StudioRtRenderer 重构为 persistent session。
2. 完成 1080p HQ baseline 和 deterministic metadata。
3. 增加 GPU accumulation、AOV、denoise/history 接口。
4. 优化 TLAS/BLAS/texture/light/scratch lifecycle。
5. 通过固定 Go2 scene/trajectory 的质量报告，再做更高 spp/final profile。

出口：HQ profile 可重现、视觉质量通过、结构性性能报告完整；不把 HQ 离线成本混入 realtime FPS gate。

## 13. 测试与验收 Gate

### 13.1 Core 不受污染

- [ ] Vulkan OFF 时 core、headless runtime、Python physics API 可构建并运行。
- [ ] core target 不包含 ImGui/GLFW/Vulkan/IPC/Python UI symbols。
- [ ] `nk::World` solver/contact/ABI 的 D1、checkpoint、hash 和现有 physics tests 不因 editor 改动改变。
- [ ] trainer 可以不启动 `nuka-editor` 运行完整 batch/headless rollout。

### 13.2 Worker 与 snapshot

- [ ] `nk::World` 只有 physics worker owner。
- [ ] UI 线程无 Step/UploadField/DownloadField。
- [ ] command 在 tick 边界按序消费；snapshot generation 不回退。
- [ ] render 慢时只丢 snapshot，不阻塞 physics 或 UI。
- [ ] worker shutdown、trainer crash、editor disconnect 不死锁。

### 13.3 Interactive

- [ ] local scene 加载、Play/Pause/Step/Reset、Go2 inference、teleop、script、Inspector、gizmo 可用。
- [ ] running/infer 时 editor 只读。
- [ ] paused edit 后 policy history、decimation、episode state reset。
- [ ] Interactive display 与 physics snapshot 的 scene/model hash 一致。

### 13.4 Train Observe

- [ ] trainer 作为唯一 physics/policy 权威，editor 不产生 physics mutation。
- [ ] scene/model/asset/profile/schema hash mismatch 明确拒绝。
- [ ] 1/16/32 env 的 pose、quat 顺序、单位、坐标系验证通过。
- [ ] 不添加 editor spacing/origin/yaw；重叠 pose 原样显示。
- [ ] display instances 不进入 collision/contact solve。
- [ ] 指标、物理状态、性能与 observation、timeline/本地录制可用。

### 13.5 Vulkan performance

- [ ] NVIDIA Go2 benchmark present 目标 30 FPS，p95 frame <= 33 ms。
- [ ] worker physics、controller、snapshot、render、present 分项可观测。
- [ ] zero-copy 可用时自动启用；fallback reason 和 host copy cost 可见。
- [ ] 正常 frame 无 `vkQueueWaitIdle`；mesh/pipeline/descriptor cache 不重复创建。
- [ ] shadow/sky/fog/ground/PBR/overlay 可以独立开关和 profiling。

### 13.6 Gizmo/UI

- [ ] zoom/FOV/DPI/viewport resize 下 gizmo handle 尺寸稳定。
- [ ] gizmo 与真实实例对齐，world/local/snap/undo/redo 一致。
- [ ] 不引入对象 scale，不改变 transform/physics ABI。
- [ ] Native ImGui UI 完成 Interactive/Train Observe mode gating 和状态显性化。

### 13.7 Offline RT

- [ ] 1080p HQ profile 使用 128 spp、16 shadow rays、8 AO rays、1 GI bounce，seed 显式记录。
- [ ] 静态 BLAS、textures、lights、scratch、AOV/accumulation 持久化；刚体只 update instance/TLAS。
- [ ] raw/AOV/denoised output 可导出，动画无明显 flicker、拖影和严重 speckle。
- [ ] scene、trajectory、profile、provider、GPU、driver、seed 可完整复现。
- [ ] 离线 RT 质量和 wall time 单独报告，不用实时 FPS 口径掩盖任一问题。

## 14. 非目标与明确限制

- 不把 `nuka-editor` 变成 Nuka Physics core 的必需启动器。
- 不让 core 依赖 Vulkan、ImGui、IPC、Python、GLFW 或 Go2 policy。
- 不在 Train Observe 中编辑 trainer world、暂停 trainer、重置 env 或发送 policy command。
- 不把 32 个观察实例复制进 physics broadphase；它们是 display-only。
- 不为重叠环境自动添加视觉间距；需要可读性时另做明确的非权威 compare mode，默认不启用。
- 不在本 spec 中增加对象 Scale；gizmo 只修手柄视觉大小。
- 不把 WebView/React/shadcn runtime 引入 nuka-editor。
- 不承诺 editor 取代 RL framework、checkpoint manager 或实验跟踪系统。
- 不为实时 viewport 使用离线 path tracer；实时 Vulkan 与离线 CUDA RT 是两个质量/时序 profile。
- 不以关闭阴影、降低分辨率、减少物理 tick 或改变真实 pose 作为性能达标手段。
- 不将现有后端 provider/OptiX 计划强行下沉到 physics core；RT provider 仍通过独立 adapter/ABI 接入。

## 15. 交付物

- `docs/architecture/nuka-runtime-editor-boundary-zh.md`：core/runtime/editor/trainer 依赖图和 ownership。
- versioned `FrameSnapshot` / `CommandEnvelope` / `nuka.observe.v1` 协议定义。
- `nuka-editor interactive` 与 `nuka-editor observe` 启动路径。
- `nuka.viewer` Python client 和 CLI 连接工具。
- PhysicsWorker、snapshot ring、host fallback 和 CUDA/Vulkan zero-copy publisher。
- Interactive/Train Observe UI、gizmo 修复、physics/inference/metrics/timeline panels。
- Vulkan realtime profile、shadow/ground/PBR/debug overlay 的 shader/pipeline/resource cache。
- CUDA offline RT HQ profile、persistent session、AOV/accumulation/denoise metadata。
- NVIDIA interactive benchmark、host fallback benchmark、Train Observe 1/16/32 env report。
- 固定 Go2 scene/trajectory 的 realtime screenshot/video 和 offline 1080p HQ frame/video。

只有 P0-P3 完成并证明 core 边界、worker ownership、IPC schema 和 zero-copy/fallback 行为后，才进入 P4-P6 的 UI 与渲染效果扩展。这样可以避免用更漂亮的 Vulkan 画面掩盖 physics/render 同步问题。
