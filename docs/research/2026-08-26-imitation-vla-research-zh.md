# Nuka 模仿学习与 pi0.5 VLA 调研记录

日期：2026-08-26  
状态：调研阶段，尚未开始本需求的源代码实现  
范围：Unitree G1 舞蹈模仿学习 + 机械臂视觉 VLA/pi0.5 运行场景  
优先级：两个方向同等重要

> 本文用于保存当前上下文和研究结论。后续实现应以本文的“已确认事实”和“待验证门”为准，不应把外部项目的 Isaac Lab/MuJoCo 能力直接视为 Nuka 已有能力。

## 1. 当前工作区状态

### 1.1 仓库基本信息

- 当前目录：`C:/Softwares/code/Nuka-Physics`
- 当前分支：`master`
- 最近提交：`b6e4327 docs: integrate Go2 skill showcases`
- 当前工作区存在用户/历史工作留下的脏改动和未跟踪文件，包括：
  - 多个已删除的 `.bat` 启动脚本；
  - `context.md`；
  - `docs/roadmap/2026-08-26-spec-04-nuka-editor-extension-zh.md`；
  - `logs/`、渲染出的 `.ppm`、若干 bundle、`tools/ref_go2_*` 等。
- 本轮调研没有修改任何原有源代码，也没有恢复或删除上述文件。
- 本文档本身是本轮新增的调研记录。

### 1.2 用户已经确认的约束

1. 模仿学习和 VLA 必须都做，不能只把其中一个当作附属验证。
2. VLA 首选 pi0.5；可以参考 Hugging Face、Physical Intelligence `openpi` 和 `every-embodied`。
3. H1 场景不可用，后续方案不依赖 H1；现有 H1 文件暂不删除，避免影响历史代码和测试。
4. 需要判断当前机器是否足以部署 VLA。
5. 需要判断 Nuka 当前是否支持 view/视觉输出；如果不完整，再参考 Newton/Isaac Lab 的设计。
6. 当前阶段先调研和记录，之后暂停 compact，再继续实现。

## 2. 当前机器与软件环境

### 2.1 已检测到的硬件

PowerShell 和 `nvidia-smi` 检测结果：

| 项目 | 当前值 |
|---|---|
| CPU | AMD Ryzen 7 9800X3D，8 核 / 16 线程 |
| 内存 | 约 33.5 GB，系统可用规格约 32 GB |
| GPU | NVIDIA GeForce RTX 5080 |
| GPU 显存 | 16,303 MiB，约 16 GB |
| NVIDIA 驱动 | 610.88，Windows WDDM |
| CUDA UMD | 13.3 |
| 系统 | Windows，Git Bash/MSYS 环境 |
| Python | 3.13.5 |
| WSL | 已安装 `wsl.exe`，当前只有 `docker-desktop`，状态为 Stopped；没有发现可用的 Ubuntu 发行版 |

### 2.2 Nuka 官方构建前提

仓库 README 和 `docs/getting-started.md` 写明：

- 主要支持平台是 Linux x86-64；
- CUDA 12.8；
- `g++-10`；
- CUDA-capable GPU；
- Python 3.10+；
- Vulkan 可选，用于 Vulkan 验证和实时显示；
- Nuka 当前也存在 Windows 构建目录和 Python `.pyd`，但这不等于所有 Linux-only 外部依赖都能在 Windows 原生运行。

### 2.3 对 VLA 部署的初步判断

结论分为三层：

- **仅运行 pi0.5 推理：有希望。** Physical Intelligence `openpi` README 给出的单卡推理要求是 `> 8 GB`，RTX 5080 的 16 GB 显存满足这个最低数字。Blackwell/RTX 50 系列还需要确认实际 PyTorch/CUDA wheel 和 kernel 支持，不能只依据显存判断。
- **在同一 GPU 上同时运行 Nuka 物理、CUDA sensor RT 和 pi0.5：不建议直接同进程部署。** Nuka 训练/批量物理和 pi0.5 的视觉语言模型会竞争显存，16 GB 留给模型、KV cache、临时张量和 Nuka RT 的空间有限。
- **pi0.5 微调：当前机器不应作为已保证目标。** 官方 LeRobot 文档给出的 pi0.5 微调基线是单张 80 GB GPU；`openpi` README 给出的 LoRA 估计为 `>22.5 GB`，全量微调为 `>70 GB`。RTX 5080 16 GB 不满足官方 LoRA 估计，更不满足全量微调。

推荐部署形态是：

```text
Windows Nuka client / simulation
        |
        | websocket: uint8 images + state + prompt
        v
Linux or WSL2 openpi server
        |
        | RTX 5080, pi05 model inference
        v
action chunk: 7 joint velocity + 1 gripper action
```

如果 WSL2 CUDA、Linux Python 3.11 和 openpi 依赖在当前机器上无法稳定配置，应把推理服务迁移到另一台 Ubuntu/NVIDIA 机器；Nuka 客户端仍可以在 Windows 上运行并连接服务。

## 3. 当前 Nuka 已有能力

## 3.1 物理、批量环境和控制

当前仓库已经具有：

- GPU-resident articulated dynamics；
- Featherstone/ABA articulation；
- general contact pipeline；
- 多环境复制和批量 stepping；
- PyTorch/JAX/DLPack 接口；
- PD position、torque、velocity、computed torque、OSC、actuator 等控制模式常量；
- writable `DRIVE_TARGET`、`TORQUE_INPUT`、`VELOCITY_TARGET`、drive gains 等设备 buffer；
- state hash、checkpoint、reset_envs 和 D1 deterministic 设计；
- Go2 的已有 RL 和 TorchScript inference 示例。

Python 绑定通过 `torch.from_dlpack(world.buffer_view(...))` 让 Torch tensor 直接 alias Nuka 的 CUDA buffer。现有 Go2 task 采用这种方式组装观测和写入驱动目标，没有每步 NumPy round trip。

### 3.2 已有机器人/demo

当前仓库最成熟的是 Go2：

- `python/nuka/tasks/go2_locomotion.py`；
- `python/nuka/tasks/go2_backflip.py`；
- `python/nuka/tasks/go2_front_handstand.py`；
- Go2 观测构造、关节顺序映射、PD target 和 torque 约定；
- `examples/demo/go2_skill_infer.py`；
- `examples/demo/go2_skill_video_cuda.cpp`；
- Go2 trajectory 导出和 CUDA RT 视频渲染。

已有 Go2 经验可以复用的是：

- vectorized Gym-like task 结构；
- action/observation contract；
- reset、episode、termination；
- DLPack zero-copy buffer 访问；
- policy loop 和视频导出；
- 关节名称到 cooked slot 的结构化映射方法。

不能直接复用的是：

- Go2 的 12 DoF 常量；
- Go2 足端检测和四足 reward；
- Go2 参考动作、默认姿态和关节顺序；
- Go2 专用 `Go2ObsBuilder` 和 `Go2GpuPolicy`。

### 3.3 H1 的处理结论

H1 相关代码和历史实验存在，例如：

- `python/nuka/tasks/h1_stand.py`；
- H1 reduced training asset generator；
- H1 grasp choreography；
- H1 相关测试和 scene。

但用户已经确认 H1 场景不可用。因此：

- VLA 机械臂不使用 H1；
- 模仿学习不使用 H1；
- 不把现有 H1 grasp scene 作为机械臂基础；
- 现有 H1 文件暂不删除，避免无关破坏。

## 4. Nuka 当前是否支持 view/视觉输出

结论：**支持视觉 sensor/AOV 输出，支持离线 beauty 图像，也有 Vulkan viewer；但 Isaac Lab 风格完整的训练观察器/跨进程 view 架构仍不是一个可以直接假定完成的现成功能。**

### 4.1 GPU camera sensor 已实现

当前代码和测试表明，Nuka 有以下设备端相机能力：

- `src/render/sensor_backend.hpp`：批量 sensor render backend；
- CUDA RT 的 batched sensor render；
- 相机可以挂到 `Link`、`Body` 或 `Base`；
- 相机由 local offset `[px, py, pz, qw, qx, qy, qz]` 定义；
- 每个 env 可以挂多个相机；
- 同一尺寸下重复 attach 会形成 `S` cameras per env；
- 相机输出逻辑形状为 `(E, S, H, W, channels)`，单相机时 Python 文档也支持 `(E, H, W, channels)` 形式；
- COLOR/NORMAL/ALBEDO 为 float32，DEPTH 为 float32，PRIM 为 uint32；
- RANGE 是独立的 lidar/ray sensor plane；
- AOV 可通过 `set_sensor_aov_mask` 选择；
- 输出 buffer 在 CUDA device 上，没有强制 host round trip；
- 现有测试验证了 device pointer、非空图像、跨 env tile 一致性、重复 render deterministic、多相机 tile 和生命周期。

公开 C ABI 包含：

```text
nuka_world_attach_camera_sensor
nuka_world_render_sensors
nuka_world_get_sensor_view
nuka_world_get_sensor_dims
```

Python 绑定包含：

```python
world.attach_camera_sensor(...)
world.render_sensors()
world.get_sensor_view(nuka.SensorChannel.COLOR)
world.sensor_dims()
```

`get_sensor_view` 返回可被 `torch.from_dlpack(...)` 消费的 zero-copy CUDA ndarray。

### 4.2 图像 dtype 和 pi0.5 的适配

pi0.5/openpi 的推理客户端通常接收 `uint8` 的 `(H, W, 3)` 图像，目标尺寸一般为 224x224。Nuka camera COLOR 当前是 device float32 AOV，通常是 `[0, 1]` 语义。

因此需要在 Python client 中做：

```python
rgb_u8 = (
    color_device[env, camera]
    .clamp(0.0, 1.0)
    .mul(255.0)
    .to(torch.uint8)
    .cpu()
    .numpy()
)
```

然后传给 `openpi_client.image_tools.resize_with_pad` 或 LeRobot 的预处理。

这会在 VLA 请求时发生一次 GPU-to-host copy，但它只发生在 policy query 频率，而不是每个 200/240 Hz physics tick。第一版应该优先保证正确性和可观察性，再优化 pinned host buffer/异步 copy。

### 4.3 已有 host beauty render

Python `World` 还提供：

```python
world.render_beauty(
    eye=..., look=..., up=..., fov_deg=..., width=..., height=..., spp=...,
    dtype="uint8",
)
```

它使用离线 CUDA path tracer，把当前 world 渲染为 host `(H, W, 3)` NumPy 图像。这个接口适合：

- 视频/截图；
- demo 展示；
- 没有 batched camera sensor 时的验证。

它不适合第一版 VLA 闭环的高频 camera observation，因为它是 host beauty render，延迟和同步成本更高。VLA 应优先使用 batched camera sensor。

### 4.4 Vulkan/viewer 当前状态

仓库有：

- Vulkan raster renderer；
- Vulkan present renderer；
- CUDA-to-Vulkan interop 相关代码；
- `runtime/app/viewer`；
- camera controller、editor scene、debug overlay、ImGui layer；
- viewer/editor scenario tests；
- Go2 skill video 和 offscreen render demo。

但未跟踪的 `docs/roadmap/2026-08-26-spec-04-nuka-editor-extension-zh.md` 明确记录了当前架构问题和目标：

- 现有 editor 主循环可能同步调用 physics、controller、pose download、UI record 和 present；
- `HostDownloadPublisher` 会产生 GPU-to-CPU pose copy；
- `CudaVulkanInteropPublisher` 是起点，但实时主路径还需要进一步接入；
- Train Observe 的 physics worker、snapshot ring、IPC、display-only instances 和最多 32 env 观察模式属于规格/后续实现方向；
- 目标架构是 physics worker 独占 `nk::World`，render/UI 只消费 snapshot，不直接 step physics。

因此，本需求第一阶段不应等待完整 editor/IPC/viewer 重构：

- 模仿学习训练先 headless；
- VLA 相机 observation 先走 Python `render_sensors + get_sensor_view`；
- demo 视频先走已有 `render_beauty` 或已有 C++ offscreen 工具；
- 后续若需要训练中实时观察，再实现或接入 Spec 04 的 snapshot/observe 体系。

## 5. 现有 articulation DOF 约束与 G1 风险

### 5.1 当前代码中的两个上限

`src/runtime/articulation/articulation_contacts.hpp` 当前包含：

```text
kMaxArticulationDof = 64
kMaxContactSolverDof = 18
```

它们语义不同：

- `kMaxArticulationDof=64` 已用于主 contact solve、CRBA/factor 等通用工作存储；
- `kMaxContactSolverDof=18` 仍用于部分 diffsim KKT/IFT/Osc-adjoint 和 computed torque/OSC dense scratch；
- 对超过 18 的路径，当前代码很多地方已经改为 loud failure，而不是静默截断；
- `tests/runtime/test_dof_above18_honesty.cpp` 专门验证超过 18 DoF 的因子化、effective mass、implicit damping 和超过 64 的 loud failure；测试注释仍保留“历史上 RED”的背景，实际是否全部通过需要在当前机器构建后验证。

### 5.2 G1 的实际广义 DoF

Unitree 官方 `g1_29dof_mode_15.urdf` 是 29 个可动关节版本。若使用 floating base，广义坐标通常是：

```text
6 floating-base DoF + 29 scalar joint DoF = 35 generalized DoF
```

这低于主通用 contact path 的 64 上限，但高于历史 18 上限。因此不能未经 smoke test 就承诺 G1 29 DoF 舞蹈能在 Nuka 中训练。

需要实际验证：

1. G1 URDF/MJCF 是否被 importer 正确解析；
2. 所有 visual/collision mesh 是否能被加载和 cook；
3. 根部是否得到 floating-base articulation，而不是错误的 fixed root；
4. 35 DoF 的 CRBA/factor/contact solve 是否完整覆盖；
5. 29 个动作关节是否能按名称映射到 drive slots；
6. 足端与平面的接触是否稳定；
7. 每步 finite、reset、state hash、批量 env 复制是否正常；
8. Nuka 现有固定 per-env contact slot 设计是否足以表达 G1 足底接触；
9. 使用 PD position/torque 模式时是否绕开 18-DoF computed torque/OSC 限制。

第一版 G1 task 建议使用主通用 path + PD position 或 torque，不依赖 computed torque/OSC/diffsim adjoint。

## 6. 模仿学习方向调研

## 6.1 目标定义

用户目标是：

- 使用宇树机器人；
- 机器人在 Nuka 场景中跳舞；
- 通过模仿学习/强化学习训练；
- 训练直到动作稳定；
- 能在 Nuka 中回放、评估和输出视频。

推荐的第一个具体目标是 **Unitree G1 EDU 29 DoF 单段舞蹈动作跟踪**，而不是一开始做 Go2 舞蹈或多人编舞。原因：公开 G1 动作数据和训练参考更多，G1 是人形机器人，舞蹈动作和“宇树机器人跳舞”的表达更直接。

## 6.2 公开参考项目

### G1 Dance Pipeline

地址：<https://github.com/guoshuhao066-code/g1-dance-pipeline>

该项目给出了完整流水线：

```text
human motion
    -> GMR retargeting
    -> 36-column G1 CSV
    -> Isaac Lab full-body NPZ
    -> BeyondMimic policy training/play
    -> ONNX
    -> MuJoCo Sim2Sim
    -> suspended G1 Sim2Real
```

README 中的基线：

- Unitree G1 EDU 29 DoF；
- 参考动作 50 Hz NPZ；
- policy observation 154；
- policy output 29；
- 训练基线 512 environments、30,000 iterations；
- 输出 raw ONNX 和 actions-only deployment bundle；
- 输入可来自 CMU、LAFAN1、Nokov、SMPL-X/GVHMR 等。

该项目依赖 Isaac Lab、GMR、Unitree RL/MuJoCo 等外部组件，不能直接当作 Nuka task，但可以作为数据格式、训练节奏、部署安全检查和验收指标的参考。

### BeyondMimic / whole_body_tracking

地址：<https://github.com/HybridRobotics/whole_body_tracking>

README 的主要流程：

1. 取得 Unitree retargeted motion CSV；
2. `csv_to_npz.py` 做 forward kinematics，生成 body pose/velocity/acceleration；
3. `Tracking-Flat-G1-v0` 或 `Tracking-Flat-G1-Wo-State-Estimation-v0` 进行 PPO tracking；
4. 通过 RSL-RL 训练；
5. play、导出和 sim2sim。

奖励和观察设计的总体思想是 DeepMimic/BeyondMimic：

- 参考关节姿态/速度；
- 当前机器人和参考动作的身体位置、姿态、线速度、角速度误差；
- action smoothing；
- 关节限位、力矩和稳定性约束；
- fall/height/orientation termination；
- actor/critic 可能使用不同观测，实机部署观测应尽量接近 IMU/encoder 可获得量。

### G1 Moves 数据集

地址：

- 代码：<https://github.com/experientialtech/g1-moves>
- 数据集：<https://huggingface.co/datasets/exptech/g1-moves>

数据集 README 标注为 CC-BY-4.0。公开数据包含：

- 60 个 Unitree G1 EDU 29 DoF clip；
- 其中 28 个 dance、27 个 karate、5 个 bonus；
- 60 FPS retargeted motion；
- PKL、CSV、NPZ、PyTorch policy、ONNX policy；
- 不把大数据和权重直接提交到主仓。

数据格式：

PKL：

```text
fps       scalar
root_pos  (N, 3)
root_rot  (N, 4), xyzw
 dof_pos  (N, 29)
```

CSV：

```text
columns 0..2   root position
columns 3..6   root quaternion, xyzw
columns 7..35  29 joint angles
```

NPZ：

```text
fps             (1,)
joint_pos       (N, 29)
joint_vel       (N, 29)
body_pos_w      (N, 30, 3)
body_quat_w     (N, 30, 4)
body_lin_vel_w  (N, 30, 3)
body_ang_vel_w  (N, 30, 3)
```

G1 Moves README 给出的 160 维 observation contract：

```text
0-28       ref_joint_pos
29-57      ref_joint_vel
58-60      motion_anchor_pos_b
61-66      motion_anchor_ori_b, rotation matrix first 2 columns
67-69      base_ang_vel
70-72      base_lin_vel
73-101     current joint_pos - default_pos
102-130    current joint_vel
131-159    last_action
```

输出为 29 个 joint position target，50 Hz，通常 200 Hz physics + decimation 4。

注意：G1 Dance Pipeline/BeyondMimic 的 README 里有 154 维无 state-estimation 观察版本，G1 Moves README 里有 160 维版本。它们不是同一个 contract，必须在实现时选定一个，并用字段级测试锁定，不能混接。

## 6.3 every-embodied 的舞蹈参考

地址：<https://github.com/datawhalechina/every-embodied>

相关页面：

- 春晚舞蹈机器人复刻：
  <https://github.com/datawhalechina/every-embodied/blob/main/07-%E6%9C%BA%E5%99%A8%E4%BA%BA%E6%93%8D%E4%BD%9C%E3%80%81%E8%BF%90%E5%8A%A8%E6%8E%A7%E5%88%B6/Locomotion/01%E6%98%A5%E6%99%9A%E8%88%9E%E8%B9%88%E6%9C%BA%E5%99%A8%E4%BA%BA%E5%A4%8D%E5%88%BB.md>
- AGILE 人形机器人 Loco-Manipulation：
  <https://github.com/datawhalechina/every-embodied/blob/main/05-%E5%85%B7%E8%BA%AB%E5%9C%BA%E6%99%AF%E7%9A%84%E6%B7%B1%E5%BA%A6%E5%92%8C%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0/03AGILE%E4%BA%BA%E5%BD%A2%E6%9C%BA%E5%99%A8%E4%BA%BALoco-Manipulation%E5%A4%8D%E7%8E%B0/README.md>

春晚页面记录的流程是：

```text
Prompt / Video -> PromptHMR -> SMPL-X -> GMR -> Robot Motion
```

它强调的实际难点：

- 人体/视频到 G1 的 retarget 精度；
- 人体动作可能超出机器人扭矩和稳定性；
- 视频复刻本身缺乏物理反馈；
- 后续需要 Isaac Sim/其他 physics 训练增强动作稳健性。

对 Nuka 的启示：

- 不必第一阶段接入 PromptHMR；
- 先用已 retarget 的 G1 CSV/NPZ，建立物理 tracking task；
- 先证明 Nuka 能把一段动作稳定地跟踪；
- 视频到机器人动作可以作为后续数据生产工具，而不是阻塞第一个 Nuka demo。

## 6.4 推荐的 Nuka 模仿学习任务

### 第一版机器人和资产

- Unitree G1 EDU 29 DoF；
- 优先使用 `g1_29dof_mode_15.urdf` 或可获得的等价 MJCF；
- 需要补齐 mesh 下载、路径重写、license/provenance 和 floating-base scene authoring；
- 先做单机器人、平地、无视觉的 headless task；
- 训练时使用 29 个 action target，根部状态由 physics integration 管理。

### 第一版控制 contract

建议：

```text
physics dt:       1/200 或 1/240
policy/control:   50 Hz
decimation:       4 或 5，按最终 dt 对齐
action:           29 joint position targets
control mode:     PD position
```

等 G1 PD tracking 稳定后，再考虑 torque policy。第一版不要依赖 computed torque/OSC，因为它们仍有 18-DoF scratch 上限。

### 第一版 observation contract

推荐先采用 G1 Moves 的 160 维字段，因为它有清晰的公开说明和现成 NPZ：

```text
reference joint position      29
reference joint velocity       29
anchor position/orientation     9
current base angular/linear    6
current joint position error   29
current joint velocity         29
previous action                29
                              ---
                              160
```

其中实际 root/base 的 frame、四元数顺序、body anchor 选择、坐标轴方向必须在 converter 中显式写出，不允许依赖“和 MuJoCo 一样”的模糊约定。

### 第一版 reward

初始 reward 可以采用 DeepMimic/BeyondMimic 的分解：

```text
r_pose       = exp(-k_pose * mean((q - q_ref)^2))
r_vel        = exp(-k_vel  * mean((qd - qd_ref)^2))
r_anchor_pos = exp(-k_pos  * ||anchor_pos_error||^2)
r_anchor_ori = exp(-k_ori  * orientation_error^2)
r_body_vel   = exp(-k_body * body_velocity_error^2)
r_upright    = upright/base-height stability reward
r_contact    = valid foot contact / support reward
r_smooth     = action-rate and torque penalty
r_limit      = joint limit / nonfinite penalty
```

训练顺序建议：

1. 固定一段短舞蹈，先解决姿态 tracking 和站稳；
2. 逐步增加 reference clip 长度和动作能量；
3. 增加 root pose/velocity、摩擦、mass、latency、PD gain randomization；
4. 最后再做多段 dance、随机起始帧和恢复能力。

### 稳定的验收标准

“训练稳定”不能只看一段视频，至少应记录：

- policy 输出和 physics state 全部 finite；
- 目标 clip 完成率；
- 20/50 Hz 控制循环无异常；
- base height、base tilt、脚接触不出现持续发散；
- joint pose RMS error、anchor position/orientation error；
- fall termination rate；
- torque saturation rate；
- 末尾稳定保持时间；
- 3 个 seed 或至少多次 deterministic replay 的结果；
- 训练 checkpoint 和评估 trajectory 可复现。

## 7. pi0.5 VLA 方向调研

## 7.1 官方 openpi

仓库：<https://github.com/Physical-Intelligence/openpi>

官方 README 明确写出三类模型：

- pi0：flow-based VLA；
- pi0-FAST：FAST action tokenizer 的 autoregressive VLA；
- pi0.5：改进的 open-world generalization VLA。

当前 openpi 对 pi0.5 支持的是 flow matching head；原论文中其他组件如 subtask prediction、action tokenization 或 RL 不属于当前开源实现的完整范围。

官方显存估计：

| 模式 | 估计显存 |
|---|---:|
| inference | `> 8 GB` |
| LoRA fine-tuning | `> 22.5 GB` |
| full fine-tuning | `> 70 GB` |

官方测试平台是 Ubuntu 22.04，README 明确说明当前不支持其他操作系统。

官方 Python 项目依赖包括：

- Python `>=3.11`；
- JAX CUDA 12；
- PyTorch 2.7.1；
- transformers 4.53.2；
- LeRobot；
- openpi-client；
- CUDA/OpenPI 对应的 uv environment。

官方支持 base checkpoint 和已微调 checkpoint，其中与当前需求最相关的是：

```text
pi05_base
pi05_droid
pi05_libero
```

官方 server 入口：

```bash
uv run scripts/serve_policy.py \
  policy:checkpoint \
  --policy.config=pi05_droid \
  --policy.dir=gs://openpi-assets/checkpoints/pi05_droid
```

server 默认监听 8000，并通过 websocket 接收 observation、返回 action chunk。

### 7.2 openpi DROID contract

官方文件：

- `src/openpi/policies/droid_policy.py`
- `examples/droid/README.md`
- `examples/droid/main.py`

DROID 示例输入：

```text
observation/exterior_image_1_left   uint8 image 224x224x3
observation/wrist_image_left        uint8 image 224x224x3
observation/joint_position          7
observation/gripper_position        1
prompt                              text
```

DROID 输入 transform 会把 state 拼成：

```text
7 joint positions + 1 gripper position = 8 state dims
```

pi0/pi05 使用：

```text
base_0_rgb
left_wrist_0_rgb
right_wrist_0_rgb = zero image, masked
```

官方 DROID outputs transform 只返回前 8 个 action 维度。

官方 DROID 示例明确使用：

```text
7 joint velocity actions + 1 gripper position action
```

动作应在客户端 clip 到 `[-1, 1]`，gripper 通常二值化。

官方示例还说明：

- policy 不必每个 environment step 都重算；
- 可以执行 action chunk 的前若干步，再重新规划；
- 默认 open-loop horizon 示例是 8；
- DROID 控制频率示例是 15 Hz；
- 网络 inference 0.5-1 秒/chunk 的延迟是可能的，因此需要 chunk/replan 策略。

注意：openpi 当前 `training/config.py` 中 `pi05_droid` 的 model 配置是 `action_dim=32`、`action_horizon=15`；DROID data transform 对外只保留 8 维。不同 openpi/LeRobot 版本中 action chunk 长度的默认值有差异，实际实现必须从加载的 checkpoint metadata/config 读取，不要硬编码 50 或 15。

### 7.3 LeRobot pi05

官方文档：<https://huggingface.co/docs/lerobot/en/pi05>

相关模型：

- <https://huggingface.co/lerobot/pi05_base>
- <https://huggingface.co/lerobot/pi05_libero_base>

安装：

```bash
pip install "lerobot[pi]@git+https://github.com/huggingface/lerobot.git"
```

或使用对应 release 的 `lerobot[pi]`。

LeRobot pi05 关键设置：

```text
policy.type=pi05
image resolution=224x224
state/action padded to max dimension 32
flow matching inference
```

LeRobot 默认配置中常见值：

```text
chunk_size = 50
action_horizon = 50
num_inference_steps = 10
max_state_dim = 32
max_action_dim = 32
dtype = float32 or bfloat16
```

但预训练 checkpoint 的配置可能覆盖这些默认值，实际运行时必须检查 `config.json`。

LIBERO 文档给出的特征：

```text
agentview image: 256x256x3 -> 224x224
wrist image:    256x256x3 -> 224x224
state:          8
action:         7, internally padded to 32
```

LeRobot 文档还强调：

- pi05 默认使用 quantile normalization；需要 `q01/q99`；
- 可以显式切换到 MEAN_STD；
- `n_action_steps`、`empty_cameras` 不能误用 pretrained_path 默认值；
- real-time chunking/RTC 是可选能力；
- pi05 base 是需要对具体 robot/task fine-tune 的 base model，不是任意机械臂上开箱即用的万能 controller。

### 7.4 pi0.5 在任意 Nuka 机械臂上的现实边界

这是 VLA 方案中最重要的风险：

- `pi05_droid` 在 DROID/Panda 类数据上训练；
- `pi05_libero_base` 在 LIBERO 的 7 action 任务上训练；
- 一个自建的、关节顺序和相机视角不同的 Nuka 机械臂，即使输入 shape 对了，也不代表预训练模型会完成有效抓取；
- action 的关节顺序、单位、归一化、速度/位置语义、gripper 语义必须和训练数据对齐；
- “模型真实运行”与“模型在自建 Nuka task 上成功”是两个不同验收层级。

建议把 VLA demo 分成两道门：

1. **Inference smoke gate**：真实加载 pi05 checkpoint，Nuka camera 产生真实图像，state 进入 server，server 返回真实 action chunk，Nuka 真实执行并记录 latency/action/state。即使 task 尚未成功，这一门也能证明 VLA 模型真实运行。
2. **Task success gate**：使用 Nuka 自己的演示数据构成 LeRobot dataset，针对 Nuka 机械臂和相机布局微调 pi05，再验收“抓红色方块放入盒子”等明确任务。当前 RTX 5080 16 GB 不足以承诺这一步的本地微调，需要外部 GPU 或更激进的显存优化实验。

## 8. 推荐的 VLA 机械臂场景

## 8.1 不使用 H1

VLA 场景不使用 H1，也不依赖 H1 grasp choreography。

## 8.2 第一版机械臂选择

推荐两种实现层级：

### 选项 A：Panda/Franka 兼容模型

参考资产：<https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda>

该仓库提供 Panda MJCF、no-hand 版本、场景和 mesh。Panda 是 DROID/LIBERO 生态中更容易对齐的 7 DoF 机械臂。

风险：

- 当前仓库没有确认已内置 Panda 资产；
- 现有 MJCF importer 支持 mesh/STL/OBJ、joint、actuator 和 collision，但必须实测完整 Panda mesh cook；
- Panda MJCF 的 visual/collision mesh 数量较多，可能需要 mesh 下载脚本和路径重写；
- 需要补一个夹爪/末端状态，才能构造 DROID 风格 8 维输入。

### 选项 B：Nuka 原生轻量 7 DoF arm

用 MJCF/NKS primitive geometry author 一个 7 revolute DoF 的桌面机械臂：

- link 使用 capsule/box/cylinder；
- end effector 使用简单夹爪；
- 场景包括 table、red cube、target bin；
- camera 包括 external/base camera + wrist camera；
- action contract 仍设计为 7 joint velocity + 1 gripper。

优点：

- 不依赖大量外部 mesh；
- 便于快速验证 Nuka importer、velocity control、camera mount 和 contact；
- 适合第一版 inference smoke 和数据采集。

缺点：

- 与 pi05_droid/libero 的真实训练 robot embodiment 不完全一致；
- 预训练模型零样本 task success 预期低；
- 后续需要 Nuka 自有 demonstration fine-tuning。

推荐顺序：先用轻量 Nuka 原生 arm 过引擎/VLA 接口门，再用 Panda 资产过 DROID/Panda 语义对齐门。

## 8.3 Nuka VLA 闭环

建议第一版控制 loop：

```text
Nuka physics: 200/240 Hz
low-level control: velocity target or PD target
camera render: 10-15 Hz
VLA replan: every K action steps
open-loop chunk: 5-10 actions
```

伪代码：

```python
world = nuka.World.create_from_scene(
    device,
    arm_scene,
    env_count=1,
    dt=1.0 / 240.0,
    control_mode=nuka.CONTROL_MODE_VELOCITY,
)

world.attach_camera_sensor(... external camera ...)
world.attach_camera_sensor(... wrist camera ...)

for step in range(max_steps):
    if step % camera_period == 0:
        world.render_sensors()
        ext = to_uint8(world.get_sensor_view(nuka.SensorChannel.COLOR)[0, 0])
        wrist = to_uint8(world.get_sensor_view(nuka.SensorChannel.COLOR)[0, 1])
        q = read_joint_position(world)
        gripper = read_gripper_state(world)
        chunk = client.infer({
            "observation/exterior_image_1_left": ext,
            "observation/wrist_image_left": wrist,
            "observation/joint_position": q,
            "observation/gripper_position": gripper,
            "prompt": instruction,
        })["actions"]

    action = chunk[action_index]
    write_velocity_target(world, action[:7])
    write_gripper_target(world, action[7])
    world.step_n(decimation)
```

实际代码必须额外处理：

- cooked DOF name 到 policy order 的映射；
- action unit/scale；
- velocity target 的限幅；
- gripper position 的二值化或限位；
- stale action、server timeout、网络断开；
- action jump check；
- workspace boundary；
- collision/fall/invalid state emergency stop；
- action chunk 的执行与重规划；
- camera RGB channel、coordinate frame、image resize/pad；
- state normalization 和训练 checkpoint metadata。

## 9. 分阶段可行方案

## Phase 0：引擎和资产 gates

### 0A. G1 gate

目标：不训练，证明 G1 在 Nuka 中真实可运行。

验收：

- G1 asset 下载和 license/provenance；
- URDF/MJCF -> SceneIR/NKS cook；
- floating base 正确；
- `base_link_count`、`action_dim`、DOF names 与期望一致；
- 35 generalized DoF 主 path 通过；
- PD target 逐关节写入；
- foot collision/ground contact；
- `reset_envs`、finite、state hash；
- 1 env / 16 env / 128 env smoke；
- camera/offline render 至少输出一帧。

### 0B. Arm gate

目标：不加载模型，证明机械臂和视觉闭环输入能真实产生。

验收：

- 7 DoF + gripper arm scene cook；
- velocity target 或 PD target 实际改变关节；
- cube/table/bin 碰撞；
- external camera + wrist camera；
- camera COLOR output 非空、finite、shape 正确；
- float device AOV -> uint8 input pipeline；
- save image and state sample。

## Phase 1：G1 reference playback

目标：不训练 policy，直接将公开 G1 NPZ reference 写入 Nuka 的 PD target，验证：

- q/ref joint order；
- root pose/quaternion；
- body anchor；
- contact and ground calibration；
- Nuka physics 是否能追踪参考动作；
- 视频输出是否能看到机器人跳舞。

这一步可以快速区分“资产/坐标/控制不对”和“RL 尚未收敛”。

## Phase 2：G1 imitation training

目标：使用 Nuka vectorized task + PPO，训练单段 dance 到稳定。

初始建议：

- 160-dim G1 Moves-style observation；
- 29 joint position target actions；
- 50 Hz policy；
- 512 env 起步；
- physics 200/240 Hz；
- headless training；
- 先不加视觉；
- 先不做 AMP discriminator，先做明确 reference tracking reward；
- 收敛后再增加 domain randomization 和多片段 curriculum。

可选后续：

- AMP discriminator；
- teacher/student；
- recurrent history policy；
- motion phase randomization；
- sim2sim with MuJoCo；
- ONNX/TorchScript deployment。

## Phase 3：pi05 inference smoke

目标：真实运行 pi0.5，不要求第一轮就完成自定义 task success。

步骤：

1. 在 Ubuntu/WSL2 或另一台 Linux GPU 机安装 openpi/LeRobot；
2. 启动 `pi05_droid` 或 `pi05_libero` policy server；
3. Nuka arm client attach two cameras；
4. 发送真实 Nuka image/state/prompt；
5. 收到真实 action chunk；
6. 写入 Nuka velocity target/gripper；
7. 记录 input image、action chunk、policy latency、physics state 和视频。

验收：

- 模型权重不是 LFS pointer 或空文件；
- server 真实完成 inference；
- action shape 和 finite；
- Nuka 真实执行动作；
- server disconnect/timeout 时安全停机；
- 完整 rollout 可复现或至少能保存日志。

## Phase 4：Nuka VLA task success

目标：明确 task，例如：

```text
pick up the red cube and place it in the blue bin
```

步骤：

1. Nuka 中实现随机 cube/bin 初始位置；
2. 用 keyboard/script/teleop 采集 demonstration；
3. 转换为 LeRobot dataset；
4. 写入 images/state/action/task metadata；
5. 生成 stats，至少支持 pi05 normalization；
6. 使用外部 >=24 GB/80 GB GPU 尝试微调；
7. server 部署 fine-tuned checkpoint；
8. Nuka 中评估多 seed、多位置、多灯光和少量 domain randomization。

当前机器不承诺能独立完成本阶段的 fine-tuning。

## 10. 需要重点验证或实现的代码面

### G1

- `src/import/urdf_importer.cpp` / `mjcf_importer.cpp` 的 G1 解析；
- floating root authoring；
- `src/runtime/articulation` 35 DoF 主 path；
- generalized contact and foot support；
- Python generic articulation task，不使用 Go2 常量；
- reference motion loader/converter；
- reward/termination/metrics；
- G1 render map 和 camera placement。

### VLA arm

- Panda 或轻量 arm scene；
- 7 joint + gripper DOF naming；
- velocity target public Python usage；
- attached camera API；
- COLOR AOV reshape and uint8 conversion；
- websocket openpi client adapter；
- action chunk queue/replan；
- safety clamps and timeout；
- rollout recorder。

### 通用 view/observe

第一版不需要先改 Nuka core。优先复用：

- `World.attach_camera_sensor`；
- `World.render_sensors`；
- `World.get_sensor_view`；
- `World.render_beauty`；
- 现有 C++ CUDA RT video demo。

如果训练中需要实时多环境可视化，再实现 Spec 04 方向：

```text
PhysicsWorker -> snapshot ring -> Vulkan consumer
external trainer -> IPC/shared-memory snapshot -> observe editor
```

不要让 VLA camera observation 直接依赖 editor UI 线程，也不要在核心 solver 中加入 openpi/LeRobot 依赖。

## 11. 主要风险清单

### 风险 A：G1 资产不是当前 Nuka 可直接运行的 asset

官方 G1 URDF 的 mesh 文件很多，root floating joint 在部分 URDF 中是注释状态；需要路径重写和 floating-base authoring。必须先过 G1 gate。

### 风险 B：G1 35 DoF 接触路径存在隐藏限制

主 path 当前目标上限是 64，但 computed torque/OSC/diffsim 仍保留 18 上限。第一版使用 PD position，仍需通过真实 G1 contact rollout 验证主 path。

### 风险 C：公开 dance policy 与 Nuka physics 存在 sim2sim gap

G1 Moves/BeyondMimic 的 policy 是在 MuJoCo/Isaac Lab 训练或导出的。即便 Nuka 正确导入 G1，直接加载外部 policy 也可能因接触、PD、坐标、关节顺序和 delay 不同而不稳定。应先用 reference playback，再训练 Nuka-native policy。

### 风险 D：pi05 预训练模型不能保证任意机械臂 zero-shot 成功

shape 对齐不等于 embodiment 对齐。DROID/LIBERO 的 joint order、normalization、action semantics 和相机分布都需要匹配。第一阶段的成功定义必须是“真实推理闭环运行”，不是直接宣称自定义 Nuka task 成功。

### 风险 E：16 GB 显存同时承载模型和 Nuka

单独 pi05 inference 有官方 `>8 GB` 依据，但 Nuka sensor RT、Torch preprocessing、模型临时 buffer 和 Windows desktop 占用会减少余量。需要实测显存峰值和 inference latency，必要时拆为 server/client 进程。

### 风险 F：Windows/openpi 支持边界

openpi 官方只测试 Ubuntu 22.04，当前 Nuka 工作区是 Windows。推荐 WSL2 Ubuntu 或单独 Linux server，不要把 openpi 依赖硬塞进当前 Windows Nuka Python 环境。

### 风险 G：数据和权重许可

- G1 Moves README 标注 CC-BY-4.0；使用时保留 attribution；
- Unitree robot description 和 mesh 需要保留各自 license/provenance；
- 不提交大尺寸动作数据、模型权重、Isaac Sim、SDK build products 或机器凭据；
- 只提交下载/转换脚本、manifest、checksum、schema 和小型 smoke fixture。

## 12. 待继续确认的问题

这些问题不阻碍本文记录，但开始实现前应逐项确定：

1. 舞蹈机器人是否确定使用 Unitree G1 29 DoF，还是希望使用 Go2/其他宇树型号？当前推荐 G1。
2. 是否已经有 Unitree G1 的本地资产、mesh 或实际机器人；如果没有，是否允许从 Unitree GitHub 下载官方 asset？
3. VLA 的机械臂第一版是优先 Panda/Franka 兼容，还是优先轻量 Nuka 原生 7 DoF arm？当前推荐先轻量 arm gate，再 Panda 对齐。
4. VLA 第一阶段的目标是“真实运行 pi05 inference”还是必须在 Nuka 中完成一个成功的 pick-and-place？两者应分成两个 gate。
5. 是否可以使用 WSL2 Ubuntu 22.04/24.04，或者是否有另一台 Linux NVIDIA 机器承担 openpi server 和微调？
6. pi05 首选 checkpoint 是 `pi05_droid` 还是 `pi05_libero_base`？若使用 DROID action contract，推荐 `pi05_droid`；若使用 7 action position/末端语义，则需要严格做 Libero adapter。
7. 是否接受第一阶段 VLA 只做仿真，不连接真实机械臂；真实硬件控制需要另行加入硬件 watchdog、限位、急停和通信层。

## 13. 外部资料索引

### pi0.5 / openpi

- Physical Intelligence openpi：<https://github.com/Physical-Intelligence/openpi>
- openpi README：<https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/README.md>
- openpi pyproject：<https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/pyproject.toml>
- openpi remote inference：<https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/docs/remote_inference.md>
- openpi DROID README：<https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/examples/droid/README.md>
- openpi DROID example：<https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/examples/droid/main.py>
- openpi DROID transform：<https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/policies/droid_policy.py>
- openpi pi0 config：<https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/models/pi0_config.py>
- openpi pi0 model：<https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/models/pi0.py>
- openpi train config：<https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/training/config.py>
- Hugging Face LeRobot pi05 docs：<https://huggingface.co/docs/lerobot/en/pi05>
- LeRobot pi05 base：<https://huggingface.co/lerobot/pi05_base>
- LeRobot pi05 Libero base：<https://huggingface.co/lerobot/pi05_libero_base>
- Hugging Face pi05 custom training guide：<https://huggingface.co/blog/Tonic/training-and-inference-with-pi05>
- LeRobot pi05 model implementation：<https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/policies/pi05/modeling_pi05.py>
- LeRobot pi05 configuration：<https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/policies/pi05/configuration_pi05.py>
- Physical Intelligence pi0.5 blog：<https://www.pi.website/blog/pi05>

### 模仿学习 / G1

- G1 Dance Pipeline：<https://github.com/guoshuhao066-code/g1-dance-pipeline>
- BeyondMimic / whole_body_tracking：<https://github.com/HybridRobotics/whole_body_tracking>
- Unitree RL Mjlab：<https://github.com/unitreerobotics/unitree_rl_mjlab>
- Unitree G1 description：<https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_description>
- Unitree G1 29 DoF mode 15 URDF：<https://raw.githubusercontent.com/unitreerobotics/unitree_ros/master/robots/g1_description/g1_29dof_mode_15.urdf>
- G1 Moves code：<https://github.com/experientialtech/g1-moves>
- G1 Moves dataset：<https://huggingface.co/datasets/exptech/g1-moves>
- DeepMimic：<https://github.com/xbpeng/DeepMimic>
- DeepMimic paper：<https://xbpeng.github.io/projects/DeepMimic/DeepMimic_2018.pdf>
- Isaac Lab humanoid imitation reference：<https://isaac-sim.github.io/IsaacLab/main/source/overview/imitation-learning/skillgen.html>

### every-embodied

- 主仓库：<https://github.com/datawhalechina/every-embodied>
- 春晚舞蹈机器人复刻：<https://github.com/datawhalechina/every-embodied/blob/main/07-%E6%9C%BA%E5%99%A8%E4%BA%BA%E6%93%8D%E4%BD%9C%E3%80%81%E8%BF%90%E5%8A%A8%E6%8E%A7%E5%88%B6/Locomotion/01%E6%98%A5%E6%99%9A%E8%88%9E%E8%B9%88%E6%9C%BA%E5%99%A8%E4%BA%BA%E5%A4%8D%E5%88%BB.md>
- AGILE Loco-Manipulation：<https://github.com/datawhalechina/every-embodied/blob/main/05-%E5%85%B7%E8%BA%AB%E5%9C%BA%E6%99%AF%E7%9A%84%E6%B7%B1%E5%BA%A6%E5%92%8C%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0/03AGILE%E4%BA%BA%E5%BD%A2%E6%9C%BA%E5%99%A8%E4%BA%BALoco-Manipulation%E5%A4%8D%E7%8E%B0/README.md>
- every-embodied Pi0/Pi05 MuJoCo material：通过仓库中 `06-策略抓取或抓取VLA/大模型控制、VLA、VLM/04mujoco复现ACT、Pi0、SmolVLA` 目录检索对应 notebooks/README。

### 机械臂资产

- Google DeepMind MuJoCo Menagerie Panda：<https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda>
- Panda no-hand MJCF：<https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/franka_emika_panda/panda_nohand.xml>

## 14. 当前建议结论

在暂停前保留以下决策作为下一轮起点：

1. 模仿学习先做 G1 29 DoF，不做 H1；先 reference playback，再 Nuka-native PPO tracking。
2. 舞蹈初始 observation 采用 G1 Moves 160-dim contract，另写字段测试；不要混用 BeyondMimic 154-dim contract。
3. 舞蹈首版使用 PD position target、50 Hz policy、200/240 Hz physics，headless batch training。
4. VLA 首版不把 pi05 模型编译进 Nuka；采用 openpi websocket server/client 结构。
5. VLA 首选 DROID-style 8-dim state/action contract：7 joint velocity + 1 gripper position；Nuka 侧用 velocity target 控制模式。
6. VLA 先用轻量 Nuka 原生 7 DoF arm 过 camera/inference/actuation gate，再决定是否引入 Panda mesh 做 embodiment 对齐。
7. 当前 RTX 5080 16 GB 可尝试单独 pi05 inference，但不承诺本地微调；模型 server 与 Nuka 最好分进程。
8. Nuka 已有 camera AOV/view 输出，不需要先做完整 Isaac Lab 风格 viewer 才能开始 VLA；实时训练观察和跨进程多环境显示属于后续 snapshot/IPC 工作。
9. 下一轮真正修改代码前，第一优先级是两个资产 smoke gate：G1 29 DoF cook/step/contact 和 7 DoF arm/camera/velocity target。

## 15. 需求修订：最终采用同一 WSL2 进程，不使用 Docker/WebSocket

用户后续确认了最终部署决策：

- 不使用 Docker；
- 不使用 Windows 原生进程 + WSL2 policy server 的 websocket 分进程方案；
- 由 Windows 直接启动/进入 WSL2 Ubuntu，在 WSL2 内编译 Nuka；
- 在同一个 WSL2 Python 环境和同一个进程中加载 Nuka、PyTorch/LeRobot pi0.5；
- Nuka 与 pi0.5 共用同一 CUDA device，VLA adapter 直接消费 Nuka 的 device tensor；
- 不使用 `openpi_client.WebsocketClientPolicy` 作为 VLA 主路径，因为它会产生序列化/传输边界，不满足本地同设备直连目标。

### 15.1 WSL2 与 Docker 的边界

Docker Desktop 官方 GPU 支持只在 WSL2 backend 上可用；这说明“Windows 直接使用 Docker 访问 GPU”技术上有路径，但本项目最终不采用 Docker。当前机器检测到的 WSL 只有 `docker-desktop`，没有可用 Ubuntu 发行版，因此实现前需要安装 Ubuntu 22.04/24.04，并在该发行版中验证：

```bash
nvidia-smi
nvcc --version
python3.11 --version
```

WSL2 仍然借助 Windows NVIDIA driver 的 GPU-PV/CUDA 通道，但从项目依赖管理角度属于原生 WSL Linux 环境，不是 Docker 部署。

### 15.2 同进程 direct-device 的准确含义

用户进一步确认：图像路径可以先按实测决定，不预先强制 lower-level adapter，也不把本地 fine-tuning 作为第一版前置条件。

当前 openpi 官方 `Policy.infer` 和 DROID transform 面向 NumPy/通用客户端，内部存在 host conversion；因此：

- 采用官方高层 API 时，路径标记为 `HOST_COMPAT`，可以用于首先验证真实模型和任务，但不能称为全链路零拷贝；
- 如果 host copy 成为瓶颈，再实现 `DEVICE_TORCH` lower-level adapter；
- 如果 camera backend 增加 `uint8` COLOR 输出能简化 pi05 兼容或减少转换，可以新增 `uint8` 路径；必须保留 dtype/range/layout/deterministic golden test；
- state/action 仍优先通过 `world.buffer_view()` + `torch.from_dlpack()` 在同一 CUDA device 连接；
- COLOR float32 AOV 到模型输入域的转换、resize/pad 和 layout 可以先使用 device Torch，也可以根据 benchmark 选择 engine-side `uint8`；
- prompt/tokenization 的 host 阶段必须单独计时；不允许隐藏在 physics tick 内；
- 只有 instrumentation 证明后，才能报告“没有 steady-state image host copy”；
- “零拷贝”仅表示 Nuka/PyTorch 之间没有不必要的 CPU/网络搬运，不表示图像格式转换、模型 activation 或 flow sampling 没有 device 临时 buffer。

### 15.3 对原方案的最终修订

- 原第 2 节中的 `Windows Nuka client -> Linux/WSL websocket -> openpi server` 只保留为远程 fallback，不再是本需求主路径；
- 原 Phase 3 的“启动 policy server + Nuka client”改为同进程加载 pi05 policy；
- 原 VLA 机械臂选项改为用户确认的直接 Panda/Franka 资产；
- 原 VLA 验收改为必须通过 pick-and-place task success，不以 inference smoke 单独结项；
- 本地 fine-tuning 从第一版硬前置降为后续选项；优先尝试已发布的第三方 Panda/Franka pi05 checkpoint；
- `ases200q2/Isaac_panda_pick_cube_pi05_20251126_100645` 成为第一候选，但其 model card 和训练步数不足以证明质量，必须本地真实验证；
- `openpi-client`、websocket、msgpack 不进入 Nuka VLA 主路径；
- WSL2 Ubuntu 是开发/运行环境，Docker 不在交付范围。

完整实施合同见：`docs/roadmap/2026-08-26-spec-05-g1-dance-pi05-vla-zh.md`。

## 16. Hugging Face pi05 微调模型调研结果

本次使用 Hugging Face Models API、模型文件列表、README、config、processor config 和 dataset README 交叉检查候选。只看模型名不作为结论；至少要求能看到权重文件和输入/输出配置。

### 16.1 第一候选：Panda pick cube

模型：<https://huggingface.co/ases200q2/Isaac_panda_pick_cube_pi05_20251126_100645>

已确认：

- `model.safetensors` 约 7.47 GB；
- `library_name=lerobot`，`type=pi05`，基于 `lerobot/pi05_base`；
- `observation.state` 为 8 维；
- 单相机 `observation.images.camera_1`，数据形状 480x640x3；
- action 为 8 维；dataset 的 action 名称是 7 个 `panda_joint*.pos` + `gripper.pos`；
- dataset <https://huggingface.co/datasets/ases200q2/Isaac_Panda_PickCube_SpaceMouse_EE_50eps> README 显示 51 个 episodes，Panda pick-cube 任务；
- `chunk_size=50`、`n_action_steps=50`；
- STATE/ACTION 使用 quantile normalization；
- processor 包含 state tokenizer、tokenizer、device processor，适合检查 LeRobot PyTorch 直接加载。

未确认：

- model card 是通用模板，没有独立 success rate；
- `train_config.json` 显示 `steps=200`，与 30,000 step schedule 不一致，可能是短训练/上传配置问题；
- 没有在 Nuka 中验证 Panda joint zero、camera pose、action scale 和成功率。

结论：C1 是当前最适合先下载和真实测试的候选，但不能称为已验证的好模型。

### 16.2 第二候选：Franka pick-place

模型：<https://huggingface.co/DorianAtSchool/pi05base-finetune-franka-pickplace>

已确认：

- LeRobot `model.safetensors`，约 7.47 GB；
- 8 维 state；
- 三路 image：`wrist`、`side`、`top`；
- 训练配置显示 10,000 steps。

风险：

- output action 配置为 7 维，没有清晰的独立 gripper action；
- model card 没有可靠成功率和 task details；
- 三路 camera 与 Nuka 当前最小外部相机方案不一致。

结论：作为 C2 审计候选，不作为默认 checkpoint。

### 16.3 第三候选：Franka insert marker

模型：<https://huggingface.co/ankile/openpi-pi05-franka-insert-marker-v2-ft>

已确认：

- Franka/DROID 形态；
- 5,000 training steps；
- 8D joint velocity + gripper；
- action horizon 16；
- 有明确 observation contract 和训练数据；
- checkpoint 是 openpi Orbax/JAX 格式，约 6.7 GB 参数，不是 LeRobot PyTorch safetensors；
- 任务是 marker insertion，不是 pick cube。

结论：适合验证 DROID velocity contract 或作为转换参考，不作为第一版 pick-place 默认模型。

### 16.4 其他候选

- `bartek-niedzielski/pi05-panda-pos-v1` / `pi05-panda-vel-40-v1`：Panda position/velocity 名称有吸引力，但主要是 Orbax/JAX checkpoint、约 44.6/44.7 GB tree、无 model card；先审计，不直接加载；
- `RLinf/RLinf-Pi05-RLCo-PandaPutOnPlateInScene25DigitalTwin-V1-SFT`：有 7.47 GB `model.safetensors`，任务名与 Panda plate 相关，但根目录缺少可直接判断的 LeRobot config/README，先不纳入主路径；
- `PolinAvA/pi05_franka_base`：有 PyTorch safetensors，但无 model card，配置为 32 action/50 horizon，缺少当前 pick-place contract；
- `phospho-app` 下的 pi0.5：大多是 Piper、SO100 或其他用户任务，机器人/action semantics 与当前 Panda Nuka 不一致。

### 16.5 执行顺序

```text
C1 ases200q2 Panda pick cube
    -> download + checksum + processor audit
    -> Nuka Panda static contract
    -> real pi05 rollout
    -> 10+ episode pick-place evaluation

C1 fails contract/task
    -> audit C2 DorianAtSchool
    -> audit C3 ankile only if velocity/Franka path is useful
    -> only then consider local fine-tuning or larger-GPU adaptation
```

本地 fine-tuning 目前只是备用方案。优先级是先利用已发布 checkpoint 完成 task validation，避免在没有证明 asset/action/preprocessing 正确之前消耗训练资源。
