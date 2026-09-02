# Spec 05：Unitree G1 舞蹈模仿学习与 Panda pi0.5 VLA

状态：需求已确认，进入实现前统筹阶段  
日期：2026-08-26  
关联调研：[Nuka 模仿学习与 pi0.5 VLA 调研记录](../research/2026-08-26-imitation-vla-research-zh.md)  
优先级：G1 模仿学习与 Panda VLA 同等重要

> 本 spec 是实现合同，不是“可能的方向列表”。凡是写成“必须”的条目，进入实现和验收时都需要有代码、测试或运行日志证据。H1 不属于本 spec 的任何路径；现有 H1 文件保留，不删除。

## 1. 已确认的用户决策

1. 舞蹈机器人固定为 Unitree G1 EDU mode 15，29 个可动关节。
2. VLA 机械臂第一版直接采用 Franka/Panda 兼容资产，不先做轻量替代臂。
3. VLA 第一版必须以真实 pick-and-place 成功作为最终任务验收；“模型成功返回 action”只能作为中间 gate，不能单独结项。
4. 运行环境采用 WSL2 Ubuntu 原生 Linux，不使用 Docker。
5. Nuka 和 pi0.5/LeRobot 在同一 WSL2 Python 环境中编译/加载，并优先在同一进程、同一 CUDA device 内直接连接。
6. VLA 主路径不使用 websocket、`openpi-client`、msgpack 或跨进程 policy server。
7. Nuka 的 physics、camera sensor、PyTorch policy 和动作执行属于同一个本地 rollout 进程；远程 server 只作为后续 fallback，不进入本版主合同。
8. G1 和 VLA 是两个独立 demo、独立 task、独立场景、独立运行入口；不能把 Go2/H1 task 改名伪装成这两个 demo。

## 2. 目标与非目标

### 2.1 目标

### G1

- 将公开的 G1 29DoF retargeted reference motion 导入 Nuka；
- 在 Nuka 中完成 G1 asset cook、floating-base dynamics、地面接触和 PD playback；
- 提供 29 关节 reference motion tracking task；
- 使用 Nuka 的批量环境和 PPO/模仿学习训练，至少使一段短舞蹈稳定完成；
- 保存 checkpoint、reference、指标和视频；
- 提供可重复的 headless train/eval/play 入口。

### Panda pi0.5 VLA

- 将 Panda/Franka 7DoF arm、夹爪、桌面、方块、目标盒和两个视觉相机导入 Nuka；
- 真实使用 Nuka camera COLOR AOV 和关节/夹爪状态构造 pi0.5 observation；如实测更合适，可新增 Nuka 原生 `uint8` COLOR 输出或等价图像适配路径；
- 在同一 WSL2 进程内真实加载 pi0.5 PyTorch checkpoint，执行真实 flow-matching inference；
- 不经过 websocket/网络边界，把 state/action 和能保持 device 的 image tensor 直接交给模型；
- 把模型返回的 action chunk 转换为 Nuka velocity/夹爪控制；
- 通过已验证的第三方 Panda/Franka pi05 fine-tuned checkpoint 完成任务；当前优先验证 `ases200q2/Isaac_panda_pick_cube_pi05_20251126_100645`；
- 本地微调作为后续选项，第一版可以暂缓；只有没有可用第三方 checkpoint，或 action/camera/asset contract 无法对齐时，才重新启动数据采集和 fine-tuning；
- 通过 scripted/teleop demonstrations 建立可复现实验和后续数据基础，让模型完成：

```text
pick up the red cube and place it in the blue bin
```

- 提供成功判定、失败安全停机、rollout 日志、图片和视频。

### 2.2 非目标

- 不在本 spec 中接入 H1；
- 不把 Nuka core 绑定到 openpi、LeRobot、PyTorch、JAX、transformers 或 sentencepiece；
- 不实现真实 Panda 硬件控制、CAN/ROS、急停硬件、机械限位或安全认证；第一版只做仿真；
- 不把 editor/UI/完整 snapshot IPC 重构作为 G1/VLA 的前置条件；
- 不提交大模型权重、完整 G1 motion 数据、Panda mesh build product 或机器凭据；
- 不承诺 RTX 5080 16 GB 可以完成官方 full/LoRA pi05 微调；必须通过实际峰值测试决定本地训练是否可行；
- 不宣称公开的 pi05 base/DROID checkpoint 在自建 Nuka Panda 场景中无需适配即可完成任务。

## 3. 最终系统架构

## 3.1 WSL2 原生部署

Windows 只负责安装/启动 WSL2 和进入 Ubuntu；源码、编译器、Nuka、PyTorch、LeRobot/openpi 依赖和运行日志均放在 WSL2 Linux 文件系统中。不要把高频源码、虚拟环境和 checkpoint 放在 `/mnt/c` 上作为性能基线。

目标拓扑：

```text
Windows host
  NVIDIA Windows driver + WSL2 GPU passthrough
             |
             v
WSL2 Ubuntu 22.04/24.04
  g++ / nvcc / CMake / Vulkan loader
  Python 3.11 virtual environment
             |
             +-- nuka Python extension (CUDA)
             +-- PyTorch CUDA
             +-- LeRobot/openpi pi05 PyTorch implementation
             |
             `-- one process: Nuka world + camera + policy + control loop
```

第一阶段环境 gate 必须执行：

```bash
nvidia-smi
nvcc --version
python3.11 --version
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

然后分别验证：

```bash
# Nuka build
cmake -S . -B build/wsl-release \
  -DNK_REQUIRE_CUDA=ON \
  -DNK_PHYSICS_BACKEND=CUDA \
  -DNK_BUILD_TESTS=ON
cmake --build build/wsl-release -j

# Nuka/PyTorch coexistence
python -c "import nuka, torch; print(nuka.__engine_version__, torch.cuda.get_device_name(0))"
```

当前机器只有 `docker-desktop` WSL 条目，没有 Ubuntu。安装 WSL Ubuntu 是实现前置，但 Docker 不属于交付路径。

## 3.2 同进程 direct-device VLA

VLA 主路径如下：

```text
Nuka World
  camera COLOR: CUDA float32 (E,S,H,W,3)
  JOINT_POSITION: CUDA buffer
  gripper state: CUDA buffer or mapped control state
       |
       | torch.from_dlpack / device tensor views
       v
Pi05DevicePolicy
  device image conversion + resize/pad + layout
  device state normalization + model input preparation
  pi05 PyTorch model.sample_actions()
       |
       | CUDA action chunk
       v
Nuka VELOCITY_TARGET / gripper drive buffers
       |
       v
world.step_n()
```

### 推理路径选择

VLA 主路径固定为 WSL2 同进程，但具体 preprocessing/API 路径按正确性、延迟和显存实测决定：

- 首先允许使用官方 openpi/LeRobot 的高层 PyTorch inference/data processor，快速验证真实 checkpoint、输入 contract 和 Panda 任务行为；
- 当前 `openpi.Policy.infer()` 不得被标记为 zero-copy 路径，因为其通用实现会把输入转为 NumPy 后再创建 PyTorch tensor；如果采用它，日志和文档必须标记 `HOST_COPY`，不能冒充 direct-device；
- 如果 host copy 成为瓶颈，或需要保持 Nuka/PyTorch 的 device alias，再实现可选的 device-aware lower-level adapter；它不是未经实测就必须启用的唯一方案；
- 图像输入允许两条实现：沿用当前 COLOR float32 AOV 后在 Torch/CUDA 上转换，或为 camera backend 增加 `uint8` COLOR 输出/适配路径；是否新增 `uint8` 路径由基准测试决定；
- state/action 的 DLPack 同 device 直连仍是优先目标；官方 API 的 NumPy 转换只影响采用该 API 时的实际零拷贝等级；
- 不调用 `openpi_client.WebsocketClientPolicy` 作为本地 VLA 主路径；远程 websocket 只保留为后续 fallback；
- Nuka camera 每个 query 不应无条件转为 CPU NumPy。若高层 API 需要 CPU 图像，则必须明确记录为兼容路径，并保留 device image 路径的测试入口；
- prompt/checkpoint 加载和必要的 tokenizer CPU 工作可以发生在初始化或指令改变时，但必须单独计时，不得隐藏在 physics tick 中；
- adapter 必须输出本次 rollout 的 `preprocess_mode`：`HOST_COMPAT`、`DEVICE_TORCH` 或 `ENGINE_UINT8`，以及实际 host copy 统计；
- 本 spec 的“同进程 direct-device”是架构目标；“全链路零 CPU copy”只有在 instrumentation 证明后才能宣称。图像 dtype cast、resize/layout、模型 activation 和 flow sampling 的 device 临时 buffer 不算违反零拷贝。

## 4. 外部资产和许可策略

### 4.1 G1

默认来源：

- Unitree G1 description：<https://github.com/unitreerobotics/unitree_ros/tree/master/robots/g1_description>
- 目标 URDF：`g1_29dof_mode_15.urdf`
- motion/reference：G1 Moves dataset 或其他有明确许可的 G1 retargeted motion

资产处理要求：

- 实际下载通过脚本完成，仓库只保存 URL、commit/revision、SHA-256、许可证和转换配置；
- mesh 路径相对于 scene/asset root 解析，不依赖开发机绝对路径；
- 保存 `THIRD_PARTY_NOTICES`/provenance；
- 不把公开数据集所有 NPZ/PT/ONNX 二进制提交进源码；
- smoke test 使用小型 fixture 或在 asset 不存在时清晰 skip/fail，并输出缺少的文件。

### 4.2 Panda

默认来源：

- MuJoCo Menagerie Franka Panda：<https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda>
- 参考文件：`panda.xml`、`panda_nohand.xml`
- Menagerie Panda 目录标注 Apache-2.0，实际复制/修改时保留许可证和来源。

Nuka 不直接承诺完整接受 MuJoCo 的所有语义。必须生成一个明确版本化的 Nuka-compatible Panda asset：

- 保留 7 个 arm hinge joints、两个可碰撞 finger pads 和足够的 visual mesh；
- 解析或转换掉仅服务 MuJoCo actuator/tendon/equality 的控制语义；
- 若保留两个 finger slide joints，Python adapter 必须将一个 gripper policy action 映射到两个 finger slots，并在 scene manifest 中说明；
- 不以 MJCF `<actuator>`、`<tendon>`、`<equality>` 在 Nuka 中“碰巧被忽略”作为正确实现；转换脚本必须明确处理或拒绝这些元素；
- 7 个 arm joint 的名字、顺序、单位和限位在 manifest 中固定；
- gripper 的开闭方向、范围 `[0,1]`、finger joint 的映射和碰撞几何通过测试锁定。

## 5. G1 舞蹈 demo 需求

## 5.1 场景和 importer gate

目标文件建议：

```text
examples/scenes/g1_29dof_mode15.nks      # 或稳定的可导入 MJCF/URDF入口
examples/assets/g1_mode15/                # 下载/转换后的本地 asset root，不强制提交大文件
examples/motions/g1/manifest.json        # reference metadata, URL, hash, license
```

实际路径可以按仓库现有资产惯例调整，但必须存在稳定 manifest，不在 Python 中散落硬编码绝对路径。

G1 gate 必须检查：

- floating base 是 6 DoF；
- 29 个目标可动 joint 名字全部存在且无重复；
- `base_link_count`、可动 slot、joint type 和 DOF index 可被程序读取；
- 总 generalized DOF 通常为 `6 + 29 = 35`，实际以 cooked topology 输出为准；
- collision mesh/foot contact 已生成；
- visual mesh 至少可被 sensor/offline RT 渲染；
- q、qd、pose、contact、drive target 每步 finite；
- 1 env、16 env 和目标批量 env 都能创建/step/reset；
- `control_mode=PD_POSITION` 可以改变 29 个目标关节；
- 不调用 computed torque、OSC 或 diffsim IFT 路径；这些路径仍有 18-DoF scratch 约束；
- 若 cook 失败，错误必须指出 importer/cook/contact/DOF 具体阶段，不允许静默降级为无机器人场景。

## 5.2 参考动作格式

第一版统一使用 G1 Moves 风格 160 维 observation，不能同时兼容 154 维而不标记版本。

reference manifest 至少包含：

```json
{
  "format": "g1_moves_npz_v1",
  "fps": 50,
  "joint_count": 29,
  "joint_order": ["..."],
  "root_quaternion": "xyzw",
  "coordinate_system": "right_handed_z_up",
  "anchor_body": "...",
  "files": {
    "joint_pos": "...",
    "joint_vel": "...",
    "body_pos_w": "...",
    "body_quat_w": "...",
    "body_lin_vel_w": "...",
    "body_ang_vel_w": "..."
  }
}
```

NPZ/reference converter 必须明确：

- root quaternion 输入是 xyzw，Nuka pose 是 wxyz；
- world frame、body frame、z-up 和 forward axis；
- body anchor 选择和索引；
- 50 Hz reference 与 physics tick 的插值/采样方式；
- joint order 从 manifest 读取，不能依赖 NumPy 文件自然顺序；
- 参考数据 finite、长度一致、joint limits 可检查；
- 启动时用 G1 cooked DOF names 与 manifest 做双向 permutation 校验。

160 维 observation 合同：

```text
[  0: 29] reference joint position       29
[ 29: 58] reference joint velocity        29
[ 58: 61] motion anchor position in body   3
[ 61: 67] motion anchor orientation        6
[ 67: 70] current base angular velocity    3
[ 70: 73] current base linear velocity     3
[ 73:102] current joint position - default 29
[102:131] current joint velocity          29
[131:160] previous action                 29
```

字段 shape、dtype、frame 和顺序必须有独立 unit test。不要将 BeyondMimic 的 154 维 observation 静默塞入 160 维网络。

## 5.3 控制和 task API

第一版默认：

```text
physics dt:        1/200 or 1/240
policy frequency:  50 Hz
policy decimation: computed from dt, normally 4 at 200 Hz
action:            29 joint position targets
control mode:      PD position
```

Nuka task 必须是独立 generic articulation task，不能导入 `Go2ObsBuilder` 的 12DoF 常量。建议文件：

```text
python/nuka/tasks/g1_motion.py
python/nuka/tasks/g1_dance.py
python/examples/training/train_g1_dance_ppo.py
python/examples/demo/g1_dance_play.py
```

实际目录以仓库现有 `examples/` 与 `python/` 组织惯例为准，但入口必须能从仓库根目录稳定解析包和 asset。

task 至少提供：

- `reset()`、`step(action)`、`close()`；
- vectorized `num_envs`；
- GPU obs/action/reward/termination tensor；
- random reference start frame；
- terminal fall、nonfinite、joint-limit 和 episode timeout；
- `reset_envs` 后 reference phase/previous action 同步清零；
- metrics：pose RMS、velocity RMS、anchor error、base height/tilt、contact、torque saturation、fall rate、clip completion；
- `play` 模式不依赖训练框架，可以直接回放 reference 或 checkpoint。

## 5.4 reward 和训练

首版采用明确的 reference tracking reward，不以 AMP discriminator 作为第一个可运行门：

```text
pose       = exp(-k_pose * mean((q - q_ref)^2))
velocity   = exp(-k_vel  * mean((qd - qd_ref)^2))
anchor_pos = exp(-k_pos  * norm(anchor_pos_error)^2)
anchor_ori = exp(-k_ori  * orientation_error^2)
body_vel   = exp(-k_body * body_velocity_error^2)
stability  = upright + height + valid_support
regularize = action_rate + effort + joint_limit penalties
```

要求：

- reward weights 在 task config 中可见，不埋在 kernel/训练脚本；
- 每个 reward component 可单独记录；
- reward 在首次运行先用 reference playback 和 zero/noise policy 做数值 sanity；
- 非 finite state 直接 terminate/reset，不能进入 PPO experience；
- 首版只训练单段短 dance；收敛后再做多段 curriculum、domain randomization 和随机初始 phase；
- policy action space 固定 `[-1,1]`，内部 target scale/limit 显式记录，避免已有 Go2 的 action rescale 类错误；
- 可复现 seed、checkpoint、reference manifest 和 engine build 信息必须写入 run metadata。

训练算法可以复用仓库已有 PPO/rl_games 连接方式，但 observation/action/reward contract 必须是 G1 专属。若现有训练入口无法容纳 29 action/160 obs，新增 adapter，不修改 Go2 行为。

## 5.5 G1 验收门

### G1-A：asset/cook/step

- 目标 asset 存在时，创建 world 成功；
- 29 个 joint name、joint type、limits 和 slot map 全部匹配；
- 35 generalized DoF 或实际 topology 通过主 ABA/contact path；
- 1000 physics steps finite；
- reset 后 q/qd/base pose 恢复；
- foot contact 与 floor 有非零、有限的可解释输出；
- 至少导出一帧 camera/offline render。

### G1-B：reference playback

- 同一 reference 和 seed 两次 playback 输出 bitwise 或按指定 tolerance 一致；
- 参考动作不会因 q/quaternion/joint order 错误立即翻转；
- 运行视频中可识别机器人主体和上/下肢动作；
- 记录 tracking metrics。

### G1-C：training/eval

- 短片段达到完成率和稳定性阈值；阈值在首个 baseline 运行后写回 config，不用视频主观判断代替；
- 至少 3 次 eval rollout 无 solver NaN/Inf；
- episode fall rate、pose RMS、anchor error、torque saturation 和 clip completion 都输出；
- checkpoint reload 后 replay 行为与保存时一致；
- headless train 与单 env play 使用同一 asset/schema。

## 6. Panda pi0.5 VLA demo 需求

## 6.1 Panda scene

目标 scene 至少包含：

```text
Panda base + 7 arm joints + parallel gripper
flat table
red cube
blue bin/target receptacle
external camera
optional wrist camera mounted on end-effector/hand
```

推荐目标 manifest：

```text
examples/scenes/panda_pi05_pick_place.xml
examples/scenes/panda_pi05_pick_place.nks     # 如需 cook 后入口
examples/assets/panda/
examples/vla/panda_pi05_manifest.json
```

manifest 必须固定：

- arm joint order：`joint1 ... joint7` 或转换后的明确名称；
- engine slot 到 policy state/action index 的双向映射；
- gripper finger joint slots；
- arm joint limits、velocity limits、control scale；
- camera index、mount frame、local offset、vertical FOV、width/height；
- RGB channel/order 和坐标约定；
- table/cube/bin 的尺寸、初始 pose 范围、成功判定几何阈值。

Panda importer gate：

- mesh 路径可在 WSL 下解析；
- visual/collision geometry 都存在；
- fixed body、MJCF class/default、quat 和 mesh scale 解析正确；
- `<actuator>/<tendon>/<equality>` 要么有 Nuka 映射，要么在预处理脚本显式移除并记录；
- two-finger collision 不穿透 table/cube；
- 7 arm joints + gripper drive 可以稳定 step；
- camera attach 到 base 和 wrist 均成功。

## 6.2 Observation contract

目标模型使用 `pi05_droid` 风格输入：

```text
observation/exterior_image_1_left: uint8 HxWx3, model target 224x224
observation/wrist_image_left:       uint8 HxWx3, model target 224x224
observation/joint_position:         float32[7]
observation/gripper_position:       float32[1], range [0,1]
prompt:                             text
```

对于同进程 device adapter，内部 representation 为：

```text
camera COLOR view: (E,S,224,224,3) float32 CUDA [0,1]
external/wrist:    (1,3,224,224) CUDA, model expected domain
state:             (1,8) CUDA, padded to model dimension when required
image masks:       CUDA bool, third camera masked if model requires it
```

注意：官方 DROID transform 对 pi05 会生成 `base_0_rgb`、`left_wrist_0_rgb` 和 masked `right_wrist_0_rgb`，并将 state 组成 7 joint + 1 gripper。适配器必须对照实际 checkpoint/config 读取 image feature names、state/action dimensions、chunk size、normalization stats；不能只根据 README 写死。

## 6.3 pi05 integration adapter

建议文件：

```text
python/nuka/vla/pi05_device.py
python/nuka/vla/pi05_contract.py
python/examples/demo/panda_pi05_pick_place.py
python/tests/test_pi05_contract.py
python/tests/test_pi05_preprocess_modes.py
```

adapter 的职责：

1. 加载真实 pi05 checkpoint；
2. 校验模型类型、action dimension、chunk size、image keys、state dimension 和 normalization metadata；
3. 从 Nuka DLPack view 建立可复用 Torch alias；
4. 支持并明确标记 `HOST_COMPAT`、`DEVICE_TORCH`、`ENGINE_UINT8` 三种图像 preprocessing mode；
5. 在 device path 中完成 image layout、resize/pad、state normalization、padding、action unnormalization；
6. 完成 pi05 state tokenization/model input preparation，并记录是否发生 host tokenizer；
7. 调用所选高层或 lower-level PyTorch model path；
8. 返回 action chunk，并将其写入 Nuka control buffer；
9. 保存 preprocessing mode、host copy、timing、model version、input shapes、action range 和 fallback reason。

### API 形态

adapter API 不应把 openpi 私有字段泄漏到 demo：

```python
policy = Pi05Policy.load(
    checkpoint=...,                 # 优先选择已验证的 Panda/Franka checkpoint
    device="cuda:0",
    expected_robot="panda_nuka_v1",
    preprocess_mode="auto",        # host_compat/device_torch/engine_uint8
    compile=False,
)

policy.reset()
chunk = policy.infer(
    exterior_rgb=color[:, 0],       # CUDA HWC/float AOV or uint8 tensor
    wrist_rgb=color[:, 1],          # optional, model config decides
    joint_position=q7,              # CUDA alias when available
    gripper_position=g1,
    prompt=instruction,
)
# chunk: CUDA or host according to selected backend, validated before actuation
```

实现策略：

- 第一阶段先以能正确加载并运行候选 checkpoint 为准，允许 `HOST_COMPAT`；
- 只有基准显示 host conversion 是主要瓶颈，或用户明确要求全链路 device alias 时，才投入 lower-level adapter；
- 如果使用 LeRobot PyTorch `PI05Policy`，优先调用其公开 processor/model contract；发现 processor 产生 host boundary 时，只在 adapter 内封装，不污染 Nuka core；
- 如果使用 openpi PyTorch lower-level `PI05Pytorch.sample_actions`，必须添加与官方高层路径的 golden input/action 对齐测试；
- 不修改 upstream 源码到 Nuka core；兼容修补集中在 adapter，记录 upstream commit/version；
- 初始关闭 `torch.compile`，完成候选模型正确性/显存 gate 后，再比较 `default`、`reduce-overhead`、`max-autotune`；
- first inference 预热、CUDA graph/autotune 和模型加载不计入 steady-state policy latency，但需单独日志记录。

### State/image preprocessing 的决策门

当前 openpi 原生 pi05 路径会将归一化 state 离散为 256 bins，再通过 SentencePiece 形成 prompt token；LeRobot processor 也可能在 device processor 之前存在 host 阶段。第一版不预先假定必须重写这套逻辑：

1. 先用官方/LeRobot processor 跑通真实候选 checkpoint，并记录 host/device 边界；
2. 若 host copy 对任务吞吐和延迟没有实质影响，保留 `HOST_COMPAT`，但日志必须如实标记；
3. 若需要 device-only image/state path，再实现 `DEVICE_TORCH` 或新增 Nuka `uint8` COLOR 输出；
4. 任何重写 tokenizer/preprocessor 都必须用官方 processor 的 golden vectors 对齐 token IDs、mask、normalization 和 action。

无论选哪条路径：

- 不能把未经校验的连续 state 直接替代 pi05 的离散 state；
- state/action normalization 必须使用 checkpoint 自带 stats；
- token length/truncation、camera keys、image dtype 和 host copy 必须可观测；
- 预处理 mode 是运行配置，不是隐式 fallback；
- 图像 `uint8` 路径只有在 camera backend 已提供正确 HWC RGB、shape、range 和 deterministic 测试后，才能作为正式输入。

## 6.3.1 Hugging Face checkpoint 优先级

本轮检索到的候选按“Panda 形态 + pick/place 任务相关性 + LeRobot/PyTorch 可加载性 + 输入输出 contract 完整度”排序：

### C1：第一候选

`ases200q2/Isaac_panda_pick_cube_pi05_20251126_100645`

- Hugging Face：<https://huggingface.co/ases200q2/Isaac_panda_pick_cube_pi05_20251126_100645>
- 类型：LeRobot，`model.safetensors`，约 7.47 GB；
- 模型：pi05，基于 `lerobot/pi05_base`；
- 输入：单个 `observation.images.camera_1`，原始形状 480x640x3；8 维 `observation.state`；
- 输出：8 维 action；配置中的 state 名字为 `panda_joint1.pos` ... `panda_joint7.pos` + `panda_finger_joint1.pos`，action 名字为 7 个 Panda joint position + `gripper.pos`；
- horizon：`chunk_size=50`、`n_action_steps=50`；
- 数据：`ases200q2/Isaac_Panda_PickCube_SpaceMouse_EE_50eps`，README 显示 51 episodes；
- normalization：STATE/ACTION quantiles；
- 优点：输入输出维度和 Panda pick-cube 场景最直接匹配，且有 PyTorch/LeRobot processor 文件；
- 风险：model card 是通用模板，没有发布独立成功率；`train_config.json` 的 `steps` 为 200，不能据此推断训练充分；必须下载后做本地真实 rollout。

C1 是第一下载和 smoke 顺序，但不是“已证明成功”的模型。

### C2：备用候选

`DorianAtSchool/pi05base-finetune-franka-pickplace`

- Hugging Face：<https://huggingface.co/DorianAtSchool/pi05base-finetune-franka-pickplace>
- 类型：LeRobot，`model.safetensors`，约 7.47 GB；
- 输入：8 维 state，三路图像 `wrist/side/top`；
- 输出：配置为 7 维 action，未清晰包含独立 gripper action；
- 训练配置显示 10,000 steps；
- 风险：模型卡无可靠评估数据，7 action 与当前 8 维 Panda gripper contract 不直接匹配，三相机布局也需要适配；除非下载检查后确认 action 语义，否则不作为默认模型。

### C3：语义有价值但格式不优先

`ankile/openpi-pi05-franka-insert-marker-v2-ft`

- Hugging Face：<https://huggingface.co/ankile/openpi-pi05-franka-insert-marker-v2-ft>
- Franka/DROID 形态，5000 steps，8D joint velocity + gripper，horizon 16；
- 有明确的 Franka insert-marker 任务和 DROID observation contract；
- checkpoint 是 openpi Orbax/JAX 格式，不是 LeRobot PyTorch safetensors；
- 任务是插 marker，不是 pick cube；适合验证 DROID velocity/action adapter，不作为第一版 Panda pick-place 默认模型。

### C4：暂不作为主路径

- `bartek-niedzielski/pi05-panda-pos-v1` 和 `pi05-panda-vel-40-v1`：名称对应 Panda position/velocity，但仓库主要是 Orbax/JAX checkpoint，分别约 44.6/44.7 GB 的 checkpoint tree，没有 model card，需先确认 config、训练数据和转换路径；
- `RLinf/RLinf-Pi05-RLCo-PandaPutOnPlateInScene25DigitalTwin-V1-SFT`：存在 7.47 GB safetensors 和 Panda plate 名称，但根目录缺少可直接用于 LeRobot 的 config/README，contract 不足，先作为待审计候选；
- `PolinAvA/pi05_franka_base`：存在 PyTorch safetensors，但没有模型卡，配置是 32 action/50 horizon，名称更接近 base/实验输出，缺少当前 pick-place contract；
- `phospho-app` 下的 pi0.5 模型：多数是 Piper、SO100 或其他用户任务，机器人和动作语义与 Panda Nuka 不一致。

### 选型规则

- 下载顺序：C1 -> C2 -> C3/C4 审计；
- C1 若真实 rollout 达到 VLA-D，直接使用其 checkpoint，第一版不做本地 fine-tuning；
- C1 只通过静态 contract 但任务失败，不立即训练，先检查 camera pose、Panda joint zero、action position/velocity 语义、normalization stats 和控制频率；
- 只有候选均无法完成任务，或需要 Nuka 专属 embodiment adaptation 时，才启动本地 demonstration/fine-tuning；
- 所有候选下载、文件 SHA-256、revision、processor config 和实测结果写入外部 artifact manifest，不提交权重。

## 6.4 Action contract

VLA action semantics 由所选 checkpoint 的 processor/config 决定，禁止通过模型名字猜测。第一候选 C1 使用 Panda position contract：

```text
C1 panda_position_8:
  action[0:7] = absolute Panda joint position target, radians
  action[7]   = gripper position, [0,1] or dataset-defined range
  state[0:7]  = current Panda joint positions, radians
  state[7]    = current finger/gripper position
  control     = Nuka PD_POSITION
  chunk       = 50, read from checkpoint/config
```

DROID fallback contract 保留：

```text
DROID velocity_8:
  action[0:7] = normalized joint velocity command, [-1,1]
  action[7]   = gripper command, converted to [0,1] / binary policy
  state       = 7 joint positions + 1 gripper position
  control     = Nuka VELOCITY_TARGET
  frequency   = 15 Hz baseline
  chunk       = read from checkpoint/config
```

Nuka adapter：

- 启动时从 `config.json`、processor config、dataset feature names 和 manifest 确定 `position`/`velocity`，否则拒绝运行；
- action 先 finite check，再按 checkpoint 约定反归一化和限幅；
- position action 写入 `DRIVE_TARGET`，velocity action 写入 `VELOCITY_TARGET`；
- two-finger gripper action 映射到两个 finger slots；
- gripper > 0.5 的二值化规则只对明确声明为 binary 的模型启用；
- target 不得突破 joint limit/workspace safety clamp；
- action chunk 只在 CUDA 上缓存，执行 horizon 由 checkpoint chunk 和实际 inference latency 决定；
- policy exception、OOM、NaN、chunk 为空、action jump 或 safety violation 都必须停止动作并将目标保持为安全值。

LeRobot/OpenPI 文档指出 DROID 原始 checkpoint 使用 joint velocity，而仿真 fine-tuning 常使用 joint position。C1 优先使用 position；如果 C1 不适配，才切换 DROID velocity 或其他已审计模型。position/velocity 不得混用同一 normalization、processor 或 action metadata。

## 6.5 Pick-and-place task

成功判定必须是几何和状态结合，而不是仅看 prompt 或末端位置：

```text
1. cube initially outside bin
2. gripper closes around cube or cube has stable grasp contact
3. cube center enters bin volume
4. cube remains inside for settle window
5. gripper may release and cube stays inside
```

建议记录：

- `cube_grasped`；
- gripper width/command；
- cube-bin center distance；
- cube in-bin volume fraction；
- stable settle frames；
- collisions/contact normals/forces；
- success/failure reason；
- episode duration；
- model inference latency；
- action chunk age；
- Nuka physics step time。

场景 reset 随机化：

- cube 在 table 上多个位置；
- bin 在有限 workspace 多个位置；
- 相机固定但允许轻微 pose/lighting randomization；
- 第一轮 train/eval 不改变 robot joint convention 或 image key。

## 6.6 Demonstration 和 fine-tuning（第一版可暂缓）

第一版优先使用已经发布的第三方 Panda/Franka pi05 checkpoint，尤其是 C1。只有候选 checkpoint 无法完成 Nuka task，或静态 contract 无法对齐时，才启动本地 demonstrations 和 fine-tuning；本地 fine-tuning 不作为第一版前置条件。

数据采集仍应保留为后续适配基础，方式可以是：

- Nuka scripted expert；
- Python keyboard/waypoint teleop；
- 后续真实硬件录制，但不在本版。

每个 episode 至少保存：

```text
external RGB
optional wrist RGB
7 joint positions
1 gripper position
action chunk / per-step action
prompt/task
success and terminal state
timestamp/frame index
scene randomization seed
```

LeRobot dataset 转换必须保留：

- image shape/channel/order；
- state/action feature names；
- action semantics（velocity 或 position）；
- fps/control frequency；
- normalization stats；
- task text；
- source asset/model hashes。

只有决定启动本地 fine-tuning 时，才使用以下显存优先配置：

```text
PyTorch pi05
bfloat16 where supported
freeze vision encoder = true initially
train expert only = true initially
batch size = 1 or smallest fitting value
gradient checkpointing = true if supported
torch.compile = false until correctness passes
```

若未来 fine-tuning 在 16 GB OOM：

- 记录实际峰值和失败阶段；
- 不能把 OOM 隐藏成“训练完成”；
- 允许在更大 Linux GPU 上训练 checkpoint，再回到当前 WSL 运行 inference/eval；
- RTX 5080 的硬件结论仍是“可尝试 inference”，不是“保证 pi05 fine-tuning”。

## 6.7 VLA 验收门

### VLA-A：Panda/importer/camera

- Panda Nuka-compatible asset cook 成功；
- 7 arm joints 和 gripper 映射通过 manifest test；
- table/cube/bin contact 有限且可解释；
- 所选 checkpoint 要求的 camera keys 全部能由 Nuka 提供；C1 只要求一个 external `camera_1`，DROID/C3 才要求 external + wrist；
- 所需 camera view 的 shape、dtype、非空和 deterministic 通过；
- 至少生成一张可人工检查的 224x224 RGB 图，若 checkpoint 要求双相机则生成两张。

### VLA-B：真实 pi05 direct inference

- 真实 checkpoint 文件可加载，不是 LFS pointer、空目录或 dummy model；
- PyTorch model 在 `cuda:0`；
- Nuka 与 policy 在同一进程；
- state DLPack pointer 与 engine view 一致；
- camera preprocessing 是否调用 `.cpu()`/`.numpy()` 取决于所选 `preprocess_mode`；若使用 `HOST_COMPAT` 必须记录 host copy，若使用 `DEVICE_TORCH` 或 `ENGINE_UINT8` 则通过 instrumentation 证明 steady-state image path 不落 CPU；
- model 返回 finite action chunk，shape 从 checkpoint/config 校验；
- 固定输入/seed 的 action 与 reference adapter 在容差内一致；
- policy latency、peak VRAM、preprocess time、model time 有日志。

### VLA-C：动作执行和安全

- action chunk 实际改变 Panda joint state；
- gripper command 实际改变 two-finger width；
- action clip、joint limits、workspace limits 生效；
- NaN/Inf/OOM/empty chunk/exception 会进入 stop state；
- reset 清除 action queue 和 policy state；
- 失败不会让下一 episode 继承旧动作。

### VLA-D：pick-and-place task success

最终必须在固定协议下完成多次 rollout：

- 至少 10 个 evaluation episodes；
- 至少 3 个不同 cube/bin placement seed；
- 成功判定全部通过几何/settle 条件；
- 记录 success rate、失败原因、平均 episode 时间、平均 inference latency；
- 保存至少一个完整成功 rollout 的 RGB、state、action、视频和 metadata；
- 使用的 checkpoint、processor config、dataset stats、asset hash、Nuka build hash 全部可追溯；
- 如果使用 C1/C2 等第三方 fine-tuned checkpoint，必须先报告其静态 contract 和下载 revision，再报告 Nuka rollout 结果。

具体成功率阈值在第一个 baseline 运行后写入 `panda_pi05_manifest.json`，但不能以“偶尔成功一次”作为结项。默认建议最低门为 10 次中 8 次成功，若模型是未微调公开 checkpoint，则先记录真实结果并明确标注未达到 task gate。

## 7. Nuka 视觉接口要求

当前 Nuka 已有能力，第一版优先复用，不重造 view core：

```python
world.attach_camera_sensor(...)
world.render_sensors()
world.get_sensor_view(nuka.SensorChannel.COLOR)
world.sensor_dims()
```

实现要求：

- camera 数量由所选 checkpoint 的 `input_features` 决定：C1 默认一个 external camera；需要 wrist/external 的模型再挂两个；
- 相同尺寸的相机可形成 `(E,S,H,W,3)` camera block，camera index 在 manifest 中显式标注；
- camera output 当前是 device float32 AOV；如基准测试证明有收益，可新增 engine `uint8` COLOR output，但必须保留 float AOV 兼容路径和 golden image test；
- 任何 image preprocessing mode 都要明确写入 rollout metadata；
- `SensorAov` 只开启需要的 COLOR，避免不必要的 AOV 分配；
- camera render cadence 与 physics cadence 分离；C1 的 dataset 是 30 Hz，实际 query 频率由 checkpoint/control contract 决定；
- 每次 query 只 render policy 所需 env 0，不因 VLA task 创建大 batch；
- 如果需要视频，使用已有 beauty/offline recorder，不将 video recorder 嵌入模型 adapter；
- `.cpu()`、`.numpy()`、`cudaMemcpyDeviceToHost` 只能出现在 `HOST_COMPAT` 或显式 recording/debug 分支，不能在声称 direct-device 的 steady-state path。

## 8. 测试设计

## 8.1 G1 tests

建议新增：

```text
python/tests/test_g1_motion_contract.py
python/tests/test_g1_asset_gate.py
python/tests/test_g1_reference_loader.py
python/tests/test_g1_task_reset.py
python/tests/test_g1_reward_metrics.py
```

覆盖：

- 29 joint names/permutation；
- 160 obs field offsets；
- NPZ dtype/shape/finite/fps；
- xyzw -> wxyz；
- reference interpolation；
- reset phase/last action；
- action range and target map；
- reward components and termination；
- optional GPU asset gate skip reason。

如需 core 端回归，新增 C++ smoke：

```text
tests/runtime/test_g1_29dof_cook_step.cpp
```

该测试只验证 importer/cook/step/contact，不启动 PPO。

## 8.2 Panda/VLA tests

建议新增：

```text
python/tests/test_panda_asset_manifest.py
python/tests/test_panda_camera_contract.py
python/tests/test_pi05_contract.py
python/tests/test_pi05_device_preprocess.py
python/tests/test_pi05_no_host_image_copy.py
python/tests/test_panda_action_execution.py
python/tests/test_panda_pick_place_success.py
```

覆盖：

- Panda arm 7 joint + gripper manifest；
- engine slot <-> policy index map；
- camera `(E,S,H,W,3)` shape、COLOR dtype、camera order；
- RGB clamp/layout/missing-camera mask；
- state/action dimension and checkpoint metadata；
- normalization quantile/mean-std selection；
- action chunk shape/finite/clip/scale；
- gripper duplication；
- direct device pointer alias；
- image preprocess 禁止 CPU fallback；
- reset/action queue；
- deterministic scripted pick-place evaluator；
- real checkpoint test在模型可下载/可用时运行，否则必须清晰标为环境 gate skip，不可用 dummy pass 代替真实验收。

## 8.3 运行期 instrumentation

VLA rollout 需要输出结构化 JSONL 或等价记录：

```text
step
physics_tick
camera_tick
policy_query_id
image_shape/image_dtype/image_device
state_shape/state_device
host_tokenizer_used
preprocess_ms
tokenize_ms
model_ms
inference_ms
chunk_length/action_dim
action_min/action_max
physics_step_ms
peak_vram_bytes
cube_grasped
cube_in_bin
success/failure_reason
```

测试或代码审查必须能回答：

1. 图像是否落到 CPU；
2. state 是否 alias Nuka buffer；
3. action 是否回写 Nuka buffer；
4. policy 是否真的是 pi05，不是随机/零模型；
5. chunk horizon 是否从 config 读取；
6. normalization 是否与 checkpoint 匹配；
7. 任务成功是否由几何条件判定。

## 9. 分阶段实施计划

该计划是普通仓库实施顺序，不依赖任何 subagent/superpower 工作流。

### Step 1：WSL2 和构建基线

- 安装 Ubuntu WSL2；
- 安装 CUDA-compatible Linux toolchain、Vulkan loader、Python 3.11、PyTorch；
- 编译当前 Nuka；
- 运行现有 C++/Python smoke；
- 记录 GPU、driver、CUDA、CMake、compiler、Python 和 Nuka engine version。

交付：`docs/research` 更新环境日志；无功能源代码变更。

### Step 2：G1 asset/importer gate

- 下载/整理 G1 29DoF asset；
- 处理 mesh root 和 floating base；
- 写 manifest/asset doctor；
- 运行 cook/step/contact/reset；
- 如果失败，先修 importer/asset conversion 或明确记录 core blocker；
- 不在 gate 未通过时写 PPO。

交付：G1 asset smoke、manifest、最小测试、失败诊断。

### Step 3：G1 reference loader/playback

- 实现 G1 joint schema；
- 实现 NPZ/CSV loader；
- 实现 reference interpolation、anchor frame 和 observation 160；
- 实现直接 PD playback；
- 导出一段可检查视频和 metrics。

交付：独立 G1 playback demo 和 contract tests。

### Step 4：G1 Nuka-native PPO

- 新增 G1 task/reward/termination；
- 接入现有训练基础设施或新增 generic 29DoF adapter；
- 先固定短片段、少量 domain randomization；
- 保存 checkpoint、evaluation metrics 和 replay metadata；
- 最后加多 seed/curriculum。

交付：训练入口、eval/play 入口、至少一个稳定短舞蹈 checkpoint（权重不提交仓库）。

### Step 5：Panda asset and task gate

- 拉取 Panda Menagerie asset；
- 生成 Nuka-compatible variant；
- 处理 gripper、table、cube、bin；
- 固定 arm/gripper manifest；
- attach external/wrist cameras；
- 做 scripted velocity/PD pick-place，先不加载 pi05。

交付：Panda scene、asset doctor、camera/action tests、scripted success baseline。

### Step 6：pi05 同进程 inference adapter

- 确认 openpi/LeRobot exact commit/version；
- 按 C1 -> C2 -> C3/C4 顺序下载并审计第三方 checkpoint；
- 优先加载 C1 的 LeRobot/PyTorch `model.safetensors`，确认 processor、stats、camera key、8D state/action 和 position semantics；
- 若候选只有 openpi Orbax/JAX 格式，再单独评估 JAX path 或官方转换，不把它当作 LeRobot/PyTorch 直接可用；
- 实现 `Pi05Policy` adapter，支持 `HOST_COMPAT`、`DEVICE_TORCH` 和可选 `ENGINE_UINT8` preprocessing mode；
- 对齐 image/state/action transforms、stats、tokenization；
- 先跑单次真实 inference，再跑多次 action execution；
- 初始关闭 compile，完成正确性和显存记录后再优化。

交付：真实 pi05 direct-inference demo、候选 checkpoint 审计表、contract tests、preprocess mode/host-copy instrumentation。

### Step 7：Panda pi05 task success

- 优先直接评估 C1/C2 等第三方 fine-tuned checkpoint；
- 不把本地 fine-tuning 作为第一版前置条件；
- 只有第三方 checkpoint 无法完成 task，或 Nuka embodiment/camera/action contract 必须重新适配时，才采集 scripted/teleop demonstrations；
- 如果启动训练，生成 LeRobot-compatible dataset/stats，并记录 action semantics；
- 在当前 5080 上可以尝试最小显存配置，但 OOM 必须保留日志；
- 必要时在更大 Linux GPU 上训练 checkpoint，再回到当前 WSL 同进程运行 inference/eval；
- 运行 10+ episode evaluator；
- 输出成功 rollout、视频、指标和 provenance。

交付：pick-and-place task gate 的可追溯结果，或明确未通过原因、候选模型缺陷和剩余硬件/数据阻塞。

## 10. 预期修改面

以下是实现阶段的候选文件，不代表现在就全部创建：

### G1

```text
python/nuka/tasks/g1_motion.py
python/nuka/tasks/g1_dance.py
python/nuka/tasks/g1_motion_contract.py
python/examples/training/train_g1_dance_ppo.py
python/examples/demo/g1_dance_play.py
python/tests/test_g1_motion_contract.py
python/tests/test_g1_reference_loader.py
python/tests/test_g1_task_reset.py
tests/runtime/test_g1_29dof_cook_step.cpp
examples/scenes/g1_29dof_mode15.*
examples/assets/g1_mode15/manifest.json
examples/motions/g1/manifest.json
```

### Panda/pi05

```text
python/nuka/vla/pi05_contract.py
python/nuka/vla/pi05_device.py
python/nuka/tasks/panda_pick_place.py
python/examples/demo/panda_pi05_pick_place.py
python/examples/demo/panda_asset_gate.py
python/tests/test_panda_asset_manifest.py
python/tests/test_panda_camera_contract.py
python/tests/test_pi05_contract.py
python/tests/test_pi05_device_preprocess.py
python/tests/test_pi05_no_host_image_copy.py
python/tests/test_panda_action_execution.py
python/tests/test_panda_pick_place_success.py
examples/scenes/panda_pi05_pick_place.*
examples/assets/panda/manifest.json
examples/vla/panda_pi05_manifest.json
```

### 文档/工具

```text
docs/research/2026-08-26-imitation-vla-research-zh.md
docs/roadmap/2026-08-26-spec-05-g1-dance-pi05-vla-zh.md
tools/assets/fetch_g1_asset.*
tools/assets/convert_panda_for_nuka.*
tools/assets/check_embodied_manifest.*
```

如果实际代码结构不同，保持功能边界和测试合同，不为遵守候选文件名而做无意义重构。

## 11. 风险与决策记录

### R1：G1 35 generalized DoF

主 contact/ABA 路径有 64 articulation 上限，但 diffsim/OSC/computed torque 仍有 18 scratch 上限。第一版只用 PD position，必须通过真实 G1 contact gate 后才能进入训练。

### R2：Panda MJCF 语义不等于 Nuka 语义

Panda Menagerie 的 MuJoCo actuator、tendon、equality 和 fixed-body 结构需要显式转换。Importer 能读取 XML 不等于 dynamics/control 已对齐。

### R3：pi05 官方 high-level API 与 zero-copy 取舍

`openpi.Policy.infer` 和官方 DROID transform 面向 NumPy/通用客户端，使用它们时必须标记 `HOST_COMPAT/HOST_COPY`，不能宣称 zero-copy。第一版先按正确性、显存和延迟实测决定是否继续使用；只有 host copy 成为瓶颈或 direct-device 是明确验收目标时，才实现 device-aware PyTorch lower-level adapter。

### R4：pi05 state tokenization 可能有 host 边界

pi05 将归一化 state 离散为语言 token。除非实现 golden-equivalent device tokenizer，否则不能宣称 state tokenization 无 host。可接受低频 host tokenizer，但必须显式记录。

### R5：预训练 embodiment/action mismatch

`pi05_droid` 是 DROID Franka setup，Panda scene 的相机、joint zero、gripper、workspace、physics 和 action scale 仍需对齐。公开 checkpoint 真实推理成功不等于 Nuka task success。

### R6：16 GB VRAM 与本地 fine-tuning

单独 inference 有希望；与 Nuka 同 device 需要测峰值。fine-tuning 官方数字高于当前显存，第一版优先使用已发布的第三方 fine-tuned checkpoint，本地 fine-tuning 暂缓。不能用 OOM 后的假完成日志结项；如果未来必须训练，允许迁移到更大 Linux GPU。

### R7：WSL Vulkan/CUDA 组合

Nuka 编译和 camera/offline RT 需要 WSL Linux 依赖；WSL `nvidia-smi` 通过不代表 Vulkan viewer/offline RT 一定可用。headless camera sensor 和 training path 应先独立于 editor/viewer 验证。

### R8：数据许可和大文件

只提交 manifest、脚本、小型 fixtures 和 provenance；权重、完整 motion、mesh 和生成视频放在外部 artifact/output 目录。

## 12. 当前结论

1. 需求现在以“G1 29DoF + Panda pi05 pick-and-place”正式收敛。
2. VLA 不走 Docker、不走 websocket；主路径是 WSL2 Ubuntu 中的同进程 PyTorch/Nuka direct-device integration。
3. Nuka 已有 camera AOV 和 DLPack buffer，足以开始，不需要先重做 editor/viewer 架构。
4. 真正的技术难点不是相机 attach，而是 Panda asset semantic conversion、pi05 state tokenization、normalization/action alignment 和 16 GB 显存。
5. G1 必须先过 asset/contact/reference playback gates，再进入 PPO；Panda 必须先过 scripted pick-place，再进入真实 pi05。
6. 最终验收同时要求：G1 稳定舞蹈 checkpoint，以及 Panda pi05 真实闭环 pick-and-place 多次成功；任何只返回随机/零 action 的 fake model 都不算完成。
7. 下一步不是继续泛化方案，而是按 Step 1 开始做 WSL 环境和两个资产 gate，并将实际失败信息回填本文档。
