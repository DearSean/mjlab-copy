# Sparse-Prompt Velocity Control Plan

本文档是一个可迭代的研究和工程路线草案，目标是在现有 mjlab
`velocity` 和 `tracking` 任务基础上，先完成一个最小闭环，再逐步接入
diffusion 或 flow matching 生成式动作先验。方案不是最终设计，后续可以随
实验结果继续删改。

## 1. 核心目标

当前 `velocity` 任务能提供速度跟踪、平衡、抗扰动和地形适应能力，但动作
表达主要由 reward shaping 决定，学术创新容易显得像工程堆叠。新的目标是：

```text
velocity command 决定往哪里走、走多快、如何保持平衡；
sparse pose/keyframe prompt 决定以什么上身姿态或动作风格走；
policy 在物理可行性约束下尽量满足动作 prompt，而不是逐帧硬模仿。
```

典型功能：

- 前进、后退、转向、侧移时保持基础速度控制。
- 行走中挥手、举手、手臂指向、弯腰、低身走。
- prompt 和行走稳定性冲突时，策略自动弱化不安全的动作部分。
- 后续扩展到全身接触：膝、手、肘、躯干与地面、墙面、箱体等障碍物的
  可控接触，用于跪姿前进、匍匐、支撑越障、贴墙支撑等动作。
- 接上传感器后，使 prompt 与地形/障碍物感知结合，支持越障和复杂环境
  下的运动策略选择。
- 后续可由 diffusion 或 flow matching 从稀疏关键帧生成软参考轨迹。

一句话论文立意：

> Velocity policies are robust and controllable but lack expressive whole-body
> behaviors, while motion tracking policies require dense and clean references.
> We propose sparse-prompted velocity control, where sparse pose/keyframe
> prompts are converted into soft whole-body references and executed by a
> residual velocity controller under physical feasibility constraints.

### 1.1 收敛版本

当前文档包含了多条后续扩展线：sparse prompt、motion matching、diffusion、
flow matching、全身接触、传感器越障。它们不应该同时作为第一篇论文目标。
建议按工作量和论文完整度收敛成以下几版。

#### V0: 最小功能闭环

目标是证明“velocity policy 可以在保持走路和平衡时响应稀疏动作 prompt”。

包含：

- `SparsePoseVelocityCommand`。
- 挥手、举手、弯腰、低身走 4 类 prompt。
- wrist height、torso pitch、pelvis height 等 sparse rewards。
- G1 flat terrain 训练和 play 展示。

不包含：

- motion 数据。
- motion matching。
- diffusion / flow matching。
- 全身接触和障碍物。

适合 2-4 周内打通工程链路，但单独作为 IROS 主论文偏弱。它应作为所有后续
版本的 proof-of-concept。

#### V1: 推荐主论文版本

目标是形成一篇个人可完成、贡献清晰的 IROS 级方案：

```text
Sparse-prompted velocity control for expressive humanoid locomotion
```

包含：

- V0 全部内容。
- 关键帧/解析 reference generator，把稀疏 prompt 转成软参考。
- residual reference action：`q_target = q_ref + policy_residual`。
- prompt weight / body mask / command composition。
- flat + rough terrain，验证速度、稳定性、动作表达的权衡。
- 轻量 contact-aware 设计：只允许 hand/knee/torso contact 的接口和指标，
  不做匍匐、跪姿、越障。

不包含：

- diffusion / flow matching。
- 完整 motion matching。
- 视觉或 RGB-D。
- 强接触技能，如匍匐、翻越、倒地起身。

论文卖点：

- 低数据或无动作数据条件下，用稀疏姿态/关键帧 prompt 扩展 velocity
  policy 的全身表达能力。
- 策略不是硬 tracking，而是在速度跟踪、平衡和动作 prompt 之间动态权衡。
- 接口天然可接上游 LLM/VLM/diffusion，但第一篇不依赖大模型。

这是当前最推荐版本。

#### V2: 加强版论文

目标是在 V1 基础上增强表现性和长时序能力：

```text
Sparse-prompt velocity control with motion-matched soft references
```

包含：

- V1 全部内容。
- 小型 motion database。
- motion matching reference generator。
- locomotion -> prompt skill -> locomotion 的技能拼接。
- entry window 和 transition blending。
- 少量接触轻任务，例如手扶墙站立、低身通过，不做 parkour。

不包含：

- diffusion / flow matching。
- 大规模数据集。
- 视觉端到端学生策略。

论文卖点：

- 在低数据条件下，motion matching 比 diffusion 更可解释、更稳。
- locomotion manifold 作为技能之间的公共过渡空间。
- 能展示更强表现性，但工作量明显高于 V1。

适合在 V1 已经跑通后升级，不建议一开始就作为第一目标。

#### V3: 后续大版本

目标是面向更大的长期项目：

```text
Contact-aware generative reference grounding for humanoid scene interaction
```

包含：

- diffusion 或 flow matching reference generator。
- contact schedule、contact points/normals、interaction keypoints。
- 跪姿、匍匐、墙面/箱体支撑。
- raycast / RGB-D 感知越障。
- teacher-student 或 DAgger + RL。

这是完整研究线，不适合个人第一篇论文直接冲。它可以作为第二篇论文或博士
课题方向。

#### 当前推荐

先做 V0，论文目标定为 V1。文档后面保留 V2/V3 作为扩展接口，但第一阶段
所有代码设计都应服务于 V1：

```text
Sparse prompt
+ soft reference
+ residual velocity control
+ clear ablations
+ no large motion dataset
+ no diffusion dependency
```

如果 V1 提前跑通，再把 motion matching 作为一项增强实验加入，而不是把
diffusion/flow 和全身接触同时纳入主线。

## 2. 现有代码接入点

优先复用已有模块，避免一开始重写训练栈。

### Velocity 侧

- `src/mjlab/tasks/velocity/velocity_env_cfg.py`
  - 当前 actor observation、critic observation、`twist` command、reward、
    terrain curriculum 都在这里组装。
  - 最小闭环应在这里新增 prompt command、prompt observation 和 prompt
    rewards。
- `src/mjlab/tasks/velocity/mdp/velocity_command.py`
  - 当前 `UniformVelocityCommand` 只输出 `[vx, vy, wz]`。
  - 新方案不要直接破坏它，先新增 `SparsePoseVelocityCommand`。
- `src/mjlab/tasks/velocity/mdp/rewards.py`
  - 当前已有 `track_linear_velocity`、`track_angular_velocity`、
    `variable_posture`、`feet_clearance`、`feet_slip` 等。
  - 新增 sparse prompt reward 时，应保持 velocity reward 为主，prompt
    reward 为软约束。
- `src/mjlab/tasks/velocity/config/g1/env_cfgs.py`
  - G1 的 body/site/joint 名称、action scale、self collision sensor 在这里
    patch。prompt 相关的关节 mask 和目标 body 名称应在这里落地。
- `src/mjlab/tasks/velocity/config/rlboy/recovery_assist.py`
  - 已有 recovery assist、unstable reset、gated reward 等思路。
  - 后续做 recovery-aware training 时优先迁移这里的设计，而不是重新写。

### Tracking 侧

- `src/mjlab/tasks/tracking/mdp/commands.py`
  - `MotionLoader` 和 `MotionCommand` 已能读取 `.npz` motion，提供
    `joint_pos`、`joint_vel`、body pose、body velocity。
  - 最小闭环可以先不依赖 motion 数据；第二阶段再复用它做 reference
    window。
- `src/mjlab/tasks/tracking/mdp/rewards.py`
  - 已有 anchor/body position/orientation/velocity tracking reward。
  - 后续 motion prior reward 可以从这里抽公共函数，避免复制太多。
- `src/mjlab/tasks/tracking/scripts/evaluate.py`
  - 已有 MPJPE/MPKPE 一类 tracking 指标计算路径，后续可扩展到新任务评估。

### RL 侧

- `src/mjlab/rl/config.py`
  - 当前主训练栈是 RSL-RL PPO，模型默认 `MLPModel`。
  - 最小闭环保持 MLP，先证明任务定义可行。
  - 后续 dynamics-conditioned attention actor 需要新增 custom model/runner。
- `src/mjlab/tasks/velocity/rl/runner.py`
  - 当前负责保存 ONNX/JIT。新增 prompt metadata 或生成器时，需要在这里补
    export metadata。

### Sensor / Terrain 侧

- `src/mjlab/sensor/`
  - 已有 raycast、terrain height、contact、camera 等传感器基础设施。
  - 后续全身接触与越障需要更多 contact sensor 分组，例如手、膝、肘、
    躯干、脚分别统计合法/非法接触。
- `src/mjlab/terrains/`
  - 已有 rough terrain generator。后续需要加入障碍物/墙面/箱体/低矮
    通道等场景，用于训练匍匐、跪姿前进、支撑越障。
- `src/mjlab/tasks/velocity/mdp/terrain_utils.py`
  - 已有 terrain normal 相关工具。后续可扩展为 obstacle/contact-aware
    prompt 的几何特征提取。

## 3. 最小闭环阶段

目标：不用 diffusion，不需要大量动作数据，先让 G1 velocity policy 支持
几个稀疏动作 prompt，并能训练、评估、可视化。

### 3.1 新增任务 ID

新增一个独立任务，避免影响已有 baseline：

```text
Mjlab-Velocity-Prompt-Flat-Unitree-G1
```

建议新增文件：

```text
src/mjlab/tasks/velocity/config/g1/prompt_env_cfgs.py
src/mjlab/tasks/velocity/config/g1/prompt_rl_cfg.py
```

也可以先在 `env_cfgs.py` 增加 factory，稳定后再拆文件。

### 3.2 新增 SparsePoseVelocityCommand

新增文件：

```text
src/mjlab/tasks/velocity/mdp/prompt_command.py
```

第一版 command 输出：

```text
[
  vx, vy, wz,
  prompt_type_onehot,
  prompt_weight,
  torso_pitch_target,
  pelvis_height_target,
  left_wrist_height_target,
  right_wrist_height_target,
  left_arm_phase,
  right_arm_phase
]
```

第一版 prompt 类型：

- `none`: 只走路。
- `left_wave`: 左手挥手。
- `right_wave`: 右手挥手。
- `both_hands_up`: 双手举高。
- `torso_bend`: 弯腰/身体前倾。
- `low_walk`: 低身走。

实现建议：

- 内部仍复用 `UniformVelocityCommand` 的速度采样逻辑，或者直接组合一个
  `twist` 子命令。
- prompt 每 2-6 秒重采样一次，速度每 3-8 秒重采样一次；两者可先同步，
  稳定后再拆成双 timer。
- `prompt_weight` 从 `{0.0, 0.25, 0.5, 1.0}` 采样，用于训练策略处理动作
  强弱。
- play 模式支持 GUI slider/button，后续再做；最小闭环可以只随机采样。

### 3.3 新增 Prompt Observations

在 `make_velocity_env_cfg()` 或 G1 prompt cfg 中新增 actor observation：

```python
"prompt": ObservationTermCfg(
  func=mdp.generated_commands,
  params={"command_name": "prompt"},
)
```

为了保持 velocity baseline 可比，建议不要删除原有 `twist`，而是：

```text
commands:
  twist: UniformVelocityCommandCfg
  prompt: SparsePoseVelocityCommandCfg

observations:
  command: twist
  prompt: prompt
```

第一版也可以让 `prompt.command` 包含 twist，减少 observation 项数量。更清晰
的做法是拆开，因为 velocity reward 仍使用 `twist`。

### 3.4 新增 Sparse Prompt Rewards

在 `src/mjlab/tasks/velocity/mdp/rewards.py` 新增：

```text
wrist_height_tracking
torso_pitch_tracking
pelvis_height_tracking
masked_joint_pose_tracking
prompt_tracking_success_metric
```

第一版重点不要追求动作完全一致，而是“尽量接近且不破坏行走”：

```text
total reward =
  velocity tracking
  + upright / posture / foot regularization
  + prompt_weight * sparse prompt reward
  - action smoothness / joint limits / self collision
```

建议权重原则：

- velocity tracking 权重保持主导。
- prompt reward 从小权重开始，例如 `0.2-0.5`。
- 弯腰和低身类 prompt 必须配合 base height、torso tilt 安全范围。
- 挥手只约束肩/肘/腕或 wrist height，不约束腿部。

### 3.5 先用解析参考，不用生成模型

最小闭环不要直接上 diffusion。先写一个 deterministic reference generator：

```text
prompt_type + phase + target strength -> sparse targets
```

例子：

- `left_wave`: 左腕高度目标随正弦相位变化，肩肘关节只给软目标。
- `both_hands_up`: 左右腕高度目标固定到胸/头附近。
- `torso_bend`: torso pitch 目标为正向前倾，pelvis height 稍降。
- `low_walk`: pelvis height 降低，torso pitch 小幅前倾。

这一步的意义是先验证：

- prompt reward 不会破坏速度跟踪。
- 上身动作可以在走路中出现。
- prompt 强弱和 velocity 命令能同时被策略理解。

### 3.6 Prompt 与关键帧文件

需要准备，但第一版只需要少量结构化配置，不需要完整动作数据集。

推荐位置：

```text
src/mjlab/tasks/velocity/config/g1/prompts/
  prompt_specs.json
  keyframes.json
  README.md
```

原因：

- V0/V1 的 prompt 与 G1 的 joint/body/site 名称强绑定，属于任务配置。
- 这些文件应随代码版本管理，便于复现实验。
- 后续如果变成完整 motion clip，再迁移到 `motions/` 或转换成 `.npz`。

第一版建议拆成两类文件。

`prompt_specs.json` 描述动作意图、body mask 和 reward 权重：

```json
{
  "left_wave": {
    "type": "periodic_pose",
    "duration_s": 2.0,
    "body_mask": ["left_arm", "torso"],
    "targets": {
      "left_wrist_height": [0.75, 1.05],
      "torso_pitch": 0.0
    },
    "reward_weights": {
      "wrist_height": 0.5,
      "masked_joint_pose": 0.2,
      "velocity": 1.0
    },
    "safety": {
      "max_torso_pitch": 0.35,
      "min_pelvis_height": 0.55
    }
  }
}
```

`keyframes.json` 描述少量参考关节姿态。建议只写相对默认姿态的 delta，而不是
整个人形所有关节的绝对值：

```json
{
  "left_wave": {
    "fps": 10,
    "loop": true,
    "frames": [
      {
        "time": 0.0,
        "joint_pos_delta": {
          "left_shoulder_pitch_joint": -0.4,
          "left_shoulder_roll_joint": 0.45,
          "left_shoulder_yaw_joint": 0.0,
          "left_elbow_joint": 0.8
        }
      },
      {
        "time": 0.5,
        "joint_pos_delta": {
          "left_shoulder_pitch_joint": -0.2,
          "left_shoulder_roll_joint": 0.7,
          "left_shoulder_yaw_joint": 0.25,
          "left_elbow_joint": 1.1
        }
      }
    ]
  }
}
```

准备关键帧时的原则：

- 优先准备上身和躯干关节，腿部先留给 velocity policy。
- 优先写 `joint_pos_delta`，以 robot default pose 为零点，减少机器人配置变化
  带来的偏差。
- 每个 prompt 先准备 2-5 个关键帧，不要一开始做长轨迹。
- 每个关键帧都要标注是否循环、持续时间、body mask。
- 关键帧只是软参考；reward 和 residual action 不应强迫逐帧完全一致。

挥手这类动作是否要准备“全身关节位置”：

- V0/V1 不建议准备完整全身绝对关节姿态。
- 建议只准备左/右肩、肘、腕，以及必要的腰部/躯干 delta。
- 腿部、脚、髋、膝、踝默认不写，由 velocity task 保持行走和平衡。
- 如果某个动作确实需要全身协调，例如低身走，可以写 pelvis height、
  torso pitch，以及少量 hip/knee 的软参考，但权重要小。

什么时候使用 `motions/`：

- 如果有连续 mocap/retargeted 轨迹，或要复用 tracking task 的
  `MotionLoader`，放到 `motions/` 或 `motions/<robot>/`。
- CSV 格式可以走 `src/mjlab/scripts/csv_to_npz.py` 转成 tracking 使用的
  `.npz`，其中包含 `joint_pos`、`joint_vel`、`body_pos_w` 等完整数组。
- 这属于 V2 之后的 motion prior / motion matching 阶段，不是 V1 必需项。

### 3.7 最小闭环训练命令

先用 flat terrain，降低变量：

```sh
uv run train Mjlab-Velocity-Prompt-Flat-Unitree-G1 \
  --env.scene.num-envs 2048
```

如果显存允许，再用 4096 envs。迭代期间优先跑较短训练，比如：

```sh
uv run train Mjlab-Velocity-Prompt-Flat-Unitree-G1 \
  --env.scene.num-envs 1024 \
  --agent.max-iterations 2000
```

### 3.8 最小闭环验收标准

必须同时满足：

- `none` prompt 下速度跟踪接近原 velocity baseline。
- `left_wave/right_wave` 下能看到明显手臂动作，速度不崩。
- `torso_bend/low_walk` 下身体姿态变化明显，fall rate 不显著上升。
- prompt weight 从 0 到 1 时，动作幅度有连续变化。
- action rate、self collision、foot slip 没有异常爆炸。

建议新增测试：

```text
tests/test_velocity_prompt_task.py
tests/test_velocity_prompt_rewards.py
```

测试内容：

- 新任务已注册。
- prompt command 输出维度正确。
- prompt reward 在目标接近时更高。
- play cfg 不破坏已有 velocity task。

运行：

```sh
uv run pytest tests/test_velocity_prompt_task.py tests/test_velocity_prompt_rewards.py
uv run ruff format
uv run ruff check --fix
```

## 4. 第二阶段：接入 motion/window，但仍不用 diffusion

目标：把 tracking 的 motion reference 能力引入 velocity prompt，但用简单的
检索和插值，不引入生成模型复杂度。

### 4.1 Motion-Prior Command

新增或扩展 command：

```text
SparsePoseVelocityCommand -> MotionPromptVelocityCommand
```

输入：

```text
twist command
prompt type
prompt strength
optional motion clip id
phase
```

输出：

```text
q_ref[t:t+H]
base_height_ref[t:t+H]
body target / wrist target[t:t+H]
confidence[t:t+H]
```

参考 `MotionLoader`：

```text
src/mjlab/tasks/tracking/mdp/commands.py
```

但注意不要直接把完整 `MotionCommand` 搬进 velocity。velocity 主目标仍然是
速度控制，motion 只是软参考。

### 4.2 Residual Reference Action

新增 action term：

```text
src/mjlab/tasks/velocity/mdp/prompt_actions.py
```

第一版：

```text
q_target = q_default + policy_action
```

第二版：

```text
q_target = q_ref + policy_action
```

需要注意：

- 对无参考的 prompt，`q_ref = q_default`。
- 对上身 prompt，只对上身 joint 使用 `q_ref`，腿部仍可围绕 default 或
  learned locomotion posture。
- action scale 对上身和下身分开配置，避免上身 prompt 破坏腿部步态。

### 4.3 Dynamics-Conditioned Command Aggregation

这一阶段先不必实现 Transformer actor，可以先把 command window flatten 后给
MLP，建立 baseline：

```text
obs_t + prompt_window_flat -> MLP actor
```

之后再实现：

```text
obs_history[t-K:t] -> dynamics embedding
prompt_window[t-L:t+L] -> token sequence
cross attention -> command embedding
actor(obs_t, command_embedding) -> action
```

这部分参考论文：

- Robust and Generalized Humanoid Motion Tracking:
  https://arxiv.org/pdf/2601.23080

### 4.4 Motion Matching 作为表现性强基线

Perceptive Humanoid Parkour (PHP) 的关键启发是：表现性和长时序能力不一定
先靠 diffusion/flow。对于低数据场景，motion matching 是一个更简单、更可控
的中间方案：把少量原子动作片段放入数据库，通过特征最近邻检索，把 locomotion
和 skill clips 拼成长时序参考。

可迁移到本项目的设计：

```text
motion database:
  locomotion clips
  wave / bend / low_walk clips
  kneel / crawl / wall_support clips (later)

query feature:
  current root velocity
  current foot/hand positions and velocities
  desired future 2D trajectory from velocity command
  prompt type / contact mode
  obstacle-relative features (later)

output:
  next reference frame index
  q_ref window
  contact schedule
  transition confidence
```

它适合放在 diffusion/flow 之前，作为 reference generator 的强基线：

```text
sparse prompt + velocity command
        -> motion matching retrieval
        -> blended reference window
        -> residual velocity policy
```

实施建议：

- 先对站立/行走 prompt 做 motion matching，不碰 parkour。
- 用 locomotion 作为所有技能之间的共享过渡空间，避免每对技能都需要单独
  采集 transition。
- 每个 skill clip 手动标注 `start_frame`、`end_frame`、`entry_window`。
- 进入 contact-rich skill 后，短时间禁用 motion matching，顺序播放该技能
  clip，避免破坏关键接触序列。
- 参考切换时做短窗口 blending，避免 `q_ref` 不连续。
- 训练数据要覆盖不同 approach distance、stride phase、速度和方向。PHP 的
  消融显示，过渡密度不足会导致关键接触时机失败。

这条路线与 diffusion/flow 的关系：

- motion matching 是可解释、低风险、低数据的 reference generator。
- diffusion/flow 可以在 motion matching 数据之上学习更连续的生成模型。
- 实验上应把 motion matching 作为 B4/B5 之间的基线，否则 diffusion 的收益
  很难讲清楚。

## 5. 第三阶段：接入 diffusion generator

目标：用生成模型替代手写插值/检索，从稀疏 prompt 生成软参考窗口。

### 5.1 推荐定位

不要让 diffusion 直接输出低层 motor action。更稳的接口是：

```text
diffusion generator:
  condition -> q_ref/body_targets over horizon

RL tracker:
  obs + q_ref/body_targets -> residual action
```

这样 diffusion 错了，低层 policy 仍有机会修正。

### 5.2 数据来源

优先用便宜数据构造 pseudo dataset：

- 手工关键帧：挥手、举手、弯腰、低身。
- IK/插值生成轨迹：关节限位、速度限位、jerk smoothing。
- 当前最小闭环 policy rollout：把稳定 rollout 记录下来，作为生成器训练数据。
- 少量 `.npz` motion：复用 tracking preprocessing。

数据格式建议：

```text
condition:
  obs summary
  vx, vy, wz
  prompt type / target
  body mask
  keyframes

target:
  q_ref[H, dof]
  body target[H, N, feature]
  optional confidence[H]
```

### 5.3 参考仓库

优先参考思路和模块，不建议直接 vendor 大量代码。

- DiffuseLoco:
  https://github.com/HybridRobotics/DiffuseLoco
  - 借鉴 receding horizon control、delayed inputs、offline dataset 训练流程。
  - 不建议直接复刻低层 diffusion locomotion policy，因为它更依赖多技能离线
    数据，并且原仓库是 IsaacGym/Cyberdog 结构。
- Diffusion Policy:
  https://github.com/real-stanford/diffusion_policy
  - 借鉴 action horizon、observation horizon、Transformer/UNet 时间序列
    diffusion policy 的工程组织。
  - 我们只生成参考动作窗口，不直接生成最终控制 action。
- Awesome Robotics Diffusion:
  https://github.com/showlab/Awesome-Robotics-Diffusion
  - 用于持续跟踪 robotics diffusion 论文和实现。

### 5.4 mjlab 中建议新增包结构

```text
src/mjlab/generative/
  __init__.py
  datasets.py
  normalizer.py
  diffusion.py
  flow_matching.py
  reference_generator.py

src/mjlab/tasks/velocity/generative/
  collect_prompt_rollouts.py
  train_prompt_generator.py
  evaluate_prompt_generator.py
```

最小训练脚本先离线运行，不接入 RSL-RL runner。等生成器稳定后，再在 command
term 中加载 frozen generator。

### 5.5 Diffusion 接入验收标准

- 同样 prompt 下，生成轨迹比线性插值更平滑或更自然。
- best-of-N 采样能降低自碰撞、joint limit、速度误差。
- frozen generator + residual velocity policy 比 handcrafted reference 更稳定或
  更 expressive。
- 推理延迟满足控制频率预算。若 generator 只每 0.2-0.5s 生成一次 reference
  window，实时压力会小很多。

## 6. 第四阶段：Flow Matching 替代或并行

Flow matching 更适合实时性和少步生成。定位同 diffusion：

```text
condition -> reference trajectory
```

不直接替代 velocity policy。

参考仓库：

- Flow Matching Policy:
  https://github.com/HRI-EU/flow-matching-policy
  - 借鉴条件 flow matching 的训练与采样组织。
- FlowPolicy:
  https://github.com/zql-kk/FlowPolicy
  - 借鉴 consistency flow matching 和少步推理思路。
- A2A Flow Matching:
  https://github.com/JIAjindou/A2A_Flow_Matching
  - 借鉴 action/history-informed initialization 的想法；对 locomotion 的连续性
    很有启发。

建议先做 rectified flow/conditional flow matching 的简单版本：

```text
x0 ~ N(0, I)
x1 = reference trajectory
xt = (1 - t) * x0 + t * x1
model(xt, t, condition) -> x1 - x0
```

推理：

```text
sample x0
ODE steps 1-8
decode reference trajectory
feasibility filter
```

对比 diffusion：

- 训练是否更稳定。
- 同样延迟下动作质量如何。
- 1/2/4/8 步采样的效果曲线。

## 7. 可行性过滤与安全层

无论 reference 来自手写、tracking motion、diffusion 还是 flow matching，都
必须经过 feasibility layer。

第一版规则：

- joint limit clamp。
- joint velocity / acceleration / jerk 限制。
- wrist/torso target 超出可达范围时缩放 prompt weight。
- torso pitch 和 pelvis height 安全区间。
- 自碰撞和非法接触 penalty。
- 对腿部 reference 使用低权重，避免上身动作干扰步态。

第二版评分：

```text
score = velocity feasibility
      + prompt target closeness
      - joint limit cost
      - self collision risk
      - support/contact inconsistency
```

生成模型可以 best-of-N 采样多个 reference window，再选择最高分候选。

## 8. 全身接触与障碍物扩展

这个方向很重要，但不建议放进第一版最小闭环。原因是全身接触会把问题从
普通 locomotion 变成 contact-rich whole-body locomotion，涉及接触模式、
碰撞几何、传感器、终止条件和 reward 的整体重写。建议把它作为独立里程碑
逐步接入。

OmniRetarget 的关键启发是：复杂交互不应只靠下游 RL reward 修补，而应在
参考生成/retargeting 阶段显式保留 agent-object-terrain 的空间关系和接触
关系。对本项目而言，这意味着 diffusion/flow 或关键帧生成器输出的不是单纯
`q_ref`，而应包含可检查的交互几何、接触计划和物理可行性标记。

### 8.1 目标能力

后续目标不只是“上身做动作”，而是允许策略主动利用身体其他部位接触环境：

- 跪姿前进：膝盖和脚作为主要支撑点，手臂可辅助平衡。
- 匍匐前进：手、肘、膝、躯干与地面形成多点支撑。
- 低矮空间通过：根据高度约束降低 pelvis/torso，并保持速度命令。
- 支撑越障：手接触箱体或墙面，辅助跨越、爬上或稳定身体。
- 贴墙/扶墙稳定：手或前臂接触墙面，抵抗侧向扰动。
- 障碍物跨越：接入 raycast/camera 后，根据障碍物高度和距离调整步态或
  切换到支撑动作。

### 8.2 接触建模原则

全身接触不能只用“非法接触惩罚”。需要区分：

```text
allowed contacts:
  feet, knees, hands, forearms, elbows, torso (depending on prompt/skill)

illegal contacts:
  head, neck, fragile links, high-impact unintended contacts

conditional contacts:
  knees allowed during kneel/crawl, penalized during normal walking
  hands allowed near wall/box support, penalized during ordinary walking
```

因此 reward 和 termination 都要由 prompt/contact mode 控制，而不是固定规则。

从 OmniRetarget 借鉴的建模原则：

- interaction representation：把身体关键点、物体/障碍物表面采样点、地形
  采样点放在同一个交互结构里，至少保留相对距离、相对方向、接触点和法向。
- hard feasibility before RL：生成参考时先检查 joint limit、velocity limit、
  non-penetration、foot/hand sticking、接触力方向等硬约束，减少 RL 被迫
  修补坏参考。
- object/environment frame：与箱体、墙面、平台交互时，接触 prompt 应尽量
  表达在物体或局部环境坐标系中，而不是固定世界坐标，便于物体平移、旋转、
  尺寸变化后复用。
- contact preservation metrics：评估时不仅看速度和姿态，还要看 penetration、
  skating、目标接触保持时间、非法接触率。

### 8.3 建议新增模块

```text
src/mjlab/tasks/velocity/mdp/contact_modes.py
src/mjlab/tasks/velocity/mdp/contact_rewards.py
src/mjlab/tasks/velocity/mdp/obstacle_commands.py
src/mjlab/tasks/velocity/config/g1/contact_env_cfgs.py
```

第一版 contact mode：

```text
walk: 只允许脚接触地面
kneel_walk: 脚 + 膝允许接触地面
crawl: 手 + 膝 + 肘/前臂 + 躯干低接触允许
wall_support: 手/前臂允许接触墙面
box_support: 手允许接触箱体，脚仍负责推进
```

### 8.4 传感器与观测

先用低维几何和接触传感器，不急着上视觉：

- 分组 contact sensors：
  - feet ground contact
  - knee ground contact
  - hand/forearm obstacle contact
  - torso ground contact
  - head illegal contact
- raycast / terrain height：
  - 当前地形高度。
  - 前方低矮障碍高度。
  - 低通道 clearance。
  - 墙面/箱体距离。
- critic privileged observations：
  - body link poses。
  - contact force。
  - obstacle relative pose。
  - terrain/obstacle type。

后续再接：

- RGB-D camera。
- 更密集的 raycast grid。
- obstacle segmentation 或 height map。

### 8.5 Reward / Termination 设计

新增 reward 族：

```text
contact_mode_match:
  目标接触部位按 prompt 出现，非目标接触部位受罚

contact_force_safety:
  接触力过大受罚，鼓励软接触

support_polygon / support_confidence:
  多点支撑时鼓励稳定支撑布局

obstacle_progress:
  在障碍物任务中鼓励越过或通过目标区域

clearance:
  低通道时鼓励身体低于高度约束，跨越时鼓励脚/膝/手有足够 clearance

recovery_to_velocity:
  全身接触动作完成后回到普通 velocity locomotion
```

termination 也要按 mode 区分：

- 普通 walking 中躯干/膝盖触地可以算失败。
- crawl/kneel mode 中躯干/膝盖触地不应提前终止。
- head/neck 高冲击接触始终应终止。
- recovery/contact-rich mode 中允许短时间低 base height，但设置超时。

对接触任务，推荐把 reference 质量指标也纳入训练/评估日志：

```text
penetration_duration / penetration_depth:
  身体、物体、地形之间的穿透时长和最大深度

contact_skating:
  目标接触点处于 sticking phase 时的切向滑移速度

contact_preservation:
  目标接触是否在期望窗口内出现并保持

interaction_deformation:
  身体关键点与物体/环境关键点的相对几何变化
```

如果这些指标在 reference 生成阶段已经很差，不应期望 RL 仅靠 reward 稳定
修复。更合理的流程是回到生成器/优化器修正 reference。

### 8.6 生成模型的新作用

接入 motion matching、diffusion 或 flow 后，生成器不仅生成姿态轨迹，还应
生成接触计划：

```text
condition:
  velocity command
  obstacle geometry / sensor embedding
  contact mode prompt
  sparse keyframes

output:
  q_ref[t:t+H]
  body target[t:t+H]
  contact schedule[t:t+H, body_group]
  contact points / normals[t:t+H, body_group]
  interaction keypoints[t:t+H]
  confidence[t:t+H]
```

低层 policy 的职责变成：

```text
根据当前动力学和传感器反馈，修正生成模型给出的姿态与接触计划。
```

这比“生成全身动作后硬 tracking”更稳，也更适合越障和复杂接触。

参考生成可分两层：

```text
geometry/contact generator:
  sparse prompt + obstacle geometry -> interaction keypoints + contact schedule

motion generator:
  interaction keypoints + contact schedule -> q_ref/body_targets
```

第一层更像 OmniRetarget 的 interaction-preserving retargeting 思路；第二层再
交给关键帧插值、IK、motion matching、diffusion 或 flow matching。这样可以
避免生成模型只学到“看起来像动作”，却破坏接触关系。

从 PHP 借鉴的接触技能拼接原则：

- contact-rich skill 应有明确 entry window。比如翻越、跪姿进入、匍匐进入，
  只允许从接近接触前的合理相位切入。
- 技能执行期间优先保持 clip 内接触时序，不要每一帧都重新检索。
- 技能结束后回到 locomotion manifold，再进行下一次 prompt/skill 切换。
- 通过 approach distance、速度、方向、障碍物位姿随机化制造过渡密度，让
  policy 学会在不同距离和步态相位下触发正确接触。

### 8.7 建议推进顺序

不要从匍匐或越障开始。建议按接触难度升级：

1. `low_walk`: 无新增接触，只降低身体高度。
2. `kneel_static`: 原地跪姿保持，允许膝盖接触。
3. `kneel_walk`: 低速跪姿前进。
4. `crawl_static`: 多点接触姿态保持。
5. `crawl_forward`: 匍匐低速前进。
6. `wall_support`: 手扶墙站立/侧向移动。
7. `box_support`: 手扶箱体稳定或跨越。
8. `sensor_obstacle_crossing`: 接 raycast/camera 后做感知越障。

每一步都建议配套一个小型数据增强：

- 障碍物位置平移/旋转。
- 箱体尺寸缩放。
- 平台或低通道高度变化。
- 地面坡度和摩擦变化。
- 初始身体相位/速度扰动。

增强时要保持交互关系。例如手扶箱体时，手部目标应随箱体坐标系变化；低通道
通过时，躯干/头部 clearance 约束应随通道高度变化。

### 8.8 与最小闭环的关系

最小闭环仍然只做站立/行走状态下的 sparse prompt。为了给 contact-rich
扩展预留接口，第一版就应避免以下设计：

- 不要把所有非脚接触永久当作失败。
- 不要把 base height 过低无条件终止，至少留 mode-dependent termination
  的接口。
- prompt command 中保留 `contact_mode` 字段，即使第一版只使用 `walk`。
- reward 中把 illegal contact 写成可配置 body group，而不是硬编码。

第一版也可以预留一个轻量 interaction prompt 格式，即使暂时不用：

```text
contact_mode
target_contact_body_group
target_surface_type
target_surface_normal
target_relative_pose
clearance_limit
```

这会让后续从“挥手/弯腰”扩展到“手扶墙/箱体、跪姿、匍匐、越障”时不需要
重写 command 接口。

## 9. 实验计划

### Baselines

```text
B0: 当前 Mjlab-Velocity-Flat-Unitree-G1
B1: velocity + sparse prompt reward, no reference generator
B2: velocity + handcrafted keyframe/interpolation reference
B3: velocity + motion-loader/reference-window
B4: B3 + motion matching reference generator
B5: B4 + dynamics-conditioned command aggregation
B6: B5 + diffusion generator
B7: B5 + flow matching generator
B8: B5 + contact-mode conditioning
B9: B8 + obstacle sensors
```

### 指标

- `velocity_rmse`: base linear velocity tracking error。
- `yaw_rmse`: yaw rate tracking error。
- `success_rate`: 未摔倒 rollout 比例。
- `prompt_error`: wrist height、torso pitch、pelvis height 误差。
- `prompt_success`: 在速度误差未超阈值时，prompt 是否达成。
- `action_rate` / `action_jerk`。
- `foot_slip`。
- `self_collision_count`。
- `energy` 或 actuator effort。
- `recovery_rate`：后续加入 recovery 后统计。
- `contact_mode_accuracy`: 目标接触部位与实际接触部位匹配程度。
- `illegal_contact_rate`: 非法接触比例。
- `contact_force_peak`: 关键接触部位最大接触力。
- `obstacle_success_rate`: 通过、跨越、支撑障碍物成功率。

### Ablations

- prompt weight: `0, 0.25, 0.5, 1.0`。
- 上身 mask vs 全身 mask。
- reference residual action vs default-pose action。
- handcrafted reference vs motion matching retrieval。
- motion matching density: sparse entry distances vs dense entry distances。
- flatten command window vs cross-attention command encoder。
- diffusion steps / flow steps。
- best-of-N candidate count。
- contact mode conditioned termination vs fixed termination。
- with / without obstacle sensors。

## 10. 推荐实施顺序

### Milestone A: 最小闭环

1. 新增 `SparsePoseVelocityCommand`。
2. 新增 prompt observation。
3. 新增 wrist/torso/pelvis sparse rewards。
4. 注册 `Mjlab-Velocity-Prompt-Flat-Unitree-G1`。
5. 训练 flat prompt policy。
6. 写 play/eval 脚本或扩展 viewer GUI，能手动切换 prompt。

成功标志：边走边挥手、边走边弯腰、低身走可见且不显著破坏速度跟踪。

### Milestone B: 参考动作窗口

1. 用关键帧/解析生成器输出 `q_ref`。
2. 新增 residual reference action。
3. 加入 command window observation。
4. 对比 no-reference 和 reference residual。

成功标志：动作更平滑、更像目标姿态，训练速度或稳定性优于 A。

### Milestone C: Tracking motion prior

1. 复用 `MotionLoader` 读取少量 `.npz`。
2. 按 velocity/prompt 检索 motion segment。
3. 用 motion reference 做软 reward 或 q_ref。
4. 增加 MPJPE/MPKPE 风格指标。

成功标志：少量 motion 数据能改善自然性，但 velocity tracking 不下降。

### Milestone D: Motion matching skill composition

1. 为 locomotion 和每个 prompt/skill 建立小型 motion database。
2. 设计 query feature：未来速度轨迹、当前脚/手状态、prompt/contact mode。
3. 实现 nearest-neighbor retrieval 和短窗口 blending。
4. 生成多种 approach distance、速度、方向、相位的训练参考。
5. 对比 uncomposed clips 和 motion-matched trajectories。

成功标志：长时序切换比直接播放孤立 clips 更平滑，prompt/skill 触发时机更稳。

### Milestone E: Dynamics-conditioned actor

1. 保持 PPO，新增 custom actor model。
2. history encoder 编码 proprioception。
3. command encoder 聚合 prompt/reference window。
4. 做 flatten/self-attention/cross-attention ablation。

成功标志：命令噪声、reference discontinuity 下更稳。

### Milestone F: Diffusion / Flow generator

1. 从 Milestone A-C 的 reference/rollout 收集数据。
2. 训练 conditional diffusion generator。
3. 训练 conditional flow matching generator。
4. 接入 frozen generator，做 best-of-N + feasibility filter。
5. 比较 handcrafted / diffusion / flow。

成功标志：生成器提供更丰富动作，同时延迟和稳定性可接受。

### Milestone G: Contact-rich whole-body locomotion

1. 增加 contact body groups 和 mode-dependent contact rewards。
2. 新增 `walk/kneel_walk/crawl/wall_support/box_support` 等 contact modes。
3. 先在无障碍平地上训练 kneel/crawl，再加入墙面和箱体。
4. 接入 raycast/terrain height/camera 等传感器，做 obstacle-conditioned
   prompt。
5. 让 diffusion/flow generator 同时生成姿态轨迹和 contact schedule。

成功标志：策略能在不同 contact mode 下稳定切换，并能利用手、膝、肘、
躯干等接触完成低姿态前进和简单障碍物交互。

## 11. 暂不建议放进第一版的内容

- 完整复刻 BFM-Zero 的 off-policy FB representation。
- 让 diffusion/flow 直接输出低层 action 并替代 PPO policy。
- 一开始就统一倒地起身、爬行、坐地、越障等强接触动作。
- 大规模 motion dataset 清洗和 retargeting。

这些都可能成为后续工作，但会显著拉高最小闭环风险。第一篇论文更适合聚焦：

```text
sparse pose/keyframe prompts
+ velocity-controlled locomotion
+ residual soft-reference tracking
+ optional generative reference model
```

## 12. 参考资料

- BFM-Zero: A Promptable Behavioral Foundation Model for Humanoid Control
  Using Unsupervised Reinforcement Learning:
  https://arxiv.org/abs/2511.04131
- Robust and Generalized Humanoid Motion Tracking:
  https://arxiv.org/pdf/2601.23080
- OmniRetarget: Interaction-Preserving Data Generation for Humanoid Whole-Body
  Loco-Manipulation and Scene Interaction:
  https://arxiv.org/abs/2509.26633
- OmniRetarget project:
  https://omniretarget.github.io/
- Perceptive Humanoid Parkour: Chaining Dynamic Human Skills via Motion Matching:
  https://arxiv.org/abs/2602.15827
- PHP project:
  https://php-parkour.github.io/
- DiffuseLoco project:
  https://diffuselo.co/
- DiffuseLoco code:
  https://github.com/HybridRobotics/DiffuseLoco
- Diffusion Policy code:
  https://github.com/real-stanford/diffusion_policy
- Flow Matching Policy:
  https://github.com/HRI-EU/flow-matching-policy
- FlowPolicy:
  https://github.com/zql-kk/FlowPolicy
- A2A Flow Matching:
  https://github.com/JIAjindou/A2A_Flow_Matching
