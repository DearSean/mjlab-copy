# 最小论文标准实现版：Sparse-Prompted Residual Velocity Control

## 强调说明

这份文档不是新的长期路线图，也不是把
`SPARSE_PROMPT_VELOCITY_CONTROL_PLAN.md` 里的 V0-V3 全部再展开一遍。

这份文档只描述一个目标：

```text
用尽量小但结构化的代码改动，把当前 velocity/recovery 工程堆叠升级为
一套能写进 IROS/CoRL 方法章节的最小论文标准实现。
```

因此第一版刻意不做以下内容：

- 不训练 diffusion 或 flow matching 模型。
- 不引入大规模 motion dataset。
- 不做完整 motion matching skill composition。
- 不做匍匐、越障、手扶墙等强接触任务。
- 不替换 PPO/RSL-RL 训练栈。
- 不重写 manager 层。

第一版只做：

```text
sparse prompt
+ analytic/keyframe soft reference
+ feasibility gate
+ residual reference action
+ adaptive recovery prompt distribution
```

这个版本的目标不是功能最多，而是让所有已有机制都服从同一个研究抽象：

```text
prompt -> soft reference -> feasibility gate -> residual velocity policy
```

只要这个抽象打通，起身、低身走、弯腰、挥手、正常速度跟踪都不再是零散奖励
和课程的拼接，而是同一个 prompt-conditioned controller 的不同实例。

## 论文核心故事

当前 velocity policy 的优势是鲁棒、可控、速度跟踪强；motion tracking policy
的优势是动作表达强，但依赖密集参考轨迹，且容易牺牲行走稳定性。

本实现的论文故事是：

```text
Velocity command 决定机器人去哪里、走多快。
Sparse prompt 决定机器人做什么身体行为。
Prompt reference generator 将稀疏意图转换为软参考。
Feasibility gate 根据当前状态决定软参考应被执行到什么程度。
Residual policy 在软参考附近输出修正动作，保证物理可行。
```

推荐题目：

```text
Sparse-Prompted Residual Velocity Control for Robust and Expressive Humanoid Locomotion
```

若要突出当前 RL_BOY 的倒地恢复，也可以作为副标题或实验亮点：

```text
with Self-Recovery as a Prompt-Conditioned Behavior
```

## 与现有路线图的区别

`SPARSE_PROMPT_VELOCITY_CONTROL_PLAN.md` 是长期路线图，包含 V0/V1/V2/V3、
motion matching、diffusion、flow matching、contact-rich locomotion 和
obstacle-conditioned prompt。

本文档是论文最小实现版，只保留能直接提升论文标准的部分：

| 方向 | 路线图版本 | 本文档版本 |
| --- | --- | --- |
| Sparse prompt | 长期主线之一 | 必做 |
| Keyframe/reference generator | V1 | 必做，但第一版只用解析/少量 keyframe |
| Residual reference action | V1 | 必做 |
| Recovery | 后续可迁移 | 立即统一成 `stand_up` prompt |
| Motion matching | V2 | 只预留接口，不实现 |
| Diffusion/flow | V3 | 只借鉴接口，不训练 |
| Contact-rich | 后续大版本 | 只保留 `contact_mode` 字段 |

## 当前代码基础

### Velocity 基线

主要文件：

- `src/mjlab/tasks/velocity/velocity_env_cfg.py`
- `src/mjlab/tasks/velocity/mdp/velocity_command.py`
- `src/mjlab/tasks/velocity/mdp/rewards.py`
- `src/mjlab/tasks/velocity/mdp/curriculums.py`
- `src/mjlab/tasks/velocity/config/g1/env_cfgs.py`
- `src/mjlab/tasks/velocity/config/rlboy/env_cfgs.py`

已有能力：

- `UniformVelocityCommand` 输出 `[vx, vy, wz]`。
- `track_linear_velocity` 和 `track_angular_velocity` 实现速度追踪。
- `variable_posture` 已有速度相关姿态容忍度。
- G1/RL_BOY 配置已经分别 patch body/site/joint 名称。
- `RewardManager`、`CommandManager`、`CurriculumManager` 已足够支持新设计，不需要改。

### Recovery 相关现状

主要文件：

- `src/mjlab/tasks/velocity/config/rlboy/recovery_assist.py`
- `src/mjlab/tasks/velocity/config/rlboy/env_cfgs.py`

已有能力：

- `RlBoyRecoveryAssist` 管理 fallen reset、辅助力、恢复成功统计、辅助退火。
- `recovery_walk_gate` 已经实现了按高度和倾角连续过渡的 gate。
- `gated_track_linear_velocity`、`gated_track_angular_velocity` 等已经证明 gate
  可以把 recovery 和 velocity tracking 连续连接。

需要改造的点：

```text
不要把 recovery 写成特殊任务分支。
把 recovery 表达为 `stand_up` prompt under fallen initial states。
```

### Tracking 侧可复用能力

主要文件：

- `src/mjlab/tasks/tracking/mdp/commands.py`
- `src/mjlab/tasks/tracking/mdp/rewards.py`
- `src/mjlab/tasks/tracking/mdp/metrics.py`

第一版不直接依赖 `MotionCommand`，但它提供了后续可复用的接口风格：

- motion/reference command term
- body/joint target cache
- MPJPE/MPKPE 风格指标
- reference visualization 思路

## 借鉴的开源项目与只借鉴的部分

这些项目只提供接口和实验设计启发。第一版不 vendor 代码，也不把它们的训练
系统搬进 mjlab。

### DiffuseLoco

仓库：

```text
https://github.com/HybridRobotics/DiffuseLoco
```

公开 README 将其定位为从离线数据训练 diffusion locomotion controller，并提到
receding-horizon control、delayed inputs、uniform observation space 等工程组织。

本项目只借鉴：

- reference/action horizon 的概念。
- 生成器不必每个控制步都运行，可以低频生成 reference window。
- generator 和低层控制器解耦。

第一版不做：

- 不训练 diffusion。
- 不让 diffusion 直接输出低层 action。
- 不依赖离线 locomotion dataset。

### Diffusion Policy

仓库：

```text
https://github.com/real-stanford/diffusion_policy
```

公开 README 里有 demonstration replay buffer、zarr 数据组织、ConditionalUnet1D、
TransformerForDiffusion 等模块说明。

本项目只借鉴：

- horizon-based sequence policy/generator 的接口。
- 后续如果收集 rollout，可使用结构化 replay dataset。
- generator 输出 reference window，而不是直接替代 PPO actor。

第一版不做：

- 不做视觉 imitation。
- 不训练 diffusion policy。

### Flow Matching Policy / FlowPolicy / A2A Flow Matching

仓库：

```text
https://github.com/HRI-EU/flow-matching-policy
https://github.com/zql-kk/FlowPolicy
https://github.com/JIAjindou/A2A_Flow_Matching
```

这些项目说明了 flow/consistency flow 可作为低步数、低延迟生成模型的候选方向。
FlowPolicy README 还强调了单步或少步推理效率。

本项目只借鉴：

- 将 future reference generation 设计成可替换后端。
- 第一版 analytic generator 的输出格式应能被未来 flow generator 复用。

第一版不做：

- 不训练 flow model。
- 不改变当前 actor 网络。

### Perceptive Humanoid Parkour

项目页：

```text
https://php-parkour.github.io/
```

项目页说明其使用 motion matching，把 atomic human skills 组合成长时序轨迹。

本项目只借鉴：

- motion matching 是比 diffusion 更低风险的未来 reference generator 基线。
- skill/prompt 应该有 entry window、transition blending、locomotion manifold。

第一版不做：

- 不实现 nearest-neighbor motion matching。
- 不做 parkour/contact-rich skill chaining。

### OmniRetarget

项目页：

```text
https://omniretarget.github.io/
```

项目页强调 interaction-preserving data generation，显式保持 agent、terrain、
object 之间的空间和接触关系。

本项目只借鉴：

- prompt/reference 结构中预留 `contact_mode` 和 body mask。
- 后续接触任务不能只靠 illegal contact penalty 修补。

第一版不做：

- 不做 object/terrain interaction mesh。
- 不做接触丰富的 loco-manipulation。

## 目标抽象

### Prompt

Prompt 表示稀疏行为意图：

```text
p = {
  prompt_type,
  prompt_weight,
  phase,
  body_mask,
  contact_mode,
}
```

第一版 prompt type：

```text
none
stand_up
low_walk
torso_bend
left_wave
right_wave
both_hands_up
```

第一版 contact mode：

```text
walk
recovery
low_posture
```

`stand_up` 是关键：它把已有 recovery 从特殊工程分支变成 prompt 的一个实例。

### Soft Reference

Reference generator 输出：

```text
q_ref:              [B, num_joints]
joint_mask:         [B, num_joints]
body_targets:       sparse targets, e.g. wrist height / torso pitch / base height
prompt_confidence:  [B]
```

第一版 `q_ref` 可以全是 default pose，只对上身/躯干关节加少量 delta。
对 `stand_up`，可以不设置完整 `q_ref`，只设置 base height / upright sparse
target，让策略自己找起身动作。

### Feasibility Gate

把当前 `recovery_walk_gate` 泛化为：

```text
confidence = feasibility_gate(s, p)
```

直观解释：

- 机器人倒地时，velocity tracking 不应强行主导。
- 机器人未站稳时，挥手和弯腰 prompt 不应强行执行。
- 机器人重新达到可行姿态后，velocity 和 expressive prompt 逐渐淡入。

### Residual Reference Action

当前 `JointPositionAction` 是：

```text
q_target = q_default + policy_action * scale
```

新 action term 是：

```text
q_target = q_ref(prompt) + policy_residual * scale
```

无 prompt 时：

```text
q_ref = q_default
```

所以它退化成当前 velocity baseline。

有 prompt 时：

- 腿部仍主要由 residual policy 保持 locomotion。
- 上身/躯干可围绕 prompt reference 做动作表达。
- `joint_mask` 决定哪些关节受 prompt reference 影响。

## 最小实现模块

### 1. Prompt command

新增文件：

```text
src/mjlab/tasks/velocity/mdp/prompt_command.py
```

建议类：

```python
class SparsePromptCommand(CommandTerm):
  @property
  def command(self) -> torch.Tensor:
    ...

@dataclass(kw_only=True)
class SparsePromptCommandCfg(CommandTermCfg):
  prompt_types: tuple[str, ...]
  prompt_probabilities: tuple[float, ...]
  prompt_weight_choices: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
  phase_speed_range: tuple[float, float] = (0.5, 1.5)
```

Command tensor 第一版：

```text
[
  prompt_type_onehot,
  contact_mode_onehot,
  prompt_weight,
  phase,
  torso_pitch_target,
  base_height_target,
  left_wrist_height_target,
  right_wrist_height_target,
]
```

不要把 `twist` 合并进 prompt command。保持：

```text
commands:
  twist
  prompt
```

原因：

- velocity baseline 可比。
- 现有 velocity rewards 不需要改。
- 后续可单独做 prompt ablation。

### 2. Prompt reference

新增文件：

```text
src/mjlab/tasks/velocity/mdp/prompt_reference.py
```

建议 API：

```python
def get_prompt_reference(
  env,
  command_name: str,
  asset_cfg: SceneEntityCfg,
) -> PromptReference:
  ...
```

第一版不一定需要 dataclass，也可以返回 dict，但推荐内部使用清晰字段：

```text
q_ref
joint_mask
base_height_target
torso_pitch_target
left_wrist_height_target
right_wrist_height_target
prompt_weight
confidence
```

解析规则示例：

```text
none:
  q_ref = default
  sparse targets disabled

left_wave:
  left wrist height = nominal + amplitude * sin(phase)
  upper-body joint deltas optional
  leg joint mask = 0

both_hands_up:
  left/right wrist target high
  upper-body joint deltas optional

torso_bend:
  torso pitch target positive
  base height lower bound stays safe

low_walk:
  base height target lower
  torso pitch mild forward
  prompt weight capped by stability confidence

stand_up:
  base height target = upright height
  upright target enabled
  velocity prompt confidence starts low and rises with recovery
```

### 3. Prompt observations

在 prompt task cfg 中加入：

```python
cfg.observations["actor"].terms["prompt"] = ObservationTermCfg(
  func=mdp.generated_commands,
  params={"command_name": "prompt"},
)
cfg.observations["critic"].terms["prompt"] = ObservationTermCfg(
  func=mdp.generated_commands,
  params={"command_name": "prompt"},
)
```

第一版不需要加入完整 `q_ref` window 到 observation。先让 policy 看到 sparse
prompt 即可。

如果 residual action 需要 `q_ref`，action term 可直接从 command/reference helper
取，不一定作为 observation 显式输入。

### 4. Prompt rewards and metrics

新增文件：

```text
src/mjlab/tasks/velocity/mdp/prompt_rewards.py
```

建议 reward：

```text
prompt_sparse_body_tracking
masked_joint_reference_tracking
prompt_feasibility_penalty
prompt_success_metric
```

第一版 priority：

1. `prompt_sparse_body_tracking`
2. `prompt_success_metric`
3. `masked_joint_reference_tracking`

不要一开始让 masked joint reward 太强。论文故事是 sparse prompt，不是 dense
motion tracking。

推荐形式：

```text
r_prompt = prompt_weight * confidence * exp(-target_error / std^2)
```

其中 `confidence` 来自 feasibility gate。

### 5. Feasibility gate

新增或迁移文件：

```text
src/mjlab/tasks/velocity/mdp/feasibility.py
```

从 RL_BOY 的 `recovery_walk_gate` 抽出通用逻辑。

建议函数：

```python
def locomotion_readiness_gate(
  env,
  height_low: float,
  height_high: float,
  tilt_low: float,
  tilt_high: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  ...

def prompt_feasibility_gate(
  env,
  command_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  ...
```

第一版可以只用 base height + projected gravity：

```text
height_gate = smoothstep(base_height, height_low, height_high)
upright_gate = 1 - smoothstep(tilt, tilt_low, tilt_high)
confidence = height_gate * upright_gate
```

`stand_up` prompt 使用 `1 - confidence` 加权 recovery target。
walking/expressive prompt 使用 `confidence` 加权 prompt tracking。

### 6. Residual reference action

新增文件：

```text
src/mjlab/tasks/velocity/mdp/prompt_actions.py
```

建议类：

```python
class ReferenceResidualJointPositionAction(JointPositionAction):
  def process_actions(self, actions):
    residual = actions * scale
    q_ref = get_prompt_reference(...).q_ref
    target = q_ref + residual
```

或者不继承 `JointPositionAction`，直接继承 `BaseAction`，更清晰。

第一版配置：

```python
cfg.actions["joint_pos"] = ReferenceResidualJointPositionActionCfg(
  entity_name="robot",
  actuator_names=(".*",),
  scale=...,
  command_name="prompt",
  use_default_reference_when_inactive=True,
)
```

需要注意：

- `q_ref` 必须按 action target joint order 对齐。
- 对未被 prompt mask 命中的关节，`q_ref = default_joint_pos`。
- 对腿部 reference 权重保持 0 或很小。
- clip 仍然要保留，避免 prompt 目标超限。

### 7. Prompt task cfg

新增文件：

```text
src/mjlab/tasks/velocity/config/g1/prompt_env_cfgs.py
src/mjlab/tasks/velocity/config/g1/prompt_rl_cfg.py
```

新增任务 ID：

```text
Mjlab-Velocity-Prompt-Flat-Unitree-G1
```

推荐先只做 G1 flat，因为：

- G1 配置更干净。
- flat terrain 降低变量。
- prompt 行为容易观察。

稳定后再迁移到：

```text
Mjlab-Velocity-Prompt-Flat-RLBOY
Mjlab-Velocity-Prompt-Recovery-RLBOY
```

### 8. Recovery prompt integration

保留：

```text
src/mjlab/tasks/velocity/config/rlboy/recovery_assist.py
```

但重命名论文概念：

```text
RlBoyRecoveryAssist -> adaptive stand_up prompt sampler / assistance scheduler
```

代码可以暂时不改类名，只在文档、日志和 config helper 中改表达。

最小改动：

- 在 curriculum state 中增加 prompt 视角日志：
  - `stand_up_probability`
  - `stand_up_success_rate`
  - `assist_force_min_n`
  - `assist_force_max_n`
  - `active_recovery_fraction`
- 将 recovery reward 命名解释为 `stand_up prompt reward`。
- 将 recovery success/timeout 解释为 prompt episode outcome。

## 文件改动清单

最小新增：

```text
src/mjlab/tasks/velocity/mdp/prompt_command.py
src/mjlab/tasks/velocity/mdp/prompt_reference.py
src/mjlab/tasks/velocity/mdp/prompt_rewards.py
src/mjlab/tasks/velocity/mdp/feasibility.py
src/mjlab/tasks/velocity/mdp/prompt_actions.py
src/mjlab/tasks/velocity/config/g1/prompt_env_cfgs.py
src/mjlab/tasks/velocity/config/g1/prompt_rl_cfg.py
tests/test_velocity_prompt_command.py
tests/test_velocity_prompt_rewards.py
tests/test_velocity_prompt_task.py
```

最小修改：

```text
src/mjlab/tasks/velocity/mdp/__init__.py
src/mjlab/tasks/velocity/config/g1/__init__.py
src/mjlab/tasks/velocity/config/rlboy/recovery_assist.py
src/mjlab/tasks/velocity/config/rlboy/env_cfgs.py
```

建议后续再拆：

```text
src/mjlab/tasks/velocity/config/rlboy/recovery_objective.py
```

把目前 `rlboy/env_cfgs.py` 中的 gated reward helpers 移出去，减少 config 文件
的“工程堆叠感”。

## 数据需求

第一版不需要大量运动数据。

推荐数据量：

```text
none: no data
stand_up: existing RL_BOY fallen poses / optional existing getup CSV
left_wave/right_wave: 2-5 hand-written upper-body keyframes each
both_hands_up: 1-3 keyframes
torso_bend: analytic target only
low_walk: analytic target only
```

可选文件：

```text
src/mjlab/tasks/velocity/config/g1/prompts/prompt_specs.json
src/mjlab/tasks/velocity/config/g1/prompts/keyframes.json
```

如果不想第一版引入 JSON loader，也可以先把 prompt specs 写成 Python dataclass
常量。论文标准更看重统一接口，不要求第一版数据格式复杂。

## 训练与验证

### 最小训练

```sh
uv run train Mjlab-Velocity-Prompt-Flat-Unitree-G1 \
  --env.scene.num-envs 1024 \
  --agent.max-iterations 2000
```

稳定后：

```sh
uv run train Mjlab-Velocity-Prompt-Flat-Unitree-G1 \
  --env.scene.num-envs 4096
```

### 必跑测试

```sh
uv run pytest tests/test_velocity_prompt_command.py \
  tests/test_velocity_prompt_rewards.py \
  tests/test_velocity_prompt_task.py
uv run ruff format
uv run ruff check --fix
```

如果改了 shared MDP/action 逻辑，再跑：

```sh
uv run pytest tests/test_action_manager.py tests/test_manager_config_immutability.py
```

## 验收标准

### 行为验收

- `none` prompt 下，速度跟踪接近原始 velocity baseline。
- `left_wave/right_wave` 下，手臂动作可见，速度不崩。
- `both_hands_up` 下，双腕高度明显上升，机器人仍能保持站立或低速行走。
- `torso_bend` 下，躯干 pitch 有响应，fall rate 不明显升高。
- `low_walk` 下，base height 有连续降低，速度跟踪仍可接受。
- `stand_up` 下，fallen initial states 的恢复成功率高于无 prompt/gate baseline。
- prompt weight 从 `0 -> 1` 时，动作幅度连续变化。

### 指标验收

必须记录：

```text
velocity_rmse
yaw_rmse
prompt_error
prompt_success
effective_prompt_weight
fall_rate
self_collision_count
action_rate
torque_saturation or torque_excess
recovery_success_rate
recovery_time
```

`prompt_success` 不应只看 prompt error，而应在速度误差未超阈值时计算：

```text
prompt_success = prompt_error < eps_prompt AND velocity_error < eps_velocity
```

这样能证明策略不是通过牺牲 locomotion 来完成 prompt。

## Ablation 设计

建议论文主表：

```text
B0: 当前 velocity baseline
B1: velocity + sparse prompt reward, no reference action
B2: B1 + feasibility gate
B3: B2 + residual reference action
B4: B3 + adaptive stand_up prompt sampler
```

每个 baseline 的意义：

- `B0` 证明原始速度能力。
- `B1` 证明单纯加 prompt reward 不够稳定，容易像工程堆叠。
- `B2` 证明 feasibility gate 对稳定性和恢复有用。
- `B3` 证明 residual reference action 提升动作表达与训练效率。
- `B4` 证明 recovery 可以被统一到 prompt-conditioned training distribution。

辅助 ablation：

```text
prompt weight: 0, 0.25, 0.5, 1.0
upper-body mask vs full-body mask
analytic sparse target vs keyframe q_ref
with vs without recovery prompt sampler
flat vs rough terrain
```

## 代码风格约束

- 不改 manager 层，除非测试证明现有接口无法表达。
- Prompt 相关逻辑放在 `velocity/mdp/`，不要塞进 robot-specific cfg。
- Robot-specific body/joint/site 名称留在 `config/g1/` 或 `config/rlboy/`。
- Gated reward helper 不要长期留在 `env_cfgs.py`。
- 第一版不要加复杂模型依赖。
- 第一版不要把 prompt command 和 velocity command 混成一个巨大 command。

## 推荐提交顺序

### PR 1: Prompt command and task shell

- 新增 `SparsePromptCommand`。
- 新增 prompt observation。
- 注册 `Mjlab-Velocity-Prompt-Flat-Unitree-G1`。
- 不改 action，不加复杂 reward。

验收：

- 环境能创建。
- command 输出维度正确。
- `none` prompt 行为等价 velocity baseline。

### PR 2: Prompt reference and sparse rewards

- 新增 analytic prompt reference。
- 新增 wrist/base/torso sparse rewards。
- 新增 prompt success metrics。

验收：

- reward 接近目标时更高。
- prompt weight 控制 reward 强度。

### PR 3: Feasibility gate

- 抽出 `feasibility.py`。
- 用 gate 调制 prompt reward。
- 把 recovery gate 的论文表达统一到 feasibility gate。

验收：

- 不可行状态下 prompt 不强行拉动作。
- gate 日志可画曲线。

### PR 4: Residual reference action

- 新增 `ReferenceResidualJointPositionAction`。
- prompt task 切换到 reference residual action。
- 做 B1/B2/B3 对比。

验收：

- 无 prompt 退化为 baseline action。
- 有 prompt 动作表达更明显或训练更快。

### PR 5: Recovery prompt integration

- 将 RL_BOY recovery 以 `stand_up` prompt 形式组织。
- 增加 recovery prompt logs。
- 做 B4 实验。

验收：

- 起身和速度跟踪不再是两个独立叙事。
- 论文图表能展示 adaptive prompt distribution 和 gate transition。

## 论文写法提示

方法章节可以按这个顺序写：

1. Problem formulation: velocity command + sparse prompt。
2. Sparse prompt reference generation。
3. State-dependent feasibility gate。
4. Residual reference action policy。
5. Adaptive prompt distribution for self-recovery。

不要按代码模块写：

```text
reward A
reward B
curriculum C
event D
```

而要按统一控制流写：

```text
prompt -> reference -> gate -> residual control -> adaptive distribution
```

这就是本实现版提升论文标准的核心。

## 以后如何升级

第一版完成后，外部项目的借鉴可以这样落地：

- DiffuseLoco / Diffusion Policy:
  将 analytic reference generator 替换成 frozen diffusion reference generator。
- FlowPolicy / Flow Matching Policy:
  将 generator 替换成少步 flow reference generator，比较延迟和动作质量。
- PHP:
  用 motion matching 生成 reference window，作为 diffusion/flow 前的低数据强基线。
- OmniRetarget:
  扩展 `contact_mode`，让 prompt reference 包含目标接触部位、接触点和环境相对几何。

这些都是第二篇或后续大版本。第一篇的最小论文标准实现不依赖它们。

