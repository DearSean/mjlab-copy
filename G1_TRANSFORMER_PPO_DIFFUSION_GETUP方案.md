# G1 RGMT–SMP–PPO 起身训练详细方案

## 1. 方案目标

本方案在当前 mjlab 项目中训练 Unitree G1 从随机倒地状态自主恢复到 velocity
环境默认屈膝站姿。`lafan-g1-getup/` 中的 G1 动作片段用于学习支撑、翻转、
抬升和姿态转换；PPO 在 MuJoCo 接触动力学中将这些局部能力组合成完整起身。

采用的方法包括：

1. [RGMT](https://arxiv.org/html/2601.23080v1) 的本体历史 Transformer、
   动力学条件参考窗口注意力、关节残差控制和非对称 Actor–Critic；
2. [SMP](https://arxiv.org/html/2512.03028v3) 的冻结动作 Diffusion、
   Ensemble Score Matching 奖励和 Reference State Initialization；
3. [Extreme-RGMT](https://arxiv.org/html/2607.20110v1) 的困难时间窗自适应
   采样。

系统链路：

```text
G1 动作片段
  ├── 参考状态初始化
  ├── 21 步参考命令窗口 ───────────┐
  └── 10 步动作窗口 → SMP Diffusion ─┤ 训练时冻结奖励
                                      ↓
机器人最近 10 步本体状态 → Transformer Actor → 29 维关节目标
随机倒地状态 ────────────────────────────┘
默认屈膝站姿 → 站立任务奖励 ──────────────┘
                                      ↓
                            MuJoCo 接触动力学
                                      ↓
                         稳定站立后接管 velocity
```

不完整轨迹可以提供任意合法的局部参考窗口和 SMP 动作窗口，不要求每条轨迹都
包含最终站立尾段。完整起身方向由默认站姿奖励给出，连续性由共享反馈策略和
仿真状态转移产生。

## 2. 公式阅读约定

文档中的公式使用普通文本书写。以下函数含义在全文保持一致：

```text
clip(x, lower, upper)
  把 x 限制在 [lower, upper] 内。

mean(x)
  x 所有元素的平均值。

mean_square(x)
  x 所有元素平方后的平均值。

RMS(x)
  √mean_square(x)。

norm(x)
  x 的欧氏长度，也就是 √Σ(x_i²)。

indicator(condition)
  条件成立时为 1，否则为 0。

smoothstep(x)
  3x² - 2x³，其中 x 已限制在 [0, 1]。

exp(x)
  自然指数函数 eˣ。

stop_gradient(x)
  x 参与数值计算，但不向其来源模型传播梯度。
```

时间下标写成 `变量(t)`；reference、default、target 分别简写为 `ref`、`0`、
`tar`。

## 3. 任务定义

### 3.1 仿真状态与动作

一个 MuJoCo 状态写成：

```text
state(t) = {
  pelvis 世界位置和旋转，
  pelvis 线速度和角速度，
  29 维关节位置 q(t)，
  29 维关节速度 dq(t)，
  link 状态和接触信息
}
```

策略动作是 29 维归一化关节残差：

```text
action(t) ∈ [-1, 1]²⁹
```

PPO 最大化折扣累计回报：

```text
单条轨迹回报
  = reward(0)
  + γ × reward(1)
  + γ² × reward(2)
  + ...

训练目标 J
  = 所有策略采样轨迹回报的平均值
```

第一版控制频率为 50 Hz：

```text
control_dt = 1 / 50 = 0.02 秒
```

### 3.2 默认站姿

`q0` 是 G1 velocity 配置的 `KNEES_BENT_KEYFRAME`，默认 pelvis 高度：

```text
h0 = 0.76 米
```

MuJoCo 中直立时，机身坐标系的投影重力约为 `(0, 0, -1)`。定义直立度：

```text
upright(t) = clip(-projected_gravity_z(t), 0, 1)

完全直立时 upright ≈ 1
水平倒地时 upright ≈ 0
```

### 3.3 稳定站立集合

状态同时满足以下条件时，记为 `is_standing(t) = 1`：

```text
pelvis_height(t) ≥ 0.85 × h0
upright(t)       ≥ 0.90
norm(base_linear_velocity)  ≤ 0.20 米/秒
norm(base_angular_velocity) ≤ 0.40 弧度/秒
RMS(joint_velocity)         ≤ 0.50 弧度/秒
左脚稳定接触地面
右脚稳定接触地面
没有手、膝、躯干或头部持续承重
RMS((q(t) - q0) / joint_pose_scale) ≤ pose_error_limit
```

最终阈值通过默认站姿扰动评测集校准。成功必须连续保持 0.5 秒：

```text
hold_steps = ceil(0.5 × 50) = 25

success(t) = 过去连续 25 步的 is_standing 全部等于 1
```

### 3.4 Episode 模式

所有模式共享一个 Actor：

| 模式 | 初始状态 | 参考窗口 | 训练目的 |
|---|---|---|---|
| reference | 动作片段中的安全帧 | 有 | 学习局部支撑和姿态转换 |
| fallen | 随机物理倒地状态 | 无，使用 null command | 学习完整自主恢复 |
| stand | 默认姿态附近的扰动状态 | 无，使用默认姿态 | 学习站立吸引域和接管 |

Actor 额外接收三维 one-hot mode。部署起身时使用 fallen mode；稳定站立后进入
stand/velocity mode。

## 4. 动作数据编译

### 4.1 数据字段

每条编译轨迹至少包含：

```text
clip_id
source_recording_id
subject_id
split
source_fps
target_fps = 50
valid_length
root_position
root_quaternion_xyzw
joint_position[29]
support_start
support_complete
success
terminal_type
source_file_sha256
feature_schema_version
```

源帧率必须由 manifest 或命令行明确给出，不能依靠隐式默认值。

### 4.2 时间重采样

源帧率为 `source_fps`，目标帧率为 50 Hz：

```text
source_time(n) = n / source_fps
target_time(k) = k / 50
```

若 `target_time(k)` 落在源帧 n 和 n+1 之间：

```text
ratio
  = (target_time(k) - source_time(n))
    / (source_time(n+1) - source_time(n))

q_target(k)
  = (1 - ratio) × q_source(n)
    + ratio × q_source(n+1)
```

root quaternion 使用最短弧球面插值：

```text
root_rotation_target(k)
  = SLERP(root_rotation(n), root_rotation(n+1), ratio)
```

### 4.3 速度计算

内部帧使用中心差分：

```text
joint_velocity(k)
  = (q(k+1) - q(k-1)) / (2 × control_dt)

root_linear_velocity(k)
  = (root_position(k+1) - root_position(k-1))
    / (2 × control_dt)
```

root 角速度由相邻旋转的 rotation log 得到：

```text
relative_rotation
  = transpose(root_rotation(k-1)) × root_rotation(k+1)

root_angular_velocity(k)
  = rotation_log(relative_rotation) / (2 × control_dt)
```

首尾帧使用单边差分。任何差分都不能越过 clip 边界。

### 4.4 机身局部坐标

设 `R_body(t)` 为 pelvis 世界旋转矩阵：

```text
body_linear_velocity(t)
  = transpose(R_body(t)) × world_linear_velocity(t)

body_angular_velocity(t)
  = transpose(R_body(t)) × world_angular_velocity(t)

projected_gravity(t)
  = transpose(R_body(t)) × (0, 0, -1)

body_endpoint_position(t, joint)
  = transpose(R_body(t))
    × (world_joint_position(t) - world_pelvis_position(t))
```

### 4.5 特征标准化

每一维只使用 train split 估计均值和标准差：

```text
normalized_feature(d)
  = (feature(d) - train_mean(d))
    / max(train_std(d), 1e-6)
```

validation 和 test 只能读取 train statistics。

### 4.6 Source-group split

同一原始 recording 的所有 subject 和裁剪片段进入同一 split。确定性哈希：

```text
bucket = hash(source_recording_id, seed) mod 100

bucket < 80       → train
80 ≤ bucket < 90  → validation
bucket ≥ 90       → test
```

由于当前 source 数量较少，最终允许在 manifest 中固定分配，但不能按单帧随机
划分。

## 5. 参考命令窗口

### 5.1 单帧 38 维命令

采用 RGMT 的命令定义：

```text
command(t) = concat(
  reference_body_linear_velocity[3],
  reference_body_angular_velocity[3],
  reference_projected_gravity[3],
  reference_joint_position[29]
)

command_dimension = 3 + 3 + 3 + 29 = 38
```

### 5.2 21 步上下文

当前参考时刻为 t，使用前后各 10 步：

```text
command_window(t)
  = [command(t-10), ..., command(t), ..., command(t+10)]

window_shape = 21 × 38
```

每个 token 有一个有效标志：

```text
valid_mask(j) = 1  → token 在当前 clip 内
valid_mask(j) = 0  → token 越过 clip 边界，只是 padding
```

Attention 和 tracking reward 都忽略无效 token。

### 5.3 Partial 尾端平滑退场

`end_time` 是片段最后有效时间，退场时间 `blend_time = 0.4 秒`：

```text
remaining_ratio(t)
  = clip((end_time - current_time) / blend_time, 0, 1)

reference_blend(t)
  = smoothstep(remaining_ratio(t))

reference_confidence(t)
  = indicator(mode == reference) × reference_blend(t)

joint_anchor(t)
  = reference_blend(t) × q_ref(t)
    + (1 - reference_blend(t)) × q0
```

该插值只用于控制目标连续化，不写入动作数据，也不进入 SMP 正样本。

fallen 和 stand mode 始终使用：

```text
reference_confidence = 0
joint_anchor = q0
```

## 6. RGMT Transformer Actor

### 6.1 93 维本体观测

```text
observation(t) = concat(
  projected_gravity[3],
  base_angular_velocity[3],
  joint_position - q0[29],
  joint_velocity[29],
  previous_action[29]
)

observation_dimension = 3 + 3 + 29 + 29 + 29 = 93
```

Actor 接收带噪观测，Critic 接收无噪声观测。最终噪声量级：

```text
projected_gravity_noise  ~ Uniform(-0.05, 0.05)
angular_velocity_noise   ~ Uniform(-0.20, 0.20) 弧度/秒
joint_position_noise     ~ Uniform(-0.01, 0.01) 弧度
joint_velocity_noise     ~ Uniform(-0.50, 0.50) 弧度/秒
```

### 6.2 最近 10 步历史

```text
history(t)
  = [observation(t-9), ..., observation(t)]

history_shape = 10 × 93
```

不足 10 步时复制 reset 后首帧，并用 history mask 标记，避免把全零解释成真实
运动。

### 6.3 Causal history encoder

每个观测先映射到 128 维 token：

```text
E
  = MLP_history(history)
    + history_positional_encoding

E_shape = 10 × 128
```

一个 causal Transformer block：

```text
H1
  = E
    + causal_multihead_attention(layer_norm(E))

H2
  = H1
    + feedforward_MLP(layer_norm(H1))

H_normalized
  = layer_norm(H2)
```

Causal mask 保证第 i 个时间 token 只能读取第 0 到 i 个 token。按 RGMT 对时间
维做逐通道最大池化：

```text
dynamics_embedding(d)
  = max over time [H_normalized(time, d)]

dynamics_embedding_dimension = 128
```

### 6.4 Dynamics-conditioned cross-attention

动力学 query：

```text
Q = MLP_dynamics(dynamics_embedding)
Q_shape = 1 × 128
```

参考命令 token：

```text
Z
  = MLP_command(command_window)
    + command_positional_encoding

Z_shape = 21 × 128
```

每个 attention head 对第 j 个命令 token 计算：

```text
raw_score(j)
  = dot(Q, key(j)) / √head_dimension

masked_score(j)
  = raw_score(j)       当 valid_mask(j) = 1
  = negative_infinity 当 valid_mask(j) = 0

attention_weight
  = softmax(masked_score)

attention_output
  = Σ [attention_weight(j) × value(j)]
```

命令序列后增加一个始终有效的 null token，防止全部参考 token 无效时出现 NaN：

```text
null_command = concat(
  zero_linear_velocity[3],
  zero_angular_velocity[3],
  upright_projected_gravity = (0, 0, -1),
  default_joint_position = q0
)
```

Cross-attention block：

```text
cross_1
  = Q
    + masked_multihead_attention(layer_norm(Q), Z)

cross_2
  = cross_1
    + feedforward_MLP(layer_norm(cross_1))

command_embedding
  = layer_norm(cross_2)
```

### 6.5 Actor 输出

```text
actor_input = concat(
  current_observation,
  dynamics_embedding,
  command_embedding,
  reference_confidence,
  episode_mode_one_hot
)
```

Actor MLP 使用 `[512, 256, 128]`，ELU 激活，输出 29 维 Gaussian mean。

```text
sampled_action
  ~ Normal(actor_mean, actor_standard_deviation)

action
  = clip(sampled_action, -1, 1)
```

部署时使用 `actor_mean`。

### 6.6 关节目标和 PD

`action_scale` 使用项目现有的 G1 逐关节尺度：

```text
joint_target(t)
  = joint_anchor(t)
    + action_scale × action(t)

requested_torque(t)
  = Kp × (joint_target(t) - joint_position(t))
    - Kd × joint_velocity(t)
```

Actuator model 再应用力矩、速度和饱和限制。训练同时记录 requested torque 与
applied torque。

### 6.7 Asymmetric Critic

Critic 输入包括：

```text
无噪声 93D 本体观测
参考命令和 valid mask
pelvis 和各 link 的位置、旋转、速度
接触力和接触对象
requested/applied torque
episode mode
reference height
```

```text
value(t) = Critic_MLP(privileged_state(t))
```

Critic MLP 建议 `[1024, 512, 256]`。特权信息不进入 Actor 和导出模型。

## 7. Reset 与探索课程

### 7.1 三类 reset 概率

训练进度：

```text
progress(iteration)
  = clip(iteration / curriculum_iterations, 0, 1)
```

概率从早期值平滑变化到后期值：

| 模式 | 早期概率 | 后期概率 |
|---|---:|---:|
| reference | 0.80 | 0.30 |
| fallen | 0.10 | 0.60 |
| stand | 0.10 | 0.10 |

```text
probability(mode, iteration)
  = (1 - progress) × early_probability(mode)
    + progress × late_probability(mode)
```

课程还必须满足固定评测集成功率阈值，不能只按 iteration 推进。

### 7.2 Reference State Initialization

```text
joint_position
  = reference_joint_position
    + Uniform(-joint_noise_limit, joint_noise_limit)

joint_velocity
  = reference_joint_velocity
    + Uniform(-velocity_noise_limit, velocity_noise_limit)

root_rotation
  = reference_root_rotation
    × rotation_from_small_axis_angle_noise
```

扰动后执行 joint-limit clip、MuJoCo forward 和碰撞合法性检查。严重穿透、NaN、
速度爆炸和 terminal-unsafe 帧重新采样。

### 7.3 Random fallen initialization

随机倒地来自：

1. 安全参考帧或默认站姿加大扰动后自由落地并 settle；
2. prone、supine、left-side、right-side、kneeling、seated 参数化模板。

```text
root_yaw ~ Uniform(-π, π)

accept_fallen_state
  = pelvis_height < 0.80 × h0
    OR upright < 0.75
```

拒绝关节硬超限、NaN 和深度穿透状态。固定 validation/test fallen bank 与训练
reset 分开保存。

### 7.4 Stand-noise initialization

```text
joint_position
  = q0 + Uniform(-joint_noise, joint_noise)

pelvis_height
  = h0 + Uniform(-height_noise, height_noise)

joint_velocity
  ~ Uniform(-joint_velocity_noise, joint_velocity_noise)
```

同时对 pelvis roll/pitch 和 base velocity 添加小扰动。关节噪声首版取对应
action scale 的 20%–40%。

### 7.5 退火向上辅助力

```text
assist_decay(iteration)
  = max(1 - iteration / assist_iterations, 0)

assist_force_z
  = Bernoulli(assist_probability)
    × Uniform(0, max_assist_force)
    × assist_decay
```

`max_assist_force = 200 N` 作为 RGMT 起始量级，但必须按 G1 质量和最大允许加速度
限幅。最终训练和全部评测中辅助力为 0。

### 7.6 终止条件

Recovery episode 中低高度、手膝接地和大倾角不是早停条件。仅在以下情况终止：

- 已连续稳定站立 0.5 秒；
- 达到 4–6 秒 recovery timeout；
- NaN、仿真发散或严重关节硬超限；
- 超过安全碰撞冲量；
- root 离开训练区域。

## 8. Reference tracking reward

### 8.1 Keypoint 位置

Keypoint 集合包括 pelvis、torso、head、双手、双脚和双膝。全部使用 pelvis
局部坐标。

```text
keypoint_position_error(t)
  = mean over keypoints [
      norm(
        simulated_body_position(t, keypoint)
        - reference_body_position(t, keypoint)
      )²
    ]

keypoint_position_reward(t)
  = exp(-keypoint_position_gain × keypoint_position_error(t))
```

### 8.2 关节姿态

```text
joint_pose_error(t)
  = mean over 29 joints [
      ((q(t) - q_ref(t)) / joint_pose_scale)²
    ]

joint_pose_reward(t)
  = exp(-joint_pose_gain × joint_pose_error(t))
```

### 8.3 Keypoint 速度

```text
keypoint_velocity_error(t)
  = mean over keypoints [
      norm(
        simulated_body_velocity(t, keypoint)
        - reference_body_velocity(t, keypoint)
      )²
    ]

keypoint_velocity_reward(t)
  = exp(-keypoint_velocity_gain × keypoint_velocity_error(t))
```

### 8.4 合成 tracking reward

```text
tracking_reward(t)
  = reference_confidence(t)
    × (
        0.50 × keypoint_position_reward(t)
        + 0.30 × joint_pose_reward(t)
        + 0.20 × keypoint_velocity_reward(t)
      )
```

partial 尾部由 `reference_confidence` 连续退场。

## 9. 起身任务奖励

### 9.1 五个站立分数

直立分数：

```text
upright_reward(t)
  = exp(-upright_gain × (1 - upright(t))²)
```

高度分数：

```text
relative_height_error(t)
  = (pelvis_height(t) - h0) / h0

height_reward(t)
  = exp(-height_gain × relative_height_error(t)²)
```

默认关节姿态分数：

```text
default_pose_error(t)
  = mean over joints [
      ((q(t) - q0) / joint_pose_scale)²
    ]

default_pose_reward(t)
  = exp(-default_pose_gain × default_pose_error(t))
```

低速度分数：

```text
motion_energy(t)
  = norm(base_linear_velocity)²
    + angular_velocity_weight × norm(base_angular_velocity)²
    + joint_velocity_weight × RMS(joint_velocity)²

still_reward(t)
  = exp(-still_gain × motion_energy(t))
```

双脚稳定支撑分数：

```text
foot_slip_speed²
  = norm(left_foot_horizontal_velocity)²
    + norm(right_foot_horizontal_velocity)²

feet_reward(t)
  = indicator(left_foot_contact AND right_foot_contact)
    × exp(-foot_slip_gain × foot_slip_speed²)
```

### 9.2 站立势函数

```text
standing_potential(t)
  = 0.25 × upright_reward(t)
    + 0.25 × height_reward(t)
    + 0.15 × default_pose_reward(t)
    + 0.15 × still_reward(t)
    + 0.20 × feet_reward(t)
```

权重首版和为 1，随后通过消融调整。

### 9.3 净进展奖励

```text
raw_progress(t)
  = γ × standing_potential(t+1)
    - standing_potential(t)

progress_reward(t)
  = clip(raw_progress(t), -progress_clip, progress_clip)
```

该项奖励向站立吸引域的净进展，并惩罚重新跌落。

### 9.4 成功保持和低位抬升

```text
hold_reward(t) = is_standing(t)
```

低位阶段增加短期抬升引导：

```text
lift_reward(t)
  = indicator(pelvis_height(t) < 0.70 × h0)
    × clip(vertical_base_velocity / target_lift_velocity, -1, 1)
```

`lift_reward` 权重随课程衰减，防止策略反复上下运动刷奖励。

任务奖励：

```text
task_reward(t)
  = potential_weight × standing_potential(t)
    + progress_weight × progress_reward(t)
    + hold_weight × hold_reward(t)
    + lift_weight(iteration) × lift_reward(t)
```

## 10. SMP Diffusion motion prior

### 10.1 G1 的 51 维动作特征

```text
smp_feature(t) = concat(
  pelvis_height / h0[1],
  projected_gravity[3],
  body_linear_velocity[3],
  body_angular_velocity[3],
  normalized_joint_position[29],
  left_hand_body_position[3],
  right_hand_body_position[3],
  left_foot_body_position[3],
  right_foot_body_position[3]
)

smp_feature_dimension
  = 1 + 3 + 3 + 3 + 29 + 12
  = 51
```

关节归一化：

```text
normalized_joint_position(j)
  = (q(j) - q0(j)) / joint_pose_scale(j)
```

动作短窗：

```text
smp_window(t)
  = [smp_feature(t-9), ..., smp_feature(t)]

smp_window_shape = 10 × 51
```

窗口必须完全位于同一 clip 内。所有合法 partial 和完整轨迹窗口都可使用。

### 10.2 Diffusion 前向加噪

总 diffusion timestep 数量为 50。对第 i 个噪声层级：

```text
alpha(i) = 1 - beta(i)

alpha_bar(i)
  = alpha(1) × alpha(2) × ... × alpha(i)
```

给干净窗口和高斯噪声加噪：

```text
noise ~ Normal(0, identity)

noised_window(i)
  = √alpha_bar(i) × clean_window
    + √(1 - alpha_bar(i)) × noise
```

### 10.3 Diffusion 训练损失

Denoiser 输入 `noised_window` 和 timestep i，输出预测噪声：

```text
predicted_noise
  = denoiser(noised_window(i), timestep=i)

diffusion_loss
  = mean_square(noise - predicted_noise)
```

模型结构：

```text
2 个 Transformer encoder blocks
4 个 attention heads
embedding dimension = 128
FFN dimension = 512
timestep embedding 通过 adaptive LayerNorm 注入
```

参数 EMA：

```text
ema_parameter
  = 0.999 × previous_ema_parameter
    + 0.001 × current_train_parameter
```

PPO 只读取 EMA denoiser，并始终冻结。

### 10.4 小数据增强

允许：

- G1 左右镜像并按关节轴修正符号；
- 小幅 joint/root rotation 扰动；
- 小幅速度尺度扰动；
- 随机选择合法窗口起点。

不使用时间反转，因为起身的反向序列是跌倒。

### 10.5 Ensemble Score Matching

固定使用 SMP 论文的三个 timestep：

```text
esm_timesteps = [22, 15, 8]
```

对策略产生的最近 10 步窗口，在每个 timestep i 上计算：

```text
noise_i ~ Normal(0, identity)

policy_noised_window_i
  = √alpha_bar(i) × policy_window
    + √(1 - alpha_bar(i)) × noise_i

predicted_noise_i
  = frozen_denoiser(policy_noised_window_i, i)

sds_error_i
  = mean_square(predicted_noise_i - noise_i)
```

### 10.6 不同 timestep 的尺度校准

先用冻结基础 PPO 策略收集校准 rollout：

```text
running_mean_error_i
  = (1 - update_rate) × previous_running_mean_error_i
    + update_rate × current_sds_error_i
```

正式训练时冻结该均值，得到归一化错误：

```text
normalized_sds_error_i
  = sds_error_i / (running_mean_error_i + 1e-6)
```

ESM 奖励：

```text
average_normalized_sds_error
  = mean over i in [22, 15, 8] [normalized_sds_error_i]

smp_reward(t)
  = exp(-smp_scale × average_normalized_sds_error)
```

`smp_scale` 初始为 1，通过校准使训练初期 SMP reward 中位数位于 0.2–0.8。

### 10.7 任务优先的 SMP 上限

防止先验奖励鼓励策略保持自然但没有起身的低位动作：

```text
smp_reward_limit(t)
  = max(clip(task_reward(t), 0, 1), 0.3)

clipped_smp_reward(t)
  = min(smp_reward(t), smp_reward_limit(t))
```

任务进展较差时，SMP 奖励上限较低；任务已经向站立推进时，SMP 可以更充分
地区分动作质量。

### 10.8 GPU 批量计算

每个环境维护 `[10, 51]` 环形动作缓冲区。一次奖励计算把所有环境和三个
timestep 合并成：

```text
denoiser_batch_shape
  = [3 × number_of_environments, 10, 51]
```

冻结 denoiser 在 `no_grad` 模式执行。Diffusion 仅训练期计算，部署模型只导出
Actor。

## 11. 安全和正则代价

### 11.1 动作变化

```text
action_rate_cost(t)
  = mean_square(action(t) - action(t-1))
```

### 11.2 力矩占比

```text
torque_cost(t)
  = mean over joints [
      (requested_torque(j) / torque_limit(j))²
    ]
```

### 11.3 软关节限位

```text
normalized_limit_distance(j)
  = abs(q(j) - joint_mid(j))
    / (joint_range(j) / 2)

joint_limit_excess(j)
  = ReLU(normalized_limit_distance(j) - soft_limit_ratio)

joint_limit_cost(t)
  = mean_square(joint_limit_excess)
```

### 11.4 足部滑移

```text
slip_cost(t)
  = left_foot_contact
    × norm(left_foot_horizontal_velocity)²
    + right_foot_contact
    × norm(right_foot_horizontal_velocity)²
```

### 11.5 非安全碰撞

```text
unsafe_contact_cost(t)
  = Σ over unsafe bodies [
      clip(contact_force / safe_force, 0, maximum_contact_cost)
    ]
```

手、前臂、膝和小腿在低位 recovery 时允许承重；接近站立后惩罚其持续承重。
头部和躯干高冲量始终惩罚。

## 12. 总奖励

```text
total_reward(t)
  = task_weight × task_reward(t)
    + tracking_weight × tracking_reward(t)
    + smp_weight(iteration) × clipped_smp_reward(t)
    - action_rate_weight × action_rate_cost(t)
    - torque_weight × torque_cost(t)
    - joint_limit_weight × joint_limit_cost(t)
    - slip_weight × slip_cost(t)
    - unsafe_contact_weight × unsafe_contact_cost(t)
```

SMP 权重平滑接入：

```text
smp_ramp_ratio
  = clip(
      (iteration - smp_start_iteration) / smp_ramp_iterations,
      0,
      1
    )

smp_weight(iteration)
  = maximum_smp_weight × smoothstep(smp_ramp_ratio)
```

第一轮按 episode 回报贡献校准，使 task、tracking、SMP、cost 的绝对贡献大约为：

```text
50% : 20% : 20% : 10%
```

所有原始奖励项和加权结果分别记录。

## 13. PPO 更新

### 13.1 GAE advantage

单步 TD error：

```text
td_error(t)
  = reward(t)
    + γ × value(t+1)
    - value(t)
```

GAE advantage：

```text
advantage(t)
  = td_error(t)
    + (γ × λ) × td_error(t+1)
    + (γ × λ)² × td_error(t+2)
    + ...

γ = 0.99
λ = 0.95
```

### 13.2 PPO clipped policy loss

```text
probability_ratio(t)
  = new_policy_probability(action(t) | actor_input(t))
    / old_policy_probability(action(t) | actor_input(t))

clipped_ratio(t)
  = clip(probability_ratio(t), 1 - 0.2, 1 + 0.2)

policy_objective(t)
  = min(
      probability_ratio(t) × advantage(t),
      clipped_ratio(t) × advantage(t)
    )

policy_loss
  = -mean(policy_objective)
```

### 13.3 Value 和总损失

```text
value_loss
  = mean_square(predicted_value - return_target)

total_ppo_loss
  = policy_loss
    + value_loss_weight × value_loss
    - entropy_weight × policy_entropy
```

初始设置：

```text
rollout_horizon = 24
ppo_epochs = 5
minibatches = 4
target_KL = 0.01
clip_ratio = 0.20
```

## 14. 困难时间窗自适应采样

每条 reference clip 按 0.4 秒划分 temporal bins；prone、supine、side、kneel、
seated 随机倒地类型也分别建立 bin。

### 14.1 单次困难度

```text
failure_flag
  = 1 当 episode timeout 或重新跌倒
  = 0 其他情况

normalized_tracking_error
  = clip(tracking_error / hard_error_threshold, 0, 1)

observed_difficulty(bin)
  = max(failure_flag, normalized_tracking_error)
```

### 14.2 EMA

采用 Extreme-RGMT 的更新形式：

```text
difficulty_ema(bin)
  = (1 - ema_rate) × previous_difficulty_ema(bin)
    + ema_rate × observed_difficulty(bin)

ema_rate = 0.01
```

### 14.3 采样概率

```text
clipped_difficulty(bin)
  = clip(difficulty_ema(bin), 0, maximum_difficulty)

normalized_difficulty
  = clipped_difficulty / sum(all_clipped_difficulties)

unnormalized_probability(bin)
  = normalized_difficulty(bin)
    + uniform_mass / number_of_bins

sampling_probability(bin)
  = unnormalized_probability(bin)
    / sum(all_unnormalized_probabilities)

uniform_mass ≥ 0.20
```

访问次数不足的 bin 强制均匀采样。该采样器只改变 reset 分布，不跨 clip 构造
相邻动作。

## 15. Velocity 接管

连续满足站立集合后，接管进度每步增加：

```text
handoff_progress(t+1)
  = clip(
      handoff_progress(t) + control_dt / handoff_duration,
      0,
      1
    )

handoff_duration = 0.5 到 1.0 秒
```

独立 recovery/velocity 策略时混合关节目标：

```text
executed_joint_target(t)
  = (1 - handoff_progress) × recovery_joint_target(t)
    + handoff_progress × velocity_joint_target(t)
```

速度命令同步开放：

```text
executed_velocity_command(t)
  = handoff_progress × user_velocity_command
```

若 `upright < 0.80` 或 `pelvis_height < 0.75 × h0`，关闭 velocity command 并
重新进入 recovery。关节目标继续有限速过渡，避免瞬时跳变。

## 16. 分阶段训练流程

### S0：数据与物理一致性

完成：

1. 确认 source fps、quaternion 顺序和 29 关节顺序；
2. 编译 50 Hz 数据与 source-group split；
3. 验证离线 FK 和 MuJoCo link pose；
4. 检查手、前臂、膝、小腿碰撞几何和切向摩擦；
5. 建立固定 reference reset bank 和 fallen reset bank。

进入下一阶段的条件：参考状态载入 MuJoCo 后没有深穿透、NaN 和速度爆炸；
离线/在线 51D SMP feature 在同一状态上数值一致。

### S1：离线训练 SMP

完成：

1. 生成全部合法 10 步窗口和左右镜像窗口；
2. 训练两层 Transformer denoiser；
3. 保存 EMA checkpoint；
4. 监控 source-held-out validation loss；
5. 比较真实窗口、时间打乱、关节打乱和随机噪声窗口的 SMP reward。

进入下一阶段的条件：validation 真实窗口的 SMP reward 中位数明显高于破坏
窗口，训练/验证 loss 没有持续分离。

### S2：Task + tracking PPO

使用最终 Transformer Actor，暂令 `smp_weight = 0`。混合 reference、fallen、
stand reset，并逐步退火辅助力。

进入下一阶段的条件：

- reference reset 能保持局部物理跟踪；
- fallen bank 出现稳定、无辅助起身；
- stand-noise bank 能保持默认姿态至少 0.5 秒；
- 结果优于零动作和固定 q0 PD；
- 同时记录 MLP PPO 作为结构消融。

### S3：冻结 SMP 接入 PPO

1. 加载并冻结 S1 EMA denoiser；
2. 用冻结 S2 policy rollout 校准三个 timestep 的 mean SDS error；
3. 从 S2 checkpoint 恢复；
4. 把 SMP 权重从 0 平滑增加到目标值。

进入下一阶段的条件：固定 fallen bank 成功率不能显著下降，并且非足部承重、
滑移、动作变化、力矩或起身时间至少有一项改善。

### S4：困难 bin 自适应采样

开启 bin difficulty EMA，逐渐使用自适应 reset 概率，但保留至少 20% 均匀
质量。固定评测仍使用均匀分布。

进入下一阶段的条件：最差倒地类别和最差 reference bin 的成功率提高，整体
成功率不下降。

### S5：随机化与 velocity 接管

依次加入：

1. friction；
2. mass 和 CoM；
3. motor strength、PD gain 和 motor zero offset；
4. 观测噪声和控制延迟；
5. 外力；
6. velocity 接管；
7. rough terrain。

每次只扩大一组随机化，并保留无随机化回归环境。

## 17. 初始参数

| 参数 | 初始值 |
|---|---:|
| 控制频率 | 50 Hz |
| 本体历史 | 10 步 |
| 参考命令窗口 | 21 步 |
| RGMT embedding | 128 |
| history Transformer | 1 block |
| cross-attention | 1 block |
| SMP 窗口 | 10 步 |
| diffusion timestep 数 | 50 |
| ESM timestep | 22、15、8 |
| SMP Transformer | 2 blocks、4 heads、embedding 128 |
| recovery timeout | 4–6 秒 |
| 成功保持 | 0.5 秒 |
| PPO γ | 0.99 |
| GAE λ | 0.95 |
| PPO clip | 0.20 |
| rollout horizon | 24 起步 |
| early assist force | 0–200 N，最终为 0 |
| difficulty EMA rate | 0.01 |
| uniform sampling mass | 至少 0.20 |

## 18. 评测与消融

固定测试类别：

- supine；
- prone；
- left/right side；
- kneeling；
- seated；
- reference partial；
- reference complete；
- stand noise；
- 质量/摩擦外推；
- velocity handoff。

每类报告：

- success rate；
- time-to-stand；
- 0.5 秒 hold success；
- 峰值和累计 requested/applied torque；
- 足部滑移；
- 手、膝、躯干承重时间；
- 头部碰撞冲量；
- 最终默认姿态误差；
- velocity 接管后 2–5 秒存活率。

必要消融：

1. MLP PPO + task reward；
2. causal history Transformer + task reward；
3. RGMT command/tracking + task reward；
4. 第 3 项 + frozen SMP；
5. 第 4 项 + adaptive reset sampling；
6. SMP 单随机 timestep 与 ESM `[22, 15, 8]`；
7. 有/无 task-priority clipping；
8. reference-only reset 与三类混合 reset。

## 19. mjlab 代码落点

```text
src/mjlab/tasks/velocity/recovery_data/
  g1_schema.py
  g1_compiler.py
  g1_windows.py

src/mjlab/tasks/velocity/config/g1/
  recovery_env_cfg.py
  recovery_events.py
  recovery_commands.py
  recovery_observations.py
  recovery_rewards.py

src/mjlab/tasks/velocity/recovery_prior/
  rgmt_actor_critic.py
  smp_model.py
  smp_reward.py
  adaptive_sampler.py

src/mjlab/scripts/
  compile_g1_recovery.py
  train_g1_smp.py
  audit_g1_recovery.py
```

RSL-RL integration 需要支持自定义 Actor–Critic、history/command mask、checkpoint
feature schema 校验和 Actor-only 导出。SMP checkpoint 单独版本化，不进入部署
模型。

## 20. 实现增量

1. I0：数据 schema、编译、重采样、source split 和窗口 mask；
2. I1：三类 reset、终止、成功集合、碰撞/摩擦和辅助力；
3. I2：RGMT history encoder、cross-attention、residual action 和 critic；
4. I3：tracking、standing potential、progress、hold 和安全 cost；
5. I4：SMP denoiser、EMA、ESM reward、校准和 clipping；
6. I5：reset 概率课程和困难 bin sampler；
7. I6：velocity handoff、随机化、导出和固定评测。

每个增量先通过 CPU 单元测试和小规模仿真 smoke test，再进入 GPU 大规模训练。

## 21. 参考来源

- [SMP: Reusable Score-Matching Motion Priors for Physics-Based Character
  Control](https://arxiv.org/html/2512.03028v3)：Diffusion 噪声预测、SMP
  reward、ESM `[22,15,8]`、adaptive normalization、10 步动作窗和 Getup task。
- [Robust and Generalized Humanoid Motion Tracking
  (RGMT)](https://arxiv.org/html/2601.23080v1)：G1 93D observation、38D
  command、10 步 causal history、21 步 cross-attention、residual PD target、
  asymmetric critic、recovery reset 和退火辅助力。
- [Extreme-RGMT](https://arxiv.org/html/2607.20110v1)：temporal-bin failure
  EMA、uniform-baseline adaptive sampling，以及 G1 PPO 和随机化量级。

整个系统中，参考窗口提供局部动作指引，SMP 评价策略生成的短时动作分布，
站立势函数规定全局恢复目标，PPO 和 MuJoCo 接触动力学负责产生完整可执行起身。
