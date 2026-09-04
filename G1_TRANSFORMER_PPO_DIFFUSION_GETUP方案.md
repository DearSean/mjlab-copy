# G1 Conditional Flow–SMP–FPO 起身训练详细方案

## 1. 方案目标

本方案在当前 mjlab 项目中训练 Unitree G1 从随机倒地状态自主恢复到 velocity
环境默认屈膝站姿。`lafan-g1-getup/` 中的 G1 动作片段用于学习支撑、翻转、
抬升和姿态转换；FPO 在 MuJoCo 接触动力学中将这些局部能力组合成完整起身。

采用的方法包括：

1. 轻量 causal Transformer 编码真机可获得的 10 步本体历史；
2. [Flow Matching Policy Gradients](https://arxiv.org/html/2507.21053) 的统一
   conditional flow Actor 和 PPO-compatible FPO clipped objective；
3. [SMP](https://arxiv.org/html/2512.03028v3) 的冻结动作 Diffusion、
   Ensemble Score Matching 奖励和 Reference State Initialization；
4. [Extreme-RGMT](https://arxiv.org/html/2607.20110v1) 的困难时间窗自适应
   采样。

系统链路：

```text
G1 动作片段
  ├── 参考状态初始化
  └── 10 步动作窗口 → SMP Diffusion ─┐ 训练时冻结奖励
                                     ↓
机器人最近 10 步本体状态 → Conditional Flow Actor → 29 维关节目标
随机倒地状态 ─────────────────────────────┘
默认屈膝站姿 → 站立任务奖励 ───────────────┘
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

FPO 与 PPO 一样最大化折扣累计回报：

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
| reference | 动作片段中的中后段安全帧 | 无 | 自主完成较短的剩余恢复 |
| fallen | 低进度起身数据帧 | 无 | 在辅助力下学习完整自主恢复 |
| stand | 默认姿态附近的扰动状态 | 无，使用默认姿态 | 学习站立吸引域和接管 |

reset mode 只用于环境统计、课程和奖励 gating，不进入 Actor 或 Critic。策略只能
根据本体状态历史判断当前恢复方式；辅助力大小和课程等级同样不进入观测。

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

## 5. 不使用参考命令

当前自主起身任务明确禁用原 21×38 reference command、valid mask、cross-attention、
reference confidence 和 reference action anchor。动作片段只承担两项训练职责：

1. 提供不同进度的 reset 初始状态；
2. 训练冻结 SMP，使在线状态短窗得到动作分布奖励。

reference、fallen、stand 三类环境使用完全相同的 Actor 输入和默认关节目标锚点。
这保证部署策略不需要知道当前状态来自哪个数据板块。

## 6. Causal Transformer Conditional Flow Actor

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

不足 10 步时复制 reset 后首帧，不向策略暴露 reset mask。

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

Causal mask 保证第 i 个时间 token 只能读取第 0 到 i 个 token。Actor 使用最后
一个 token 作为包含完整历史的条件向量：

```text
history_context
  = H_normalized(last_time)

dynamics_embedding_dimension = 128
```

### 6.4 Conditional flow 动作生成

同一个向量场维护完整条件分布，不存在专家编号或离散行为标签：

```text
noise z ~ Normal(0, 0.25² I)
x(0) = z

dx/dτ
  = velocity_field(history_context, x(τ), time_embedding(τ))

bounded_velocity = 1.5 × tanh(velocity_field / 1.5)
latent_action = x(1)
environment_action = tanh(latent_action)
```

向量场 MLP 为 `[512, 256, 128]`，ELU 激活，输出 29 维速度。当前使用 32 个固定
midpoint Euler 积分步；所有环境在每个积分步做一次批量前向，因此不改变并行
rollout 结构。训练 rollout 从基础分布随机采样，以保留多峰探索；验证、导出和
真机部署默认使用 `z = 0` 的 zero-sampling，使同一状态下的控制输出可复现，仍由
观测历史决定采用哪条恢复路径。需要检查同一状态的其他可行模式时，可以显式
重复随机采样。连续 `tanh` 动作映射替代硬截断，避免长期更新后大量动作黏在
`±1` 且截断外没有梯度。较小的初始噪声避免未训练策略以全幅随机关节目标破坏
reference 和 stand reset；1.5 的向量场上限防止单位流时间内把 latent action
推入 `tanh` 深饱和区。本体
输入在每个时间步使用 LayerNorm，不更新跨 rollout 的运行统计量。

FPO 的 CFM target 和 old/new loss 始终使用未压缩的 `latent_action`；只有送入
环境的动作经过 `tanh`。这样 rollout 的 ODE 终点、CFM 监督目标和 likelihood
ratio 位于同一个空间，避免把压缩后的环境动作错误地当作流终点而产生系统性
训练漂移。

### 6.6 关节目标和 PD

`action_scale` 使用项目现有的 G1 逐关节尺度：

```text
joint_target(t)
  = q0
    + action_scale × action(t)

requested_torque(t)
  = Kp × (joint_target(t) - joint_position(t))
    - Kd × joint_velocity(t)
```

Actuator model 再应用力矩、速度和饱和限制。训练同时记录 requested torque 与
applied torque。

### 6.7 Asymmetric Critic

Critic 可以保留标准 velocity 配置中能够由机载估计器或接触传感器提供的状态，
但明确删除：

```text
reset mode
辅助力大小
课程等级
零速度 command
reference frame / confidence
```

```text
value(t) = Critic_MLP(privileged_state(t))
```

Critic MLP 建议 `[1024, 512, 256]`。特权信息不进入 Actor 和导出模型。

## 7. Reset 与探索课程

### 7.1 三类 reset 概率

当前实现不按 iteration 推进，而使用显式成功率课程。初期四级全部来自
reference 后半段，并把最小进度从 0.70 依次扩展到 0.55、0.40 和 0.25；只有
已激活的非 stand 恢复 episode 总成功率达到 90%，才开放下一级：

| posture level | reference / fallen / stand | reference progress 下界 |
|---:|---:|---:|
| 0 | 1.00 / 0.00 / 0.00 | 0.70 |
| 1 | 1.00 / 0.00 / 0.00 | 0.55 |
| 2 | 1.00 / 0.00 / 0.00 | 0.40 |
| 3 | 1.00 / 0.00 / 0.00 | 0.35 |
| 4 | 1.00 / 0.00 / 0.00 | 0.30 |
| 5 | 1.00 / 0.00 / 0.00 | 0.25 |
| 6 | 1.00 / 0.00 / 0.00 | 0.20 |
| 7 | 1.00 / 0.00 / 0.00 | 0.15 |
| 8 | 1.00 / 0.00 / 0.00 | 0.10 |
| 9 | 0.85 / 0.15 / 0.00 | 0.10 |
| 10 | 0.70 / 0.30 / 0.00 | 0.10 |
| 11 | 0.60 / 0.35 / 0.05 | 0.10 |
| 12 | 0.45 / 0.45 / 0.10 | 0.10 |

stand 仅用于稳定站立覆盖，不计入课程成功率分母，避免其天然高成功率虚增进阶
指标。reference 模式只从当前进度带载入初始状态，随后由策略自主恢复；不提供
逐帧 reference command、tracking reward 或 reference action anchor。

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

课程拆成严格串行的两阶段。姿态课程期间固定采样 `160–200 N`，逐级扩大
reference 进度范围并开放 fallen/stand；姿态课程满级后，保持最终姿态分布，
再启动辅助力课程：

```text
assist_force_z ~ Uniform(force_min(assist_level), force_max(assist_level))

160–200 → 120–160 → 90–120 → 65–90 → 45–65
→ 25–45 → 10–25 → 0–10 → 0 N
```

辅助力对 reference 和 fallen 回合从开始到终止始终施加本回合采样的完整值，
不再根据 `height × uprightness` 在回合内释放；stand 不施加辅助力。成功允许在
当前课程辅助力下达成，随后由 90% 成功率推动下一级减力，最终等级才要求全程
0 N。姿态课程窗口为 `500、750、1000、1250、1500、1750、2000、2500、
3000、3500、4000、5000`，辅助力课程窗口继续增长为 `6000、7000、8000、
9000、10000、12000、14000、16000`。

### 7.6 终止条件

Recovery episode 中低高度、手膝接地和大倾角不是早停条件。仅在以下情况终止：

- 已连续稳定站立 0.5 秒；
- 达到 4–6 秒 recovery timeout；
- NaN、仿真发散或严重关节硬超限；
- 超过安全碰撞冲量；
- root 离开训练区域。

## 8. Reference tracking reward（当前自主起身配置禁用）

以下 tracking 结构保留为后续 RGMT 消融设计，不进入当前
`Mjlab-Velocity-Flat-Unitree-G1-Recovery`。当前参考数据只用于 reset 初始化和
离线 SMP 训练，防止 reference 子任务占据 rollout 样本却不给 fallen 起身提供奖励。

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

### 9.2 当前站立势函数与奖励交接

```text
progress(t) = normalized_height(t) × uprightness(t)

stand_gate(t) = smoothstep(progress(t), 0.55, 0.85)

recovery_potential(t)
  = (1 - stand_gate(t))
    × normalized_height(t)
    × (0.2 + 0.8 × uprightness(t))
```

`uprightness` 的 0.2 下限保证机器人侧躺时抬高身体仍能获得梯度。恢复期间
pose、action-rate、body angular velocity 和 angular momentum 分别保留
10%、30%、20% 和 20%，随后由 `stand_gate` 平滑恢复到完整权重。

### 9.3 净进展奖励

```text
recovery_progress_potential(t)
  = 0.6 × normalized_height(t) + 0.4 × uprightness(t)

raw_progress_rate(t)
  = (recovery_progress_potential(t)
     - recovery_progress_potential(t-1)) / control_dt

progress_reward(t)
  = clip(raw_progress_rate(t), -5, 5)
```

除以 `control_dt` 用来抵消 RewardManager 的 dt 缩放，使完整 `0→1` 进展不会
再次缩小 50 倍。成功奖励和超时失败惩罚同样先除以 dt，实际一次性权重分别为
`+10` 和 `-1`。课程与成功判定仍使用 `height × uprightness ≥ 0.85`，只有密集
进展奖励拆开高度和直立度，避免侧躺时乘积接近零而没有可用学习信号。

### 9.4 成功保持和低位抬升

```text
hold_reward(t)
  = smoothstep(height × uprightness, 0.65, 0.85)
    × indicator(assistant_force = 0)
```

低位阶段增加短期抬升引导：

```text
lift_reward(t)
  = indicator(pelvis_height(t) < 0.70 × h0)
    × clip(vertical_base_velocity / target_lift_velocity, 0, 1)
```

`lift_reward` 只在非 stand 且高度低于阈值时工作，权重为 0.5；`hold_reward`
权重为 2.0，并明确要求辅助力已经归零。向下速度不再通过 lift 项重复惩罚，状态
回退已经由 `progress_reward` 表达。

任务奖励：

```text
task_reward(t)
  = 1.0 × recovery_potential(t)
    + 2.0 × progress_reward(t)
    + 0.5 × lift_reward(t)
    + 2.0 × hold_reward(t)
    + 10.0 × success_event(t)
    - 1.0 × timeout_failure_event(t)
    - 0.02 × fallen_duration_cost(t)
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

FPO 只读取 EMA denoiser，并始终冻结。

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
noise_i ~ Normal(0, identity), 每个 episode 固定

policy_noised_window_i
  = √alpha_bar(i) × policy_window
    + √(1 - alpha_bar(i)) × noise_i

predicted_noise_i
  = frozen_denoiser(policy_noised_window_i, i)

sds_error_i
  = mean_square(predicted_noise_i - noise_i)
```

### 10.6 不同 timestep 的尺度校准

先用冻结基础策略收集校准 rollout：

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

### 10.8 分布外返回引导

绝对 SMP 分数只衡量当前窗口是否位于动作分布内。为了让分布外姿态也获得明确
的运动趋势，在线奖励同时保留归一化能量：

```text
smp_energy(t) = smp_scale × average_normalized_sds_error(t)

energy_descent(t)
  = clip(
      (smp_energy(t-1) - smp_energy(t)) / dt / max_energy_rate,
      -1,
      1
    )
```

同一环境在一个 episode 内复用固定的三个噪声张量，因此相邻能量可以直接
比较；reset 时才重新采样。能量下降获得正奖励，远离分布得到对称负奖励。

每个 timestep 的去噪结果为：

```text
estimated_clean_window_i
  = (policy_noised_window_i
     - sqrt(1 - alpha_bar(i)) × predicted_noise_i)
    / sqrt(alpha_bar(i))

denoising_direction(t)
  = mean_i(estimated_clean_window_i[-1]) - policy_window[-1]
```

实际特征变化与上一时刻去噪方向的余弦一致性构成
`direction_alignment`，并按实际运动 RMS 抑制接近静止时的虚假方向奖励。
两个返回引导只在 SMP 分数低于 `0.60` 时启用，在 `0.25` 以下达到完整强度；
同时随恢复进度在 `0.65--0.85` 平滑退为零，避免 support 较多的数据把已经接近
站立的机器人重新拉回低位。clean reset 的首帧仍使用环形缓冲区回填，
立即提供绝对 SMP 分数；抬高的 noisy reset 则不评分这种静态回填窗口，
而是等待落地后收集满10个真实帧再启用 SMP。能量下降和方向奖励在
第一个有效 SMP 分数之后才启用，防止 reset 奖励尖峰。

```text
smp_total(t)
  = smp_weight(progress) × clipped_smp_reward(t)
    + 2.0 × guidance_gate(t) × energy_descent(t)
    + 1.0 × guidance_gate(t) × direction_alignment(t)
```

这些量只属于训练奖励，不加入 Actor 或 Critic 观测，也不成为 reference
tracking command。

### 10.9 GPU 批量计算

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
    + smp_total(t)
    - action_rate_weight × action_rate_cost(t)
    - torque_weight × torque_cost(t)
    - joint_limit_weight × joint_limit_cost(t)
    - slip_weight × slip_cost(t)
    - unsafe_contact_weight × unsafe_contact_cost(t)
```

SMP 与辅助力课程解耦，从 FPO 第一步就工作；其权重只按机器人当前恢复进度
与末段 pose 奖励平滑交接：

```text
smp_weight = 10.0,                         progress <= 0.65
smp_weight = smoothstep(10.0 -> 2.5),      0.65 < progress < 0.85
smp_weight = 2.5,                          progress >= 0.85
```

held-out 专家窗口的 SMP 中位分数约为 0.76，当前失败在线轨迹约为 0.006；固定
恢复权重修正首轮在线训练中 SMP 平均贡献只有约0.02、而自碰撞成本达到约2.22
的失衡。接近站立后降到2.5，避免以support为主的数据分布压制默认站姿。
自碰撞基础权重同步降为 `-0.2`，低位阶段只启用10%，接近站立后恢复。
目标仍是使 task、SMP、cost 的绝对贡献大约为：

```text
60% : 25% : 15%
```

所有原始奖励项和加权结果分别记录。

## 13. Flow Policy Optimization 更新

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

### 13.2 FPO conditional flow matching ratio

```text
对每个 rollout action 固定采样 Nmc 组：

tau(i)   ~ Uniform(0, 1)
noise(i) ~ Normal(0, 0.25² I)

noisy_action(i)
  = tau(i) × latent_action + (1 - tau(i)) × noise(i)

target_velocity(i)
  = latent_action - noise(i)

cfm_loss(theta, i)
  = mean_square(
      velocity_field_theta(noisy_action(i), tau(i), history_context)
      - target_velocity(i)
    )

raw_log_ratio(t, i)
  = clamp(cfm_loss(old_theta, t, i), max=3)
    - clamp(cfm_loss(theta, t, i), max=3)

bounded_log_ratio(t, i)
  = straight_through_clip(raw_log_ratio(t, i), -3, 3)

fpo_ratio(t, i) = exp(bounded_log_ratio(t, i))

ppo_positive(t, i)
  = min(
      fpo_ratio(t, i) × advantage(t),
      clip(fpo_ratio(t, i), 0.95, 1.05) × advantage(t)
    )

spo_negative(t, i)
  = fpo_ratio(t, i) × advantage(t)
    - abs(advantage(t)) / (2 × 0.05)
      × (fpo_ratio(t, i) - 1)²

aspo_objective(t, i)
  = ppo_positive(t, i), if advantage(t) >= 0
  = spo_negative(t, i), otherwise

policy_loss
  = -mean_t,i(aspo_objective)
```

旧 loss、tau 和 noise 在 rollout 时存储，所有 PPO epoch 复用同一组 Monte Carlo
样本。这样当前参数在首次更新前严格得到 `fpo_ratio = 1`，避免额外估计噪声。
`Nmc = 16`。FPO++ 不先平均 MC loss，而是让每个固定样本维护独立 ratio。正优势
采用 PPO clipping；负优势采用带二次恢复项的 ASPO/SPO，策略偏离旧动作时始终
存在拉回梯度。数值边界使用直通梯度 clip，避免 `exp` 溢出但不在边界外切断
梯度；新旧 CFM loss 在作差前各自截断到 3，优势截断到 `[-5, 5]`。原来的整批
skip 和 smooth-L1 CFM guard 已移除：它们只能阻止单次大更新，无法阻止多轮小
更新造成的累计漂移，也是 500 轮实验中 flow loss 和动作饱和持续上升的主要
算法缺口。

### 13.3 Value 和总损失

```text
value_loss
  = mean_square(predicted_value - return_target)

total_fpo_loss
  = policy_loss
    + value_loss_weight × value_loss
    + 0.01 × flow_diversity_loss
```

FPO 不使用 Gaussian entropy；探索来自 flow 的基础噪声。Actor 的观测 normalizer
固定关闭，避免 rollout 内更新统计量导致 old/new CFM loss 不再处于同一条件。
`flow_diversity_loss` 对同一条本体观测分别采样两组基础噪声，以两条 endpoint
动作差估计条件标准差，并只惩罚其低于 0.12 的部分。每个 minibatch 最多使用
64 条观测，因此不向 Actor 添加任何输入，也不暴露辅助力、reset mode、课程或
reference 信息。训练同步记录 `flow_pair_std` 和动作饱和比例。

初始设置：

```text
rollout_horizon = 24
ppo_epochs = 3
minibatches = 8
CFM Monte Carlo samples = 16
flow integration steps = 32
clip_ratio = 0.05
learning_rate = 5e-5
optimizer = AdamW(weight_decay=5e-4)
flow diversity target std = 0.12
flow diversity coefficient = 0.01
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
4. 使用与 G1 velocity 任务相同的全身切向摩擦检查手、前臂、膝、小腿碰撞；
5. 保留原始动作供 SMP 使用，另行建立经过接触投影和短时物理静置验证的
   reference/fallen reset bank。

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

### S2/S3：Task + autonomous reset + frozen SMP 联合 FPO

当前可训练配置使用 causal Transformer conditional flow Actor，从 FPO 训练开始
即启用已冻结 SMP。混合
reference-initialization、fallen、stand reset；辅助力和 reset 难度由成功率
推进，SMP 权重与它们解耦；单回合内仅按当前progress与pose平滑交接。

```text
smp_weight(progress <= 0.65) = 10.0
smp_weight(progress >= 0.85) = 2.5
```

每级只在当前非 stand 恢复总成功率达到 90% 后进阶。第一阶段在固定
`160–200 N` 完整辅助力下逐步增加姿态难度；姿态课程满级后，第二阶段固定
最终 reset 分布，并按 `160–200 → 120–160 → 90–120 → 65–90 → 45–65 →
25–45 → 10–25 → 0–10 → 0 N` 退火辅助力。两阶段证据窗口全局严格增长。
SMP 任务优先上限使用：

```text
smp_cap = 0.3 + 0.7 × height × uprightness
clipped_smp_reward = min(normalized_smp_reward, smp_cap)
```

reference reset 从进度 `[0.70, 0.85)` 开始，后期把下界扩展到 0.10；其后
完全自主执行，不推进 reference target。三类 reset 从 `100/0/0` 分级迁移到
`45/45/10`，避免训练最初就让尚不可解的 fallen 样本淹没正反馈。进入下一
阶段的条件：

- reference initialization 能自主完成剩余恢复；
- fallen bank 出现稳定、无辅助起身；
- stand-noise bank 能保持默认姿态至少 0.5 秒；
- 结果优于零动作和固定 q0 PD；
- 同时记录原 Gaussian MLP PPO 作为结构消融。

### S3：联合阶段内的 SMP 校准与验收

S3 不再单独重启 FPO。训练前用 held-out reference window 固定校准
ESM `[22, 15, 8]` 的 mean SDS error，之后 denoiser 和校准均保持冻结。
固定 fallen bank 成功率不能显著下降，并且非足部承重、滑移、
动作变化、力矩或起身时间至少有一项改善。

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
| 并行环境（8 GiB GPU 默认） | 1024 |
| 本体历史 | 10 步 |
| 参考命令窗口 | 禁用 |
| Flow history embedding | 128 |
| history Transformer | 1 block |
| Flow vector field MLP | 512、256、128 |
| Flow integration steps | 32 |
| Flow base noise std | 0.25 |
| Flow velocity bound | 1.5 |
| FPO MC samples | 16 |
| FPO CFM loss clamp | 3.0 |
| FPO per-sample log-ratio numerical bound | 3.0 |
| FPO advantage clamp | -5.0、5.0 |
| FPO negative-advantage objective | ASPO/SPO |
| SMP 窗口 | 10 步 |
| diffusion timestep 数 | 50 |
| ESM timestep | 22、15、8 |
| SMP Transformer | 2 blocks、4 heads、embedding 128 |
| recovery timeout | 4–6 秒 |
| 成功保持 | 0.5 秒 |
| PPO γ | 0.99 |
| GAE λ | 0.95 |
| FPO clip | 0.05 |
| optimizer | AdamW，weight decay 5e-4 |
| learning epochs / minibatches | 3 / 8 |
| learning rate | 5e-5 |
| 同状态 Flow diversity target / coefficient | 0.12 / 0.01 |
| rollout horizon | 24 起步 |
| early assist force | 160–200 N，最终为 0 |
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
- reset mode 占比和各 mode 独立 success numerator/rate；
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
  recovery_rewards.py

src/mjlab/rl/
  flow_policy.py
  fpo.py

src/mjlab/tasks/velocity/recovery_prior/
  smp_model.py
  smp_reward.py
  adaptive_sampler.py

src/mjlab/scripts/
  compile_g1_recovery.py
  train_g1_smp.py
  audit_g1_recovery.py
```

RSL-RL integration 通过完整模块路径加载自定义 Flow Actor 和 FPO；rollout storage
的 distribution parameters 保存 old CFM loss、tau 和 noise。SMP checkpoint 单独
版本化，不进入部署模型。

## 20. 实现增量

1. I0：数据 schema、编译、重采样、source split 和窗口 mask；
2. I1：三类 reset、终止、成功集合、碰撞/摩擦和辅助力；
3. I2：causal history encoder、conditional flow Actor、FPO ratio 和 critic；
4. I3：tracking、standing potential、progress、hold 和安全 cost；
5. I4：SMP denoiser、EMA、ESM reward、校准和 clipping；
6. I5：固定 reset 分层比例、姿态难度课程和困难 bin sampler；
7. I6：velocity handoff、随机化、导出和固定评测。

每个增量先通过 CPU 单元测试和小规模仿真 smoke test，再进入 GPU 大规模训练。

## 21. 2026-08-30 算法超参数 500 轮验证

运行目录：`2026-08-30_13-30-24_flow_diversity_hparam_fresh_500`。该实验从零
训练 500 轮，使用 `max_flow_velocity=1.5`、学习率 `5e-5`、3 个 learning
epochs，以及 target std 0.12、权重 0.01 的同状态双噪声 diversity loss。

| 100 轮窗口 | mean reward | flow loss | pair std | saturation | reference success | fallen success | stand success |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–99 | 5.21 | 0.131 | 0.127 | 0 | 39.0% | 0.57% | 94.3% |
| 400–499 | 0.79 | 0.304 | 0.072 | 0 | 5.75% | 0.24% | 51.6% |

结论：降低向量场上限彻底阻止了 `tanh` 饱和，但 0.01 的 diversity 权重不足以
保持基础噪声影响；策略转而坍缩为低方差、非饱和但无效的动作。末段 recovery
progress reward 为 -0.189，课程始终停留在 level 0。该候选配置未通过任务验收，
不得从 `model_499.pt` 继续长训。结果表明只调整 Flow/PPO 超参数不能解决当前
奖励和自采样 CFM 共同造成的行为坍缩。

## 22. 2026-08-30 FPO++ 对齐补丁

500 轮失败实验显示策略的平均 `|log ratio|` 仅约 0.0021，同状态
pair std 从 0.127 降到 0.072，而末段 SMP raw score 仅为 0.033。
在线 SMP 公式重测得到参考窗口 0.422、时间打乱 0.181、关节打乱
0.106 和随机特征约 `9e-6`，证明主要问题是 FPO 未将有效 SMP 信号
转化为行为，而非 SMP 缺乏区分力。

已实施的最小对齐补丁：

- CFM 动作维度 reduction 从 `mean` 改为方差保持的 `sqrt`；
- Flow 积分步数从 32 增加到 64，base noise 从 0.25 增加到 0.5；
- 使用 0.02 的训练期 latent action perturbation 替代无效的 pair-std 目标损失；
- MC samples 从 16 增加到 32，learning epochs 从 3 增加到 8，
  学习率从 `5e-5` 恢复到 `1e-4`；
- 关闭 clipped value loss，AdamW weight decay 恢复到 `1e-4`；
- SMP 固定权重从 5 增加到 10，恢复早期 pose gate minimum 从 0.1 降为 0；
- Actor 观测、torso 辅助力、三类 reset 和成功率课程保持不变。

1024 环境、RTX 4060 Ti 8 GiB 的 5 轮 smoke 验证通过：无 OOM、NaN
或动作饱和，第 4 轮 `|log ratio|=0.0074`，相比失败配置的末段
更新强度提高约 3.5 倍。该 smoke 只验收显存与数值链路，不代表长期
起身成功率验收。

## 23. 2026-08-30 连续初始姿态课程修正

FPO++ 对齐后的 500 轮实验中，mean reward 从 4.07 升至 14.46，SMP raw score
从 0.044 升至 0.222，reference 成功率达到约 47%，但 fallen 成功率末段仍只有
约 0.5%。Flow pair std 保持在 0.227，说明算法更新和多峰性已经稳定，瓶颈转为
reset 课程：旧 level 0 同时采样 `[0.70,0.85)` reference 和 `[0,0.25)` fallen，
中间存在难度断层，且进阶只统计 fallen 成功率。

修正后第一阶段在固定完整辅助力下由 pure-reference 后段逐步开放更早姿态和
fallen；姿态满级后第二阶段才逐级降低辅助力。进阶统计覆盖 reference 与 fallen
的全部恢复 episode，排除 stand，阈值固定为 90%，证据窗口跨两个阶段持续增长。
辅助力不再在单回合内按进度释放，成功判定也不要求当前辅助力为零；最终课程
等级仍为全程 0 N。该课程只改变环境 reset 和训练统计，不向 Actor 或 Critic
增加 reset mode、辅助力或课程等级观测。

后续 500 轮验证在旧 posture level 3（直接从 progress 0.40 下探到 0.25）停留：
末段成功率约 76%，而 `[0.25,0.40)` 一次新增 701 个帧，占新采样池约 49%。
因此把 reference 下探细化为 `0.40→0.35→0.30→0.25→0.20→0.15→0.10`，
再按 `100/0/0→85/15/0→70/30/0→60/35/5→45/45/10` 开放其他模式。
新增十组互斥 initial-progress bin denominator/success 指标，使每个区间的条件
成功率可以由 `recovery_success_bin_x / recovery_bin_x` 直接计算。

细化后的 500 轮验证在 progress `0.30–0.35` 的课程前沿停滞：该区间末段
成功率约 60%，但容易历史区间稀释了训练信号。因此当一级新开放更低的
reference progress 时，采用 50% 当前新开放区间和 50% 已学历史区间的
前沿平衡采样。例如当前 level 为 `0.30` 时，分别采样 `[0.30,0.35)` 和
`[0.35,0.85)`。该补丁不改变总体 90% 进阶条件：新区间 85% 且已学区间
95% 时，平衡后总成功率正好为 90%。当 progress 下界不再降低、课程开始加入
fallen/stand 后，reference 恢复对整个 `[0.10,0.85)` 区间均匀采样。

前沿平衡实验进一步显示 posture level 3 的 `[0.35,0.40)` 成功率从约 72%
提升到 80%，但末段总成功率稳定在 86% 而无法达到 90%。为避免继续对整个
progress 区间盲目加权，当前实现把 42 个训练 clip 按 0.2 秒分成 315 个
temporal bins，在不改变“前沿/已学 50:50”配额的前提下，对每个配额内的
bin 使用失败 EMA 自适应采样：

```text
failure(bin) = 1 - episode_success

failure_ema(bin)
  = (1 - 0.01) × previous_failure_ema
    + 0.01 × failure

sampling_probability
  = 0.5 × normalized_failure_ema
    + 0.5 × uniform_probability

sampling_probability(bin)
  <= 3 × uniform_probability(bin)
```

所有 bin 的初始 difficulty 相同，因此新课程刚开始时仍为均匀采样；随着成功
样本使对应 failure EMA 下降，采样自动集中到持续失败片段。50% 均匀基线
保证已学 bin 不会被完全遗忘。每累计 10000 个恢复回合固定一次 top-5
困难 bin 报告，TensorBoard 记录 clip index、temporal-bin index、原始 CSV 帧范围、
尝试数、成功率和 failure EMA。训练后可使用：

```bash
uv run python -m mjlab.tasks.velocity.scripts.report_g1_recovery_hard_bins \
  --run-dir logs/rsl_rl/g1_recovery_s2/<run>
```

该自适应状态只存在环境 reset/curriculum 内，不向 Actor 或 Critic 添加 clip、
bin、reset mode 或辅助力观测。

首次 500 轮自适应实验发现，原 80% 自适应权重使一个成功率仅 0.8% 的
bin 占据前沿约 15% 采样，累计 8436 次尝试仍无明显改善，课程因此比
均衡采样版本更早停在 posture level 2。上述 50% 均匀基线和单 bin 3 倍
概率上限用于防止近乎不可解的片段劫持 rollout。

训练 reset 中另有 25% 非 stand 回合使用固定 probe 分布：仍保持前沿/已学
50:50，但区间内不应用失败 EMA。90% 课程进阶只使用这些 probe 的全部
reference + fallen 成功率；其余 75% 回合使用有上限的自适应采样训练。
这使训练器可以持续查找难点，而课程成功率仍对应固定、可跨时间比较的
reset 分布。probe 标志只用于环境统计，不进入策略观测。

带上限自适应采样的 500 轮实验最终到达 posture level 3，但固定 probe 的
完整窗口成功率长期停在约 84%--86%，而自适应训练分布成功率约 74%。为验证
前沿平衡策略是否只需要更长的收敛时间，下一次长训练恢复到自适应采样之前的
固定分布：每一级仍按 50% 新开放 frontier 和 50% 已学区间分组，各组内部按
符合 progress 条件的帧均匀采样。所有非 stand 恢复回合都进入 90% 课程窗口；
failure EMA、0.2 秒 temporal bin 和 top-5 报告继续记录，但不再影响 reset
概率。对应配置为 `curriculum_probe_probability=1.0`、
`adaptive_uniform_probability=1.0` 和 `adaptive_max_probability_ratio=1.0`。
Actor/Critic 观测、SMP、Flow/FPO、辅助力和两阶段课程均保持不变。

固定 50:50 分组的 3000 轮实验在第 575 轮以 1262 个完整窗口样本、90.02%
成功率从 posture level 3 进入 level 4，证明前一实验偶尔达到 90% 并非完全
无效信号。但 level 4 此后约 2400 轮停在 73%--76%，其中新开放的
`[0.30,0.35)` 区间约 63%。原 `recovery_lift` 直接奖励 0.7 名义高度以下的
正向 root z 速度，既可被上下振荡重复获得，也会把 reference reset 继承的初始
速度和 torso 辅助力误归因为策略抬升。现改为 episode 内不可重复的恢复里程碑：

```text
progress(t) = height(t) × uprightness(t)
best(t) = max(best(t-1), progress(t))
milestone_lift(t) = max(best(t) - best(t-1), 0) / dt
```

`best` 在 reset 后由首个状态初始化，因此初始 reference 速度不产生奖励；回落后
重新达到旧高度也不重复计奖。该项权重由 0.5 调到 1.0，单步 rate 上限为 5.0。

恢复任务也已采用进度滑动权重：pose 在低 progress 时关闭，接近站立时平滑
增强。由于 get-up 数据包含大量 support 阶段且整体 root 高度偏低，pose 现在从
progress 0.65 开始平滑开启，到 0.85 达到满权重；满权重由 1.0 提高到 2.0。
这样早期翻身、撑地和跪起仍由 SMP 与恢复进展主导，而成功保持的最后 0.5 秒会
更强地把 29 个关节拉向 velocity 任务默认站姿。

里程碑lift与末段pose的800轮消融仍停在posture level 3：末段完整窗口约
87%--88%，`[0.35,0.40)` frontier约82.5%，与旧版本突破前基本相同。日志同时
显示SMP raw约0.18，远低于随progress上升到约0.87的task cap，因此原cap实际
没有承担末段交接。当前补丁保留progress 0.65前的SMP权重10，并在
`[0.65,0.85]`用与pose相同的smoothstep降到2.5；pose同期从0升到2。权重交接
不依赖reset mode、辅助力或课程等级，也不新增策略观测或验收指标。

## 24. 物理初始化库修正

3000 轮训练中的四个长期零成功 temporal bin 经可视化确认存在跪姿手掌悬空，
且原 audit 只对无地面的 G1 XML 执行 `mj_forward`，不能验证地面支撑或重力下
稳定性。两个相关 clip 还分别存在 2.86 cm 和 8.44 cm 自穿透，但旧配置没有把
深穿透作为硬失败。

现保留 `clips/*.npz` 原始轨迹供 SMP 使用，新增独立
`physical_init.npz` reset bank。构建器对训练 split 的每个候选帧执行：

1. 使用 G1 velocity 全身 `condim=3`、摩擦系数 0.6 的碰撞模型和真实地面；
2. 拒绝超过 1 cm 的初始自穿透；
3. 对距地面 5 cm 内的脚、胫、手、腕和肘支撑点做有界 root/关节接触投影；
4. 用真实 PD 执行器在重力下保持姿态并运行 0.4 秒；
5. 检查接触持续率、接触力、地面/自穿透以及 root 沉降位移和旋转；
6. 保存仿真沉降后的 qpos 和零速度，reset 只从通过状态采样。

默认标准从 2949 个 train 帧中保留 910 个（30.9%）。修正后各 progress 区间
均有候选：`[0.30,0.35)` 33 帧、`[0.35,0.40)` 29 帧、`[0.40,0.55)`
28 帧、`[0.55,0.70)` 59 帧和 `[0.70,0.85)` 70 帧。原始SMP动作、Actor/Critic
观测、课程成功率和辅助力均不改变。生成命令：

```bash
uv run --python .venv/bin/python --no-sync python -m \
  mjlab.tasks.velocity.scripts.build_g1_physical_init
```

训练模式要求该文件存在，防止无意间退回未经物理验证的原始reset；play模式在
文件存在时同样使用它。

在纯净物理库之上另建 `noisy_physical_init.npz`。每个候选噪声方向的关节位置
范围为 `±0.1 rad`，关节速度范围为 `±0.1 rad/s`；root 整体抬高 3 cm，使训练
从无地面接触的短暂自由落体开始。构建器在噪声尺度
`[0, 0.25, 0.5, 0.75, 1]` 上检查初始自碰撞和地面相交，只保存整条尺度方向
通过的候选。训练 reset 再为所选安全方向连续采样 `λ∈[0,1]`，因此每个父帧
能够产生无限多个局部状态，而不是固定复用少量离散变体：

```text
q_reset = q_clean + λ × delta_q_safe
qdot_reset = qdot_clean + λ × delta_qdot_safe
```

抬高产生的 3 cm 不参与课程 progress 划分，noisy 状态始终继承纯净父帧的
`height × uprightness` 难度标签。SMP 内部使用一个不进入 Actor/Critic 观测的
全身—地面接触传感器；连续3个 control step 有接触后确认落地，清空落地前
历史，再收集10个落地后真实帧才开始计算 SMP。生成命令：

```bash
uv run --python .venv/bin/python --no-sync python -m \
  mjlab.tasks.velocity.scripts.build_g1_noisy_physical_init
```

13 个原始姿态等级展开为 26 个 clean/noisy 子等级。除第一个 noisy 等级使用
50% 当前 clean 和 50% 当前 noisy 外，此后的 clean 等级使用 50% noisy history
和 50% clean frontier，noisy 等级则两部分全部使用 noisy 状态。clean 子等级
保留 24 control-step 冷却；只有 noisy 子等级要求至少停留
`60 × 24 = 1440` control steps，并与 90% 总成功率窗口同时满足后才进阶。证据
窗口在冷却期间持续轮换，避免刚进入 noisy 等级时的早期失败永久压低成功率。
最终 noisy 姿态等级通过后才进入辅助力退火。

物理初始化查看器支持三种来源：`motion` 播放 SMP 使用的原始轨迹，`physical`
只播放 `physical_init.npz` 中训练 reset 实际加载的沉降姿态，`compare` 将两者
按相同 target frame 在 Y 方向并排显示。修正模式会列出区间内真正通过审核的帧、
投影/沉降/穿透审计范围，以及每帧与地面接触的具体碰撞几何；MuJoCo 窗口同时
开启 contact point 和 contact force。由此可区分“原动作视觉上像手撑地”和“训练
状态中手掌碰撞体确实接触地面”。当前物理审核的 `support_fraction` 只证明至少
一个机器人碰撞体持续接触地面，尚不表达动作阶段所期望的手、膝或脚接触语义；
该差异必须在后续重编 reset bank 前单独处理，不能用任意支撑接触替代。

## 25. 大并行课程窗口与 Fallen 分段开放

4096 环境远程训练在 10 个 optimizer iteration 内从 posture level 0 到达 level
9，随后约 330 轮停在 63% 左右。该阶段 reference 条件成功率约 70%，fallen
约 22%；按 level 9 的 85:15 混合后正好得到约 63%。分 progress 统计显示
`[0.15,0.25)` 已有约 50%--77% 成功率，但 `[0,0.10)` 仅约 2%，而原 level 9
会一次性从整个 `[0,0.25)` fallen 库采样，且物理初始化库中 52.9% 的 fallen
帧集中在 `[0,0.10)`。因此这里的瓶颈是课程分布突变，不是 FPO 数值失稳；同期
ratio、clipping、gradient、动作饱和度和 value loss 均正常，所以不调整
Actor/FPO 超参数和奖励权重。

课程统计和采样现采用以下补丁：

1. 每个回合在 reset 时记录 posture level 与 assist level。只有在当前等级 reset
   的回合才进入当前 90% 进阶窗口；切级时尚未结束的旧回合仍进入 temporal-bin
   诊断，但记为 `excluded_stale_attempts`，不能为新等级提供成功证据。
2. 成功窗口以 1024 环境为基准随并行数只增不减。4096 环境下所有窗口扩大 4
   倍，例如 level 0 从 500 变为 2000，level 9 从 3500 变为 14000。日志同时
   输出 `base_required_window`、`curriculum_window_scale` 和实际
   `required_window`。
3. 两次进阶至少相隔 24 个环境 step，即默认一个完整 rollout，使新等级至少经过
   一次策略更新后才能再次被验证。进阶仍必须达到 90% 总非 stand 成功率，这个
   间隔不是按轮次替代成功率课程。
4. Fallen 比例仍按原计划在 level 9--12 为 15%、30%、35%、45%，但 progress
   下界分别为 0.15、0.10、0.05、0.00，上界保持 0.25。最难的
   `[0,0.05)` 状态只在姿态课程最后一级开放；辅助力课程仍要等姿态课程满级后
   才开始，并在每个回合施加当前等级的完整辅助力。

Actor/Critic 观测、SMP/pose 权重、reference 的 50:50 frontier 采样、FPO 和
0.9 成功率阈值均未改变。

## 26. 参考来源

- [SMP: Reusable Score-Matching Motion Priors for Physics-Based Character
  Control](https://arxiv.org/html/2512.03028v3)：Diffusion 噪声预测、SMP
  reward、ESM `[22,15,8]`、adaptive normalization、10 步动作窗和 Getup task。
- [Flow Matching Policy Gradients](https://arxiv.org/html/2507.21053)：以固定
  timestep/noise 对估计 CFM loss ratio，在保留 GAE、Critic 和 PPO clipping 的
  同时训练统一多峰 flow policy。
- [FPO++: Flow Policy Optimization for Scalable Robot
  Learning](https://arxiv.org/abs/2602.02481)：每个 Monte Carlo 样本独立计算
  ratio、正优势使用 PPO、负优势使用 ASPO/SPO，并在评测和部署使用
  zero-sampling，修正 vanilla FPO 在机器人控制中的训练不稳定。
- [Robust and Generalized Humanoid Motion Tracking
  (RGMT)](https://arxiv.org/html/2601.23080v1)：G1 93D observation、38D
  command、10 步 causal history、21 步 cross-attention、residual PD target、
  asymmetric critic、recovery reset 和退火辅助力。
- [Extreme-RGMT](https://arxiv.org/html/2607.20110v1)：temporal-bin failure
  EMA、uniform-baseline adaptive sampling，以及 G1 PPO 和随机化量级。

当前系统中，参考数据提供由易到难的初始状态并训练 SMP；冻结 SMP 从训练初期
评价策略生成的短时状态运动分布，站立势函数和净进展规定全局恢复方向，FPO 和
MuJoCo 接触动力学负责产生完整可执行起身。逐帧 reference tracking 仅作为后续
消融，不进入当前自主起身任务。
