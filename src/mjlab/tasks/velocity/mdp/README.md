# Velocity MDP 机器人接入说明

`mjlab.tasks.velocity.mdp` 提供与机器人形态无关的速度跟踪、足式行走和
倒地起身 MDP 项。机器人配置不应重新实现这些状态机和奖励函数，
只需提供本机器人的模型、名称映射、物理参数和训练范围。

## 公共 MDP 已提供的内容

- `UniformVelocityCommandCfg`：平面线速度和偏航角速度指令。
- 速度跟踪、直立、姿态、足端腾空、离地高度、打滑和落地冲击奖励。
- 观测、reset、push、地形课程和速度指令课程。
- `PosturePhaseEstimator`：基于相对站立高度和直立度的形态归一化
  姿态阶段估计。
- `RecoveryAssist`：倒地 reset、向上辅助力、成功/失败判定和辅助力
  课程。
- `recovery_potential_progress`、`recovery_time_penalty`和成功/失败信号。
- `recovery_gates.py`：在倒地、起身和正常行走之间连续缩放行走奖励。

## 每台机器人必须准备的基础内容

### 1. 机器人资产

在 `src/mjlab/asset_zoo/robots/<robot>/` 中准备：

- MJCF/XML 模型和它引用的 mesh、texture 等资产。
- 精确的 link 质量、惯量、关节轴、关节限位和碰撞体。
- 执行器类型、减速比、力矩/速度限制、刚度、阻尼和转子反射惯量。
- 站立初始状态 `HOME_KEYFRAME`，包含 root pose、默认关节位置和关节速度。
- 关节位置动作的 scale 和物理 target clip。
- 创建新 `EntityCfg` 的工厂函数，例如 `get_<robot>_robot_cfg()`。

模型中的关节、body、site 和 geom 名称是任务配置的公共接口，应保持
稳定。

### 2. 行走任务映射

在 `src/mjlab/tasks/velocity/config/<robot>/env_cfgs.py` 中从
`make_velocity_env_cfg()` 创建基础配置，然后至少覆盖：

- `cfg.scene.entities["robot"]`：本机器人的 `EntityCfg`。
- 主躯干 body：用于视角、IMU、直立奖励和躯干角速度奖励。
- 左右足 body/site/geom：用于足地接触、足高、腾空时间、打滑和落地
  冲击。
- 地形扫描传感器的跟随 frame，以及适合机器人脚尺寸的扫描半径。
- 脚部摩擦随机化的 geom 列表、躯干 COM 随机化的 body。
- 关节位置 action scale 和 clip。
- 站立、行走和跑步速度下的逐关节姿态容差。
- 奖励权重、速度指令范围、终止角度、push 范围和负载范围。
- 与机器人动作维度匹配的 PPO 训练配置。

## 启用倒地起身时每台机器人必须准备的内容

### 1. 规范倒地姿态

至少提供一个倒地 keyframe，通常包括：

- 仰卧。
- 俯卧。
- 左侧卧，可选增加右侧卧。

`poses` 中的姿态格式为：

```python
{"pos": (x, y, z), "quat": (qw, qx, qy, qz)}
```

root 高度应保证碰撞体不在地面下方，但也不应让机器人悬空。

### 2. 辅助力配置

每台机器人需要单独设置：

- `asset_cfg.body_names`：施加向上辅助力的单个躯干 body。
- `force_ranges`：从强辅助到 `0 N` 的课程阶段。
- `force_ramp_up_s` 和 `force_ramp_down_s`：避免辅助力突变。
- `fall_confirm_s`：防止正常步态中的短暂低姿态被判定为倒地。
- `upright_hold_s`：必须在无辅助力下保持直立的成功确认时间。
- `recovery_timeout_s`：单次起身尝试的最长时间。

`force_ranges` 不能直接从其他机器人复制。它至少应根据机器人质量、
躯干受力点、关节力矩和起身时间重新标定。

### 3. 倒地状态随机化

需要根据关节结构配置：

- `root_height_range`。
- `root_lin_vel_range` 和 `root_ang_vel_range`。
- `joint_position_ranges`：按关节名称正则表达式设定。
- `joint_velocity_ranges`：按关节类型设定。
- 关节软限位和 action target clip，避免随机状态或起身动作超过物理
  限位。

### 4. 起身动作帧（可选）

动作帧用于从较干净、可恢复的中间状态开始训练。它们是 reset 状态，
不是要求策略跟踪的动作轨迹。

每行 CSV 不带表头，所有值都是浮点数：

```text
root_x, root_y, root_z,
quat_x, quat_y, quat_z, quat_w,
joint_0_pos, joint_1_pos, ..., joint_N_pos
```

注意：

- CSV 四元数顺序是 `xyzw`；MJCF keyframe 和 `poses` 使用 `wxyz`。
- loader 当前只使用 CSV root 位置的 `z`，每次 reset 会重新随机化 x/y 方向
  和全局 yaw。
- `csv_joint_names` 的顺序必须与 CSV 后面的关节列一致。
- CSV 列数必须等于 `7 + len(csv_joint_names)`。
- CSV 没有关节速度；关节速度由 `joint_velocity_ranges` 在 reset 时采样。
- `frame_files` 中每个 glob pattern 是一个采样源。
- `source_names` 和 `pose_stage_source_weights` 的列数必须等于
  `len(frame_files) + 1`；最后一列始终是规范倒地姿态源。

### 5. 没有起身动作文件的配置

没有 CSV 不会阻止接入。可以只使用规范倒地姿态：

```python
params = {
  # 其他 RecoveryAssist 参数省略。
  "poses": canonical_fallen_poses,
  "frame_dir": "",
  "frame_files": (),
  "csv_joint_names": (),
  "source_names": ("canonical",),
  "pose_stage_source_weights": (
    (1.0,),
    (1.0,),
    (1.0,),
  ),
}
```

这种方式可以运行完整的辅助力课程，但从完全倒地姿态开始探索更难。
建议先使用较大辅助力、低姿态噪声和保守的关节范围，再逐步降低辅助。

## 恢复阶段和行走奖励的衔接

不应在机器人躺倒时强制其跟踪速度指令。启用起身时，需要把通用
行走奖励替换为 `recovery_gates.py` 中对应的 gated 版本：

```text
track_linear_velocity          -> gated_track_linear_velocity
track_angular_velocity         -> gated_track_angular_velocity
upright                        -> gated_upright
variable_posture               -> gated_variable_posture
body_angular_velocity_penalty  -> gated_body_angular_velocity_penalty
angular_momentum_penalty       -> gated_angular_momentum_penalty
action_rate_l2                 -> gated_action_rate_l2
feet_air_time                  -> gated_feet_air_time
feet_clearance                 -> gated_feet_clearance
feet_swing_height              -> gated_feet_swing_height
feet_slip                      -> gated_feet_slip
```

各机器人需单独确定每项奖励的 `gate_min_scale`。倒地时应完全关闭的步态目标
通常使用 `0.0`；直立、动作平滑等对起身仍有帮助的项可以保留少量权重。

## 姿态阶段参数

`PosturePhaseEstimatorCfg` 默认使用：

```python
PosturePhaseEstimatorCfg(
  recovery_confidence=0.60,
  ready_confidence=0.85,
)
```

这两个值基于相对高度和直立度，不是米制高度。如果为某台机器人覆盖
这些阈值，必须保证 `RecoveryAssist` 的成功/倒地判定和行走奖励门控使用
同一组阈值。

## 当前内置机器人状态

| 机器人 | Velocity 行走 | 起身配置 | 起身动作帧 |
|---|---|---|---|
| Unitree G1 | 已接入 Flat/Rough | 未接入 | 未提供 |
| qlmini2 | 已接入 Flat | 未接入 | 未提供 |
| RL_BOY | 已接入 | 已接入 | 已提供 |

RL_BOY 是当前完整起身接入的参考实现：

- 机器人专用参数：
  `src/mjlab/tasks/velocity/config/rlboy/env_cfgs.py`
- 公共起身 MDP：
  `src/mjlab/tasks/velocity/mdp/recovery.py`
- 公共恢复到行走奖励门控：
  `src/mjlab/tasks/velocity/mdp/recovery_gates.py`

## 接入后的最小验证清单

- XML/MJCF 能够单独 compile。
- 动作维度、关节顺序、action scale 和 target clip 一致。
- 所有传感器引用的 body/site/geom 名称都存在。
- 平地 zero-action reset 不会产生 NaN 或穿透。
- 站立状态下 `PosturePhaseEstimator.is_ready` 为真。
- 所有规范倒地姿态下 `needs_recovery` 为真。
- CSV 模式、列数、四元数顺序和关节顺序有自动化测试。
- 辅助力会平滑上升和下降，且不会施加到错误 body。
- 恢复成功后辅助力为零，行走奖励门控恢复到 `1.0`。
- 无法在 `recovery_timeout_s` 内起身时会正确结束 episode。
- 至少完成一次 CPU 配置构建测试和一次 GPU rollout。
