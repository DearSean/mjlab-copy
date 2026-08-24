# G1 Transformer–PPO–Diffusion 联合起身方案（修订版）

## 1. 文档目的与结论

本文给出一套能够在当前 mjlab 项目中逐步落地的 Unitree G1 自主起身方案。
方案以 `lafan-g1-getup/` 中已经重定向到 G1 的高精度轨迹为主要动作先验，
以 G1 velocity 环境的默认屈膝站姿为最终目标，并联合使用：

- **Transformer**：编码一段本体感知历史，识别当前倒地姿态、支撑状态和
  恢复进度；
- **Diffusion**：根据当前状态和最终站立目标，生成短时、多模态的恢复路标；
- **PPO**：在 MuJoCo 闭环动力学中把路标转化为 29-DoF 关节动作，并负责
  接触、平衡、力矩约束和未被数据覆盖的动作连接；
- **G1 velocity 策略目标**：稳定起身后平滑进入站立或速度跟踪，而不是把
  “到达某条动作最后一帧”当作成功。

结论是：该方案在本项目中可实现，但不能把 Diffusion 当成缺失轨迹的真实
补帧器，也不能直接把不完整轨迹末帧和默认站姿拼接成监督样本。推荐采用
“**短期 latent 路标 Diffusion + 目标条件 PPO + 成功轨迹回灌**”的闭环方案。

第一版应先完成 G1 recovery 环境和无 Diffusion 的 PPO 基线，再依次加入
Transformer 和 Diffusion。三者从第一天同时端到端更新，难以区分数据、
接触、奖励和生成模型各自的问题，不应作为首个可运行版本。

## 2. 当前项目与数据基础

### 2.1 G1 控制基础

当前项目已经具备：

- G1 29-DoF 模型、位置执行器和逐关节 action scale；
- G1 flat/rough velocity 环境；
- 关节位置、关节速度、重力投影、历史动作等本体观测；
- 足部接触传感器、姿态奖励、速度奖励和域随机化；
- ObservationManager 历史窗口；
- RSL-RL PPO 训练、checkpoint、play 和导出框架；
- RL_BOY recovery 中已经实现的姿态阶段估计、恢复辅助、成功保持和阶段
  门控逻辑，可复用算法思想，但不能照搬机器人参数。

G1 velocity 实际使用
`src/mjlab/asset_zoo/robots/unitree_g1/g1_constants.py` 中的
`KNEES_BENT_KEYFRAME` 作为默认状态，而不是 `HOME_KEYFRAME`。其主要值为：

| 项目 | 默认值 |
|---|---:|
| pelvis 高度 | 0.76 m |
| 左右 hip pitch | -0.312 rad |
| 左右 knee | 0.669 rad |
| 左右 ankle pitch | -0.363 rad |
| elbow | 0.6 rad |
| joint velocity | 0 |

这是一种便于平衡和接管 velocity 的屈膝站姿，适合作为起身后的目标中心。

### 2.2 `lafan-g1-getup` 数据事实

以当前 `lafan-g1-getup/index.json` 和 CSV 静态审计结果为准：

- 63 条 G1 CSV，共 3345 帧；
- 每帧 36 列：root position 3、root quaternion `xyzw` 4、G1 joint 29；
- 轨迹来自 6 个 source recording、5 个 subject；
- 30 条标记为 `success`，33 条标记为 `partial`；
- 32 条 `stationary`、18 条 `support_only`、13 条 `locomotion`；
- 严格定义为 `success` 且终点为 `stationary/locomotion` 的完整恢复共有
  26 条；
- 62 条 `terminal_safe=true`，1 条为 false；
- 所有轨迹均有人工确认的 `support_start` 和 `support_complete`；
- 43/63 条轨迹从局部第 0 帧就已经进入 `support_start`，说明数据对可靠
  支撑阶段覆盖较好，但对“倒地静止到首次建立支撑”覆盖不足；
- 一些轨迹在 `support_complete` 后没有足够长的稳定站立尾段；
- 当前帧率未写入 manifest。已有转换脚本默认输入 30 Hz，但正式编译前必须
  从数据来源确认，不能把默认值当作数据事实。

这些数据的正确定位不是“63 条完整起身示范”，而是：

1. 26 条完整恢复示范；
2. 大量高质量局部支撑与姿态转换示范；
3. 若干右删失轨迹：它们已有部分是有效正样本，但没有观测到最终站立；
4. 不足以独立监督所有倒地初态到默认站姿的完整动力学连接。

### 2.3 现有配置中阻碍起身的部分

当前 velocity 通用配置包含 `fell_over` 终止：机器人倾角超过约 70° 就结束
episode。恢复训练必须移除它，否则策略没有机会从地面起身。

当前 G1 完整碰撞配置中，脚部碰撞为 `condim=3`，其他碰撞几何主要为
`condim=1`。手、前臂、膝/小腿需要在起身中提供切向摩擦，因此必须为
recovery 任务建立专用碰撞配置并进行物理验证。

## 3. 任务定义

### 3.1 最终目标不是单帧，而是稳定站立吸引域

最终目标以 `KNEES_BENT_KEYFRAME` 为中心，但定义成容许小范围扰动的状态
分布 `G_stand`：

```text
G_stand = {
  q       接近 default_joint_pos，
  pelvis  接近默认高度，
  torso   基本竖直，
  dq      足够小，
  v_base  足够小，
  contact 双脚形成稳定支撑，
  unsafe  无持续非预期碰撞、无明显滑移和超限
}
```

推荐第一版 flat terrain 成功判据：

- pelvis 相对地面高度不低于默认高度的 85%；
- torso uprightness 不低于 0.90；
- base 线速度和角速度低于配置阈值；
- 双脚接触稳定，手、膝、躯干不再持续承重；
- 关节位置在软限位内，且总体接近默认屈膝站姿；
- 上述条件连续保持 0.5 s。

具体数值应通过初始 PPO 基线校准，不能用单个关节角误差替代动力学稳定性。

### 3.2 episode 阶段

恢复 episode 分为四个连续阶段，阶段值用于奖励门控和课程，不应作为只能
人工获得的 actor 输入：

1. **fallen**：高度低、躯干不竖直，尚未建立有效支撑；
2. **support/recovering**：手、肘、膝、脚等正在建立和转换支撑；
3. **upright-unstable**：已经接近站高，但仍有较大运动或非足部承重；
4. **ready/velocity**：稳定站立，逐步恢复 velocity 命令。

可以复用 `PosturePhaseEstimator` 的连续高度、uprightness、motion stability
和 walk gate 思路，增加 G1 专用支撑接触状态。

### 3.3 控制频率与时间定义

当前 velocity 环境通常以 50 Hz policy 频率运行。原 CSV 很可能为 30 Hz，
所以所有窗口、保持时间和 Diffusion horizon 必须用秒定义，再由编译器映射
到目标帧率。禁止直接把“10 帧”在 30 Hz 数据和 50 Hz 仿真之间等同使用。

建议：

- policy 频率：50 Hz；
- Transformer 历史：0.4–0.8 s；
- Diffusion 短期路标：0.2–0.4 s；
- 每 0.1–0.2 s 重规划一次，可保留上一规划并采用 receding horizon；
- PPO rollout 初始使用 48–96 steps/env，使 rollout 覆盖 0.96–1.92 s；
- 成功保持：0.5 s。

## 4. 总体架构

```text
                    离线/训练期
  lafan-g1-getup ──► G1 数据编译器 ──► 有效窗口、mask、事件、分组 split
                              │
                              ├──► 冻结结构编码器 E_ref
                              │
                              └──► 条件短期 Latent Diffusion D_theta

                    在线 PPO rollout
  G1 历史本体状态 ──► Transformer E_policy ──► h_t
              │                              │
              │                              ▼
              │                 D_theta(h_t, z_stand, support, goal)
              │                              │
              │                         z_goal / waypoint
              │                              │
              └──────────────────────────────┤
                                             ▼
                              PPO Actor π(a_t | h_t, z_goal, command)
                                             │
                                             ▼
                              MuJoCo + 29-DoF position action
                                             │
                         ┌───────────────────┴───────────────────┐
                         ▼                                       ▼
                  PPO Critic V(s_priv)                  成功轨迹 replay
                                                                 │
                                                                 └─► 再训练 Diffusion
```

部署时至少包含 `E_policy + Actor`。若在线使用 Diffusion，则还包含一个轻量
denoiser 和低步数 DDIM/DPM sampler；也可以在训练完成后把 Diffusion 路标
蒸馏进 Actor，使部署不再运行 Diffusion。

## 5. 数据编译与监督规则

### 5.1 建立版本化 G1 recovery dataset

不能让训练代码直接遍历 CSV 并依赖隐式顺序。应新增确定性编译步骤，将
`lafan-g1-getup` 转成统一 NPZ 或 shard：

- 明确记录输入 FPS、输出 FPS=50、坐标系、单位和 quaternion 顺序；
- quaternion 从 CSV `xyzw` 转为项目内部使用的 `wxyz`；
- root XY 平移归一化到 clip 起点，并做 heading/yaw 归一化；
- quaternion 用 SLERP 重采样，位置和关节角使用时间一致的插值；
- 通过 G1 FK 计算关键 body 的位置、速度和离地高度；
- 记录每个 clip 的来源、subject、recording、原始帧范围和哈希；
- 输出 `valid_mask`、`support_mask`、`terminal_mask`、`terminal_safe`、
  `has_standing_terminal` 和事件索引；
- 检查 NaN/Inf、关节硬限位、速度尖峰、地面穿透、接触冲击和 quaternion
  连续性；
- 编译可重复，相同输入与配置必须得到相同产物哈希。

现有 `src/mjlab/scripts/csv_to_npz.py` 可以复用插值、SLERP、速度估计和 G1
FK，但要改成批量、本地、确定性编译器，不能继续依赖固定 `/tmp` 输出和
W&B 上传作为核心数据路径。

### 5.2 防止数据泄漏

数据有效多样性主要来自 6 个 source recording，而不是 63 条彼此独立的
轨迹。Train/Validation/Test 必须按 source recording 分组；同一 recording
切出的 clip、同一时段的相邻窗口不能跨集合。随机按窗口划分会严重高估
泛化能力。

### 5.3 不完整轨迹的损失规则

不完整轨迹采用右删失处理：

- 有效范围内所有局部转移都可参与 Transformer 和局部 Diffusion 训练；
- `support_start/support_complete` 可参与阶段和支撑辅助任务；
- clip 边界以后不存在的帧不补零、不复制最后一帧、不插值到默认站姿；
- `partial` 或 `support_only` 不计算“必须到达站立”的 terminal loss；
- 只有 `success` 且 `stationary/locomotion` 的完整轨迹计算真实终端连接损失；
- `terminal_safe=false` 的轨迹默认不作为正向控制目标，可用于失败/质量判别；
- locomotion 终点可以作为进入 velocity 接管的正样本，不必强迫先完全静止。

### 5.4 默认站姿数据扩充

单纯重复一帧默认姿态会让模型学习到过窄的终点。应在 MuJoCo 中收集
`G_stand` buffer：

- 从默认屈膝姿态开始；
- 加入小关节扰动、root 姿态扰动、速度扰动和轻微外力；
- 使用现有 velocity policy 或站立 PPO 稳定 1–3 s；
- 只保留满足稳定判据的窗口；
- 覆盖不同摩擦、质量、COM、执行器增益和观测噪声。

该 buffer 用于定义 `z_stand` 分布和站立终端判别器，但不能凭空提供
“支撑末端到站立”的中间动作监督。

### 5.5 仿真成功轨迹回灌

PPO 发现的连接轨迹只有通过以下过滤后才能加入 replay：

- 完成稳定站立或安全进入 velocity；
- 无硬限位违规和持续地面穿透；
- 峰值力矩、速度、功率和接触力在设定范围；
- 无明显手/膝滑移和高频 action 抖动；
- 初始状态和恢复路线具有足够多样性。

replay 必须标记来源为 `retargeted_demo` 或 `sim_validated`，评测时分别报告，
避免把模型自己生成的数据当成人工真值。

## 6. Transformer 表征

### 6.1 Actor 可部署观测

建议每个 50 Hz 时刻的 actor 观测至少包含：

- projected gravity / torso orientation 的可部署表示；
- base angular velocity；
- 29 个 joint position 相对默认姿态；
- 29 个 joint velocity；
- 上一步 29 维 action；
- velocity command；
- 可由实机可靠获得的足部接触或估计量；
- 必要时加入由 IMU/运动学估计的 base linear velocity。

绝对 pelvis 高度、精确全身接触力和仿真真值 base linear velocity 只能在
实机确实具备相应估计器时进入 actor。否则只放入 privileged critic。

### 6.2 结构建议

第一版可使用：

- 历史长度：20–40 帧；
- model dim：128 或 256；
- 3–4 层 causal Transformer；
- 4 heads；
- latent `h_t`：64 或 128 维；
- reset 时清空历史并使用有效长度 mask，禁止跨 episode 污染。

Transformer 只编码到当前时刻，在线路径不能使用未来帧。训练期允许另有
双向 `E_ref` 编码完整短窗口，用来建立稳定的动作度量。

### 6.3 `E_policy` 与 `E_ref` 分离

不建议一个不断被 PPO 更新的编码器同时承担以下三件事：

1. actor 的策略特征；
2. Diffusion 的训练坐标系；
3. latent 奖励的度量坐标系。

否则 reward 坐标会随 PPO 一起漂移。推荐：

- `E_ref`：在 G1 数据和仿真稳定窗口上预训练，版本化后冻结；
- `E_policy`：供 Actor 使用，可在 PPO 中更新；
- 用蒸馏/对齐损失使 `E_policy` 不偏离 `E_ref` 的结构信息；
- Diffusion target 和 latent 距离始终使用固定版本的 `E_ref`；
- 若采用 EMA，只能把 EMA checkpoint 定期冻结成新的数据版本，不能在同一
  rollout 中让奖励目标持续变化。

### 6.4 预训练辅助目标

可在所有安全有效窗口上训练：

- 下一段 latent/状态预测；
- masked joint/body state reconstruction；
- support phase 分类或连续支撑强度回归；
- recovery progress 排序；
- time-to-support / time-to-ready 的 censored regression；
- 同一轨迹相邻窗口的对比一致性。

辅助标签只在训练期使用，不应成为实机部署所需的人工输入。

## 7. 条件短期 Latent Diffusion

### 7.1 推荐角色

Diffusion 不直接生成电机 action，而是生成未来 0.2–0.4 s 的一个或多个
latent waypoint：

```text
z_goal ~ D_theta(
  noisy_future_latent,
  diffusion_step,
  h_t,
  z_stand,
  posture_condition,
  support_condition,
  remaining_time
)
```

这样利用了 Diffusion 对多模态恢复路线的优势，又把动态可行性留给 PPO。
从相似侧卧状态出发，模型可以提出“先翻俯卧”“单膝跪地”或“双手撑起”
等不同候选，而不是把多种路线平均成不可执行姿势。

### 7.2 训练样本

局部样本由同一 clip 内的前缀窗口和后续窗口组成：

```text
condition = E_ref(x[t-H+1:t]) + support/posture + z_stand
target    = E_ref(x[t+1:t+K])
```

要求 `t+K` 不超过有效 clip 边界。完整恢复轨迹可以额外采样跨越
`support_complete → ready` 的窗口。不完整轨迹只提供其实际存在的局部未来，
不能被伪造为到站立的条件对。

### 7.3 默认站姿的用法

`z_stand` 是条件和终端引导，不是把每条轨迹最后一帧替换成默认姿态：

- 从稳定站立 buffer 抽取 0.3–0.5 s 窗口，经 `E_ref` 得到终端 latent 分布；
- 条件 Diffusion 始终知道长期目标是 `G_stand`；
- 真实完整轨迹监督如何进入该分布；
- PPO 在仿真中探索数据缺失的连接；
- 经物理验证的连接回灌后，Diffusion 才逐步学会更多完整路线。

可以离线做“固定前缀 + 固定终点”的 diffusion inpainting 作为候选生成和
可视化，但 inpainting 结果必须经过 MuJoCo rollout/优化验证，不能直接视为
可执行 reference。

### 7.4 在线采样与候选选择

每次重规划可生成 2–4 个候选短期路标，由 feasibility/value 网络选择：

```text
score = predicted_success
      - torque_risk
      - contact_impact_risk
      - joint_limit_risk
      - deviation_from_stand_progress
```

首版建议 4–8 步 DDIM，并每 5–10 个 policy step 重规划一次，不要每个 50 Hz
控制步完整运行几十步 denoising。采样延迟必须计入实机控制预算。

### 7.5 不推荐的做法

- Diffusion 直接输出 29-DoF torque/action；
- 给每条 partial 轨迹强行附加默认站姿；
- 对缺失区间做关节角线性插值并当真值；
- 把随机采样的 `z_goal` 只放进 reward，而不输入 Actor/Critic；
- 使用随 PPO 在线漂移的 encoder 定义 diffusion target；
- 一次生成 4–8 s 完整恢复并要求低层 policy 盲目跟踪。

## 8. PPO 控制与联合目标

### 8.1 Actor 和 Critic

Actor：

```text
a_t ~ pi_phi(a | h_t, z_goal, velocity_command, planner_age)
```

其中 `planner_age` 表示当前 waypoint 已执行多久，避免 Actor 不知道路标的
时间相位。action 仍使用项目现有 G1 关节位置 action scale。

Critic 可以使用 privileged 状态：

- root height 和线速度真值；
- 全身接触和接触力；
- posture phase；
- 关节力矩/功率；
- Diffusion 候选质量和目标；
- 域随机化参数。

Actor 不能依赖实机不可获得的 privileged 状态。

### 8.2 因果路标奖励

原始“在时刻 `t` 比较未来 `x[t+1:t+K]`”的奖励无法由当前 RewardManager
直接计算。修订方案使用以下两种可实现方式之一：

1. Diffusion 输出按时间索引的短期 waypoint 序列，每个环境步比较当前状态
   与该步目标；
2. Diffusion 输出 chunk 终点，只在 K 步结束时比较刚刚完成的历史窗口和
   `z_goal`，并把目标在这 K 步内持续输入 Actor/Critic。

第一版推荐方法 1，奖励更稠密，也不需要修改 PPO storage 来追溯未来。

### 8.3 奖励结构

总奖励建议写成阶段门控形式：

```text
r = w_progress * r_progress
  + w_support  * r_support
  + w_goal     * r_waypoint
  + w_ready    * r_stable_stand
  + g_walk     * r_velocity
  - w_limit    * c_joint_limit
  - w_tau      * c_torque
  - w_power    * c_power
  - w_rate     * c_action_rate
  - w_impact   * c_contact_impact
  - w_slip     * c_support_slip
  - w_self     * c_self_collision
```

建议含义：

- `r_progress`：pelvis 归一化高度 × torso uprightness 的改善量和绝对值；
- `r_support`：阶段相关的有效支撑，不把手/膝接触一概当错误；
- `r_waypoint`：当前结构状态与 Diffusion 路标的相似度；
- `r_stable_stand`：进入并保持 `G_stand`；
- `r_velocity`：由 `walk_gate` 平滑开启；
- `c_support_slip`：承重手、膝、脚的切向滑移；
- `c_contact_impact`：高冲击接触，不惩罚正常建立支撑；
- `c_torque/c_power`：使用真实执行器约束，防止仿真可行但实机危险。

注意：恢复期不能始终强力奖励默认关节姿态，否则策略会在躺地时直接把腿
拉向站姿，与建立支撑冲突。默认 pose reward 应随 upright/ready 阶段逐步
增强；速度跟踪奖励同样在稳定站立后开启。

### 8.4 PPO 与辅助损失

联合优化建议为：

```text
L_total = L_PPO
        + lambda_align * L_policy_ref_align
        + lambda_pred  * L_local_prediction
        + lambda_phase * L_phase_aux
```

Diffusion 不应在每个 PPO minibatch 上被策略梯度直接更新。建议：

- PPO rollout 期间冻结 `E_ref` 和 Diffusion；
- `E_policy` 与 Actor/Critic 按 PPO 更新；
- Diffusion 使用独立 replay dataloader 周期性离线/异步更新；
- 每个 Diffusion 版本在一批 PPO rollout 内保持固定；
- checkpoint 同时记录 policy、encoder、diffusion、normalizer 和数据哈希。

这样仍属于联合系统训练，但避免 PPO 非平稳分布和 diffusion denoising loss
在同一 minibatch 中互相破坏。

## 9. G1 Recovery 环境改造

建议新建独立 task，例如：

```text
Mjlab-Velocity-Recovery-Flat-Unitree-G1
Mjlab-Velocity-Recovery-Rough-Unitree-G1   # 后续阶段
```

不要直接改变普通 G1 velocity 的训练行为。

### 9.1 必须修改

1. 移除 recovery task 的 `fell_over` 终止；
2. 增加 `recovery_succeeded` 和 `recovery_timed_out`；
3. 将 episode 长度设置为足够覆盖起身和站立保持，首版建议 6–8 s；
4. 增加手、前臂、膝/小腿、脚与地面的 contact sensor；
5. 为真实承重支撑几何配置 `condim=3` 和合理摩擦；
6. 增加持续穿透、危险碰撞、关节硬限位或数值异常终止；
7. 恢复期关闭/降低普通 velocity 的 foot clearance、air time、严格默认 pose
   和速度跟踪项，进入 ready 后平滑恢复；
8. 增加 fallen reset、RSI reset、外力和历史失败状态 reset；
9. 记录 posture/support phase、起身时间、力矩、接触冲击、滑移和成功保持。

### 9.2 接触建模

不能盲目把全部非足部几何都设成高摩擦三维接触，这可能造成衣物式粘地和
数值负担。应通过轨迹 FK 和 MuJoCo 可视化确定实际承重几何，只为手掌、
必要的前臂区域、膝/小腿和足部启用合理的切向摩擦，并分别监控：

- 法向力；
- 切向速度/滑移；
- 接触持续时间；
- 接触冲击；
- 自碰撞。

完成该审计前，重定向轨迹的“运动学精确”不能等同于“G1 动力学可执行”。

### 9.3 Reset 分布

三类 reset 混合使用：

- **RSI**：从 62 条安全轨迹的有效帧重置，重点利用支撑阶段；
- **物理 fallen reset**：随机仰卧、俯卧、左右侧卧、关节扰动后 settling；
- **failure replay**：从当前策略高失败状态重置。

由于 43/63 条数据开始时已在支撑阶段，物理 fallen reset 是弥补“首次建立
支撑”数据缺口的必要组成，而不是可选数据增广。

### 9.4 训练辅助力

可复用 RL_BOY `RecoveryAssist` 的课程思想，在最初阶段用小幅方向性辅助力
帮助 PPO 学会离地和建立支撑，然后根据成功率退火到零。但 G1 的质量、
执行器、尺寸和支撑几何不同，不能照搬 RL_BOY 的 50 N 力范围、阈值和
关节集合。最终所有正式评测必须为零辅助力。

## 10. 分阶段训练路线

### 阶段 0：数据与物理审计

目标：证明 CSV 语义、时序和 G1 接触模型正确。

- 确认 FPS、坐标系、单位、joint order 和 quaternion order；
- 批量编译 63 条数据到 50 Hz；
- 在 MuJoCo 中逐帧/PD 回放，统计穿透、接触、速度、力矩和功率；
- 建立 recording-level split；
- 生成稳定站立 buffer；
- 输出数据审计报告和可视化。

通过标准：全部数组有限、无关节硬限位违规，所有问题帧可追溯；明确哪些
帧允许 RSI、哪些只能用于 representation。

### 阶段 1：无动作先验的 G1 Recovery PPO 基线

目标：先验证环境、奖励、终止和接触设置。

- 使用 MLP 或现有 actor；
- 从随机 fallen reset 和默认站姿附近 reset 训练；
- 加入阶段门控、恢复进度、稳定站立和安全成本；
- 不使用 Transformer/Diffusion。

该基线若不能稳定训练，优先修复环境和奖励，而不是加入生成模型掩盖问题。

### 阶段 2：RSI 与局部参考基线

目标：验证 63 条数据是否能提高支撑与恢复效率。

- 从安全轨迹有效帧 RSI；
- 使用 nearest-future-frame 或确定性 latent predictor 提供短期路标；
- partial 轨迹只用有效未来；
- 比较无数据 PPO、RSI PPO 和局部路标 PPO。

这一阶段是判断 Diffusion 是否真正优于简单预测器的必要基线。

### 阶段 3：Transformer PPO

目标：让策略基于历史识别接触转换和运动趋势。

- 预训练并冻结 `E_ref`；
- 将 causal `E_policy` 接入 Actor；
- 先保持简单路标或不使用路标；
- 加入对齐和阶段辅助损失；
- 验证导出模型的历史缓存和 reset 行为。

### 阶段 4：局部条件 Diffusion + PPO

目标：利用多模态数据生成短期恢复路标。

- 使用所有安全有效局部窗口训练 Diffusion；
- 使用 26 条完整轨迹训练真实 terminal transition；
- 使用稳定站立 buffer 作为 `z_stand` 分布；
- 在线低步数采样并将 `z_goal` 输入 Actor/Critic；
- 对比确定性 predictor、单候选 diffusion、多候选 value selection。

### 阶段 5：成功连接回灌

目标：逐步补足数据缺失的“支撑→默认站姿”连接。

- 收集 PPO 成功轨迹；
- 严格过滤动力学安全性；
- 保留真实/仿真来源和版本；
- 重新训练/微调 Diffusion；
- 扩大 fallen reset 难度并重复。

### 阶段 6：Velocity 联合接管和 Rough 泛化

目标：起身后自然进入命令速度。

- 训练零命令稳定站立；
- 再加入小速度命令；
- 通过 `walk_gate` 平滑开启完整 velocity reward；
- flat 通过后再加入 rough terrain、摩擦变化和高度差；
- 最后做 system identification 和 sim-to-real 随机化。

## 11. 推荐的初始超参数范围

以下只是首轮实验范围，最终需由基线和消融确定：

| 模块 | 初始建议 |
|---|---|
| policy frequency | 50 Hz |
| Transformer history | 20–40 帧 |
| Transformer dim/layers | 128–256 / 3–4 层 |
| `E_ref` latent | 64 或 128 |
| Diffusion horizon | 10–20 policy steps |
| replanning interval | 5–10 policy steps |
| online denoising | 4–8 DDIM steps |
| candidates | 1 起步，稳定后增加到 2–4 |
| PPO rollout | 48–96 steps/env |
| episode length | 6–8 s |
| success hold | 0.5 s |
| action | 现有 G1 position action scale |
| training terrain | flat 起步，rough 后置 |

原 G1 PPO 的 `num_steps_per_env=24` 只覆盖约 0.48 s，对包含恢复阶段的优势
估计可能过短，建议从 48 或 96 开始对照，同时重新调节 batch、学习率和显存。

## 12. 评测与消融

### 12.1 固定评测集合

评测初态必须与训练采样器分离并固定种子，至少包含：

- 仰卧；
- 俯卧；
- 左侧卧；
- 右侧卧；
- 四肢交叉/不利关节姿态；
- 策略历史失败状态；
- 不同摩擦、质量、COM、执行器强度和传感噪声；
- held-out source recording 的 RSI 状态。

### 12.2 指标

- 5 s 内稳定起身成功率；
- 成功时间分布；
- 站立保持和二次跌倒率；
- support_start、support_complete、ready 的到达率和时间；
- 峰值/均方关节力矩、速度和功率；
- action rate 和高频能量；
- 手、膝、脚接触冲击与滑移；
- 自碰撞和地面穿透；
- 起身后 3 s 的 velocity tracking RMSE；
- Diffusion 采样延迟和整体控制实时率；
- 按姿态、数据来源和 domain randomization 难度分层的成功率。

### 12.3 必做消融

1. 普通 recovery PPO；
2. PPO + RSI；
3. PPO + RSI + 确定性短期 predictor；
4. Transformer PPO；
5. Transformer PPO + 单候选 Diffusion；
6. Transformer PPO + 多候选 Diffusion + value selection；
7. 去掉 partial/support-only 数据；
8. 去掉默认站立 buffer；
9. 去掉成功轨迹回灌；
10. 一个共享漂移 encoder 对比 `E_ref/E_policy` 分离；
11. Diffusion 直接动作对比 latent waypoint（只作研究对照，不作默认方案）。

至少使用多个训练种子，报告均值、标准差和置信区间。不能只展示一条看起来
成功的视频。

## 13. 相对原《Transformer PPO 端到端联合训练》方案的修改

| 原思路或假设 | 本项目中的问题 | 修订方案 |
|---|---|---|
| 只有人体/通用恢复数据 | 当前已有高精度 G1 重定向数据 | 主先验改用 `lafan-g1-getup` 的 G1 原生状态 |
| 每条训练序列都有完整未来 | 许多轨迹没有完整站立尾段 | 右删失 + valid mask；仅有效局部窗口参与未来监督 |
| partial 末端也可视为恢复终点 | 会把截断误当失败或成功 | 单独保留 `outcome/terminal_mode/terminal_safe` |
| Diffusion 补全后可直接执行 | 运动学补全不保证接触和力矩可行 | Diffusion 生成短期 latent 路标，PPO 闭环执行 |
| 默认姿态是一帧终点 | 单帧不代表稳定 | 建立含高度、姿态、速度和接触的 `G_stand` 分布 |
| 所有 partial endpoint 配对默认站姿 | 会产生虚假跳跃监督 | 只以默认站姿作长期条件；连接由完整数据和 PPO replay 学习 |
| `z_next` 只进入 reward | 策略面对随机隐藏目标，形成部分可观测问题 | `z_goal` 必须输入 Actor 和 Critic |
| 当前时刻直接奖励未来 chunk | 当前 RewardManager 无法因果访问未来 | 逐步 waypoint 奖励，或 chunk 完成时延迟比较 |
| 同一个在线 Transformer 定义策略、Diffusion 和奖励 | PPO 更新导致 latent 坐标和奖励漂移 | 冻结 `E_ref`，单独训练 `E_policy` 并对齐 |
| PPO、Transformer、Diffusion 每个 minibatch 同步端到端更新 | 三个非平稳目标互相干扰，难以调试 | 版本化交替联合：PPO rollout 固定 prior，独立 replay 更新 Diffusion |
| 直接沿用普通 G1 velocity 环境 | 倒地即终止，奖励会阻碍支撑动作 | 新建 G1 recovery task，移除 `fell_over` 并阶段门控 |
| 足部接触足以训练起身 | 手、膝、前臂是关键支撑 | 增加支撑传感器和 task-specific `condim=3` 摩擦 |
| 复制 RL_BOY recovery 参数 | 质量、几何、关节和执行器均不同 | 只复用算法结构，重新标定 G1 力、阈值和关节集合 |
| 用帧数直接定义窗口 | 数据与环境频率不同 | 全部用秒定义，编译时映射到 50 Hz |
| 24-step PPO rollout 足够 | 只覆盖约 0.48 s | 对照 48–96 steps/env |
| 随机切窗口划分数据 | 同源 recording 严重泄漏 | 按 source recording 分组 split |
| 绝对高度/精确接触均可给 Actor | 实机未必可获得 | 不可部署信息只给 privileged Critic |
| 一开始训练完整 4–8 s Diffusion | 当前有效数据量和完整轨迹不足 | 先训练 0.2–0.4 s 局部 successor Diffusion |

## 14. 建议的代码落点

最终实现时建议保持模块边界清晰：

```text
src/mjlab/tasks/velocity/
  config/g1/
    recovery_env_cfgs.py       # G1 recovery 环境配置
    recovery_rl_cfg.py         # Transformer PPO 配置
  mdp/
    g1_recovery.py             # G1 reset、成功、指标、支撑逻辑
    recovery_waypoint.py       # waypoint command/缓存/重规划
  recovery_g1_data/
    schema.py                  # 36D 输入、派生特征和 manifest schema
    compiler.py                # CSV → 50 Hz 数据集
    dataset.py                 # clip/window/mask/group split
    audit.py                   # 运动学与动力学审计
  recovery_policy/
    transformer.py             # E_ref / E_policy
    diffusion.py               # conditional latent denoiser
    planner.py                 # 低步数采样和候选选择
    replay.py                  # sim-validated replay
  scripts/
    build_g1_recovery_dataset.py
    train_g1_recovery_encoder.py
    train_g1_recovery_diffusion.py
    evaluate_g1_recovery.py
```

RSL-RL 当前标准配置主要面向 MLP/CNN/RNN，Actor 与 Critic 默认独立构造。
接入共享/对齐 Transformer、Diffusion planner、独立 auxiliary dataloader 和
联合 checkpoint 时，需要自定义 model/runner 或在 mjlab RL 适配层增加明确
扩展点，不能只修改 `RslRlModelCfg.hidden_dims` 完成。

## 15. Checkpoint、可复现性与部署

每个训练 checkpoint 至少记录：

- Actor、Critic、`E_policy`、固定 `E_ref`；
- Diffusion、sampler 和 candidate selector；
- optimizer/scheduler 和 observation normalizer；
- 数据 manifest、split、schema、CSV 输入和站立 buffer 哈希；
- G1 XML/碰撞配置哈希；
- reward、curriculum、domain randomization 配置；
- git commit、随机种子、全局 step 和 prior 版本；
- replay 数据来源与过滤配置。

部署提供两种模式：

1. **在线 planner**：`E_policy + Diffusion + Actor`，保留多模态重规划能力；
2. **蒸馏 actor**：把 planner 输出蒸馏到 Actor，仅部署 Transformer Actor，
   延迟更低，适合作为第一版实机方案。

实机前必须进行 action/velocity/torque safety clamp、站立状态机保护、急停、
软垫/吊绳测试和由易到难的姿态课程。仿真 90% 成功率不等于可直接无保护
上实机。

## 16. 主要风险与对策

| 风险 | 对策 |
|---|---|
| 完整起身示范少 | 局部 Diffusion、物理 fallen reset、成功 replay，避免伪补全 |
| 首次建立支撑覆盖不足 | 随机 fallen reset + failure replay + 逐步退火辅助力 |
| 重定向轨迹动力学不可行 | MuJoCo PD 回放审计，只把安全状态用于 RSI/控制监督 |
| Diffusion 生成 OOD 路标 | 短 horizon、候选筛选、PPO 拒绝/重规划、uncertainty gate |
| latent reward 漂移 | 冻结版本化 `E_ref` |
| 普通 velocity 奖励阻碍起身 | recovery phase 门控，ready 后再恢复速度和默认 pose |
| 手/膝摩擦错误 | 专用支撑几何和传感器，按真实承重部位配置 |
| 训练成功但实机观测缺失 | 从一开始限制 Actor 为可部署观测，Critic 才使用真值 |
| 在线 Diffusion 延迟过高 | 低步数采样、低频重规划、候选数课程、最终蒸馏 |
| 同源数据泄漏 | recording-level split 和固定 held-out 评测 |

## 17. 最终推荐实施顺序

按收益、风险和可诊断性排序：

1. **G1 数据编译与动力学审计**；
2. **G1 Recovery PPO 环境和无先验基线**；
3. **RSI + 确定性局部 successor 基线**；
4. **Transformer 历史策略与冻结 `E_ref`**；
5. **0.2–0.4 s 条件 latent Diffusion**；
6. **PPO 成功连接过滤与 replay 回灌**；
7. **稳定站立到 velocity 的联合接管**；
8. **rough terrain、强随机化和 sim-to-real**；
9. **在线 Diffusion 与蒸馏 Actor 的部署对照**。

这一路线能够利用所有有价值的支撑片段，又不会假设它们包含不存在的站立
尾段。默认屈膝站姿为系统提供统一的长期目标，Diffusion 负责表达“下一步
可以怎么恢复”的多模态先验，PPO 负责回答“在当前 G1 动力学和接触条件下
怎样安全做到”。三者分工明确后，联合训练才具有可实现性和可验证性。
