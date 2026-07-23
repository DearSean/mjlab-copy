# RL_BOY 动力学历史编码器添加方案

## 1. 文档目的

本文给出在现有 `Mjlab-Velocity-Flat-RL_BOY` 起身—行走一体化任务上增加
动力学历史编码器的工程方案。

方案的核心不是用新的网络替换原有工作，而是保留现有任务定义、恢复课程、
奖励门控、扭矩反馈和站立后速度跟踪逻辑，仅改变 actor 对历史观测的建模方式。
最终形成一个独立的新任务：

```text
Mjlab-Velocity-Flat-RL_BOY-DynHist
```

原任务和已有 GRU 对照任务继续保持可训练、可复现：

```text
Mjlab-Velocity-Flat-RL_BOY
Mjlab-Velocity-Flat-RL_BOY-GRU
```

本文参考 RGMT 的历史编码器设计：

- 论文：Robust and Generalized Humanoid Motion Tracking
- 链接：<https://arxiv.org/abs/2601.23080>
- 采用的结构元素：10 帧历史、逐帧 MLP、128 维 token、正弦位置编码、
  轻量因果 Transformer、pre-LN 残差结构和时间维最大池化。

RGMT 中的历史编码器服务于运动跟踪命令窗口的交叉注意力。本项目是速度命令
控制，命令只有当前 `[v_x, v_y, yaw_rate]`，不存在高维参考运动窗口。因此第一版
只迁移“动力学历史编码器”，不照搬命令交叉注意力。

## 2. 与现有工作的关系

现有 flat RL_BOY 任务已经解决的是一个比单独站起或单独行走更完整的问题：

1. 普通环境学习速度跟踪和抗扰动行走。
2. 恢复环境从倒地、失稳和恢复轨迹附近状态开始。
3. `RlBoyRecoveryAssist` 在训练早期提供可退火的恢复辅助。
4. 恢复奖励、速度奖励和姿态奖励按恢复阶段进行门控。
5. flat 任务移除普通的 `fell_over` 终止，使策略能够在同一 episode 内经历
   倒地、起身、重新稳定和恢复速度跟踪。
6. actor 已显式接收动作历史和三类扭矩反馈，能够感知控制器输出是否被执行、
   是否接近连续扭矩限制以及请求是否超过峰值限制。

动力学历史编码器应当被定义为这套方案的增强层：

> 原方案构造了覆盖“正常行走—受扰失稳—倒地恢复—任务恢复”的状态分布，
> 动力学历史编码器负责从这段连续状态中推断当前接触、负载、执行器饱和和
> 恢复阶段，使同一个策略更准确地选择适合当前动力学状态的控制模式。

因此，下列内容在第一版中必须保持不变：

- `recovery_assist.py` 的恢复辅助状态机和课程；
- 普通/恢复双种群以及恢复初始状态采样；
- 起身成功、超时、失败和进度相关奖励；
- 恢复阶段的奖励门控；
- 速度命令生成方式；
- 动作空间、PD 控制和动作缩放；
- actor-critic 的 PPO 训练目标；
- critic 的特权观测；
- flat 任务允许倒地后继续恢复的终止逻辑。

## 3. 设计目标与非目标

### 3.1 设计目标

- 用结构化时序模型替代 actor 当前的逐项四帧扁平拼接。
- 显式编码 0.2 秒动力学历史，同时保持 50 Hz 控制频率。
- 使用部署可获得的本体感知和控制反馈，不依赖仿真专有 actor 输入。
- 保持 actor 为非递归模型，使 PPO 继续使用普通随机 mini-batch。
- 保持 critic 和恢复训练逻辑不变，隔离网络结构变量。
- 提供确定的 ONNX/JIT 输入契约和部署侧历史缓冲语义。
- 通过新任务 ID 实现可回滚、可消融和对旧 checkpoint 零影响。

### 3.2 第一版非目标

- 不将 velocity 任务改造成 motion tracking 任务。
- 不增加参考动作窗口或 RGMT 的 command cross-attention。
- 不替换现有恢复辅助、恢复课程或奖励系统。
- 不给 critic 增加 Transformer。
- 不在第一版加入动力学参数回归、下一状态预测等辅助损失。
- 不将固定窗口模型实现为带内部隐状态的 RNN。
- 不修改原 `Mjlab-Velocity-Flat-RL_BOY` 的观测或网络配置。

## 4. 当前基线与新方案的接口

### 4.1 当前 MLP actor

当前 flat actor 使用以下信息：

- 当前 base linear velocity；
- 当前 base angular velocity；
- 当前 projected gravity；
- 四帧 joint position；
- 四帧 joint velocity；
- 四帧 previous action；
- 当前三维速度命令；
- 四帧 applied continuous torque ratio；
- 四帧 applied peak torque ratio；
- 四帧 requested peak torque ratio。

对 20 自由度 RL_BOY，当前 flat actor 的拼接维数为 492。这个设计已经包含
动力学历史，但历史在送入 MLP 前被直接扁平化，网络没有显式时间结构。

### 4.2 当前 GRU actor

`Mjlab-Velocity-Flat-RL_BOY-GRU`：

- 移除所有显式观测历史；
- actor 移除 `base_lin_vel`；
- actor 移除与 continuous ratio 线性冗余的 applied peak ratio；
- 使用 256 维单层 GRU 表示时序状态；
- critic 保持 MLP 和完整特权观测。

该任务应继续作为“隐状态递归时序建模”基线。

### 4.3 新 DynHist actor

新 actor 的数据流如下：

```text
最近 10 帧可部署观测 [B, 10, 106]
                │
                ▼
       逐帧归一化 + 两层 MLP
                │
                ▼
      128 维 token + 正弦位置编码
                │
                ▼
       单层因果 Transformer block
                │
                ▼
          时间维 element-wise max
                │
                └──────────────┐
                               ▼
最新归一化帧 [106] ──┐   dynamics embedding [128]
当前命令 [3] ────────┼─────────┘
                     ▼
              concat [237]
                     ▼
          MLP 512 → 256 → 128
                     ▼
          Gaussian action distribution
                     ▼
                20 维动作
```

critic 继续接收当前 flat 任务的完整 `critic` group，不经过历史编码器。

## 5. Actor 观测设计

### 5.1 每帧历史向量

建议每一帧使用以下 106 维信号：

| 顺序 | term | 维数 | 含义 |
|---:|---|---:|---|
| 1 | `base_ang_vel` | 3 | IMU 机体系角速度 |
| 2 | `projected_gravity` | 3 | 机体系重力方向 |
| 3 | `joint_pos` | 20 | 相对默认位姿的关节位置 |
| 4 | `joint_vel` | 20 | 关节速度 |
| 5 | `actions` | 20 | 上一控制周期的策略动作 |
| 6 | `applied_torque_continuous_ratio` | 20 | 实际扭矩/连续扭矩限制 |
| 7 | `requested_torque_peak_ratio` | 20 | 限幅前请求扭矩/峰值限制 |
|  | 合计 | 106 |  |

其中：

- `base_lin_vel` 不进入 actor，避免依赖实际机器人上通常不可直接可靠测量的
  机体系线速度，并与现有 GRU 的可部署观测口径一致。
- `applied_torque_peak_ratio` 不进入 actor。它和
  `applied_torque_continuous_ratio` 使用相同的实际扭矩分子，只是归一化常数
  不同，信息上基本线性冗余。
- `requested_torque_peak_ratio` 保留，因为它与 applied torque 的差值能够反映
  扭矩裁剪、执行器能力不足和控制请求不可实现程度。
- 现有 observation term 的 noise、clip 和 scale 继续生效，不另建一套传感器
  预处理。

### 5.2 命令向量

当前速度命令单独作为：

```text
actor_command: [B, 3] = [v_x, v_y, yaw_rate]
```

命令不重复写入十帧历史，原因是：

- 速度命令通常在多个控制周期内保持不变，重复十次没有新增信息；
- 将命令与动力学历史分开，可以明确区分“机器人现在能做什么”和“用户要求
  做什么”；
- 新模型可分别维护 history 和 command 的归一化统计；
- ONNX 输入契约更清楚。

### 5.3 critic 观测

`critic` group 完全沿用 `rlboy_flat_env_cfg()` 的结果，包括当前任务已有的
base linear velocity、足端状态、接触信息、扭矩反馈和历史设置。

这保持 asymmetric actor-critic 的原始设定，也保证实验中唯一主要变化是
actor 的历史表示方式。

### 5.4 历史长度和时间顺序

仿真 timestep 为 0.005 秒，decimation 为 4，因此控制周期为 0.02 秒，即
50 Hz。十帧覆盖 0.2 秒：

```text
history_length = 10
history[0]     = 最旧帧
history[-1]    = 最新帧
```

该顺序与 `CircularBuffer.buffer` 的现有实现一致。

环境 reset 后，第一帧会回填到该环境的全部十个历史槽位，而不是在前九帧填零。
部署端必须复现这一语义，否则 reset 后仿真和实机策略输入会不一致。

## 6. ObservationManager 配置方案

在 `rlboy_flat_env_cfg()` 之上新增：

```python
def rlboy_flat_dyn_history_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  ...
```

实现时不要原地修改与 critic 共享的 `ObservationTermCfg`。基础配置中的
`critic_terms = {**actor_terms, ...}` 是浅层字典复制，term 配置对象可能共享。
应使用 `deepcopy` 构造新的 actor groups。

建议的配置伪代码：

```python
from copy import deepcopy

from mjlab.managers.observation_manager import ObservationGroupCfg


_DYN_HISTORY_TERMS = (
  "base_ang_vel",
  "projected_gravity",
  "joint_pos",
  "joint_vel",
  "actions",
  "applied_torque_continuous_ratio",
  "requested_torque_peak_ratio",
)


def rlboy_flat_dyn_history_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = rlboy_flat_env_cfg(play=play)
  source_actor = cfg.observations["actor"]

  history_terms = {
    name: deepcopy(source_actor.terms[name])
    for name in _DYN_HISTORY_TERMS
  }
  command_term = deepcopy(source_actor.terms["command"])

  cfg.observations["actor"] = ObservationGroupCfg(
    terms=history_terms,
    concatenate_terms=True,
    enable_corruption=source_actor.enable_corruption,
    history_length=10,
    flatten_history_dim=False,
  )
  cfg.observations["actor_command"] = ObservationGroupCfg(
    terms={"command": command_term},
    concatenate_terms=True,
    enable_corruption=False,
  )
  return cfg
```

预期输出：

```text
actor         -> [num_envs, 10, 106]
actor_command -> [num_envs, 3]
critic        -> 保持当前 flat critic shape
```

注意事项：

- group-level `history_length=10` 会覆盖各 term 原有的四帧 history。
- `flatten_history_dim=False` 使各 term 生成 `[B, 10, D_i]`。
- group 在最后一维拼接，得到 `[B, 10, 106]`。
- training 继承原 actor 的 corruption 开关。
- play 模式继承 `rlboy_flat_env_cfg(play=True)` 的无噪声 actor 设置。
- `actor_command` 没有噪声，因此始终关闭 corruption。
- 任务初始化时应断言历史长度、term 顺序和最终 shape，尽早发现配置漂移。

## 7. RSL-RL 模型接入

### 7.1 配置扩展

在 `src/mjlab/rl/config.py` 的 `RslRlModelCfg` 中增加可选字段：

```python
dynamics_history_cfg: dict[str, Any] | None = None
```

该字段只向新模型传递，建议内容为：

```python
{
  "history_group": "actor",
  "command_group": "actor_command",
  "history_length": 10,
  "frame_dim": 106,
  "command_dim": 3,
  "token_dim": 128,
  "token_hidden_dim": 256,
  "num_heads": 4,
  "ffn_dim": 256,
  "num_layers": 1,
  "dropout": 0.0,
  "pooling": "max",
}
```

RGMT 论文明确给出十帧、128 维、因果注意力和 max pooling，但没有完整公开
attention head 数和 FFN 宽度。因此 `num_heads=4`、`ffn_dim=256` 是本项目的
首版工程选择，必须作为超参数记录，而不是表述成论文原始设置。

在 `MjlabOnPolicyRunner.__init__()` 中，将该字段加入可选配置清理逻辑：

```python
for opt in ("cnn_cfg", "distribution_cfg", "dynamics_history_cfg"):
  if train_cfg[key].get(opt) is None:
    train_cfg[key].pop(opt, None)
```

这样标准 MLP、CNN 和 GRU 不会收到不支持的关键字参数。

### 7.2 新模型文件

新增：

```text
src/mjlab/rl/dynamics_history_model.py
```

主类：

```text
DynamicsHistoryModel
```

RSL-RL 通过限定路径解析：

```text
mjlab.rl.dynamics_history_model:DynamicsHistoryModel
```

构造函数必须兼容 RSL-RL 5.2 的模型接口：

```python
def __init__(
  self,
  obs: TensorDict,
  obs_groups: dict[str, list[str]],
  obs_set: str,
  output_dim: int,
  hidden_dims: tuple[int, ...] | list[int],
  activation: str,
  obs_normalization: bool,
  distribution_cfg: dict | None,
  dynamics_history_cfg: dict,
) -> None:
  ...
```

推荐继承 `rsl_rl.models.MLPModel` 以复用 distribution 和 PPO 需要的属性，
但必须覆盖以下方法：

- `_get_obs_dim()`：接受一个三维 history group 和一个二维 command group；
- `_get_latent_dim()`：返回 `106 + 3 + 128 = 237`；
- `get_latent()`：执行历史编码和特征融合；
- `update_normalization()`：分别更新逐帧和 command 的统计；
- `as_jit()`：返回双输入的确定性导出 wrapper；
- `as_onnx()`：返回双输入的确定性导出 wrapper。

调用父类构造函数时，应让父类使用 `obs_normalization=False`，然后由新模型创建
自己的两个 normalizer。否则父类会尝试用一个扁平统计量处理三维历史。

### 7.3 输入校验

模型初始化时应直接校验：

```text
obs["actor"].ndim == 3
obs["actor"].shape[1:] == (10, 106)
obs["actor_command"].ndim == 2
obs["actor_command"].shape[1] == 3
obs_groups["actor"] == ["actor", "actor_command"]
```

shape 不符时抛出带 group 名、期望 shape 和实际 shape 的 `ValueError`。
不要自动 flatten 或猜测输入，这类错误在训练后才发现会造成昂贵返工。

### 7.4 归一化

使用两个独立的 `EmpiricalNormalization`：

```text
history_normalizer: shape 106
command_normalizer: shape 3
```

历史归一化时：

```python
flat_history = history.reshape(-1, 106)
normalized = history_normalizer(flat_history)
normalized = normalized.reshape(batch_size, 10, 106)
```

更新统计时同样将 batch 和 time 合并。这样每个物理量在所有环境、所有历史帧
上共享统计，不会为十个相对时间位置学习十套尺度。

`history[:, -1, :]` 必须在归一化后作为 current-frame feature 使用。

### 7.5 History encoder

建议首版结构：

```text
Token MLP:
  Linear(106, 256)
  ELU
  Linear(256, 128)

Position:
  固定正弦位置编码 [1, 10, 128]

Transformer block × 1:
  x = x + MHA(LayerNorm(x), causal_mask)
  x = x + FFN(LayerNorm(x))
  x = LayerNorm(x)

FFN:
  Linear(128, 256)
  ELU 或 GELU
  Linear(256, 128)

Pooling:
  dynamics = x.amax(dim=1)
```

首版参数：

```text
token_dim = 128
num_heads = 4
num_layers = 1
ffn_dim = 256
dropout = 0
```

因果 mask：

```python
causal_mask = torch.triu(
  torch.ones(10, 10, dtype=torch.bool),
  diagonal=1,
)
```

mask 应注册为 buffer，随模型迁移设备并进入 checkpoint。对于
`nn.MultiheadAttention(batch_first=True)`，布尔 mask 中的 `True` 表示禁止
关注未来 token。

### 7.6 Actor head 和分布

融合特征：

```python
latent = torch.cat(
  (
    normalized_history[:, -1, :],
    normalized_command,
    dynamics_embedding,
  ),
  dim=-1,
)
```

维数：

```text
106 + 3 + 128 = 237
```

继续使用原 actor head：

```text
237 → 512 → 256 → 128 → distribution input
activation = ELU
```

动作分布保持：

```python
{
  "class_name": "GaussianDistribution",
  "init_std": 1.0,
  "std_type": "scalar",
}
```

模型还必须保持 `MLPModel` 已有的 PPO 接口：

- `output_mean`
- `output_std`
- `output_entropy`
- `output_distribution_params`
- `get_output_log_prob()`
- `get_kl_divergence()`
- `reset()`
- `get_hidden_state()`
- `detach_hidden_state()`

新模型是固定窗口模型：

```text
is_recurrent = False
hidden_state = None
```

历史已经是 observation TensorDict 的一部分，因此 rollout storage 会保存每个
样本对应的完整窗口。PPO 可以继续随机打乱 mini-batch，不需要 recurrent mask
或按轨迹顺序训练。

## 8. RL 任务配置

在 `src/mjlab/tasks/velocity/config/rlboy/rl_cfg.py` 新增：

```python
def rlboy_dyn_history_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = rlboy_ppo_runner_cfg()
  cfg.actor.class_name = (
    "mjlab.rl.dynamics_history_model:DynamicsHistoryModel"
  )
  cfg.actor.dynamics_history_cfg = {
    ...
  }
  cfg.obs_groups = {
    "actor": ("actor", "actor_command"),
    "critic": ("critic",),
  }
  cfg.experiment_name = "rlboy_velocity_dyn_history"
  return cfg
```

在 `src/mjlab/tasks/velocity/config/rlboy/__init__.py` 注册：

```python
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-RL_BOY-DynHist",
  env_cfg=rlboy_flat_dyn_history_env_cfg(),
  play_env_cfg=rlboy_flat_dyn_history_env_cfg(play=True),
  rl_cfg=rlboy_dyn_history_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
```

不要让新函数成为原 flat 配置的默认分支，也不要改动原任务注册参数。

## 9. 训练数据流

每个控制周期的数据流应为：

1. action manager 使用上一策略输出形成关节目标。
2. actuator 计算限幅前 requested effort。
3. simulation 执行四个 0.005 秒 physics substeps。
4. observation terms 读取最新传感器、关节、实际 actuator force 和上一动作。
5. ObservationManager 将新帧写入各环境自己的十帧 circular buffer。
6. policy 接收 `[B, 10, 106]` 和 `[B, 3]`。
7. history encoder 产生 128 维 dynamics embedding。
8. actor 输出下一控制周期的 20 维动作。
9. critic 使用原 privileged observation 估值。
10. PPO 按当前配置优化 actor 和 critic。

历史中的 torque feedback 应明确对齐到“刚执行完的控制周期”。部署侧也必须在
物理步进和扭矩计算后更新历史，再进行下一次策略推理。

## 10. ONNX、JIT 与部署

### 10.1 导出接口

新模型的导出 wrapper 使用两个显式输入：

```text
proprio_history: float32 [1, 10, 106]
command:          float32 [1, 3]
actions:          float32 [1, 20]
```

建议 `input_names`：

```python
["proprio_history", "command"]
```

建议 `output_names`：

```python
["actions"]
```

`get_dummy_inputs()` 返回两个张量。现有
`MjlabOnPolicyRunner.export_policy_to_onnx()` 已能把 tuple 传给
`torch.onnx.export()`，但必须增加新 wrapper 的导出测试。

历史缓冲放在模型外部，不导出成有状态 ONNX：

- partial reset 的语义更明确；
- 仿真多环境和实机单机器人可共用同一策略图；
- ONNX 每次调用是确定性的；
- history 可以被记录、回放和逐元素检查；
- 不需要额外的 `h_in/h_out` 接口。

### 10.2 ONNX 兼容性

首选 `nn.MultiheadAttention`，opset 继续使用当前的 18。需要用实际项目环境
验证 ONNX Runtime 数值一致性。

若当前 PyTorch 导出器无法稳定展开 `MultiheadAttention`，备选方案是实现显式
Q/K/V projection、scaled dot-product、mask、softmax 和 output projection。
不得为了导出而删除 causal mask 或改变 pooling。

### 10.3 元数据

`src/mjlab/rl/exporter_utils.py` 当前默认读取：

```python
env.observation_manager.active_terms["actor"]
```

它需要扩展为同时记录：

- observation group 名；
- `actor` term 名和固定顺序；
- `actor_command` term 名；
- history length；
- history frame order：`oldest_to_newest`；
- 每帧维数 106；
- command 维数 3；
- ONNX input 名和 shape；
- continuous/peak torque limits；
- action scale；
- joint names 和默认关节角。

元数据应允许部署脚本拒绝不匹配的策略，而不是只依据总输入维数猜测格式。

### 10.4 部署侧历史缓冲

至少更新：

```text
src/mjlab/deploy/rlboy_mujoco.py
src/mjlab/deploy/rlboy_mujoco_rmcmd.py
```

如果 motion 脚本也要加载该 velocity policy，再单独适配
`rlboy_mujoco_motion.py`；否则不要把两类 policy 的输入协议混在一起。

部署端维护：

```text
history: np.ndarray, shape [10, 106], dtype float32
```

reset 后：

```python
history[:] = first_frame
```

正常更新：

```python
history[:-1] = history[1:]
history[-1] = current_frame
```

部署帧的 term 顺序必须与第 5.1 节完全一致。

扭矩反馈定义：

- applied continuous ratio：
  实际执行或测量扭矩除以各电机连续扭矩限制；
- requested peak ratio：
  限幅前 PD/控制器请求扭矩除以各电机峰值扭矩限制。

纯 MuJoCo 部署可从限幅后的 `data.ctrl` 构造 applied torque，并保存限幅前
`pd_control()` 输出构造 requested torque。若仿真 actuator 还包含速度相关降额，
部署代码必须复现同一电机模型，不能只用简单常数 clip。

实机上优先使用电机测量或可靠估计的实际扭矩；requested torque 来自低层控制器
限幅前请求。若某项无法获得，应先完成相应的 no-torque 消融并重新训练，不应在
部署时静默填零。

当前 `rlboy_mujoco_rmcmd.py` 仍按单个扁平输入构造观测，不能直接运行新策略。
适配完成前，加载器应对输入数量、输入名和 shape 做严格校验并显式报错。

## 11. 建议修改文件

### 11.1 必须修改

| 文件 | 修改 |
|---|---|
| `src/mjlab/rl/config.py` | 增加 `dynamics_history_cfg` |
| `src/mjlab/rl/runner.py` | 清理非 DynHist 模型的空配置 |
| `src/mjlab/rl/dynamics_history_model.py` | 新模型及 ONNX/JIT wrapper |
| `src/mjlab/tasks/velocity/config/rlboy/env_cfgs.py` | 新 observation 配置函数 |
| `src/mjlab/tasks/velocity/config/rlboy/rl_cfg.py` | 新 runner 配置 |
| `src/mjlab/tasks/velocity/config/rlboy/__init__.py` | 注册新任务 |
| `src/mjlab/rl/exporter_utils.py` | 增加多 group 历史元数据 |
| `src/mjlab/deploy/rlboy_mujoco.py` | 新输入协议和历史缓冲 |
| `src/mjlab/deploy/rlboy_mujoco_rmcmd.py` | 新输入协议和历史缓冲 |

### 11.2 建议新增测试

```text
tests/test_dynamics_history_model.py
tests/test_rlboy_dynamics_history_cfg.py
tests/test_rlboy_dynamics_history_export.py
tests/test_rlboy_dynamics_history_deploy_obs.py
```

## 12. 测试方案

### 12.1 配置测试

验证：

- 新任务能通过 registry 查到；
- 原 MLP 和 GRU 配置没有变化；
- actor term 名及顺序严格等于第 5.1 节；
- `actor` shape 为 `[B, 10, 106]`；
- `actor_command` shape 为 `[B, 3]`；
- critic 的 term 配置与 `rlboy_flat_env_cfg()` 一致；
- play actor 不含噪声；
- 新任务继续使用 `VelocityOnPolicyRunner`；
- 恢复事件、课程、奖励和终止项与原 flat 任务一致。

### 12.2 模型单元测试

用合成 TensorDict 验证：

- batch size 1、多个环境和 PPO 展平 batch 均能 forward；
- deterministic 和 stochastic 输出 shape 为 `[B, 20]`；
- distribution 的 log probability、entropy 和 KL 接口可用；
- history/command normalizer 分别更新；
- 最新帧取 `history[:, -1]`；
- causal mask 的未来区域确实被屏蔽；
- max pooling 发生在 time 维；
- `reset()` 不创建或修改 hidden state；
- 错误 group 名和错误 shape 会明确报错；
- forward 和 normalization 中不产生 NaN/Inf。

因果性测试可以固定模型权重，修改某个未来 token，并验证更早 token 的 block
输出不发生变化。不要只检查 mask tensor 的形状。

### 12.3 历史缓冲测试

验证：

- 顺序始终是 oldest-to-newest；
- 滑窗满后覆盖最旧帧；
- 单环境 reset 不影响其他环境；
- reset 后第一次 append 回填十帧；
- actor 内所有 term 在同一个时间索引上对齐。

### 12.4 PPO 集成测试

至少完成：

- CPU 小环境 rollout；
- 一次 PPO update；
- actor 和 critic loss 都能反向传播；
- encoder、attention 和 actor head 均有非零有限梯度；
- checkpoint save/load；
- load 后同一 observation 的 deterministic action 一致；
- 原 MLP 和 GRU task 的 smoke test 继续通过。

### 12.5 导出测试

验证：

- JIT 可导出并加载；
- ONNX opset 18 可导出；
- ONNX 输入名和 shape 正确；
- PyTorch、JIT 和 ONNX Runtime deterministic action 数值一致；
- 建议 `atol <= 1e-5`，必要时针对 ONNX 放宽到 `1e-4`；
- metadata 包含 term 顺序、history length 和 torque limits；
- 错误输入顺序或 shape 会失败，而不是隐式 reshape。

### 12.6 部署观测一致性测试

保存一段仿真状态和控制量，同时通过：

1. 环境 observation terms；
2. NumPy 部署 observation builder；

构造历史和命令输入，逐 term、逐帧比较。该测试是 sim-to-sim/real 部署前最重要
的协议测试，尤其要覆盖：

- quaternion 和 projected gravity 约定；
- angular velocity 坐标系；
- joint 顺序；
- default joint position；
- previous action 的时刻；
- applied/requested torque 的限幅时刻；
- reset 后回填。

### 12.7 建议命令

```bash
uv run pytest tests/test_dynamics_history_model.py -q
uv run pytest tests/test_rlboy_dynamics_history_cfg.py -q
uv run pytest tests/test_rlboy_dynamics_history_export.py -q
uv run pytest tests/test_rlboy_dynamics_history_deploy_obs.py -q
FORCE_CPU=1 uv run pytest -m "not slow" -q
uv run ruff format
uv run ruff check --fix
uv run ty check
uv run pyright
```

开发阶段优先运行单文件测试；合并前再运行项目级检查。

## 13. 分阶段实施

### 阶段 M0：冻结原始基线

- 保存原 MLP 和 GRU 的配置快照、checkpoint 和评估结果；
- 记录参数量、训练吞吐和部署推理时间；
- 固定训练 seeds 和评估场景；
- 确认原任务工作树无意外变化。

完成标准：能够用统一评估脚本复现实验基线。

### 阶段 M1：观测协议

- 增加 DynHist env config；
- 注册新 task；
- 加入 actor/history/command shape 断言；
- 完成配置和 buffer 测试。

完成标准：随机策略可以 step，原恢复环境行为不变。

### 阶段 M2：模型

- 实现 history encoder；
- 接入 distribution 和 normalization；
- 完成 model 单元测试；
- 检查参数量和显存占用。

完成标准：合成数据和真实环境 observation 均可 forward/backward。

### 阶段 M3：PPO

- 完成短 rollout 和一次 update；
- 检查梯度、KL、entropy 和 action std；
- 进行短训练观察是否稳定；
- 保存和恢复 checkpoint。

完成标准：无 NaN，学习曲线能超过随机策略，并可稳定续训。

### 阶段 M4：导出和部署

- 实现 JIT/ONNX wrapper；
- 扩展 metadata；
- 更新 MuJoCo velocity 部署脚本；
- 完成环境/部署 observation parity。

完成标准：PyTorch、JIT、ONNX 输出一致，部署推理满足 50 Hz。

### 阶段 M5：完整实验

- 与 MLP、GRU 和扁平十帧 MLP 对比；
- 完成 torque/no-torque 和 history length 消融；
- 完成正常行走、倒地恢复、任务恢复和 OOD 测试；
- 汇总统计显著性和失败案例。

完成标准：可以判断收益来自结构化动力学历史，而不是参数量或更长输入窗口。

## 14. 验收指标

### 14.1 工程验收

- 原 MLP、GRU task 配置和 checkpoint 兼容性不变；
- 新 task 能训练、保存、恢复、play 和导出；
- actor 输入 shape 固定且有 metadata；
- partial reset 的 history 行为正确；
- PyTorch/JIT/ONNX 数值一致；
- 训练无 NaN/Inf；
- 部署 observation 与环境逐元素一致；
- 单次推理显著低于 20 ms 控制周期，建议目标为部署硬件上 2–4 ms 内。

### 14.2 研究验收

至少报告：

- 正常速度跟踪误差；
- 外部推扰下存活率和速度恢复；
- 零辅助力起身成功率；
- recovery time；
- 起身后重新进入命令跟踪容差的 Task Resumption Time；
- 恢复后的速度跟踪误差；
- 连续扭矩超限率和峰值请求超限率；
- action rate/action acceleration；
- OOD payload、摩擦、PD gain、控制延迟和传感噪声下表现；
- 推理延迟、参数量和训练吞吐。

需要分别统计：

- 普通初始状态；
- 倒地初始状态；
- 运行中被击倒；
- 恢复成功后的固定时间窗口。

仅报告 episode reward 不足以证明动力学编码器改善了起身—行走连续控制。

## 15. 必要消融

建议至少训练以下对照：

| 编号 | Actor | 历史 | 扭矩反馈 | 目的 |
|---|---|---:|---|---|
| A | 当前 MLP | 4 帧扁平 | 有 | 原始方法 |
| B | 当前 GRU | 隐状态 | 有，去冗余 | 递归时序基线 |
| C | MLP | 10 帧扁平 | 有 | 排除“只是窗口更长” |
| D | DynHist | 10 帧结构化 | 无 | 检查纯本体历史收益 |
| E | DynHist | 10 帧结构化 | 有 | 完整方案 |

进一步消融：

- history length：4、10、20；
- pooling：latest token、mean、max；
- causal attention 与无 mask self-attention；
- torque 中只用 applied、只用 requested、两者都用；
- token dimension：64、128；
- 1 层和 2 层 Transformer；
- 参数量匹配的 MLP/GRU。

每个主要结论应使用相同训练预算和至少三个 seeds。参数量明显不同时，应报告
参数匹配对照，避免把更大网络的收益归因于编码器结构。

## 16. 风险与处理

| 风险 | 表现 | 处理 |
|---|---|---|
| attention 不优于 GRU | 收益小或训练更慢 | 加入参数匹配和十帧扁平 MLP 对照 |
| torque 实机不可得 | sim 好、实机输入失真 | 先确认低层接口；训练 no-torque 备选 |
| reset 历史不一致 | 起步瞬间动作异常 | 部署首帧回填并做 parity 测试 |
| observation term 错位 | 能运行但性能严重下降 | metadata、固定顺序和逐 term 测试 |
| ONNX MHA 导出失败 | checkpoint 无法部署 | 显式 Q/K/V attention fallback |
| 参数量导致不公平 | DynHist 表面占优 | 参数匹配消融并报告 FLOPs/延迟 |
| GPU memory/吞吐下降 | 并行环境数降低 | 单层、十帧、128 维；profile 后再扩展 |
| max pooling 弱化顺序 | 不同阶段混淆 | 保留位置编码；与 latest/mean 做消融 |
| torque 时刻错一拍 | 学到错误执行器关系 | 固定数据流时刻并做记录回放检查 |
| history 噪声过强 | attention 学不到稳定模式 | 先继承原噪声，再单独消融相关噪声 |

## 17. 第二阶段可选扩展

只有在纯 PPO 版本稳定、完成主要消融后，再考虑辅助监督：

- 从 dynamics embedding 预测随机 payload；
- 预测 friction 或 PD gain 扰动；
- 预测足端接触状态；
- 预测下一帧 joint velocity；
- 预测 requested-applied torque gap；
- 使用 privileged dynamics 参数做训练期表征约束。

这些扩展会引入额外 target、loss、rollout storage 或自定义 PPO 逻辑，容易混淆
第一篇实验中“历史编码结构本身”的贡献。建议以配置开关隔离，并保持部署时只用
actor history 和 command。

## 18. 推荐的首版固定配置

```text
Task ID:
  Mjlab-Velocity-Flat-RL_BOY-DynHist

Control:
  physics dt = 0.005 s
  decimation = 4
  policy rate = 50 Hz

History:
  length = 10
  duration = 0.2 s
  order = oldest_to_newest
  frame dim = 106

Encoder:
  token MLP = 106 -> 256 -> 128
  positional encoding = sinusoidal
  transformer layers = 1
  attention heads = 4
  FFN = 128 -> 256 -> 128
  dropout = 0
  mask = causal
  pooling = element-wise max over time

Fusion:
  current frame = 106
  command = 3
  dynamics embedding = 128
  total latent = 237

Actor head:
  237 -> 512 -> 256 -> 128 -> Gaussian distribution

Critic:
  unchanged MLP and unchanged privileged observations

Training:
  unchanged PPO and unchanged recovery/velocity task logic
```

## 19. 实施完成定义

只有同时满足以下条件，才视为“动力学历史编码器已添加完成”：

1. 新任务独立注册，原任务行为未改变。
2. actor 使用严格定义的 `[B, 10, 106]` history 和 `[B, 3]` command。
3. history encoder、actor head、distribution 和 normalization 全部参与训练。
4. critic、恢复辅助、课程、奖励和终止逻辑保持原实现。
5. PPO 训练、checkpoint save/load 和 play 全流程通过。
6. JIT/ONNX 导出通过并与 PyTorch 数值一致。
7. 部署端复现 term 顺序、时刻和 reset 回填语义。
8. 至少完成 MLP、GRU、十帧扁平 MLP 和 DynHist 的公平对照。
9. 结果同时覆盖行走、起身、恢复后任务重启和 OOD 鲁棒性。

该实现边界能保证论文叙事仍以原有起身—行走一体化方法为主体，动力学历史
编码器作为针对连续恢复过程、接触变化和执行器受限状态的结构化增强，而不是
把原工作重新包装成一个通用 Transformer 控制器。
