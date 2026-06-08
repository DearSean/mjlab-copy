# 在 Tracking 任务中新增机器人说明文档

以 **Unitree G1** 为参考范例，说明除机器人模型文件（MJCF/XML）之外，还需要在代码仓库中创建/修改哪些文件，以及每个文件的作用和关键内容。

---

## 一、文件总览

新增一个机器人（假设机器人目录名为 `my_robot`），除模型资产和动作捕捉数据外，共需准备/修改 **7 个 Python 文件**：

| 序号 | 文件路径 | 作用 |
|:---:|---|---|
| 1 | `src/mjlab/asset_zoo/robots/my_robot/__init__.py` | 机器人资产包的 `__init__.py` |
| 2 | `src/mjlab/asset_zoo/robots/my_robot/my_robot_constants.py` | 机器人执行器、碰撞、初始姿态等常量配置 |
| 3 | `src/mjlab/asset_zoo/robots/__init__.py` | 导出机器人配置工厂函数和动作缩放表 |
| 4 | `src/mjlab/tasks/tracking/config/my_robot/env_cfgs.py` | 运动模仿任务的**环境配置**（传感器、命令、事件、终止条件等） |
| 5 | `src/mjlab/tasks/tracking/config/my_robot/rl_cfg.py` | 运动模仿任务的**强化学习超参数配置** |
| 6 | `src/mjlab/tasks/tracking/config/my_robot/__init__.py` | 将任务注册到 `mjlab` 任务注册表 |
| 7 | `src/mjlab/tasks/tracking/config/__init__.py` | （若之前为空，通常无需修改；Python 包结构需要即可） |

---

## 二、各文件详细说明

### 1. 机器人资产常量文件

**路径**：`src/mjlab/asset_zoo/robots/my_robot/my_robot_constants.py`

这是机器人的**核心定义文件**，负责把 MJCF 模型接入 mjlab 的实体系统。

#### 需要定义的内容（参考 G1 的 `g1_constants.py`）：

**a) MJCF 路径与加载函数**

```python
from pathlib import Path
import mujoco
from mjlab import MJLAB_SRC_PATH

MY_ROBOT_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "robots" / "my_robot" / "xmls" / "my_robot.xml"
)
assert MY_ROBOT_XML.exists()

def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(MY_ROBOT_XML))
```

**b) 执行器配置（Actuator）**

一般流程：

1. 根据电机参数计算 `reflected_inertia`（转动惯量反映到关节端）
2. 给定自然频率 `NATURAL_FREQ` 和阻尼比 `DAMPING_RATIO`
3. 计算刚度 `stiffness = armature * ω²` 和阻尼 `damping = 2ζ·armature·ω`
4. 用 `BuiltinPositionActuatorCfg` 为每组关节创建配置

示例（单组电机简化版）：

```python
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia

ROTOR_INERTIA = 0.0001
GEAR_RATIO = 10
ARMATURE = reflected_inertia(ROTOR_INERTIA, GEAR_RATIO)

ACTUATOR = ElectricActuator(
    reflected_inertia=ARMATURE,
    velocity_limit=30.0,
    effort_limit=50.0,
)

NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0
STIFFNESS = ARMATURE * NATURAL_FREQ**2
DAMPING = 2.0 * DAMPING_RATIO * ARMATURE * NATURAL_FREQ

MY_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_joint", ".*_knee_joint"),  # 正则匹配关节名
    stiffness=STIFFNESS,
    damping=DAMPING,
    effort_limit=ACTUATOR.effort_limit,
    armature=ACTUATOR.reflected_inertia,
)
```

> **注意**：如果一台机器人有多种电机（如 G1 有 5020、7520、4010 等），需要为每种电机分别计算并创建对应的 `BuiltinPositionActuatorCfg`。

**c) 初始姿态（Keyframe）**

```python
from mjlab.entity import EntityCfg

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.78),  # 初始基座高度
    joint_pos={
        ".*_hip_pitch_joint": -0.1,
        ".*_knee_joint": 0.3,
    },
    joint_vel={".*": 0.0},
)
```

**d) 碰撞配置（CollisionCfg）**

```python
from mjlab.utils.spec_config import CollisionCfg

FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={r"^.*_foot_collision$": 3, ".*_collision": 1},
    priority={r"^.*_foot_collision$": 1},
    friction={r"^.*_foot_collision$": (0.6,)},
)
```

- `geom_names_expr`：哪些 geom 参与碰撞
- `condim`：接触维度（3 为点接触，6 为面接触）
- `priority`：接触优先级
- `friction`：摩擦系数

**e) 汇总为机器人配置**

```python
from mjlab.entity import EntityArticulationInfoCfg

MY_ROBOT_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(MY_ACTUATOR_CFG,),
    soft_joint_pos_limit_factor=0.9,
)

def get_my_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=MY_ROBOT_ARTICULATION,
    )
```

**f) 动作缩放表（Action Scale）**

`tracking` 任务使用 `JointPositionActionCfg`，需要为每个关节提供 `scale` 值：

```python
MY_ROBOT_ACTION_SCALE: dict[str, float] = {}
for a in MY_ROBOT_ARTICULATION.actuators:
    assert isinstance(a, BuiltinPositionActuatorCfg)
    e = a.effort_limit
    s = a.stiffness
    names = a.target_names_expr
    assert e is not None
    for n in names:
        MY_ROBOT_ACTION_SCALE[n] = 0.25 * e / s
```

---

### 2. 机器人包 `__init__.py`

**路径**：`src/mjlab/asset_zoo/robots/my_robot/__init__.py`

内容通常只有文档字符串即可，因为常量文件中的函数/变量会在外层 `robots/__init__.py` 显式导入。

```python
"""My Robot humanoid/quadruped."""
```

---

### 3. 导出到 `asset_zoo` 公共 API

**路径**：`src/mjlab/asset_zoo/robots/__init__.py`

需要把工厂函数和动作缩放表导出，供 `tracking` 任务引用：

```python
from mjlab.asset_zoo.robots.my_robot.my_robot_constants import (
    MY_ROBOT_ACTION_SCALE as MY_ROBOT_ACTION_SCALE,
)
from mjlab.asset_zoo.robots.my_robot.my_robot_constants import (
    get_my_robot_cfg as get_my_robot_cfg,
)
```

---

### 4. Tracking 任务环境配置

**路径**：`src/mjlab/tasks/tracking/config/my_robot/env_cfgs.py`

这是**最关键**的任务定制文件。它基于 `make_tracking_env_cfg()` 工厂函数提供的通用运动模仿配置，然后针对当前机器人的关节名、身体名、动作捕捉数据等进行**覆盖和定制**。

#### 典型定制项（以 G1 为例）：

**a) 替换机器人**

```python
from mjlab.asset_zoo.robots import MY_ROBOT_ACTION_SCALE, get_my_robot_cfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

def my_robot_flat_tracking_env_cfg(
    has_state_estimation: bool = True,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    cfg = make_tracking_env_cfg()
    cfg.scene.entities = {"robot": get_my_robot_cfg()}
```

**b) 配置自碰撞传感器**

tracking 任务的奖励函数中使用了 `self_collision_cost`，需要在场景中注册对应传感器：

```python
from mjlab.sensor import ContactMatch, ContactSensorCfg

self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
)
cfg.scene.sensors = (self_collision_cfg,)
```

**c) 设置动作缩放**

```python
from mjlab.envs.mdp.actions import JointPositionActionCfg

joint_pos_action = cfg.actions["joint_pos"]
assert isinstance(joint_pos_action, JointPositionActionCfg)
joint_pos_action.scale = MY_ROBOT_ACTION_SCALE
```

**d) 配置 MotionCommand（核心）**

`MotionCommandCfg` 是 tracking 任务的核心，需要指定：

- `motion_file`：动作捕捉数据文件路径（`.npz` 格式）
- `anchor_body_name`：锚点身体名称（通常是躯干/骨盆，用于对齐根节点）
- `body_names`：参与运动模仿的身体名称元组（**顺序很重要**，必须与 motion file 中的身体顺序一致）

```python
from mjlab.tasks.tracking.mdp import MotionCommandCfg

motion_cmd = cfg.commands["motion"]
assert isinstance(motion_cmd, MotionCommandCfg)
motion_cmd.motion_file = str(
    MJLAB_SRC_PATH / "asset_zoo" / "robots" / "my_robot" / "motions" / "my_robot.npz"
)
motion_cmd.anchor_body_name = "torso_link"
motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    # ... 其他需要模仿的身体
)
```

> **Motion file 格式**：`.npz` 文件需包含以下数组：
> - `joint_pos`: `(T, num_joints)`
> - `joint_vel`: `(T, num_joints)`
> - `body_pos_w`: `(T, num_bodies, 3)`
> - `body_quat_w`: `(T, num_bodies, 4)`
> - `body_lin_vel_w`: `(T, num_bodies, 3)`
> - `body_ang_vel_w`: `(T, num_bodies, 3)`
>
> `body_names` 中身体的顺序必须与 `body_pos_w` 等数组中的索引顺序一致。

**e) 配置事件参数**

```python
cfg.events["foot_friction"].params["asset_cfg"].geom_names = r"^(left|right)_foot[1-7]_collision$"
cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
```

**f) 配置终止条件**

```python
cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
```

**g) 设置 Viewer 视角**

```python
cfg.viewer.body_name = "torso_link"
```

**h) State Estimation 开关**

如果关闭状态估计，需要从观测中移除依赖外部估计的项：

```python
from mjlab.managers.observation_manager import ObservationGroupCfg

if not has_state_estimation:
    new_actor_terms = {
        k: v
        for k, v in cfg.observations["actor"].terms.items()
        if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
        terms=new_actor_terms,
        concatenate_terms=True,
        enable_corruption=True,
    )
```

**i) Play 模式覆盖**

```python
if play:
    # 无限时长
    cfg.episode_length_s = int(1e9)

    # 关闭观测噪声
    cfg.observations["actor"].enable_corruption = False

    # 移除外部推挤
    cfg.events.pop("push_robot", None)

    # 关闭 RSI（Reference State Initialization）随机化
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"
```

---

### 5. RL 训练配置

**路径**：`src/mjlab/tasks/tracking/config/my_robot/rl_cfg.py`

定义 PPO 的网络结构、算法参数和训练时长：

```python
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def my_robot_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="my_robot_tracking",
        save_interval=500,
        num_steps_per_env=24,
        max_iterations=30_000,
    )
```

> 不同机器人的网络结构可以相同，但 `experiment_name`、`max_iterations`、`obs_normalization` 等可能需要根据机器人复杂度调整。

---

### 6. 任务注册文件

**路径**：`src/mjlab/tasks/tracking/config/my_robot/__init__.py`

把环境注册到 mjlab 的任务系统中。tracking 任务使用 `MotionTrackingOnPolicyRunner`（支持 ONNX 导出，会自动打包 motion reference 数据）：

```python
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import my_robot_flat_tracking_env_cfg
from .rl_cfg import my_robot_tracking_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Tracking-Flat-My-Robot",
    env_cfg=my_robot_flat_tracking_env_cfg(),
    play_env_cfg=my_robot_flat_tracking_env_cfg(play=True),
    rl_cfg=my_robot_tracking_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Tracking-Flat-My-Robot-No-State-Estimation",
    env_cfg=my_robot_flat_tracking_env_cfg(has_state_estimation=False),
    play_env_cfg=my_robot_flat_tracking_env_cfg(has_state_estimation=False, play=True),
    rl_cfg=my_robot_tracking_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)
```

注册后，即可通过 CLI 使用：

```sh
uv run train Mjlab-Tracking-Flat-My-Robot --env.scene.num-envs 4096
uv run play Mjlab-Tracking-Flat-My-Robot --wandb-run-path ...
```

---

## 三、快速核对清单

在提交代码前，请确认：

- [ ] MJCF/XML 模型文件已放入 `asset_zoo/robots/my_robot/xmls/`
- [ ] 动作捕捉数据 `.npz` 已放入 `asset_zoo/robots/my_robot/motions/`（或指定路径）
- [ ] `<robot>_constants.py` 正确定义了执行器、初始姿态、碰撞配置和 `get_*_robot_cfg()` 工厂函数
- [ ] `*_ACTION_SCALE` 已正确计算并导出
- [ ] `env_cfgs.py` 中所有**按机器人定制**的字段都已覆盖：
  - `scene.entities`
  - `scene.sensors`（`self_collision` ContactSensorCfg）
  - `actions["joint_pos"].scale`
  - `commands["motion"].motion_file`
  - `commands["motion"].anchor_body_name`
  - `commands["motion"].body_names`（**顺序与 motion file 一致**）
  - `viewer.body_name`
  - `events["foot_friction"]` 和 `events["base_com"]` 的 `geom_names` / `body_names`
  - `terminations["ee_body_pos"].params["body_names"]`
- [ ] `rl_cfg.py` 中 `experiment_name` 已修改
- [ ] `__init__.py` 中 `task_id` 命名符合 `Mjlab-Tracking-{Flat|Rough}-<Robot>` 规范
- [ ] 运行 `make check` 通过格式和类型检查
- [ ] 运行 `uv run list-envs` 能看到新注册的任务

---

## 四、核心思路总结

- **常量文件**定义"机器人是什么"
- **环境配置文件**定义"机器人在这个任务里怎么模仿运动"
- **RL 配置文件**定义"怎么训练它"

tracking 任务与 velocity 任务的核心差异在于：

| 维度 | Velocity | Tracking |
|---|---|---|
| 核心命令 | `VelocityCommand` | `MotionCommand`（读取 motion file） |
| 关键配置 | 地形传感器、足端传感器 | `anchor_body_name`、`body_names`、自碰撞传感器 |
| Runner | `VelocityOnPolicyRunner` | `MotionTrackingOnPolicyRunner`（支持 ONNX 导出） |
| Play 模式 | 关闭噪声、移除 push | 额外关闭 RSI 随机化、设置 `sampling_mode="start"` |

按这个三层结构逐个准备，即可在 tracking 任务中接入一个新的机器人。
