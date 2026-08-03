# 在 Velocity 任务中新增机器人说明文档

以 **Unitree G1** 为参考范例，说明除机器人模型文件（MJCF/XML）之外，还需要在代码仓库中创建/修改哪些文件，以及每个文件的作用和关键内容。

---

## 一、文件总览

新增一个机器人（假设机器人目录名为 `my_robot`），除模型资产外，共需准备/修改 **7 个 Python 文件**：

| 序号 | 文件路径 | 作用 |
|:---:|---|---|
| 1 | `src/mjlab/asset_zoo/robots/my_robot/__init__.py` | 机器人资产包的 `__init__.py` |
| 2 | `src/mjlab/asset_zoo/robots/my_robot/my_robot_constants.py` | 机器人执行器、碰撞、初始姿态等常量配置 |
| 3 | `src/mjlab/asset_zoo/robots/__init__.py` | 导出机器人配置工厂函数和动作缩放表 |
| 4 | `src/mjlab/tasks/velocity/config/my_robot/env_cfgs.py` | 速度追踪任务的**环境配置**（传感器、奖励、终止条件等） |
| 5 | `src/mjlab/tasks/velocity/config/my_robot/rl_cfg.py` | 速度追踪任务的**强化学习超参数配置** |
| 6 | `src/mjlab/tasks/velocity/config/my_robot/__init__.py` | 将任务注册到 `mjlab` 任务注册表 |
| 7 | `src/mjlab/tasks/velocity/config/__init__.py` | （若之前为空，通常无需修改；Python 包结构需要即可） |

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

G1 使用了多种电机型号，按关节分组配置。一般流程：

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

`velocity` 任务使用 `JointPositionActionCfg`，需要为每个关节提供 `scale` 值。G1 中的计算方式：

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

需要把工厂函数和动作缩放表导出，供 `velocity` 任务引用：

```python
from mjlab.asset_zoo.robots.my_robot.my_robot_constants import (
    MY_ROBOT_ACTION_SCALE as MY_ROBOT_ACTION_SCALE,
)
from mjlab.asset_zoo.robots.my_robot.my_robot_constants import (
    get_my_robot_robot_cfg as get_my_robot_robot_cfg,
)
```

---

### 4. Velocity 任务环境配置

**路径**：`src/mjlab/tasks/velocity/config/my_robot/env_cfgs.py`

这是**最关键**的任务定制文件。它基于 `make_velocity_env_cfg()` 工厂函数提供的通用速度追踪配置，然后针对当前机器人的关节名、身体名、传感器等进行**覆盖和定制**。

#### 典型定制项（以 G1 为例）：

**a) 替换机器人和仿真参数**

```python
from mjlab.asset_zoo.robots import MY_ROBOT_ACTION_SCALE, get_my_robot_robot_cfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

def my_robot_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = make_velocity_env_cfg()

    # 仿真参数
    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.contact_sensor_maxmatch = 500

    # 替换机器人实体
    cfg.scene.entities = {"robot": get_my_robot_robot_cfg()}
```

**b) 配置传感器坐标系**

`terrain_scan`（地形射线扫描）和 `foot_height_scan`（足端高度扫描）需要绑定到机器人具体部位：

```python
# 把地形扫描绑定到机器人躯干
for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
        assert isinstance(sensor, RayCastSensorCfg)
        assert isinstance(sensor.frame, ObjRef)
        sensor.frame.name = "torso_link"  # G1 使用 "pelvis"

# 足端高度扫描绑定到左右脚 site
site_names = ("left_foot", "right_foot")
for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
        assert isinstance(sensor, TerrainHeightSensorCfg)
        sensor.frame = tuple(
            ObjRef(type="site", name=s, entity="robot") for s in site_names
        )
        sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)
```

**c) 添加接触传感器**

```python
feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
        mode="subtree",
        pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
        entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
)
cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground_cfg,)项目
```

**d) 设置动作缩放**

```python
joint_pos_action = cfg.actions["joint_pos"]
assert isinstance(joint_pos_action, JointPositionActionCfg)
joint_pos_action.scale = MY_ROBOT_ACTION_SCALE
```

**e) 设置 Viewer 视角**
项目
```python
cfg.viewer.body_name = "torso_link"
```

**f) 配置事件（摩擦随机化、质心偏移）**

```python
geom_names = tuple(f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8))

cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
```

**g) 配置奖励函数参数**

最复杂的一部分。`pose` 奖励需要为不同运动状态（站立/行走/奔跑）设置各关节的标准差：

```python
cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*knee.*": 0.35,
    # ... 更多关节
}

cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_link",)
cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_link",)
```

**h) 调整奖励权重**

```python
cfg.rewards["body_ang_vel"].weight = -0.05
cfg.rewards["angular_momentum"].weight = -0.02
cfg.rewards["air_time"].weight = 0.0
```

**i) Play 模式覆盖**

```python
if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    # ...
```

**j) 平地版本（Flat）**

通常从 Rough 版本继承，然后：
- 改为平面地形
- 移除地形扫描相关传感器和观测
- 调整终止条件和课程

```python
def my_robot_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    cfg = my_robot_rough_env_cfg(play=play)

    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # 移除地形扫描
    cfg.scene.sensors = tuple(s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan")
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]
    # ...
    return cfg
```

---

### 5. RL 训练配置

**路径**：`src/mjlab/tasks/velocity/config/my_robot/rl_cfg.py`

定义 PPO 的网络结构、算法参数和训练时长：

```python
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

def my_robot_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
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
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="my_robot_velocity",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=30_000,
    )
```

> 不同机器人的网络结构可以相同，但 `experiment_name`、`max_iterations`、`obs_normalization` 等可能需要根据机器人复杂度调整。

---

### 6. 任务注册文件

**路径**：`src/mjlab/tasks/velocity/config/my_robot/__init__.py`

RL_BOY 使用四个无机器人后缀的规范任务 ID。其他机器人应使用机器人名称
后缀，避免覆盖现有注册；Unitree G1 的 Flat/Rough 配置即采用这种方式：

```python
from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import my_robot_flat_env_cfg, my_robot_rough_env_cfg
from .rl_cfg import my_robot_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-My-Robot",
    env_cfg=my_robot_rough_env_cfg(),
    play_env_cfg=my_robot_rough_env_cfg(play=True),
    rl_cfg=my_robot_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-My-Robot",
    env_cfg=my_robot_flat_env_cfg(),
    play_env_cfg=my_robot_flat_env_cfg(play=True),
    rl_cfg=my_robot_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
```

注册后，即可通过 CLI 使用：

```sh
uv run train Mjlab-Velocity-Rough-My-Robot --env.scene.num-envs 4096
uv run play Mjlab-Velocity-Flat-My-Robot --wandb-run-path ...
```

---

## 三、快速核对清单

在提交代码前，请确认：

- [ ] MJCF/XML 模型文件已放入 `asset_zoo/robots/my_robot/xmls/`
- [ ] `<robot>_constants.py` 正确定义了执行器、初始姿态、碰撞配置和 `get_*_robot_cfg()` 工厂函数
- [ ] `*_ACTION_SCALE` 已正确计算并导出
- [ ] `env_cfgs.py` 中所有**按机器人定制**的字段都已覆盖：
  - `scene.entities`
  - `terrain_scan` 的 `frame.name`
  - `foot_height_scan` 的 `frame` 和 `pattern`
  - 接触传感器的名称和模式
  - `actions["joint_pos"].scale`
  - `viewer.body_name`
  - `events["foot_friction"]` 和 `events["base_com"]` 的 `geom_names` / `body_names`
  - `rewards["pose"]` 的三个 `std_*` 字典
  - `rewards["upright"]`、`rewards["body_ang_vel"]` 的 `body_names`
  - `rewards["foot_clearance"]`、`rewards["foot_slip"]` 的 `site_names`
- [ ] `rl_cfg.py` 中 `experiment_name` 已修改
- [ ] `__init__.py` 使用不会与已有机器人冲突的任务 ID
- [ ] 运行 `make check` 通过格式和类型检查
- [ ] 运行 `uv run list-envs` 能看到新注册的任务

---

## 四、核心思路总结

- **常量文件**定义"机器人是什么"
- **环境配置文件**定义"机器人在这个任务里怎么用"
- **RL 配置文件**定义"怎么训练它"

按这个三层结构逐个准备，即可在 velocity 任务中接入一个新的机器人。
