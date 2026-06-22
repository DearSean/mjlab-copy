"""RL Boy velocity environment configurations."""

from mjlab.asset_zoo.robots import (
  RL_BOY_ACTION_SCALE,
  get_rlboy_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def rlboy_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create RL Boy rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()
  assert cfg is not None 

  # 仿真参数
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 70
  from mjlab.utils.nan_guard import NanGuardCfg

  cfg.sim.nan_guard = NanGuardCfg(
  enabled=True,
  buffer_size=100,      # 保留 NaN 前多少步
  output_dir="/tmp/mjlab/nan_dumps",
  max_envs_to_dump=5,   # 最多导出的 env 数量
  ) 

  # 替换机器人实体
  cfg.scene.entities = {"robot": get_rlboy_robot_cfg()}

  # 设置地形射线扫描传感器帧为 RL_BOY 的 base_link
  # 注意: RL_BOY 没有 pelvis，使用 base_link 作为主躯干
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "base_link"

  # 足端高度扫描绑定到左右脚踝 site
  # 注意: RL_BOY 只有 ankle_pitch，没有 ankle_roll
  # 需要在 XML 中添加 foot site，或者使用现有的 ankle link
  # RL_BOY 机器人脚部 site 名称（已在 XML 的 ankle body 下添加 left_foot / right_foot）
  site_names = ("left_foot", "right_foot")
  foot_body_names = ("left_ankle_pitch_link", "right_ankle_pitch_link")
  foot_geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="body", name=s, entity="robot") for s in foot_body_names
      )
      # 使用更小的 ring radius 因为 RL_BOY 的脚较小
      sensor.pattern = RingPatternCfg.single_ring(radius=0.02, num_samples=6)

  # 脚部接触传感器
  # RL_BOY 使用 foot geom 作为脚部
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_pitch_link|right_ankle_pitch_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  # 自碰撞检测传感器
  # RL_BOY 使用 base_link 作为主躯干
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  # 设置动作缩放
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = RL_BOY_ACTION_SCALE

  # 设置 Viewer 视角主体
  # RL_BOY 有 base_link 作为视觉参考点
  cfg.viewer.body_name = "base_link"

  # 命令可视化偏移
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  # TODO: 根据实际机器人高度调整
  twist_cmd.viz.z_offset = 0.5  # RL_BOY 站立高度约 0.45m

  # 事件配置 - 摩擦随机化
  # RL_BOY 脚部 geom 名称
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  # ===== 新增域随机化 (DR) =====
  # 1) PD 增益随机化 ±10%：让策略适应电机响应的实物差异
  #    asset_cfg 不指定 actuator_names 时默认 actuator_ids=slice(None)，覆盖所有 20 个执行器
  cfg.events["pd_gains"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "operation": "scale",
      "kp_range": (0.9, 1.1),
      "kd_range": (0.9, 1.1),
    },
  )
  # 2) 全身 link 质量随机化 ±5%（pseudo_inertia 同步缩放 mass+inertia，物理上一致）
  #    alpha = log(density_ratio), ±5% 质量变化 ≈ alpha_range = (-0.05, 0.05)
  cfg.events["link_mass"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "alpha_range": (-0.05, 0.05),
    },
  )
  # 3) 负载随机化：在 base_link 上叠加 0~2 kg 质量
  #    注：body_mass 单独使用时不修改 inertia，docstring 明确指出仅适用于「在 COM 添加点质量」的场景
  cfg.events["base_payload"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
      "operation": "add",
      "ranges": (0.0, 2.0),
    },
  )
  # ===== 新增域随机化结束 =====

  # 姿态奖励标准差配置
  # RL_BOY 关节结构:
  # - 腿部: hip_yaw, hip_roll, hip_pitch, knee_pitch, ankle_pitch (无 ankle_roll)
  # - 腰部: waist_yaw
  # - 手臂: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch (无 _joint 后缀)
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # 腿部关节
    r".*hip_pitch.*": 0.3,
    r".*hip_roll.*": 0.15,
    r".*hip_yaw.*": 0.15,
    r".*knee_pitch.*": 0.35,
    r".*ankle_pitch.*": 0.25,
    # 注意: RL_BOY 没有 ankle_roll 关节
    # 腰部关节
    r".*waist_yaw.*": 0.2,
    # 头部关节（锁定，不主动运动）
    r".*head_yaw.*": 0.05,
    # 手臂关节 
    r".*shoulder_pitch.*": 0.15,
    r".*shoulder_roll.*": 0.15,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow_pitch.*": 0.15,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # 腿部关节
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.2,
    r".*hip_yaw.*": 0.2,
    r".*knee_pitch.*": 0.6,
    r".*ankle_pitch.*": 0.35,
    # 腰部关节
    r".*waist_yaw.*": 0.3,
    # 头部关节（锁定，不主动运动）
    r".*head_yaw.*": 0.05,
    # 手臂关节 
    r".*shoulder_pitch.*": 0.5, 
    r".*shoulder_roll.*": 0.2,
    r".*shoulder_yaw.*": 0.15,
    r".*elbow_pitch.*": 0.35,
  }

  # 躯干直立奖励主体
  # RL_BOY 使用 base_link 作为主躯干
  cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)

  # 足部清洁和滑动奖励
  # 注意: RL_BOY 没有 ankle_roll，使用 ankle_pitch 作为脚部参考
  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  # 奖励权重调整
  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.2

  # 自碰撞惩罚
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )

  # Play 模式覆盖
  if play:
    # 无限episode长度
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def rlboy_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:


  """Create RL Boy flat terrain velocity configuration."""
  cfg = rlboy_rough_env_cfg(play=play)

  cfg.sim.njmax = 500
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # 切换到平地地形
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # 移除地形扫描传感器 (无地形可扫描)
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  cfg.terminations.pop("out_of_terrain_bounds", None)

  # 禁用地形课程
  cfg.curriculum.pop("terrain_levels", None)

  # 定制速度指令课程学习
  # 从小范围开始，逐步提升线速度与角速度指令范围
  # step 单位为环境步(decimation 后的步进), 24 = 4096 envs 跑一个 iteration 的近似步数
  # 经验值参考: 0, ~120k, ~300k, ~600k 步切换
  cfg.curriculum["command_vel"] = CurriculumTermCfg(
    func=mdp.commands_vel,
    params={
      "command_name": "twist",
      "velocity_stages": [
        # 阶段 0: 起步 —— 小范围、低速
        {
          "step": 0,
          "lin_vel_x": (-0.6, 0.8),
          "lin_vel_y": (-0.3, 0.3),
          "ang_vel_z": (-0.4, 0.4),
        },
        # 阶段 1: 提升 x 方向速度上限
        {
          "step": 5000 * 24,
          "lin_vel_x": (-1.0, 1.2),
          "lin_vel_y": (-0.5, 0.5),
          "ang_vel_z": (-0.6, 0.6),
        },
        # 阶段 2: 进一步提速并扩大侧向与偏航
        {
          "step": 10000 * 24,
          "lin_vel_x": (-1.5, 1.8),
          "lin_vel_y": (-0.7, 0.7),
          "ang_vel_z": (-0.8, 0.8),
        },
        # 阶段 3: 接近最终能力上限
        {
          "step": 20000 * 24,
          "lin_vel_x": (-2.0, 2.5),
          "lin_vel_y": (-1.0, 1.0),
          "ang_vel_z": (-1.0, 1.0),
        },
      ],
    },
  )

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    # TODO: 根据 RL_BOY 的实际能力调整速度范围
    twist_cmd.ranges.lin_vel_x = (-1.0, 1.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)


  return cfg
