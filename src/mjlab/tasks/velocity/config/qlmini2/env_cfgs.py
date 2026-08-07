"""qlmini2 flat-terrain velocity environment configuration."""

from mjlab.asset_zoo.robots import QLMINI2_ACTION_SCALE, get_qlmini2_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import SceneEntityCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def qlmini2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the qlmini2 flat velocity task."""
  cfg = make_velocity_env_cfg()

  cfg.sim.njmax = 600
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  cfg.scene.entities = {"robot": get_qlmini2_robot_cfg()}

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  site_names = ("left_foot", "right_foot")
  foot_geom_names = tuple(
    f"{side}_foot{index}_collision"
    for side in ("left", "right")
    for index in range(1, 8)
  )

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=name, entity="robot") for name in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.025, num_samples=6)

  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name != "terrain_scan"
  ) + (
    ContactSensorCfg(
      name="feet_ground_contact",
      primary=ContactMatch(
        mode="subtree",
        pattern=r"^(left|right)_foot$",
        entity="robot",
      ),
      secondary=ContactMatch(mode="body", pattern="terrain"),
      fields=("found", "force"),
      reduce="netforce",
      num_slots=1,
      track_air_time=True,
    ),
    ContactSensorCfg(
      name="self_collision",
      primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
      fields=("found", "force"),
      reduce="none",
      num_slots=1,
      history_length=4,
    ),
  )

  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = QLMINI2_ACTION_SCALE

  cfg.viewer.body_name = "waist_yaw_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 0.7
  twist_cmd.ranges.lin_vel_x = (-0.3, 0.5)
  twist_cmd.ranges.lin_vel_y = (-0.2, 0.2)
  twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  # Keep the waist close to its neutral pose while leaving enough arm range for
  # counter-swing. Whole-body angular-momentum and waist-velocity costs below
  # make the learned arm motion a balance strategy rather than unconstrained
  # flailing.
  cfg.rewards["pose"].params["std_standing"] = {
    r".*_(hip_pitch|hip_roll|hip_yaw|knee|foot_pitch)_joint": 0.05,
    r".*waist_yaw.*": 0.06,
    r".*shoulder_pitch.*": 0.12,
    r".*shoulder_(roll|yaw).*": 0.08,
    r".*elbow.*": 0.12,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_pitch.*": 0.25,
    r".*hip_roll.*": 0.12,
    r".*hip_yaw.*": 0.12,
    r".*knee.*": 0.3,
    r".*foot_pitch.*": 0.2,
    r".*waist_yaw.*": 0.14,
    r".*shoulder_pitch.*": 0.45,
    r".*shoulder_roll.*": 0.2,
    r".*shoulder_yaw.*": 0.3,
    r".*elbow.*": 0.60,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_pitch.*": 0.4,
    r".*hip_roll.*": 0.18,
    r".*hip_yaw.*": 0.18,
    r".*knee.*": 0.5,
    r".*foot_pitch.*": 0.3,
    r".*waist_yaw.*": 0.22,
    r".*shoulder_pitch.*": 0.65,
    r".*shoulder_roll.*": 0.2,
    r".*shoulder_yaw.*": 0.3,
    r".*elbow.*": 0.85,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("waist_yaw_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("waist_yaw_link",)
  cfg.rewards["body_ang_vel"].weight = -0.10
  cfg.rewards["angular_momentum"].weight = -0.08
  cfg.rewards["waist_yaw_velocity"] = RewardTermCfg(
    func=envs_mdp.joint_vel_l2,
    weight=-0.04,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=("waist_yaw_joint",))},
  )
  shoulder_pitch_cfg = SceneEntityCfg(
    "robot",
    joint_names=("left_shoulder_pitch_joint", "right_shoulder_pitch_joint"),
  )
  cfg.rewards["arm_swing_antiphase"] = RewardTermCfg(
    func=mdp.paired_joint_antiphase_l2,
    weight=-0.06,
    params={
      "asset_cfg": shoulder_pitch_cfg,
      "std": 0.45,
      "command_name": "twist",
      "command_threshold": 0.15,
    },
  )
  cfg.rewards["arm_pitch_bias"] = RewardTermCfg(
    func=mdp.filtered_joint_bias_l2,
    weight=-0.03,
    params={
      "asset_cfg": shoulder_pitch_cfg,
      "time_constant_s": 0.75,
      "std": 0.25,
      "command_name": "twist",
      "command_threshold": 0.15,
    },
  )
  cfg.rewards["action_rate_l2"].weight = -0.05
  cfg.rewards["air_time"].weight = 0.25
  cfg.rewards["foot_slip"].weight = -0.20
  cfg.rewards["soft_landing"].weight = -2.0e-5
  cfg.rewards["foot_clearance"].params["target_height"] = 0.06
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.06
  for reward_name in ("foot_clearance", "foot_slip"):
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": "self_collision", "force_threshold": 10.0},
  )

  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.curriculum.pop("terrain_levels", None)
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {"step": 0, "lin_vel_x": (-0.3, 0.5), "ang_vel_z": (-0.3, 0.3)},
    {"step": 750 * 24, "lin_vel_x": (-0.5, 0.8), "ang_vel_z": (-0.5, 0.5)},
    {"step": 1500 * 24, "lin_vel_x": (-0.8, 1.2), "ang_vel_z": (-1.0, 1.0)},
  ]

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    twist_cmd.ranges.lin_vel_x = (-0.5, 0.8)
    twist_cmd.ranges.lin_vel_y = (-0.3, 0.3)
    twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)

  return cfg
