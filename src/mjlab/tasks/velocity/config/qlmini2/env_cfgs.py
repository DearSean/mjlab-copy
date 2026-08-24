"""qlmini2 flat-terrain velocity environment configuration."""

from mjlab.asset_zoo.robots import QLMINI2_ACTION_SCALE, get_qlmini2_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import CurriculumTermCfg, SceneEntityCfg
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
  twist_cmd.ranges.lin_vel_y = (-0.3, 0.3)
  twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)
  twist_cmd.ranges.heading = None
  twist_cmd.heading_command = False
  twist_cmd.rel_heading_envs = 0.0
  twist_cmd.rel_standing_envs = 0.0
  twist_cmd.rel_world_envs = 0.0
  twist_cmd.rel_forward_envs = 0.0
  twist_cmd.mode_probabilities = UniformVelocityCommandCfg.ModeProbabilities(
    standing=0.10,
    forward=0.15,
    backward=0.15,
    lateral=0.15,
    yaw=0.20,
    mixed=0.25,
  )
  twist_cmd.validate()

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_geom_names
  cfg.events["foot_friction"].params["ranges"] = (1.0, 2.0)
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
    r".*hip_roll.*": 0.25,
    r".*hip_yaw.*": 0.22,
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
    r".*hip_roll.*": 0.35,
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
  torque_cfg = SceneEntityCfg("robot", actuator_names=[".*"])
  cfg.rewards["torque_continuous_excess"] = RewardTermCfg(
    func=mdp.requested_torque_limit_penalty,
    weight=-0.05,
    params={
      "asset_cfg": torque_cfg,
      "limit": "continuous",
    },
  )
  cfg.rewards["torque_peak_usage"] = RewardTermCfg(
    func=mdp.requested_torque_limit_penalty,
    weight=-0.02,
    params={
      "asset_cfg": torque_cfg,
      "limit": "peak",
      "peak_threshold": 0.95,
    },
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
  cfg.rewards["air_time"].params["command_threshold"] = 0.05
  cfg.rewards["foot_clearance"].weight = -0.75
  cfg.rewards["soft_landing"].weight = 0.0
  cfg.rewards["soft_landing"].params["force_threshold"] = 200.0
  cfg.rewards["soft_landing"].params["force_scale"] = 140.0
  cfg.rewards["soft_landing"].params["squared"] = True
  cfg.rewards["foot_clearance"].params["target_height"] = 0.06
  cfg.rewards["foot_clearance"].params["normalize_by_target"] = True
  cfg.rewards["foot_swing_height"].weight = -0.75
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.06
  cfg.rewards["foot_slip"].weight = -0.35
  cfg.rewards["foot_slip"].params["l1_weight"] = 0.10
  for reward_name in ("foot_clearance", "foot_slip"):
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": "self_collision", "force_threshold": 10.0},
  )

  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.curriculum.pop("terrain_levels", None)
  cfg.events["push_robot"].func = mdp.staged_push_by_setting_velocity
  cfg.curriculum["command_vel"] = CurriculumTermCfg(
    func=mdp.adaptive_velocity_progression,
    params={
      "command_name": "twist",
      "push_event_name": "push_robot",
      # Stage boundaries are PPO iteration * 24.  They are earliest boundaries:
      # a stage only advances after its current policy has completed a full
      # success window, so hard commands never arrive during a falling phase.
      "window_size": 4096,
      "stages": [
        {
          "step": 0,
          "success_threshold": None,
          "lin_vel_x": (-0.25, 0.40),
          "lin_vel_y": (-0.12, 0.12),
          "ang_vel_z": (-0.15, 0.15),
          "mode_probabilities": {
            "standing": 0.25,
            "forward": 0.60,
            "backward": 0.00,
            "lateral": 0.00,
            "yaw": 0.05,
            "mixed": 0.10,
            "min_linear_speed": 0.10,
            "min_angular_speed": 0.10,
          },
          "interval_range_s": (1.0, 3.0),
          "velocity_range": {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
          },
          "reward_weights": {
            "soft_landing": 0.0,
            "torque_continuous_excess": 0.0,
            "torque_peak_usage": 0.0,
          },
        },
        {
          "step": 500 * 24,
          "success_threshold": 0.80,
          "lin_vel_x": (-0.35, 0.55),
          "lin_vel_y": (-0.15, 0.15),
          "ang_vel_z": (-0.25, 0.25),
          "mode_probabilities": {
            "standing": 0.20,
            "forward": 0.50,
            "backward": 0.05,
            "lateral": 0.03,
            "yaw": 0.10,
            "mixed": 0.12,
            "min_linear_speed": 0.12,
            "min_angular_speed": 0.12,
          },
          "interval_range_s": (10.0, 14.0),
          "velocity_range": {
            "x": (-0.08, 0.08),
            "y": (-0.08, 0.08),
            "z": (-0.05, 0.05),
            "roll": (-0.10, 0.10),
            "pitch": (-0.10, 0.10),
            "yaw": (-0.10, 0.10),
          },
          "reward_weights": {
            "soft_landing": 0.0,
            "torque_continuous_excess": 0.0,
            "torque_peak_usage": 0.0,
          },
        },
        {
          "step": 1000 * 24,
          "success_threshold": 0.80,
          "lin_vel_x": (-0.45, 0.70),
          "lin_vel_y": (-0.25, 0.25),
          "ang_vel_z": (-0.40, 0.40),
          "mode_probabilities": {
            "standing": 0.18,
            "forward": 0.40,
            "backward": 0.10,
            "lateral": 0.07,
            "yaw": 0.12,
            "mixed": 0.13,
            "min_linear_speed": 0.15,
            "min_angular_speed": 0.15,
          },
          "interval_range_s": (8.0, 12.0),
          "velocity_range": {
            "x": (-0.15, 0.15),
            "y": (-0.15, 0.15),
            "z": (-0.10, 0.10),
            "roll": (-0.20, 0.20),
            "pitch": (-0.20, 0.20),
            "yaw": (-0.20, 0.20),
          },
          "reward_weights": {
            "soft_landing": 0.0,
            "torque_continuous_excess": 0.0,
            "torque_peak_usage": 0.0,
          },
        },
        {
          "step": 1750 * 24,
          "success_threshold": 0.78,
          "lin_vel_x": (-0.55, 0.85),
          "lin_vel_y": (-0.35, 0.35),
          "ang_vel_z": (-0.55, 0.55),
          "mode_probabilities": {
            "standing": 0.15,
            "forward": 0.32,
            "backward": 0.14,
            "lateral": 0.10,
            "yaw": 0.15,
            "mixed": 0.14,
            "min_linear_speed": 0.15,
            "min_angular_speed": 0.20,
          },
          "interval_range_s": (6.0, 10.0),
          "velocity_range": {
            "x": (-0.22, 0.22),
            "y": (-0.22, 0.22),
            "z": (-0.15, 0.15),
            "roll": (-0.28, 0.28),
            "pitch": (-0.28, 0.28),
            "yaw": (-0.28, 0.28),
          },
          "reward_weights": {
            "soft_landing": -0.025,
            "torque_continuous_excess": -0.02,
            "torque_peak_usage": -0.005,
          },
        },
        {
          "step": 2400 * 24,
          "success_threshold": 0.75,
          "lin_vel_x": (-0.65, 1.00),
          "lin_vel_y": (-0.50, 0.50),
          "ang_vel_z": (-0.70, 0.70),
          "mode_probabilities": {
            "standing": 0.12,
            "forward": 0.25,
            "backward": 0.16,
            "lateral": 0.13,
            "yaw": 0.18,
            "mixed": 0.16,
            "min_linear_speed": 0.15,
            "min_angular_speed": 0.22,
          },
          "interval_range_s": (4.0, 7.0),
          "velocity_range": {
            "x": (-0.35, 0.35),
            "y": (-0.35, 0.35),
            "z": (-0.25, 0.25),
            "roll": (-0.42, 0.42),
            "pitch": (-0.42, 0.42),
            "yaw": (-0.42, 0.42),
          },
          "reward_weights": {
            "soft_landing": -0.05,
            "torque_continuous_excess": -0.035,
            "torque_peak_usage": -0.01,
          },
        },
        {
          "step": 3000 * 24,
          "success_threshold": 0.72,
          "lin_vel_x": (-0.80, 1.20),
          "lin_vel_y": (-0.80, 0.80),
          "ang_vel_z": (-1.00, 1.00),
          "mode_probabilities": {
            "standing": 0.10,
            "forward": 0.15,
            "backward": 0.15,
            "lateral": 0.15,
            "yaw": 0.20,
            "mixed": 0.25,
            "min_linear_speed": 0.15,
            "min_angular_speed": 0.25,
          },
          "interval_range_s": (1.0, 3.0),
          "velocity_range": {
            "x": (-0.50, 0.50),
            "y": (-0.50, 0.50),
            "z": (-0.40, 0.40),
            "roll": (-0.52, 0.52),
            "pitch": (-0.52, 0.52),
            "yaw": (-0.78, 0.78),
          },
          "reward_weights": {
            "soft_landing": -0.10,
            "torque_continuous_excess": -0.05,
            "torque_peak_usage": -0.02,
          },
        },
      ],
    },
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    twist_cmd.ranges.lin_vel_x = (-0.5, 0.8)
    twist_cmd.ranges.lin_vel_y = (-0.8, 0.8)
    twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)

  return cfg
