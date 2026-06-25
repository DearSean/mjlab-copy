"""RL_BOY flat tracking environment configurations."""

import torch

from mjlab.asset_zoo.robots import (
  RL_BOY_ACTION_SCALE,
  get_rlboy_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sensor.contact_sensor import ContactSensor
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

def feet_air_time_jump(
  env,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.5,
  base_height_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward feet air time for jumping motion.

  Encourages the robot to lift its feet off the ground. Unlike the velocity
  version, this does not gate on velocity commands since pure jumping
  motions have no commanded velocity.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None

  asset = env.scene["robot"]
  base_height = asset.data.root_link_pos_w[:, 2]

  is_jumping = base_height > base_height_threshold

  in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
  reward = torch.sum(in_range.float(), dim=1)

  reward *= is_jumping.float()

  in_air = current_air_time > 0
  num_in_air = torch.sum(in_air.float())
  mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
    num_in_air, min=1
  )
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time

  return reward


def rlboy_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create RL_BOY flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_rlboy_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
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
  cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = RL_BOY_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "base_link"
  motion_cmd.body_names = (
    "base_link",
    "left_hip_pitch_link",
    "left_knee_pitch_link",
    "left_ankle_pitch_link",
    "right_hip_pitch_link",
    "right_knee_pitch_link",
    "right_ankle_pitch_link",
    "waist_yaw_link",
    "left_shoulder_roll_link",
    "left_elbow_pitch_link",
    "left_wrist_link",
    "right_shoulder_roll_link",
    "right_elbow_pitch_link",
    "right_wrist_link",
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  # 跌倒/躺倒姿态随机化：训练 2000 轮迭代后，以 0.01 的概率在 reset 时
  # 从仰卧、俯卧、左侧卧、右侧卧中随机采样一个姿态，并叠加随机 z 轴旋转。
  # 按当前 runner 的 num_steps_per_env=24 计算，2000 轮 = 48000 环境步。
  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_wrist_link",
    "right_wrist_link",
  )

  # Air time reward for jumping
  cfg.rewards["air_time"] = RewardTermCfg(
    func=feet_air_time_jump,
    weight=0.2,
    params={
      "sensor_name": "feet_ground_contact",
      "threshold_min": 0.05,
      "threshold_max": 0.5,
      "base_height_threshold": 0.05,
    },
  )

  cfg.viewer.body_name = "base_link"

  # Modify observations if we don't have state estimation.
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

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg
