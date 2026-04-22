"""
WebSocket policy client for websocket_inference.

Connects a lerobot Robot to a remote WebSocket policy server using
the openpi runtime pattern (Environment / Agent / Runtime).

Example command:
```shell
python -m lerobot.websocket_inference.scripts.client_policy \
    --robot.type=bi_so101_follower \
    --robot.left_arm_port=/dev/ttyACM0 \
    --robot.right_arm_port=/dev/ttyACM1 \
    --robot.id=bimanual_follower \
    --robot.disable_arm=left \
    --robot.cameras='{
      "front_cam": {"type": "intelrealsense", "serial_number_or_name": "148522070056", "width": 1280, "height": 720, "fps": 30},
      "hand_cam": {"type": "opencv", "index_or_path": 12, "width": 640, "height": 480, "fps": 30},
      "side_cam": {"type": "opencv", "index_or_path": 10, "width": 640, "height": 480, "fps": 30}
    }' \
    --reset_pose='[-10.305, -99.649, 67.330, 95.787, -1.880, 3.119]' \
    --task="Put the toy into the box" \
    --server_host=127.0.0.1 \
    --server_port=8000 \
    --action_horizon=10 \
    --max_hz=30 \
    --num_episodes=1 \
    --max_episode_steps=1000
```
"""

import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import draccus
import numpy as np
from typing_extensions import override

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so101_follower,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import enter_pressed, init_logging, log_say

from lerobot.websocket_inference.client import image_tools
from lerobot.websocket_inference.client.action_chunk_broker import ActionChunkBroker
from lerobot.websocket_inference.client.runtime import environment as _environment
from lerobot.websocket_inference.client.runtime.agents.policy_agent import PolicyAgent
from lerobot.websocket_inference.client.runtime.runtime import Runtime
from lerobot.websocket_inference.client.websocket_client_policy import WebsocketClientPolicy

def _format_array(values: np.ndarray | list[float], precision: int = 3) -> str:
    array = np.asarray(values, dtype=np.float32)
    return np.array2string(array, precision=precision, floatmode="fixed")


class LeRobotEnvironment(_environment.Environment):
    """Wraps a lerobot Robot as an openpi-style Environment.

    Automatically introspects the robot's observation/action features
    to convert between the robot's flat dict format and the WebSocket
    server's {state, images, prompt} protocol.
    """

    def __init__(
        self,
        robot: Robot,
        task: str = "",
        image_height: int = 0,
        image_width: int = 0,
        play_sounds: bool = True,
        end_episode_on_enter: bool = True,
        log_reset_pose_candidates: int = 5,
        reset_position: list[float] | None = None,
    ) -> None:
        self._robot = robot

        obs_features = robot.observation_features
        self._motor_keys = [k for k, v in obs_features.items() if not isinstance(v, tuple)]
        self._camera_keys = [k for k, v in obs_features.items() if isinstance(v, tuple)]
        self._action_keys = list(robot.action_features.keys())

        self._task = task
        self._resize = image_height > 0 and image_width > 0
        self._image_height = image_height
        self._image_width = image_width
        self._play_sounds = play_sounds
        self._end_episode_on_enter = end_episode_on_enter and sys.stdin.isatty()
        self._log_reset_pose_candidates = log_reset_pose_candidates
        self._remaining_reset_pose_candidate_logs = 0
        self._episode_complete = False
        self._reset_position = None if reset_position is None else np.asarray(reset_position, dtype=np.float32)

        if end_episode_on_enter and not self._end_episode_on_enter:
            logging.warning("stdin is not a TTY; disabling Enter-to-end-episode support for this run.")

        if self._reset_position is not None and len(self._reset_position) != len(self._action_keys):
            raise ValueError(
                "reset_position length must match the number of robot action dimensions, "
                f"got {len(self._reset_position)} and {len(self._action_keys)}"
            )

        if self._reset_position is None:
            logging.info("No reset_position configured; reset will only play the environment reminder.")
        else:
            logging.info(
                "Configured reset_position | dims=%d | action_keys=%s | target=%s",
                len(self._reset_position),
                self._action_keys,
                _format_array(self._reset_position),
            )

    @override
    def reset(self) -> None:
        self._episode_complete = False
        self._remaining_reset_pose_candidate_logs = self._log_reset_pose_candidates
        log_say("Reset the environment", play_sounds=self._play_sounds)

        raw_obs = self._robot.get_observation()
        current_position = np.array([float(raw_obs[k]) for k in self._action_keys], dtype=np.float32)
        logging.info(
            "Current joint state at reset | action_keys=%s | pose=%s",
            self._action_keys,
            _format_array(current_position),
        )

        if self._reset_position is None:
            return

        logging.info(
            "Resetting robot to reset_position | current=%s | target=%s",
            _format_array(current_position),
            _format_array(self._reset_position),
        )
        trajectory = np.linspace(current_position, self._reset_position, num=51, dtype=np.float32)

        for pose in trajectory[1:]:
            action_dict = {k: float(pose[i]) for i, k in enumerate(self._action_keys)}
            self._robot.send_action(action_dict)
            time.sleep(0.015)

        final_obs = self._robot.get_observation()
        final_position = np.array([float(final_obs[k]) for k in self._action_keys], dtype=np.float32)
        logging.info("Reset finished | final=%s", _format_array(final_position))

    @override
    def is_episode_complete(self) -> bool:
        if self._episode_complete:
            return True

        if self._end_episode_on_enter and enter_pressed():
            logging.info("Episode marked complete by user")
            self._episode_complete = True

        return self._episode_complete

    @override
    def get_observation(self) -> dict:
        raw_obs = self._robot.get_observation()
        current_position = np.array([float(raw_obs[k]) for k in self._action_keys], dtype=np.float32)

        if self._remaining_reset_pose_candidate_logs > 0:
            sample_idx = self._log_reset_pose_candidates - self._remaining_reset_pose_candidate_logs + 1
            logging.info(
                "Reset pose candidate sample %d/%d | action_keys=%s | pose=%s",
                sample_idx,
                self._log_reset_pose_candidates,
                self._action_keys,
                _format_array(current_position),
            )
            self._remaining_reset_pose_candidate_logs -= 1

        state = np.array([raw_obs[k] for k in self._motor_keys], dtype=np.float32)

        images: dict[str, Any] = {}
        for cam_key in self._camera_keys:
            img = raw_obs[cam_key]
            img = image_tools.convert_to_uint8(img)
            if self._resize:
                img = image_tools.resize_with_pad(
                    img[np.newaxis], self._image_height, self._image_width
                )[0]
            images[cam_key] = img

        obs: dict[str, Any] = {"state": state, "images": images}
        if self._task:
            obs["prompt"] = self._task
        return obs

    @override
    def apply_action(self, action: dict) -> None:
        actions = action["actions"]
        action_dict = {k: float(actions[i]) for i, k in enumerate(self._action_keys)}
        self._robot.send_action(action_dict)


@dataclass
class PolicyClientConfig:
    """Configuration for the WebSocket policy client."""

    robot: RobotConfig = field(metadata={"help": "Robot configuration"})
    task: str = field(
        default="",
        metadata={"help": "Optional task instruction; when empty, the server default prompt is used"},
    )
    server_host: str = field(default="127.0.0.1", metadata={"help": "WebSocket policy server host"})
    server_port: int = field(default=8000, metadata={"help": "WebSocket policy server port"})
    action_horizon: int = field(
        default=10,
        metadata={"help": "Number of actions to execute from each chunk before re-querying the server"},
    )
    max_hz: float = field(default=30, metadata={"help": "Maximum control loop frequency in Hz"})
    num_episodes: int = field(default=1, metadata={"help": "Number of episodes to run"})
    max_episode_steps: int = field(
        default=0,
        metadata={"help": "Maximum steps per episode (0 = unlimited, use Ctrl+C to stop)"},
    )
    play_sounds: bool = field(default=True, metadata={"help": "Use text-to-speech prompts for lifecycle events"})
    end_episode_on_enter: bool = field(
        default=True,
        metadata={"help": "Allow pressing Enter to end the current episode"},
    )
    reset_pose: list[float] | None = field(
        default=None,
        metadata={"help": "Optional joint positions used to reset the robot between episodes"},
    )
    log_reset_pose_candidates: int = field(
        default=5,
        metadata={"help": "Log this many initial joint-state samples each episode to help build --reset-pose"},
    )
    image_height: int = field(default=0, metadata={"help": "Resize images to this height (0 = no resize)"})
    image_width: int = field(default=0, metadata={"help": "Resize images to this width (0 = no resize)"})

    def __post_init__(self):
        if self.action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {self.action_horizon}")
        if self.max_hz <= 0:
            raise ValueError(f"max_hz must be positive, got {self.max_hz}")
        if self.log_reset_pose_candidates < 0:
            raise ValueError(
                f"log_reset_pose_candidates must be non-negative, got {self.log_reset_pose_candidates}"
            )


@draccus.wrap()
def main(cfg: PolicyClientConfig) -> None:
    init_logging()

    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    logging.info("Robot connected")

    policy = WebsocketClientPolicy(
        host=cfg.server_host,
        port=cfg.server_port,
    )
    metadata = policy.get_server_metadata()
    logging.info("Connected to policy server | metadata: %s", metadata)

    server_default_prompt = metadata.get("default_prompt")
    if cfg.task:
        if isinstance(server_default_prompt, str) and server_default_prompt and server_default_prompt != cfg.task:
            logging.info("Using client task prompt and overriding server default prompt")
        else:
            logging.info("Using client task prompt")
    elif isinstance(server_default_prompt, str) and server_default_prompt:
        logging.info("Using server default prompt")
    else:
        raise ValueError("No task prompt configured. Set --task on the client or --default-prompt on the server.")

    reset_position = cfg.reset_pose
    if reset_position is not None:
        logging.info("Using client reset_pose: %s", reset_position)
    else:
        logging.info("No client reset_pose configured.")

    broker = ActionChunkBroker(policy=policy, action_horizon=cfg.action_horizon)
    agent = PolicyAgent(policy=broker)

    env = LeRobotEnvironment(
        robot=robot,
        task=cfg.task,
        image_height=cfg.image_height,
        image_width=cfg.image_width,
        play_sounds=cfg.play_sounds,
        end_episode_on_enter=cfg.end_episode_on_enter,
        log_reset_pose_candidates=cfg.log_reset_pose_candidates,
        reset_position=reset_position,
    )

    runtime = Runtime(
        environment=env,
        agent=agent,
        subscribers=[],
        max_hz=cfg.max_hz,
        num_episodes=cfg.num_episodes,
        max_episode_steps=cfg.max_episode_steps,
    )

    try:
        logging.info(
            "Starting runtime | action_horizon=%d | max_hz=%.0f | episodes=%d | max_steps=%d",
            cfg.action_horizon,
            cfg.max_hz,
            cfg.num_episodes,
            cfg.max_episode_steps,
        )
        runtime.run()
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        policy.close()
        robot.disconnect()
        logging.info("Cleanup complete")


if __name__ == "__main__":
    register_third_party_plugins()
    main()
