"""
WebSocket robot client for websocket_inference.

Connects to a WebsocketPolicyServer, sends robot observations,
receives action chunks, and executes them on the robot.

Example command:
```shell
python -m lerobot.websocket_inference.robot_client \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --robot.id=black \
    --task="pick up the cup" \
    --server_host=127.0.0.1 \
    --server_port=8000 \
    --fps=30 \
    --actions_to_execute=10
```
"""

import dataclasses
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import draccus
import numpy as np
import websockets.sync.client

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
from lerobot.utils.utils import init_logging

from lerobot.websocket_inference.client import msgpack_numpy

logger = logging.getLogger("ws_robot_client")


@dataclass
class WsRobotClientConfig:
    """Configuration for WebSocket robot client.

    Mirrors relevant parameters from ``async_inference.configs.RobotClientConfig``
    but adapted for the simpler WebSocket request-response protocol.
    """

    robot: RobotConfig = field(metadata={"help": "Robot configuration"})
    task: str = field(default="", metadata={"help": "Task instruction / prompt sent with every observation"})
    server_host: str = field(default="127.0.0.1", metadata={"help": "WebSocket policy server host"})
    server_port: int = field(default=8000, metadata={"help": "WebSocket policy server port"})
    fps: int = field(default=30, metadata={"help": "Control loop frequency in Hz"})

    actions_to_execute: int | None = field(
        default=None,
        metadata={
            "help": (
                "Number of actions from each chunk to execute before requesting a new observation. "
                "When None, execute all actions returned by the server."
            )
        },
    )

    shutdown_server_on_disconnect: bool = field(
        default=False,
        metadata={"help": "Send an exit command to the server when the client disconnects"},
    )

    def __post_init__(self):
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.actions_to_execute is not None and self.actions_to_execute <= 0:
            raise ValueError(f"actions_to_execute must be positive when provided, got {self.actions_to_execute}")


class WsRobotClient:
    """Robot client that communicates with a ``WebsocketPolicyServer`` over WebSocket.

    Unlike the gRPC-based ``async_inference.RobotClient`` which streams observations
    and actions in parallel threads, this client uses a synchronous request-response
    loop: capture observation -> send -> receive action chunk -> execute actions.
    """

    def __init__(self, config: WsRobotClientConfig):
        self.config = config
        self.robot: Robot = make_robot_from_config(config.robot)

        obs_features = self.robot.observation_features
        self._motor_keys = [k for k, v in obs_features.items() if not isinstance(v, tuple)]
        self._camera_keys = [k for k, v in obs_features.items() if isinstance(v, tuple)]
        self._action_keys = list(self.robot.action_features.keys())

        self._ws: websockets.sync.client.ClientConnection | None = None
        self._packer = msgpack_numpy.Packer()
        self._server_metadata: dict[str, Any] = {}
        self._running = False

    @property
    def server_url(self) -> str:
        return f"ws://{self.config.server_host}:{self.config.server_port}"

    def connect(self) -> None:
        """Connect the robot and establish a WebSocket connection to the policy server."""
        self.robot.connect()
        logger.info("Robot connected")

        self._ws = websockets.sync.client.connect(
            self.server_url,
            compression=None,
            max_size=None,
        )
        self._server_metadata = msgpack_numpy.unpackb(self._ws.recv(), raw=False)
        logger.info("Connected to policy server at %s | metadata: %s", self.server_url, self._server_metadata)

    def disconnect(self) -> None:
        """Disconnect the robot and close the WebSocket."""
        self._running = False

        if self._ws is not None:
            if self.config.shutdown_server_on_disconnect:
                self._ws.send(self._packer.pack({"command": "exit"}))
                logger.info("Sent exit command to server")
            self._ws.close()
            self._ws = None
            logger.info("WebSocket closed")

        self.robot.disconnect()
        logger.info("Robot disconnected")

    def _raw_obs_to_ws_payload(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        """Convert a robot raw observation to the format expected by ``WebsocketPolicyServer``.

        Returns ``{state: np.ndarray, images: {cam_name: np.ndarray(H,W,C)}, prompt: str}``.
        """
        state = np.array([raw_obs[k] for k in self._motor_keys], dtype=np.float32)
        images = {k: raw_obs[k] for k in self._camera_keys}
        payload: dict[str, Any] = {"state": state, "images": images}
        if self.config.task:
            payload["prompt"] = self.config.task
        return payload

    def _action_array_to_dict(self, action: np.ndarray) -> dict[str, float]:
        """Convert an action numpy array to the ``{motor_name: value}`` dict expected by the robot."""
        return {k: float(action[i]) for i, k in enumerate(self._action_keys)}

    def step(self) -> np.ndarray:
        """Single inference round-trip: capture observation, send to server, return action chunk.

        Returns:
            numpy array of shape ``(chunk_size, action_dim)``.
        """
        raw_obs = self.robot.get_observation()
        payload = self._raw_obs_to_ws_payload(raw_obs)
        self._ws.send(self._packer.pack(payload))

        response = msgpack_numpy.unpackb(self._ws.recv(), raw=False)
        if isinstance(response, str):
            raise RuntimeError(f"Server returned error: {response}")
        return response["actions"]

    def run(self) -> None:
        """Main control loop: repeatedly infer and execute action chunks at the configured FPS."""
        self._running = True
        dt = 1.0 / self.config.fps
        step_count = 0
        logger.info("Starting control loop at %d FPS (dt=%.4fs)", self.config.fps, dt)

        while self._running:
            try:
                t0 = time.perf_counter()
                action_chunk = self.step()
                infer_ms = (time.perf_counter() - t0) * 1000

                n_actions = len(action_chunk)
                if self.config.actions_to_execute is not None:
                    n_actions = min(n_actions, self.config.actions_to_execute)

                logger.debug(
                    "Step %d | chunk_size=%d | executing=%d | infer=%.1fms",
                    step_count,
                    len(action_chunk),
                    n_actions,
                    infer_ms,
                )

                for i in range(n_actions):
                    loop_start = time.perf_counter()
                    action_dict = self._action_array_to_dict(action_chunk[i])
                    self.robot.send_action(action_dict)
                    elapsed = time.perf_counter() - loop_start
                    sleep_time = max(0.0, dt - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                step_count += 1

            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except websockets.ConnectionClosed:
                logger.info("Server closed connection")
                break


@draccus.wrap()
def main(cfg: WsRobotClientConfig) -> None:
    init_logging()
    logger.info("Config: %s", dataclasses.asdict(cfg))

    client = WsRobotClient(cfg)
    client.connect()
    try:
        client.run()
    finally:
        client.disconnect()


if __name__ == "__main__":
    register_third_party_plugins()
    main()
