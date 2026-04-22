import asyncio
import http
import importlib
import logging
import time
import traceback
from contextlib import nullcontext
from typing import Any

import torch
import websockets
import websockets.asyncio.server
import websockets.frames
import websockets.http11
from torch.profiler import ProfilerActivity, profile, record_function

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import PolicyProcessorPipeline

from lerobot.websocket_inference.client import msgpack_numpy

class WebsocketPolicyServer:
    """Serve a policy over websockets.

    The protocol mirrors the older websocket inference server:
    the server sends metadata immediately after connection, then accepts
    msgpack-encoded observations and responds with msgpack-encoded actions.
    """

    def __init__(
        self,
        policy: PreTrainedPolicy,
        preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
        postprocessor: PolicyProcessorPipeline[torch.Tensor, torch.Tensor],
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._stop_event = asyncio.Event()

        self._trace_enable = bool(self._metadata.get("trace_enable", False))
        self._wandb_enable = bool(self._metadata.get("wandb_enable", False))
        self._profiler = None
        self._wandb = None

        if self._wandb_enable:
            self._drop_first_n_frames = int(self._metadata.get("drop_first_n_frames", 0))
            self._infer_cnt = 0
            self._wandb = self._load_wandb()

        if self._trace_enable:
            self._profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_flops=True,
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=5),
                on_trace_ready=lambda prof: prof.export_chrome_trace(
                    f"tmp/trace_schedule_{prof.step_num}.json"
                ),
            )

    @property
    def _default_prompt(self) -> str | None:
        prompt = self._metadata.get("default_prompt")
        return prompt if isinstance(prompt, str) else None

    @property
    def _actions_per_chunk(self) -> int | None:
        value = self._metadata.get("actions_per_chunk")
        return value if isinstance(value, int) and value > 0 else None

    def _load_wandb():
        try:
            return importlib.import_module("wandb")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "wandb logging is enabled in metadata, but wandb is not installed. "
                "Install lerobot with the training extra or disable wandb_enable."
            ) from exc

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ):
            logging.info(
                "WebSocket server started on %s:%s. Waiting for shutdown command.",
                self._host,
                self._port,
            )
            await self._stop_event.wait()
            logging.info("Shutdown event received. Server exiting.")

    async def preprocess_observation(self, observations: dict[str, Any]) -> dict[str, Any]:
        images = observations["images"]
        processed_observations: dict[str, Any] = {}

        for image_key, image in images.items():
            tensor = torch.from_numpy(image)
            if tensor.ndim != 3:
                raise ValueError(f"Expected image {image_key} to have shape (H, W, C), got {tensor.shape}")
            h, w, c = tensor.shape
            if not (c < h and c < w):
                raise ValueError(f"Expected channel-last image for {image_key}, got {tensor.shape}")
            if tensor.dtype != torch.uint8:
                raise ValueError(f"Expected uint8 image for {image_key}, got {tensor.dtype}")

            tensor = tensor.permute(2, 0, 1).contiguous().type(torch.float32) / 255.0
            processed_observations[f"observation.images.{image_key}"] = tensor.unsqueeze(0)

        state = torch.from_numpy(observations["state"]).float()
        if state.ndim == 1:
            state = state.unsqueeze(0)
        processed_observations["observation.state"] = state
        prompt = observations.get("prompt", self._default_prompt)
        if prompt is None:
            raise KeyError("Observation must include 'prompt' or the server must be configured with default_prompt.")
        processed_observations["task"] = prompt

        return self._preprocessor(processed_observations)

    def _predict_action_chunk(self, observation: dict[str, Any]) -> torch.Tensor:
        action_chunk = self._policy.predict_action_chunk(observation)
        if action_chunk.ndim == 2:
            action_chunk = action_chunk.unsqueeze(0)

        actions_per_chunk = self._actions_per_chunk
        if actions_per_chunk is not None:
            action_chunk = action_chunk[:, :actions_per_chunk, :]

        _, chunk_size, _ = action_chunk.shape
        processed_actions = []
        for i in range(chunk_size):
            single_action = action_chunk[:, i, :]
            processed_action = self._postprocessor(single_action)
            if not isinstance(processed_action, torch.Tensor):
                processed_action = torch.as_tensor(processed_action)
            processed_actions.append(processed_action)

        return torch.stack(processed_actions, dim=1)

    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection) -> None:
        logging.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time: float | None = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv(), raw=False)

                if isinstance(obs, dict) and obs.get("command") == "exit":
                    logging.info(
                        "Received exit command from %s. Shutting down server.",
                        websocket.remote_address,
                    )
                    await websocket.send(packer.pack({"status": "Server is shutting down."}))
                    self._stop_event.set()
                    continue

                obs = await self.preprocess_observation(obs)

                infer_start = time.monotonic()

                profiler_context = (
                    record_function("eval_policy") if self._trace_enable else nullcontext()
                )
                with torch.inference_mode(), profiler_context:
                    if self._trace_enable and self._profiler is not None:
                        self._profiler.step()
                    action = self._predict_action_chunk(obs)

                infer_ms = (time.monotonic() - infer_start) * 1000

                if self._wandb_enable and self._wandb is not None:
                    if self._infer_cnt < self._drop_first_n_frames:
                        self._infer_cnt += 1
                    else:
                        self._wandb.log({"infer_cost_ms": infer_ms})

                action = action.squeeze(0)
                response: dict[str, Any] = {"actions": action.cpu().numpy()}
                response["server_timing"] = {"infer_ms": infer_ms}
                if prev_total_time is not None:
                    response["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(response))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logging.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(
    connection: websockets.asyncio.server.ServerConnection,
    request: websockets.http11.Request,
) -> websockets.http11.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
