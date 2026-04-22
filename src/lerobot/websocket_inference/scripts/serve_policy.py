"""
WebSocket policy server for websocket_inference.

Loads a pretrained policy and serves it over WebSocket for remote inference.

Example command:
```shell
python -m lerobot.websocket_inference.scripts.serve_policy \
  --host=127.0.0.1 \
  --port=8000 \
  --policy.policy-type=smolvla \
  --policy.pretrained-name-or-path=/path/to/model \
  --policy.device=cuda \
  --default-prompt="Put the toy into the box" \
  --actions-per-chunk=10 \
  --rename_map='{
    "observation.images.front_cam": "observation.images.camera1",
    "observation.images.hand_cam": "observation.images.camera2",
    "observation.images.side_cam": "observation.images.camera3"
  }' \
  --wandb.no-enable \
  --trace.no-enable
```
"""

import dataclasses
import importlib
import json
import logging
import socket
from pathlib import Path
from typing import Any

import torch
import tyro

from lerobot.async_inference.constants import SUPPORTED_POLICIES
from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

from lerobot.websocket_inference.serving.websocket_policy_server import WebsocketPolicyServer

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PolicyConfig:
    """Policy loading options.

    The names `policy_type` and `pretrained_name_or_path` align with
    `lerobot.async_inference.policy_server`.
    """

    pretrained_name_or_path: str
    policy_type: str
    device: str | None = None
    local_files_only: bool = False
    strict: bool = False


@dataclasses.dataclass
class WandbConfig:
    """WanDB logging options."""

    enable: bool = False
    drop_first_n_frames: int = 1


@dataclasses.dataclass
class TraceConfig:
    """Torch profiler options."""

    enable: bool = False
    drop_first_n_frames: int = 1


@dataclasses.dataclass
class Args:
    """Arguments for websocket policy serving."""

    host: str = "0.0.0.0"
    port: int = 8000
    seed: int = 1000

    # If provided, will be used when incoming observations omit the "prompt" key.
    default_prompt: str | None = None

    # Similar to async inference: optionally truncate the emitted action chunk.
    actions_per_chunk: int | None = None
    rename_map: str | None = dataclasses.field(
        default=None,
        metadata={"help": "Optional JSON string mapping observation keys to policy feature keys"},
    )

    # Reserved for parity with the old script.
    record: bool = False

    policy: PolicyConfig = dataclasses.field(
        default_factory=lambda: PolicyConfig(pretrained_name_or_path="", policy_type="")
    )
    wandb: WandbConfig = dataclasses.field(default_factory=WandbConfig)
    trace: TraceConfig = dataclasses.field(default_factory=TraceConfig)


def _load_wandb():
    try:
        return importlib.import_module("wandb")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "wandb logging was requested, but wandb is not installed. "
            "Install lerobot with a wandb-providing extra or disable wandb.enable."
        ) from exc


def _get_hardware_platform() -> tuple[str | None, str | None]:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(torch.cuda.current_device()), "cuda"

    try:
        torch_npu = importlib.import_module("torch_npu")
    except ModuleNotFoundError:
        return None, None

    if hasattr(torch_npu, "npu") and torch_npu.npu.is_available():
        return torch_npu.npu.get_device_name(torch_npu.npu.current_device()), "npu"

    return None, None


def _build_metadata(args: Args) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "wandb_enable": args.wandb.enable,
        "drop_first_n_frames": args.wandb.drop_first_n_frames,
        "trace_enable": args.trace.enable,
    }

    if args.default_prompt is not None:
        metadata["default_prompt"] = args.default_prompt

    if args.actions_per_chunk is not None:
        metadata["actions_per_chunk"] = args.actions_per_chunk

    return metadata


def _parse_rename_map(rename_map: str | None) -> dict[str, str]:
    if rename_map is None or not rename_map.strip():
        return {}

    try:
        parsed = json.loads(rename_map)
    except json.JSONDecodeError as exc:
        raise ValueError(f"rename_map must be valid JSON, got: {rename_map!r}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"rename_map must decode to a JSON object, got {type(parsed).__name__}")

    result: dict[str, str] = {}
    for key, value in parsed.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("rename_map JSON object must contain only string-to-string entries")
        result[key] = value
    return result


def _init_wandb(args: Args, policy) -> None:
    wandb = _load_wandb()

    tags = [args.policy.policy_type]
    config = dataclasses.asdict(policy.config)

    train_config_path = Path(args.policy.pretrained_name_or_path) / "train_config.json"
    if train_config_path.is_file():
        with train_config_path.open("r", encoding="utf-8") as f:
            train_config = json.load(f)
        dataset = train_config.get("dataset", {})
        repo_id = dataset.get("repo_id")
        if isinstance(repo_id, str) and repo_id:
            tags.append(repo_id)

    hardware_platform, hardware_tag = _get_hardware_platform()
    if hardware_platform is not None:
        config["hardware_platform"] = hardware_platform
    if hardware_tag is not None:
        tags.append(hardware_tag)

    wandb.init(project="model-inference-monitoring", config=config, tags=tags)
    wandb.define_metric("infer_cost_ms", summary="min,max,mean")


def _load_policy(args: Args):
    policy_class = get_policy_class(args.policy.policy_type)

    config = PreTrainedConfig.from_pretrained(
        args.policy.pretrained_name_or_path,
        local_files_only=args.policy.local_files_only,
    )
    if args.policy.device is not None:
        config.device = args.policy.device

    logger.info(
        "Loading websocket inference policy | policy_type=%s | path=%s | device=%s",
        args.policy.policy_type,
        args.policy.pretrained_name_or_path,
        config.device,
    )

    policy = policy_class.from_pretrained(
        args.policy.pretrained_name_or_path,
        config=config,
        local_files_only=args.policy.local_files_only,
        strict=args.policy.strict,
    )
    policy.eval()
    return policy


def main(args: Args) -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(args.seed)

    if not args.policy.pretrained_name_or_path:
        raise ValueError("policy.pretrained_name_or_path cannot be empty")
    if not args.policy.policy_type:
        raise ValueError("policy.policy_type cannot be empty")

    if args.record:
        logger.warning("record=True is not implemented in serve_policy and will be ignored.")

    if args.actions_per_chunk is not None and args.actions_per_chunk <= 0:
        raise ValueError(f"actions_per_chunk must be positive when provided, got {args.actions_per_chunk}")

    if args.policy.policy_type not in SUPPORTED_POLICIES:
        logger.warning(
            "Policy type %s is not in async_inference SUPPORTED_POLICIES=%s. "
            "Continuing because get_policy_class may still support it.",
            args.policy.policy_type,
            SUPPORTED_POLICIES,
        )

    if args.wandb.enable and args.trace.enable:
        logger.warning("Enabling wandb logging and trace profiling together will skew latency measurements.")

    rename_map = _parse_rename_map(args.rename_map)
    policy = _load_policy(args)
    device_override = {"device": str(policy.config.device)}
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=args.policy.pretrained_name_or_path,
        preprocessor_overrides={
            "device_processor": device_override,
            "rename_observations_processor": {"rename_map": rename_map},
        },
        postprocessor_overrides={"device_processor": device_override},
    )

    if args.wandb.enable:
        _init_wandb(args, policy)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logger.info("Creating server (host: %s, ip: %s, port: %d)", hostname, local_ip, args.port)

    server = WebsocketPolicyServer(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        host=args.host,
        port=args.port,
        metadata=_build_metadata(args),
    )
    server.serve_forever()


if __name__ == "__main__":
    init_logging()
    main(tyro.cli(Args))
