"""
Websocket inference server/client.

Requires: ``pip install -e ".[websocket_inference]"``

Available modules (import directly)::

    from lerobot.websocket_inference.client.base_policy import BasePolicy
    from lerobot.websocket_inference.client.action_chunk_broker import ActionChunkBroker
    from lerobot.websocket_inference.client.image_tools import resize_with_pad, convert_to_uint8
    from lerobot.websocket_inference.client.msgpack_numpy import ...
    from lerobot.websocket_inference.client.websocket_client_policy import WebsocketClientPolicy
    from lerobot.websocket_inference.client.runtime.agent import Agent
    from lerobot.websocket_inference.client.runtime.environment import Environment
    from lerobot.websocket_inference.client.runtime.subscriber import Subscriber
    from lerobot.websocket_inference.client.runtime.runtime import Runtime
    from lerobot.websocket_inference.client.runtime.agents.policy_agent import PolicyAgent
    from lerobot.websocket_inference.scripts.serve_policy import ...
    from lerobot.websocket_inference.scripts.client_policy import ...
    from lerobot.websocket_inference.robot_client import ...
    from lerobot.websocket_inference.serving.websocket_policy_server import WebsocketPolicyServer
"""
