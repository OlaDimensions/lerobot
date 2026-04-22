"""WebSocket client policy for communicating with a WebsocketPolicyServer.

Adapted from openpi-client's websocket_client_policy.py for use within lerobot.
See ``lerobot.websocket_inference.serving.websocket_policy_server`` for the
corresponding server implementation.
"""

import logging
import time
from typing import Any

import websockets.sync.client

from lerobot.websocket_inference.client import msgpack_numpy

logger = logging.getLogger(__name__)


class WebsocketClientPolicy:
    """Implements a policy interface by communicating with a server over WebSocket.

    On construction the client connects to the server (retrying indefinitely
    until the server is reachable), receives server metadata, and is then
    ready to call :meth:`infer`.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int | None = None,
    ) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> dict[str, Any]:
        return self._server_metadata

    def _wait_for_server(
        self,
    ) -> tuple[websockets.sync.client.ClientConnection, dict[str, Any]]:
        logger.info("Waiting for server at %s...", self._uri)
        while True:
            try:
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logger.info("Still waiting for server...")
                time.sleep(5)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Send an observation to the server and return the response (e.g. action chunk)."""
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def reset(self) -> None:
        """Reset the policy to its initial state (no-op by default)."""

    def close(self) -> None:
        """Close the WebSocket connection."""
        if self._ws is not None:
            self._ws.close()
            logger.info("WebSocket connection closed")
