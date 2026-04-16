# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Bash Env Environment.

This module creates an HTTP server that exposes the BashEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.
"""

import os

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from exc

try:
    from ..models import BashAction, BashObservation
    from .bash_env_environment import BashEnvironment
except ImportError:
    from models import BashAction, BashObservation
    from server.bash_env_environment import BashEnvironment


max_concurrent = int(os.getenv("MAX_CONCURRENT_ENVS", "8"))

app = create_app(
    BashEnvironment,
    BashAction,
    BashObservation,
    env_name="bash_env",
    max_concurrent_envs=max_concurrent,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
