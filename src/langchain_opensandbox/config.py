"""Connection settings for an OpenSandbox server.

Values resolve from the process environment first, then from a LangGraph
``RunnableConfig["configurable"]`` mapping, then from the field default.

This module deliberately does **not** call ``dotenv.load_dotenv()``. Loading a
``.env`` file is an application decision, not a library one — do it in your own
entrypoint if you want it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Self

from pydantic import BaseModel, Field


class OpenSandboxSettings(BaseModel):
    """Connection settings for an OpenSandbox server."""

    opensandbox_url: str = Field(
        default="localhost:8080",
        description=(
            "OpenSandbox server address (host:port). "
            "Used as ConnectionConfig.domain when creating sandbox containers."
        ),
    )

    opensandbox_api_key: str | None = Field(
        default=None,
        description="OpenSandbox server API key (leave empty if server has no auth).",
    )

    opensandbox_image: str = Field(
        default="python:3.11-slim",
        description=(
            "Docker image for sandbox containers. "
            "Override with a custom image that has pandas, numpy, ta-lib, etc. "
            "pre-installed for faster startup."
        ),
    )

    opensandbox_use_server_proxy: bool = Field(
        default=True,
        description=(
            "Route sandbox traffic through the OpenSandbox server instead of "
            "connecting to sandbox containers directly. Required whenever the agent "
            "cannot reach sandbox container ports directly — e.g. Docker Swarm "
            "overlay / bridge deployments where the server spawns sandboxes on the "
            "host via docker.sock (the server proxies via the Docker API). Set "
            "OPENSANDBOX_USE_SERVER_PROXY=false only for host/flat-network setups "
            "where direct access is faster and reachable."
        ),
    )

    @classmethod
    def from_runnable_config(cls, config: Mapping[str, Any] | None) -> Self:
        """Build settings from env vars and a LangGraph ``RunnableConfig``.

        For each field the first value found wins: the upper-cased env var
        (``OPENSANDBOX_URL``, ``OPENSANDBOX_API_KEY``, ``OPENSANDBOX_IMAGE``,
        ``OPENSANDBOX_USE_SERVER_PROXY``), then the same lower-cased key in
        ``config["configurable"]``, then the field default. Unknown
        ``configurable`` keys are ignored.

        Args:
            config: A LangGraph ``RunnableConfig`` (or any mapping carrying a
                ``configurable`` key). ``None`` resolves from the environment
                only.

        Returns:
            A populated settings instance.
        """
        configurable: Mapping[str, Any] = (config or {}).get("configurable") or {}

        values: dict[str, Any] = {}
        for name in cls.model_fields:
            raw = os.environ.get(name.upper(), configurable.get(name))
            if raw is not None:
                values[name] = raw

        return cls(**values)
