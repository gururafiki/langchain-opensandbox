"""Thread-scoped sandbox discovery and creation for LangGraph agents.

This module is the LangGraph-specific half of the package. Import it only if
you want sandboxes managed for you; :mod:`langchain_opensandbox.backend` alone
has no LangGraph coupling and works with any sandbox you create yourself.

The pattern: a sandbox is tagged with the current ``thread_id`` at creation, so
every call in the same conversation reuses the same container and parallel
conversations stay isolated. Nothing is created at import time or at graph
build time — the lookup happens on first use.

``get_backend`` is a deepagents ``BackendFactory``, so it can be handed
straight to ``create_deep_agent(backend=...)``.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from langgraph.config import get_config
from langgraph.prebuilt import ToolRuntime
from langgraph.runtime import Runtime

from .backend import OpenSandboxSandbox
from .config import OpenSandboxSettings


class SandboxFactory:
    """Find or create an OpenSandbox container for the current thread.

    Discovers sandboxes by ``thread_id`` metadata via the OpenSandbox API.
    When no running sandbox is found, creates a new one tagged with the
    current ``thread_id``.
    """

    def __init__(self, runtime: Runtime | ToolRuntime) -> None:
        """Initialize with a LangGraph runtime context."""
        self._runtime = runtime
        self._config = self._get_settings()

    def _runnable_config(self) -> Any:
        """Return the runnable config from the runtime or the langgraph context."""
        cfg = getattr(self._runtime, "config", None)
        return cfg if isinstance(cfg, dict) else get_config()

    def _get_settings(self) -> OpenSandboxSettings:
        """Build settings from the runtime config or the langgraph context."""
        return OpenSandboxSettings.from_runnable_config(self._runnable_config())

    def _get_thread_id(self) -> str:
        """Extract ``thread_id`` from the runtime config or langgraph context."""
        cfg = self._runnable_config()
        return (cfg.get("configurable") or {}).get("thread_id") or "default"

    def _make_sync_connection(self) -> Any:
        """Build a ``ConnectionConfigSync`` from the resolved settings."""
        from opensandbox.config.connection_sync import ConnectionConfigSync

        return ConnectionConfigSync(
            domain=self._config.opensandbox_url,
            api_key=self._config.opensandbox_api_key,
            protocol="http",
            use_server_proxy=self._config.opensandbox_use_server_proxy,
        )

    def _make_async_connection(self) -> Any:
        """Build an async ``ConnectionConfig`` from the resolved settings."""
        from opensandbox.config.connection import ConnectionConfig

        return ConnectionConfig(
            domain=self._config.opensandbox_url,
            api_key=self._config.opensandbox_api_key,
            protocol="http",
            use_server_proxy=self._config.opensandbox_use_server_proxy,
        )

    def _find_sandbox_id(self) -> str | None:
        """Find a running sandbox tagged with the current thread_id."""
        from opensandbox.models.sandboxes import SandboxFilter
        from opensandbox.sync.manager import SandboxManagerSync

        manager = SandboxManagerSync.create(
            connection_config=self._make_sync_connection(),
        )
        try:
            result = manager.list_sandbox_infos(
                SandboxFilter(
                    states=["Running"],
                    metadata={"thread_id": self._get_thread_id()},
                ),
            )
            return result.sandbox_infos[0].id if result.sandbox_infos else None
        finally:
            manager.close()

    async def _afind_sandbox_id(self) -> str | None:
        """Find a running sandbox tagged with the current thread_id (async)."""
        from opensandbox.manager import SandboxManager
        from opensandbox.models.sandboxes import SandboxFilter

        manager = await SandboxManager.create(
            connection_config=self._make_async_connection(),
        )
        try:
            result = await manager.list_sandbox_infos(
                SandboxFilter(
                    states=["Running"],
                    metadata={"thread_id": self._get_thread_id()},
                ),
            )
            return result.sandbox_infos[0].id if result.sandbox_infos else None
        finally:
            await manager.close()

    def get_sandbox(self) -> Any:
        """Find or create a sandbox (sync)."""
        from opensandbox.sync.sandbox import SandboxSync

        sandbox_id = self._find_sandbox_id()
        if sandbox_id:
            return SandboxSync.connect(
                sandbox_id,
                connection_config=self._make_sync_connection(),
                skip_health_check=False,
            )
        return SandboxSync.create(
            self._config.opensandbox_image,
            connection_config=self._make_sync_connection(),
            timeout=timedelta(hours=1),
            env={"PYTHONUNBUFFERED": "1"},
            metadata={"thread_id": self._get_thread_id()},
        )

    async def aget_sandbox(self) -> Any:
        """Find or create a sandbox (async)."""
        from opensandbox.sandbox import Sandbox

        sandbox_id = await self._afind_sandbox_id()
        if sandbox_id:
            return await Sandbox.connect(
                sandbox_id,
                connection_config=self._make_async_connection(),
                skip_health_check=False,
            )
        return await Sandbox.create(
            self._config.opensandbox_image,
            connection_config=self._make_async_connection(),
            timeout=timedelta(hours=1),
            env={"PYTHONUNBUFFERED": "1"},
            metadata={"thread_id": self._get_thread_id()},
        )


def get_backend(runtime: Runtime | ToolRuntime) -> OpenSandboxSandbox:
    """Find or create the thread's sandbox and wrap it as a backend.

    Implements the deepagents ``BackendFactory`` protocol — pass directly as
    ``backend=get_backend`` to ``create_deep_agent``.
    """
    return OpenSandboxSandbox(sandbox=SandboxFactory(runtime).get_sandbox())


def get_sandbox(runtime: Runtime | ToolRuntime) -> Any:
    """Find or create a raw sandbox for the current thread (sync)."""
    return SandboxFactory(runtime).get_sandbox()


async def aget_sandbox(runtime: Runtime | ToolRuntime) -> Any:
    """Find or create a raw sandbox for the current thread (async)."""
    return await SandboxFactory(runtime).aget_sandbox()
