"""Unit tests for SandboxFactory and public factory functions."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_PATCH_PREFIX = "langchain_opensandbox.factory"


def _make_mock_config():
    """Build a mock OpenSandboxSettings."""
    config = MagicMock()
    config.opensandbox_url = "localhost:8080"
    config.opensandbox_api_key = None
    config.opensandbox_image = "python:3.11-slim"
    config.opensandbox_use_server_proxy = True
    return config


def _make_paged_result(sandbox_infos=None):
    """Build a mock PagedSandboxInfos."""
    result = MagicMock()
    result.sandbox_infos = sandbox_infos or []
    return result


def _make_sandbox_info(sandbox_id="sb-123"):
    """Build a mock SandboxInfo."""
    info = MagicMock()
    info.id = sandbox_id
    return info


# ---------------------------------------------------------------------------
# Tests — SandboxFactory._get_thread_id
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSandboxFactoryThreadId:
    def test_from_runtime_config(self):
        from langchain_opensandbox.factory import SandboxFactory

        runtime = MagicMock()
        runtime.config = {"configurable": {"thread_id": "thread-abc"}}

        factory = SandboxFactory(runtime)
        assert factory._get_thread_id() == "thread-abc"

    def test_falls_back_to_get_config(self):
        from langchain_opensandbox.factory import SandboxFactory

        # Runtime without config attr (like Runtime from langgraph.runtime)
        runtime = MagicMock(spec=[])

        with patch(
            f"{_PATCH_PREFIX}.get_config",
            return_value={"configurable": {"thread_id": "thread-xyz"}},
        ):
            factory = SandboxFactory(runtime)
            assert factory._get_thread_id() == "thread-xyz"

    def test_defaults_to_default(self):
        from langchain_opensandbox.factory import SandboxFactory

        runtime = MagicMock()
        runtime.config = {"configurable": {}}

        factory = SandboxFactory(runtime)
        assert factory._get_thread_id() == "default"


# ---------------------------------------------------------------------------
# Tests — connection config carries use_server_proxy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSandboxFactoryConnection:
    """The sync/async ConnectionConfig must forward opensandbox_use_server_proxy.

    Required for Swarm/bridge deployments where the agent cannot reach sandbox
    container ports directly and must proxy through the server (the default).
    """

    @pytest.mark.parametrize("proxy", [True, False])
    def test_sync_connection_forwards_proxy(self, proxy):
        from langchain_opensandbox.factory import SandboxFactory

        runtime = MagicMock()
        runtime.config = {"configurable": {"thread_id": "t"}}
        cfg = _make_mock_config()
        cfg.opensandbox_use_server_proxy = proxy
        with patch(
            f"{_PATCH_PREFIX}.OpenSandboxSettings.from_runnable_config",
            return_value=cfg,
        ):
            conn = SandboxFactory(runtime)._make_sync_connection()
        assert conn.use_server_proxy is proxy

    @pytest.mark.parametrize("proxy", [True, False])
    def test_async_connection_forwards_proxy(self, proxy):
        from langchain_opensandbox.factory import SandboxFactory

        runtime = MagicMock()
        runtime.config = {"configurable": {"thread_id": "t"}}
        cfg = _make_mock_config()
        cfg.opensandbox_use_server_proxy = proxy
        with patch(
            f"{_PATCH_PREFIX}.OpenSandboxSettings.from_runnable_config",
            return_value=cfg,
        ):
            conn = SandboxFactory(runtime)._make_async_connection()
        assert conn.use_server_proxy is proxy


# ---------------------------------------------------------------------------
# Tests — SandboxFactory.get_sandbox (sync)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSandboxFactoryGetSandbox:
    def test_connects_when_found(self):
        from langchain_opensandbox.factory import SandboxFactory

        runtime = MagicMock()
        runtime.config = {"configurable": {"thread_id": "thread-abc"}}

        mock_sandbox = MagicMock()
        mock_sandbox.id = "found-sb"

        mock_manager = MagicMock()
        mock_manager.list_sandbox_infos.return_value = _make_paged_result(
            [_make_sandbox_info("found-sb")],
        )

        with (
            patch(
                f"{_PATCH_PREFIX}.OpenSandboxSettings.from_runnable_config",
                return_value=_make_mock_config(),
            ),
            patch(
                "opensandbox.sync.manager.SandboxManagerSync.create",
                return_value=mock_manager,
            ),
            patch(
                "opensandbox.sync.sandbox.SandboxSync.connect",
                return_value=mock_sandbox,
            ) as mock_connect,
        ):
            factory = SandboxFactory(runtime)
            result = factory.get_sandbox()

        mock_connect.assert_called_once()
        assert mock_connect.call_args.args[0] == "found-sb"
        assert result.id == "found-sb"

    def test_creates_when_not_found(self):
        from langchain_opensandbox.factory import SandboxFactory

        runtime = MagicMock()
        runtime.config = {"configurable": {"thread_id": "thread-abc"}}

        mock_sandbox = MagicMock()
        mock_sandbox.id = "new-sb"

        mock_manager = MagicMock()
        mock_manager.list_sandbox_infos.return_value = _make_paged_result([])

        with (
            patch(
                f"{_PATCH_PREFIX}.OpenSandboxSettings.from_runnable_config",
                return_value=_make_mock_config(),
            ),
            patch(
                "opensandbox.sync.manager.SandboxManagerSync.create",
                return_value=mock_manager,
            ),
            patch(
                "opensandbox.sync.sandbox.SandboxSync.create",
                return_value=mock_sandbox,
            ) as mock_create,
        ):
            factory = SandboxFactory(runtime)
            result = factory.get_sandbox()

        mock_create.assert_called_once()
        assert result.id == "new-sb"
        # Verify thread_id metadata is passed
        assert mock_create.call_args.kwargs["metadata"] == {"thread_id": "thread-abc"}


# ---------------------------------------------------------------------------
# Tests — SandboxFactory.aget_sandbox (async)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSandboxFactoryAgetSandbox:
    @pytest.mark.asyncio
    async def test_connects_when_found(self):
        from langchain_opensandbox.factory import SandboxFactory

        runtime = MagicMock()
        runtime.config = {"configurable": {"thread_id": "thread-abc"}}

        mock_sandbox = MagicMock()
        mock_sandbox.id = "found-sb"

        mock_manager = MagicMock()
        mock_manager.list_sandbox_infos = AsyncMock(
            return_value=_make_paged_result([_make_sandbox_info("found-sb")]),
        )
        mock_manager.close = AsyncMock()

        with (
            patch(
                f"{_PATCH_PREFIX}.OpenSandboxSettings.from_runnable_config",
                return_value=_make_mock_config(),
            ),
            patch(
                "opensandbox.manager.SandboxManager.create",
                AsyncMock(return_value=mock_manager),
            ),
            patch(
                "opensandbox.sandbox.Sandbox.connect",
                AsyncMock(return_value=mock_sandbox),
            ) as mock_connect,
        ):
            factory = SandboxFactory(runtime)
            result = await factory.aget_sandbox()

        mock_connect.assert_called_once()
        assert mock_connect.call_args.args[0] == "found-sb"
        assert result.id == "found-sb"

    @pytest.mark.asyncio
    async def test_creates_when_not_found(self):
        from langchain_opensandbox.factory import SandboxFactory

        runtime = MagicMock()
        runtime.config = {"configurable": {"thread_id": "thread-abc"}}

        mock_sandbox = MagicMock()
        mock_sandbox.id = "new-sb"

        mock_manager = MagicMock()
        mock_manager.list_sandbox_infos = AsyncMock(
            return_value=_make_paged_result([]),
        )
        mock_manager.close = AsyncMock()

        with (
            patch(
                f"{_PATCH_PREFIX}.OpenSandboxSettings.from_runnable_config",
                return_value=_make_mock_config(),
            ),
            patch(
                "opensandbox.manager.SandboxManager.create",
                AsyncMock(return_value=mock_manager),
            ),
            patch(
                "opensandbox.sandbox.Sandbox.create",
                AsyncMock(return_value=mock_sandbox),
            ) as mock_create,
        ):
            factory = SandboxFactory(runtime)
            result = await factory.aget_sandbox()

        mock_create.assert_called_once()
        assert result.id == "new-sb"
        assert mock_create.call_args.kwargs["metadata"] == {"thread_id": "thread-abc"}


# ---------------------------------------------------------------------------
# Tests — get_backend() public function
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGetBackend:
    def test_returns_backend_wrapping_sandbox(self):
        from langchain_opensandbox.factory import get_backend

        runtime = MagicMock()
        runtime.config = {"configurable": {"thread_id": "thread-abc"}}

        mock_sandbox = MagicMock()
        mock_sandbox.id = "sb-456"

        with patch(
            f"{_PATCH_PREFIX}.SandboxFactory.get_sandbox",
            return_value=mock_sandbox,
        ):
            backend = get_backend(runtime)

        assert backend.id == "sb-456"

    def test_works_with_runtime_no_config(self):
        """get_backend works when runtime has no config attribute (Runtime)."""
        from langchain_opensandbox.factory import get_backend

        mock_sandbox = MagicMock()
        mock_sandbox.id = "sb-789"

        # Simulate Runtime object — no 'config' attribute
        runtime = MagicMock(spec=[])

        with (
            patch(
                f"{_PATCH_PREFIX}.get_config",
                return_value={"configurable": {}},
            ),
            patch(
                f"{_PATCH_PREFIX}.SandboxFactory.get_sandbox",
                return_value=mock_sandbox,
            ),
        ):
            backend = get_backend(runtime)

        assert backend.id == "sb-789"
