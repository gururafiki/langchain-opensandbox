"""Unit tests for LazyOpenSandboxSandbox."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_opensandbox import LazyOpenSandboxSandbox


def _make_execution(stdout_texts=(), stderr_texts=(), error=None, cmd_id="cmd-1"):
    from opensandbox.models.execd import Execution, ExecutionLogs, OutputMessage

    logs = ExecutionLogs()
    for t in stdout_texts:
        logs.add_stdout(OutputMessage(text=t, timestamp=0))
    for t in stderr_texts:
        logs.add_stderr(OutputMessage(text=t, timestamp=0))

    return Execution(id=cmd_id, result=[], error=error, logs=logs)


def _make_async_client(*, exit_code=0, stdout="hi\n", read_bytes=b"data"):
    from opensandbox.models.execd import CommandStatus

    client = MagicMock()
    client.id = "async-sb"
    client.commands.run = AsyncMock(return_value=_make_execution([stdout]))
    client.commands.get_command_status = AsyncMock(
        return_value=CommandStatus(exit_code=exit_code)
    )
    client.files.write_file = AsyncMock()
    client.files.read_bytes = AsyncMock(return_value=read_bytes)
    client.close = AsyncMock()
    return client


def _make_sync_client(*, exit_code=0, stdout="sync\n"):
    from opensandbox.models.execd import CommandStatus

    client = MagicMock()
    client.id = "sync-sb"
    client.commands.run.return_value = _make_execution([stdout])
    client.commands.get_command_status.return_value = CommandStatus(exit_code=exit_code)
    return client


def _make_backend(sync_client=None, async_client=None):
    connect = MagicMock(return_value=sync_client or _make_sync_client())
    aconnect = AsyncMock(return_value=async_client or _make_async_client())
    return LazyOpenSandboxSandbox(connect=connect, aconnect=aconnect), connect, aconnect


@pytest.mark.unit
class TestLaziness:
    def test_construction_connects_nothing(self):
        # The whole point: deepagents resolves the backend on every model call.
        _, connect, aconnect = _make_backend()
        connect.assert_not_called()
        aconnect.assert_not_called()

    def test_sync_use_connects_only_the_sync_client(self):
        backend, connect, aconnect = _make_backend()

        backend.execute("echo hi")

        connect.assert_called_once()
        aconnect.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_use_connects_only_the_async_client(self):
        backend, connect, aconnect = _make_backend()

        await backend.aexecute("echo hi")

        aconnect.assert_awaited_once()
        connect.assert_not_called()

    def test_sync_client_is_reused(self):
        backend, connect, _ = _make_backend()

        backend.execute("one")
        backend.execute("two")

        connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_client_is_reused(self):
        backend, _, aconnect = _make_backend()

        await backend.aexecute("one")
        await backend.aexecute("two")

        aconnect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_concurrent_first_use_connects_once(self):
        backend, _, aconnect = _make_backend()

        await asyncio.gather(*(backend.aexecute("cmd") for _ in range(5)))

        aconnect.assert_awaited_once()


@pytest.mark.unit
class TestAsyncOperations:
    """The async half must go to the async client, never asyncio.to_thread.

    If these ever start passing while the sync client is the one being used,
    the blocking-call problem is back.
    """

    @pytest.mark.asyncio
    async def test_aexecute_returns_output_and_exit_code(self):
        client = _make_async_client(stdout="hello\n", exit_code=7)
        backend, connect, _ = _make_backend(async_client=client)

        result = await backend.aexecute("echo hello")

        assert result.output == "hello\n"
        assert result.exit_code == 7
        connect.assert_not_called()

    @pytest.mark.asyncio
    async def test_aexecute_forwards_timeout(self):
        client = _make_async_client()
        backend, _, _ = _make_backend(async_client=client)

        await backend.aexecute("sleep 1", timeout=30)

        opts = client.commands.run.call_args.kwargs["opts"]
        assert opts.timeout.total_seconds() == 30

    @pytest.mark.asyncio
    async def test_aexecute_error_without_status_is_exit_1(self):
        from opensandbox.models.execd import ExecutionError

        client = _make_async_client()
        client.commands.run = AsyncMock(
            return_value=_make_execution(
                error=ExecutionError(name="NameError", value="boom", timestamp=0),
                cmd_id=None,
            )
        )
        backend, _, _ = _make_backend(async_client=client)

        assert (await backend.aexecute("python bad.py")).exit_code == 1

    @pytest.mark.asyncio
    async def test_aupload_files_reports_per_file_status(self):
        client = _make_async_client()
        client.files.write_file = AsyncMock(
            side_effect=[None, PermissionError("denied")]
        )
        backend, _, _ = _make_backend(async_client=client)

        results = await backend.aupload_files([("/a.py", b"x"), ("/b.py", b"y")])

        assert results[0].error is None
        assert results[1].error == "permission_denied"

    @pytest.mark.asyncio
    async def test_adownload_files_reports_per_file_status(self):
        client = _make_async_client(read_bytes=b"payload")
        backend, _, _ = _make_backend(async_client=client)

        results = await backend.adownload_files(["/out.csv"])

        assert results[0].content == b"payload"
        assert results[0].error is None

    @pytest.mark.asyncio
    async def test_adownload_missing_file(self):
        client = _make_async_client()
        client.files.read_bytes = AsyncMock(side_effect=FileNotFoundError("nope"))
        backend, _, _ = _make_backend(async_client=client)

        results = await backend.adownload_files(["/missing.txt"])

        assert results[0].error == "file_not_found"
        assert results[0].content is None

    @pytest.mark.asyncio
    async def test_derived_async_file_ops_use_aexecute(self):
        # BaseSandbox derives als/aread/awrite/aedit/agrep/aglob from aexecute,
        # so overriding aexecute is what takes the whole async surface off the
        # worker thread. Assert the derivation actually holds.
        client = _make_async_client(stdout="")
        backend, connect, _ = _make_backend(async_client=client)

        await backend.als("/")

        client.commands.run.assert_awaited()
        connect.assert_not_called()


@pytest.mark.unit
class TestLifecycle:
    @pytest.mark.asyncio
    async def test_aclose_closes_only_what_was_opened(self):
        async_client = _make_async_client()
        sync_client = _make_sync_client()
        backend, _, _ = _make_backend(sync_client, async_client)

        await backend.aexecute("cmd")
        await backend.aclose()

        async_client.close.assert_awaited_once()
        sync_client.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_aclose_is_safe_when_nothing_connected(self):
        backend, connect, aconnect = _make_backend()

        await backend.aclose()

        connect.assert_not_called()
        aconnect.assert_not_called()

    def test_close_is_safe_when_nothing_connected(self):
        backend, connect, _ = _make_backend()

        backend.close()

        connect.assert_not_called()
