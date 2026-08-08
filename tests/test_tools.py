"""Unit tests for the execute_python tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_PATCH_AGET = "langchain_opensandbox.tools.aget_sandbox"


def _make_execution(stdout_texts=(), stderr_texts=(), error=None, cmd_id="cmd-1"):
    """Build a mock async Execution object."""
    from opensandbox.models.execd import Execution, ExecutionLogs, OutputMessage

    logs = ExecutionLogs()
    for t in stdout_texts:
        logs.add_stdout(OutputMessage(text=t, timestamp=0))
    for t in stderr_texts:
        logs.add_stderr(OutputMessage(text=t, timestamp=0))

    return Execution(id=cmd_id, result=[], error=error, logs=logs)


def _make_sandbox(
    *,
    write_raises=False,
    exec_output="42\n",
    exec_exit_code=0,
):
    """Build a mock async Sandbox usable as an async context manager."""
    from opensandbox.models.execd import CommandStatus

    sandbox = MagicMock()
    sandbox.__aenter__ = AsyncMock(return_value=sandbox)
    sandbox.__aexit__ = AsyncMock(return_value=False)

    if write_raises:
        sandbox.files.write_file = AsyncMock(side_effect=PermissionError("denied"))
    else:
        sandbox.files.write_file = AsyncMock()

    run_result = _make_execution(stdout_texts=[exec_output] if exec_output else [])
    cleanup_result = _make_execution()
    sandbox.commands.run = AsyncMock(side_effect=[run_result, cleanup_result])

    sandbox.commands.get_command_status = AsyncMock(
        return_value=CommandStatus(exit_code=exec_exit_code)
    )

    return sandbox


def _make_runtime(store=None):
    """Return a mock ToolRuntime."""
    runtime = MagicMock()
    runtime.config = {"configurable": {}}
    runtime.store = store
    return runtime


async def _invoke(code):
    """Call execute_python's underlying coroutine directly."""
    from langchain_opensandbox.tools import execute_python

    runtime = _make_runtime()
    return await execute_python.coroutine(code, runtime)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestExecutePythonTool:
    def test_is_named_execute_python(self):
        from langchain_opensandbox.tools import execute_python

        assert execute_python.name == "execute_python"

    @pytest.mark.asyncio
    async def test_successful_execution_returns_output(self):
        sandbox = _make_sandbox(exec_output="hello world\n", exec_exit_code=0)

        with patch(_PATCH_AGET, AsyncMock(return_value=sandbox)):
            result = await _invoke("print('hello world')")

        assert result == "hello world\n"
        sandbox.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_write_failure_returns_error_message(self):
        sandbox = _make_sandbox(write_raises=True)

        with patch(_PATCH_AGET, AsyncMock(return_value=sandbox)):
            result = await _invoke("print(1)")

        assert "Failed to write" in result
        sandbox.commands.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_nonzero_exit_code_returns_error_with_output(self):
        err_output = "NameError: name 'x' is not defined\n"
        sandbox = _make_sandbox(exec_output=err_output, exec_exit_code=1)

        with patch(_PATCH_AGET, AsyncMock(return_value=sandbox)):
            result = await _invoke("print(x)")

        assert "Execution failed" in result
        assert "exit 1" in result
        assert "NameError" in result

    @pytest.mark.asyncio
    async def test_no_output_returns_placeholder(self):
        sandbox = _make_sandbox(exec_output="", exec_exit_code=0)

        with patch(_PATCH_AGET, AsyncMock(return_value=sandbox)):
            result = await _invoke("x = 1 + 1")

        assert result == "(no output)"

    @pytest.mark.asyncio
    async def test_cleanup_removes_temp_file(self):
        sandbox = _make_sandbox(exec_output="ok\n", exec_exit_code=0)

        with patch(_PATCH_AGET, AsyncMock(return_value=sandbox)):
            await _invoke("print('ok')")

        assert sandbox.commands.run.call_count == 2
        cleanup_cmd = sandbox.commands.run.call_args_list[1].args[0]
        assert cleanup_cmd.startswith("rm -f /tmp/langchain_opensandbox_")

    @pytest.mark.asyncio
    async def test_sandbox_discovered_via_aget_sandbox(self):
        sandbox = _make_sandbox(exec_output="found\n", exec_exit_code=0)

        with patch(_PATCH_AGET, AsyncMock(return_value=sandbox)) as mock_aget:
            result = await _invoke("print('found')")

        mock_aget.assert_called_once()
        assert result == "found\n"
