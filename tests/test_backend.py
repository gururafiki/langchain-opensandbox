"""Unit tests for OpenSandboxSandbox."""

from unittest.mock import MagicMock

import pytest
from deepagents.backends.protocol import (
    ExecuteResponse,
)

# ---------------------------------------------------------------------------
# Helpers — build mock SandboxSync objects
# ---------------------------------------------------------------------------


def _make_execution(stdout_texts=(), stderr_texts=(), error=None, cmd_id="cmd-1"):
    """Build a mock Execution object with logs populated."""
    from opensandbox.models.execd import Execution, ExecutionLogs, OutputMessage

    logs = ExecutionLogs()
    for t in stdout_texts:
        logs.add_stdout(OutputMessage(text=t, timestamp=0))
    for t in stderr_texts:
        logs.add_stderr(OutputMessage(text=t, timestamp=0))

    return Execution(id=cmd_id, result=[], error=error, logs=logs)


def _make_sandbox(
    *,
    sandbox_id="sandbox-abc",
    run_result=None,
    cmd_status_exit_code=0,
    upload_raises=False,
    download_content=b"file-content",
    download_raises=False,
):
    """Build a mock SandboxSync with commands and files services."""
    from opensandbox.models.execd import CommandStatus

    sandbox = MagicMock()
    sandbox.id = sandbox_id

    if run_result is None:
        run_result = _make_execution(stdout_texts=["hello\n"])
    sandbox.commands.run.return_value = run_result

    sandbox.commands.get_command_status.return_value = CommandStatus(
        exit_code=cmd_status_exit_code
    )

    if upload_raises:
        sandbox.files.write_file.side_effect = RuntimeError("permission denied")
    else:
        sandbox.files.write_file.return_value = None

    if download_raises:
        sandbox.files.read_bytes.side_effect = FileNotFoundError("not found")
    else:
        sandbox.files.read_bytes.return_value = download_content

    return sandbox


# ---------------------------------------------------------------------------
# Tests — id property
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOpenSandboxSandboxId:
    def test_id_returns_sandbox_id(self):
        from langchain_opensandbox import OpenSandboxSandbox

        sandbox = _make_sandbox(sandbox_id="test-id-123")
        backend = OpenSandboxSandbox(sandbox=sandbox)
        assert backend.id == "test-id-123"
        backend.close()


# ---------------------------------------------------------------------------
# Tests — execute()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOpenSandboxSandboxExecute:
    def test_execute_combines_stdout(self):
        from langchain_opensandbox import OpenSandboxSandbox

        run_result = _make_execution(stdout_texts=["line1\n", "line2\n"])
        sandbox = _make_sandbox(run_result=run_result, cmd_status_exit_code=0)
        backend = OpenSandboxSandbox(sandbox=sandbox)

        result = backend.execute("echo test")

        assert isinstance(result, ExecuteResponse)
        assert "line1\n" in result.output
        assert "line2\n" in result.output
        assert result.exit_code == 0
        backend.close()

    def test_execute_combines_stderr(self):
        from langchain_opensandbox import OpenSandboxSandbox

        run_result = _make_execution(stderr_texts=["err line\n"])
        sandbox = _make_sandbox(run_result=run_result, cmd_status_exit_code=1)
        backend = OpenSandboxSandbox(sandbox=sandbox)

        result = backend.execute("bad-command")

        assert "err line\n" in result.output
        assert result.exit_code == 1
        backend.close()

    def test_execute_exit_code_from_status(self):
        from langchain_opensandbox import OpenSandboxSandbox

        run_result = _make_execution(stdout_texts=["ok\n"])
        sandbox = _make_sandbox(run_result=run_result, cmd_status_exit_code=42)
        backend = OpenSandboxSandbox(sandbox=sandbox)

        result = backend.execute("something")

        assert result.exit_code == 42
        backend.close()

    def test_execute_exit_code_none_when_status_fails(self):
        from langchain_opensandbox import OpenSandboxSandbox

        run_result = _make_execution(stdout_texts=["out\n"])
        sandbox = _make_sandbox(run_result=run_result)
        sandbox.commands.get_command_status.side_effect = RuntimeError("fail")
        backend = OpenSandboxSandbox(sandbox=sandbox)

        result = backend.execute("cmd")

        assert result.exit_code is None
        backend.close()

    def test_execute_passes_timeout_to_run_command(self):
        from opensandbox.models.execd import RunCommandOpts

        from langchain_opensandbox import OpenSandboxSandbox

        run_result = _make_execution()
        sandbox = _make_sandbox(run_result=run_result)
        backend = OpenSandboxSandbox(sandbox=sandbox)

        backend.execute("sleep 1", timeout=30)

        call_args = sandbox.commands.run.call_args
        opts: RunCommandOpts = call_args.kwargs["opts"]
        assert opts.timeout is not None
        assert opts.timeout.total_seconds() == 30
        backend.close()

    def test_execute_no_timeout_when_not_set(self):
        from langchain_opensandbox import OpenSandboxSandbox

        run_result = _make_execution()
        sandbox = _make_sandbox(run_result=run_result)
        backend = OpenSandboxSandbox(sandbox=sandbox)

        backend.execute("ls")

        call_args = sandbox.commands.run.call_args
        opts = call_args.kwargs["opts"]
        assert opts.timeout is None
        backend.close()

    def test_execute_error_sets_exit_code_1_when_no_status(self):
        from opensandbox.models.execd import ExecutionError

        from langchain_opensandbox import OpenSandboxSandbox

        err = ExecutionError(
            name="NameError", value="name 'x' is not defined", timestamp=0
        )
        run_result = _make_execution(error=err, cmd_id=None)
        sandbox = _make_sandbox(run_result=run_result)
        backend = OpenSandboxSandbox(sandbox=sandbox)

        result = backend.execute("python bad.py")

        assert result.exit_code == 1
        backend.close()


# ---------------------------------------------------------------------------
# Tests — upload_files()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOpenSandboxSandboxUploadFiles:
    def test_upload_success(self):
        from langchain_opensandbox import OpenSandboxSandbox

        sandbox = _make_sandbox()
        backend = OpenSandboxSandbox(sandbox=sandbox)

        results = backend.upload_files(
            [("/tmp/a.py", b"print(1)"), ("/tmp/b.py", b"x=2")]
        )

        assert len(results) == 2
        assert all(r.error is None for r in results)
        assert results[0].path == "/tmp/a.py"
        assert results[1].path == "/tmp/b.py"
        backend.close()

    def test_upload_partial_failure(self):
        from langchain_opensandbox import OpenSandboxSandbox

        sandbox = _make_sandbox(upload_raises=True)
        backend = OpenSandboxSandbox(sandbox=sandbox)

        results = backend.upload_files([("/tmp/file.py", b"code")])

        assert results[0].error == "permission_denied"
        backend.close()


# ---------------------------------------------------------------------------
# Tests — download_files()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOpenSandboxSandboxDownloadFiles:
    def test_download_success(self):
        from langchain_opensandbox import OpenSandboxSandbox

        sandbox = _make_sandbox(download_content=b"result data")
        backend = OpenSandboxSandbox(sandbox=sandbox)

        results = backend.download_files(["/tmp/out.csv"])

        assert len(results) == 1
        assert results[0].content == b"result data"
        assert results[0].error is None
        backend.close()

    def test_download_file_not_found(self):
        from langchain_opensandbox import OpenSandboxSandbox

        sandbox = _make_sandbox(download_raises=True)
        backend = OpenSandboxSandbox(sandbox=sandbox)

        results = backend.download_files(["/tmp/missing.txt"])

        assert results[0].error == "file_not_found"
        assert results[0].content is None
        backend.close()
