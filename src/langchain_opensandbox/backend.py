"""OpenSandbox sandbox backend for deep agents.

Implements the deepagents ``BaseSandbox`` contract on top of the OpenSandbox
sync SDK, giving a deep agent an isolated container for shell commands, file
transfer, and code execution.

``BaseSandbox`` derives every filesystem operation (``ls`` / ``read`` /
``write`` / ``edit`` / ``grep`` / ``glob``) from :meth:`execute` and
:meth:`upload_files`, so those two methods plus :meth:`download_files` are the
whole provider surface.
"""

from datetime import timedelta
from typing import Any

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox


class OpenSandboxSandbox(BaseSandbox):
    """Deep-agent backend backed by an OpenSandbox container.

    Wraps a connected ``SandboxSync`` instance. All operations are synchronous
    and safe to call from a thread — ``BaseSandbox`` dispatches every ``a*``
    method through :meth:`aexecute`, which runs :meth:`execute` via
    ``asyncio.to_thread``, so the event loop is never blocked.

    Example::

        from opensandbox.sync.sandbox import SandboxSync
        from langchain_opensandbox import OpenSandboxSandbox

        sandbox = SandboxSync.create("python:3.11-slim")
        agent = create_deep_agent(
            model=model,
            backend=OpenSandboxSandbox(sandbox=sandbox),
        )
    """

    def __init__(self, *, sandbox: Any) -> None:
        """Initialize with a connected sandbox instance.

        Args:
            sandbox: Connected and ready ``SandboxSync`` instance.
        """
        self._sandbox = sandbox

    @property
    def id(self) -> str:
        """Return the unique sandbox container ID."""
        return str(self._sandbox.id)

    # ------------------------------------------------------------------
    # BaseSandbox abstract methods
    # ------------------------------------------------------------------

    def execute(
        self,
        command: str,
        *,
        timeout: int | None = None,
    ) -> ExecuteResponse:
        """Execute a shell command in the sandbox and return the result.

        Args:
            command: Full shell command string to execute.
            timeout: Maximum seconds to wait for completion.

        Returns:
            ExecuteResponse with combined output, exit code, and truncation flag.
        """
        from opensandbox.models.execd import RunCommandOpts

        opts = RunCommandOpts(
            timeout=timedelta(seconds=timeout) if timeout else None,
        )
        execution = self._sandbox.commands.run(command, opts=opts)

        stdout = "".join(m.text for m in execution.logs.stdout)
        stderr = "".join(m.text for m in execution.logs.stderr)
        output = stdout + stderr

        exit_code: int | None = None
        if execution.id:
            try:
                status = self._sandbox.commands.get_command_status(execution.id)
                exit_code = status.exit_code
            except Exception:
                pass  # Non-fatal; exit_code stays None

        if execution.error and exit_code is None:
            exit_code = 1

        return ExecuteResponse(output=output, exit_code=exit_code)

    def upload_files(
        self,
        files: list[tuple[str, bytes]],
    ) -> list[FileUploadResponse]:
        """Upload files to the sandbox filesystem.

        Args:
            files: List of (remote_path, content_bytes) tuples.

        Returns:
            List of FileUploadResponse with per-file success/error status.
        """
        results: list[FileUploadResponse] = []
        for path, content in files:
            try:
                self._sandbox.files.write_file(path, content)
                results.append(FileUploadResponse(path=path))
            except Exception:
                results.append(FileUploadResponse(path=path, error="permission_denied"))
        return results

    def download_files(
        self,
        paths: list[str],
    ) -> list[FileDownloadResponse]:
        """Download files from the sandbox filesystem.

        Args:
            paths: List of remote file paths to download.

        Returns:
            List of FileDownloadResponse with per-file content or error.
        """
        results: list[FileDownloadResponse] = []
        for path in paths:
            try:
                content = self._sandbox.files.read_bytes(path)
                results.append(FileDownloadResponse(path=path, content=content))
            except Exception:
                results.append(FileDownloadResponse(path=path, error="file_not_found"))
        return results

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release local HTTP resources for this sandbox.

        Does not terminate the remote container — call ``sandbox.kill()`` for
        that.
        """
        self._sandbox.close()
