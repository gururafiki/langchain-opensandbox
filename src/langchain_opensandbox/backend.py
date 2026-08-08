"""OpenSandbox sandbox backend for deep agents.

Implements the deepagents ``BaseSandbox`` contract on top of the OpenSandbox
SDK, giving a deep agent an isolated container for shell commands, file
transfer, and code execution.

``BaseSandbox`` derives every filesystem operation (``ls`` / ``read`` /
``write`` / ``edit`` / ``grep`` / ``glob``) from :meth:`execute` and
:meth:`upload_files`, so those two methods plus :meth:`download_files` are the
whole provider surface.

Two classes:

- :class:`OpenSandboxSandbox` wraps a sandbox you already hold.
- :class:`LazyOpenSandboxSandbox` wraps a pair of connect callables, dials
  nothing until the agent actually touches the sandbox, and serves the async
  half of the protocol from the async client instead of a worker thread.
"""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Any

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox


def _combine_output(execution: Any) -> str:
    """Join an execution's stdout and stderr streams into one string."""
    stdout = "".join(m.text for m in execution.logs.stdout)
    stderr = "".join(m.text for m in execution.logs.stderr)
    return stdout + stderr


def _run_opts(timeout: int | None) -> Any:
    """Build ``RunCommandOpts`` for a command with an optional timeout."""
    from opensandbox.models.execd import RunCommandOpts

    return RunCommandOpts(timeout=timedelta(seconds=timeout) if timeout else None)


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

    def _client(self) -> Any:
        """Return the sync sandbox client.

        The single point every sync operation goes through, so a subclass can
        make acquisition lazy without touching the operations themselves.
        """
        return self._sandbox

    @property
    def id(self) -> str:
        """Return the unique sandbox container ID."""
        return str(self._client().id)

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
        client = self._client()
        execution = client.commands.run(command, opts=_run_opts(timeout))
        output = _combine_output(execution)

        exit_code: int | None = None
        if execution.id:
            try:
                status = client.commands.get_command_status(execution.id)
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
        client = self._client()
        results: list[FileUploadResponse] = []
        for path, content in files:
            try:
                client.files.write_file(path, content)
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
        client = self._client()
        results: list[FileDownloadResponse] = []
        for path in paths:
            try:
                content = client.files.read_bytes(path)
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
        if self._sandbox is not None:
            self._sandbox.close()


class LazyOpenSandboxSandbox(OpenSandboxSandbox):
    """A backend that connects on first use and prefers the async client.

    Two problems this solves, both of which only appear once the backend is
    produced by a factory rather than handed in:

    **It connects too eagerly.** deepagents resolves the backend on *every*
    model call — ``FilesystemMiddleware`` needs to know whether to expose the
    ``execute`` tool — so an eager factory pays a sandbox lookup per model call
    even when the model never touches a file. Here the connect callables run on
    the first actual operation, and a backend the agent never uses costs
    nothing.

    **It connects synchronously.** That per-model-call lookup was a blocking
    ``socket.connect`` on an event loop, which ASGI servers rightly complain
    about (``langgraph dev`` refuses it without ``--allow-blocking``). The
    async half of the protocol — :meth:`aexecute`, :meth:`aupload_files`,
    :meth:`adownload_files` — now resolves through *aconnect* and awaits the
    async client. ``BaseSandbox`` derives ``als`` / ``aread`` / ``awrite`` /
    ``aedit`` / ``agrep`` / ``aglob`` from those three, so the whole async
    surface stops touching a worker thread.

    The two clients are independent and resolved on demand: a purely-async
    caller never opens a sync one. Both find the same container, because both
    connect callables discover it the same way.

    Args:
        connect: Zero-arg callable returning a connected sync sandbox.
        aconnect: Zero-arg async callable returning a connected async sandbox.
    """

    def __init__(
        self,
        *,
        connect: Callable[[], Any],
        aconnect: Callable[[], Awaitable[Any]],
    ) -> None:
        """Initialize with the two connect callables. Neither is called yet."""
        super().__init__(sandbox=None)
        self._connect = connect
        self._aconnect = aconnect
        self._async_sandbox: Any = None
        self._alock = asyncio.Lock()

    def _client(self) -> Any:
        """Return the sync client, connecting on first call."""
        if self._sandbox is None:
            self._sandbox = self._connect()
        return self._sandbox

    async def _aclient(self) -> Any:
        """Return the async client, connecting on first call."""
        if self._async_sandbox is None:
            async with self._alock:
                # Re-check: a concurrent caller may have connected while this
                # one waited for the lock.
                if self._async_sandbox is None:
                    self._async_sandbox = await self._aconnect()
        return self._async_sandbox

    # ------------------------------------------------------------------
    # Async protocol — served by the async client, not asyncio.to_thread
    # ------------------------------------------------------------------

    async def aexecute(
        self,
        command: str,
        *,
        timeout: int | None = None,  # noqa: ASYNC109
    ) -> ExecuteResponse:
        """Execute a shell command in the sandbox (async)."""
        client = await self._aclient()
        execution = await client.commands.run(command, opts=_run_opts(timeout))
        output = _combine_output(execution)

        exit_code: int | None = None
        if execution.id:
            try:
                status = await client.commands.get_command_status(execution.id)
                exit_code = status.exit_code
            except Exception:
                pass  # Non-fatal; exit_code stays None

        if execution.error and exit_code is None:
            exit_code = 1

        return ExecuteResponse(output=output, exit_code=exit_code)

    async def aupload_files(
        self,
        files: list[tuple[str, bytes]],
    ) -> list[FileUploadResponse]:
        """Upload files to the sandbox filesystem (async)."""
        client = await self._aclient()
        results: list[FileUploadResponse] = []
        for path, content in files:
            try:
                await client.files.write_file(path, content)
                results.append(FileUploadResponse(path=path))
            except Exception:
                results.append(FileUploadResponse(path=path, error="permission_denied"))
        return results

    async def adownload_files(
        self,
        paths: list[str],
    ) -> list[FileDownloadResponse]:
        """Download files from the sandbox filesystem (async)."""
        client = await self._aclient()
        results: list[FileDownloadResponse] = []
        for path in paths:
            try:
                content = await client.files.read_bytes(path)
                results.append(FileDownloadResponse(path=path, content=content))
            except Exception:
                results.append(FileDownloadResponse(path=path, error="file_not_found"))
        return results

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def aclose(self) -> None:
        """Release whichever clients were actually opened.

        Does not terminate the remote container. Safe to call when nothing was
        ever connected.
        """
        if self._async_sandbox is not None:
            await self._async_sandbox.close()
            self._async_sandbox = None
        self.close()
        self._sandbox = None
