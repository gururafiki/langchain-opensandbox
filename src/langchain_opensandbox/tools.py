"""A LangChain tool that runs Python inside the thread's sandbox.

Deep agents built on a sandbox backend already get deepagents' built-in
``execute`` shell tool. This module is for the other case: a plain ReAct agent
that needs code execution without adopting the whole deep-agent filesystem.

The sandbox is discovered by ``thread_id`` metadata (see
:mod:`langchain_opensandbox.factory`); if none is running for the current
thread, one is created.
"""

from __future__ import annotations

import uuid

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from .factory import aget_sandbox


@tool
async def execute_python(code: str, runtime: ToolRuntime) -> str:
    """Execute Python code in a secure isolated sandbox.

    Use this tool for work that is better done by running code than by
    reasoning in prose — numerical computation, dataframe manipulation,
    statistical analysis, parsing or reshaping structured data, and simulation.

    The sandbox has Python 3 with the standard library available.
    Use print() to produce output — return values alone are not captured.

    Examples::

        code = '''
        values = [100, 110, 121, 133, 146]
        growth = [(b - a) / a for a, b in zip(values, values[1:])]
        print(f"mean growth: {sum(growth) / len(growth):.4f}")
        '''

    Args:
        code: Python source code to execute. Multi-line strings supported.
        runtime: Injected by LangGraph ToolNode. Provides config and state.

    Returns:
        Captured stdout/stderr from the execution, or an error message if
        the process exits with a non-zero code.
    """
    sandbox = await aget_sandbox(runtime)
    path = f"/tmp/langchain_opensandbox_{uuid.uuid4().hex}.py"  # noqa: S108

    async with sandbox:
        try:
            await sandbox.files.write_file(path, code)
        except Exception as exc:
            return f"Failed to write code to sandbox: {exc}"

        execution = await sandbox.commands.run(f"python3 {path}")
        await sandbox.commands.run(f"rm -f {path}")

        stdout = "".join(m.text for m in execution.logs.stdout)
        stderr = "".join(m.text for m in execution.logs.stderr)
        output = stdout + stderr

        exit_code: int | None = None
        if execution.id:
            try:
                status = await sandbox.commands.get_command_status(execution.id)
                exit_code = status.exit_code
            except Exception:
                pass
        if execution.error and exit_code is None:
            exit_code = 1

        if exit_code is not None and exit_code != 0:
            return f"Execution failed (exit {exit_code}):\n{output}"

        return output or "(no output)"
