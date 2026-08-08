"""OpenSandbox backend for LangChain deep agents.

The package is split in two halves:

**Core** — :class:`~langchain_opensandbox.backend.OpenSandboxSandbox`, exported
here. It implements the deepagents ``SandboxBackendProtocol`` over a sandbox
you create and own, exactly like the other provider backends::

    from opensandbox.sync.sandbox import SandboxSync
    from langchain_opensandbox import OpenSandboxSandbox

    agent = create_deep_agent(
        model=model,
        backend=OpenSandboxSandbox(sandbox=SandboxSync.create("python:3.11-slim")),
    )

**LangGraph helpers** — :mod:`langchain_opensandbox.factory` and
:mod:`langchain_opensandbox.tools`, imported explicitly::

    from langchain_opensandbox.factory import get_backend

    agent = create_deep_agent(model=model, backend=get_backend)

They manage one sandbox per LangGraph ``thread_id`` for you. They are kept out
of this namespace on purpose: the backend itself takes no runtime, no config
and no thread, so it stays drivable from a plain script or a test. (LangGraph
is imported into the process either way — ``deepagents`` depends on it — so
this is a source-level boundary, enforced by ``tests/test_package_boundary.py``,
not a saved dependency.)

Limitations:
    If a sandbox dies mid-conversation (its timeout elapses, the container
    crashes), the factory transparently creates a new one on the next call —
    any in-sandbox state (installed packages, written files) is lost.
"""

from .backend import OpenSandboxSandbox
from .config import OpenSandboxSettings

__all__ = [
    "OpenSandboxSandbox",
    "OpenSandboxSettings",
]
