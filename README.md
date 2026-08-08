# langchain-opensandbox

[OpenSandbox](https://github.com/alibaba/OpenSandbox) backend for
[LangChain deep agents](https://docs.langchain.com/oss/python/deepagents/overview).

OpenSandbox is an Apache-2.0 sandbox runtime for AI agents that you host yourself — Docker or
Kubernetes, your hardware, no per-second billing and no account. This package makes one usable as a
deep agent's filesystem and shell, the same way `langchain-e2b`, `langchain-modal` and
`langchain-daytona` do for their hosted providers.

```bash
pip install langchain-opensandbox
```

## Quickstart

```python
from deepagents import create_deep_agent
from opensandbox.sync.sandbox import SandboxSync

from langchain_opensandbox import OpenSandboxSandbox

sandbox = SandboxSync.create("python:3.11-slim")
agent = create_deep_agent(
    model="anthropic:claude-sonnet-4-5",
    backend=OpenSandboxSandbox(sandbox=sandbox),
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Write fib.py and run it for n=30."}]}
)
```

`OpenSandboxSandbox` implements deepagents' `SandboxBackendProtocol` by subclassing `BaseSandbox`, so
the agent gets `ls` / `read_file` / `write_file` / `edit_file` / `grep` / `glob` and the `execute`
shell tool. You own the sandbox's lifetime: create it, and `kill()` it when you're done.

## Managed sandboxes for LangGraph

The above leaves sandbox lifecycle to you, which is awkward in a server where each conversation
should get its own container. `langchain_opensandbox.factory` handles that: it tags a sandbox with
the current LangGraph `thread_id` at creation and finds it again on every later call, so one
conversation reuses one container and parallel conversations stay isolated.

```python
from langchain_opensandbox.factory import get_backend

agent = create_deep_agent(model=model, backend=get_backend)
```

`get_backend` performs **no I/O**. That matters more than it sounds: deepagents resolves the backend
on every model call — `FilesystemMiddleware` has to know whether to expose the `execute` tool — so a
factory that connects eagerly pays a sandbox lookup per model call, synchronously, on the event loop.
(ASGI servers object, correctly: `langgraph dev` refuses the blocking `socket.connect` unless you pass
`--allow-blocking`.) What you get back is a `LazyOpenSandboxSandbox`, which connects on the first
operation the agent actually performs, and serves the async half of the protocol from the async
OpenSandbox client rather than a worker thread. A model call that never touches the sandbox costs
nothing at all.

If the sandbox has since died (timeout, crash) a fresh one is created transparently on the next call.
In-sandbox state — installed packages, written files — does not survive that.

For a plain ReAct agent that wants code execution without adopting the deep-agent filesystem, there
is also a standalone tool driving the same thread-scoped sandbox:

```python
from langchain.agents import create_agent
from langchain_opensandbox.tools import execute_python

agent = create_agent(model, tools=[execute_python])
```

### Settings

`factory` resolves connection settings per call. For each field the first value found wins: the
environment variable, then the same lower-cased key in the run's `config["configurable"]`, then the
default.

| Env var | Default | Meaning |
|---|---|---|
| `OPENSANDBOX_URL` | `localhost:8080` | Server address (`host:port`). |
| `OPENSANDBOX_API_KEY` | — | API key; omit if the server has no auth. |
| `OPENSANDBOX_IMAGE` | `python:3.11-slim` | Image for new containers. Use one with your libraries pre-installed for faster startup. |
| `OPENSANDBOX_USE_SERVER_PROXY` | `true` | Route sandbox traffic through the server rather than dialling container ports directly. Keep it on for Docker Swarm / bridge-network deployments where those ports are unreachable; set `false` only for host/flat networks, where direct is faster. |

This package does not read `.env` files — that is an application's decision. Load one in your own
entrypoint if you want it.

## Package layout

`backend.py` and `config.py` are provider-neutral: they take no LangGraph runtime, config or thread,
so the backend is drivable from a plain script or a test — `LazyOpenSandboxSandbox` takes two plain
connect callables, not a runtime. `factory.py` and `tools.py` are the LangGraph-specific half and are
imported explicitly rather than re-exported from the package root.
`tests/test_package_boundary.py` enforces the split.

(LangGraph is imported into your process either way — `deepagents` depends on `langchain`, which
depends on `langgraph`. The boundary is about API surface, not about saving a dependency.)

## Compatibility

- **Python** ≥ 3.11 (deepagents' floor).
- **deepagents** `>=0.6.12,<0.7`. The cap is load-bearing for `factory.py` only: `get_backend` is a
  `BackendFactory` (`deepagents.backends.protocol.BackendFactory`), a type alias that deepagents
  `main` has since dropped. `backend.py` depends on nothing but `BaseSandbox` and will outlive it.
- **opensandbox** `>=0.1.15`.

## Development

```bash
pip install -e ".[dev]"
ruff check src/ tests/ && ruff format --check src/ tests/
mypy src/
pytest
```

The suite mocks the OpenSandbox client, so it needs no server. To exercise the wire protocol, run a
server (`docker run -d -p 8080:8080 ghcr.io/alibaba/opensandbox/server:latest`) and drive an agent
against it.

## License

MIT — see [LICENSE](LICENSE).
