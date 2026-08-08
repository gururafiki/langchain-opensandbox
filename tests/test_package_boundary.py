"""The core modules must not depend on LangGraph.

`backend.py` and `config.py` are the provider-neutral half: a `BaseSandbox`
implementation over a sandbox the caller owns, plus its settings. The
thread-scoped factory and the `execute_python` tool are LangGraph-specific and
live in modules you import explicitly.

Note this is a *source-level* boundary, not a runtime one. LangGraph is
imported into the process either way: `deepagents` depends on `langchain`,
which depends on `langgraph`, and `BaseSandbox` lives in deepagents. What the
split buys is that the backend has no LangGraph concepts in its API — you can
construct and drive it from a plain script, a FastAPI handler, or a test, with
no runtime, no config, and no thread. Only a static check can state that, so
that is what this asserts.
"""

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src" / "langchain_opensandbox"

_CORE_MODULES = ("backend.py", "config.py", "__init__.py")


def _imported_roots(path: Path) -> set[str]:
    """Return the top-level package name of every import in *path*."""
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.unit
class TestPackageBoundary:
    @pytest.mark.parametrize("module", _CORE_MODULES)
    def test_core_module_does_not_import_langgraph(self, module):
        roots = _imported_roots(_SRC / module)
        assert "langgraph" not in roots, (
            f"{module} imports langgraph; the core backend must stay usable "
            f"without a LangGraph runtime"
        )

    def test_langgraph_modules_are_the_ones_that_use_it(self):
        # The converse: if nothing imported langgraph, the split would be
        # describing a distinction that no longer exists.
        assert "langgraph" in _imported_roots(_SRC / "factory.py")
        assert "langgraph" in _imported_roots(_SRC / "tools.py")

    def test_root_exports_only_the_provider_neutral_surface(self):
        import langchain_opensandbox

        assert langchain_opensandbox.__all__ == [
            "OpenSandboxSandbox",
            "OpenSandboxSettings",
        ]
