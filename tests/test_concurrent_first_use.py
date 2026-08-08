"""Concurrent first use must produce ONE container, not one per caller.

The harness resolves a `BackendFactory` per tool call, so an agent that issues
three `write_file` calls in a single turn gets three backend instances racing
find-or-create on a thread that has no sandbox yet. Without a lock all three
see "nothing running", all three create one, and each file lands in a different
container — every write reports success and `ls` afterwards shows exactly one
file.

Observed in production on 2026-08-08 (three writes, `ls` returned
`['/work/beta.txt']`) and reproduced against a live server: 3 containers for
one thread_id, 1/3 files visible.

The eager backend used to hide this. `FilesystemMiddleware` resolves the
backend on every *model* call too, so the model call preceding the tool calls
created the sandbox first and the parallel tool calls all found it. Making
resolution lazy removed that accidental serialisation and exposed the
underlying race, which was always there.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from langchain_opensandbox.factory import SandboxFactory

_PATCH = "langchain_opensandbox.factory"
_SETTINGS = f"{_PATCH}.OpenSandboxSettings.from_runnable_config"


def _runtime(thread_id="t-1"):
    return SimpleNamespace(config={"configurable": {"thread_id": thread_id}})


def _settings():
    cfg = MagicMock()
    cfg.opensandbox_url = "localhost:8080"
    cfg.opensandbox_api_key = None
    cfg.opensandbox_image = "python:3.11-slim"
    cfg.opensandbox_use_server_proxy = True
    return cfg


class _FakeServer:
    """Sandboxes per thread_id, answering lookups from that state.

    One instance serves every thread, so a single set of patches covers a
    concurrent multi-thread test. Nested `patch.object` calls inside coroutines
    that overlap would restore in the wrong order and leak the mock — hence one
    patch context for the whole test.
    """

    def __init__(self, *, create_delay=0.02):
        self.created: dict[str, list[str]] = {}
        self._create_delay = create_delay

    def ids(self, thread_id):
        return self.created.get(thread_id, [])

    # -- find (patched with autospec, so `self` is the SandboxFactory) --------

    async def afind(self, factory):
        return next(iter(self.ids(factory._get_thread_id())), None)

    def find(self, factory):
        return next(iter(self.ids(factory._get_thread_id())), None)

    # -- create / connect (routed by the thread_id in the metadata kwarg) -----

    async def acreate(self, *_a, **kw):
        await asyncio.sleep(self._create_delay)  # a real create is not instant
        return MagicMock(id=self._register(kw))

    def create(self, *_a, **kw):
        # The delay is load-bearing, not padding: without it the pool threads
        # never interleave and this test passes even with the lock removed.
        time.sleep(self._create_delay)
        return MagicMock(id=self._register(kw))

    def _register(self, kw):
        thread_id = kw["metadata"]["thread_id"]
        bucket = self.created.setdefault(thread_id, [])
        sandbox_id = f"{thread_id}-sb{len(bucket)}"
        bucket.append(sandbox_id)
        return sandbox_id

    async def aconnect(self, sandbox_id, *_a, **_kw):
        return MagicMock(id=sandbox_id)

    def connect(self, sandbox_id, *_a, **_kw):
        return MagicMock(id=sandbox_id)


@contextmanager
def _served(server, *, is_async):
    """Apply every patch once, for the whole test."""
    find_attr = "_afind_sandbox_id" if is_async else "_find_sandbox_id"
    find_impl = server.afind if is_async else server.find
    target = (
        "opensandbox.sandbox.Sandbox"
        if is_async
        else ("opensandbox.sync.sandbox.SandboxSync")
    )
    create_impl = server.acreate if is_async else server.create
    connect_impl = server.aconnect if is_async else server.connect

    with (
        patch(_SETTINGS, return_value=_settings()),
        patch.object(SandboxFactory, find_attr, autospec=True, side_effect=find_impl),
        patch(f"{target}.create", side_effect=create_impl),
        patch(f"{target}.connect", side_effect=connect_impl),
    ):
        yield


@pytest.mark.unit
class TestConcurrentFirstUse:
    @pytest.mark.asyncio
    async def test_parallel_aget_sandbox_creates_one_container(self):
        server = _FakeServer()
        runtime = _runtime("race-thread")

        with _served(server, is_async=True):
            results = await asyncio.gather(
                *(SandboxFactory(runtime).aget_sandbox() for _ in range(5))
            )

        assert len(server.ids("race-thread")) == 1, (
            f"{len(server.ids('race-thread'))} containers created for one "
            f"thread_id; every caller must land on the same sandbox"
        )
        assert {r.id for r in results} == {"race-thread-sb0"}

    def test_parallel_get_sandbox_creates_one_container(self):
        server = _FakeServer()
        runtime = _runtime("race-sync")

        with _served(server, is_async=False), ThreadPoolExecutor(5) as pool:
            results = list(
                pool.map(lambda _: SandboxFactory(runtime).get_sandbox(), range(5))
            )

        assert len(server.ids("race-sync")) == 1
        assert {r.id for r in results} == {"race-sync-sb0"}

    @pytest.mark.asyncio
    async def test_existing_sandbox_is_reused_without_creating(self):
        server = _FakeServer()
        server.created["warm"] = ["warm-sb-existing"]

        with _served(server, is_async=True):
            results = await asyncio.gather(
                *(SandboxFactory(_runtime("warm")).aget_sandbox() for _ in range(3))
            )

        assert server.ids("warm") == ["warm-sb-existing"]
        assert {r.id for r in results} == {"warm-sb-existing"}

    @pytest.mark.asyncio
    async def test_different_threads_get_their_own_container(self):
        # The lock is per thread_id: isolation between conversations must
        # survive the fix, including when they start at the same moment.
        server = _FakeServer()

        with _served(server, is_async=True):
            await asyncio.gather(
                *(
                    SandboxFactory(_runtime(t)).aget_sandbox()
                    for t in ("a", "a", "b", "b")
                )
            )

        assert server.ids("a") == ["a-sb0"]
        assert server.ids("b") == ["b-sb0"]
