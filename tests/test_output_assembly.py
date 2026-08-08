"""Output reassembly must put the stripped line terminators back.

OpenSandbox's execd emits one `OutputMessage` per line with the newline
stripped. Verified against a live server (opensandbox-server 0.1.x, docker
runtime):

    printf 'a\\nb\\nc\\n'          -> stdout ['a', 'b', 'c']
    printf 'no-trailing-newline'  -> stdout ['no-trailing-newline']
    print('x'); print('y')        -> stdout ['x', 'y']
    ... ; echo err >&2            -> stderr ['err']

Joining those on "" is what the original implementation did, and it silently
broke every newline-delimited-JSON parse in `BaseSandbox`: `ls` on a directory
with two or more entries returned `LsResult(error=None, entries=[])` — not an
error, just a wrong answer that reads as "empty directory".

These fixtures deliberately mirror the live shape (no trailing newlines) rather
than the convenient one, because the convenient one is what hid the bug.
"""

import pytest

from langchain_opensandbox.backend import _combine_output


def _execution(stdout=(), stderr=()):
    from opensandbox.models.execd import Execution, ExecutionLogs, OutputMessage

    logs = ExecutionLogs()
    for t in stdout:
        logs.add_stdout(OutputMessage(text=t, timestamp=0))
    for t in stderr:
        logs.add_stderr(OutputMessage(text=t, timestamp=0))
    return Execution(id="cmd-1", result=[], error=None, logs=logs)


@pytest.mark.unit
class TestCombineOutput:
    def test_lines_are_rejoined_with_newlines(self):
        assert _combine_output(_execution(["a", "b", "c"])) == "a\nb\nc"

    def test_single_line_is_unchanged(self):
        assert _combine_output(_execution(["only"])) == "only"

    def test_empty_output(self):
        assert _combine_output(_execution()) == ""

    def test_stderr_follows_stdout_on_its_own_line(self):
        assert _combine_output(_execution(["out"], ["err"])) == "out\nerr"

    def test_ndjson_stays_parseable(self):
        # The exact failure: BaseSandbox's ls/glob/grep split output on
        # newlines and json.loads each line.
        import json

        records = [
            {"path": "/tmp/a.txt", "is_dir": False},
            {"path": "/tmp/b.txt", "is_dir": False},
        ]
        execution = _execution([json.dumps(r) for r in records])

        parsed = [json.loads(line) for line in _combine_output(execution).splitlines()]

        assert parsed == records
