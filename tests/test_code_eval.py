from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from regret_remasking import code_eval


def _completed(stderr: str, returncode: int = 1) -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stderr=stderr)


def _python_stderr(source: str) -> str:
    completed = subprocess.run(
        [sys.executable, "-I", "-c", source],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert completed.stderr
    return completed.stderr


def test_bubblewrap_launcher_diagnostic_is_sandbox_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(code_eval.shutil, "which", lambda name: "/usr/bin/bwrap")
    monkeypatch.setattr(
        code_eval.subprocess,
        "run",
        lambda *args, **kwargs: _completed("bwrap: creating new namespace failed: Operation not permitted\n"),
    )

    result = code_eval.evaluate_humaneval_program("def f():\n    return 1", "", "def check(f): pass", "f")

    assert result.status == "sandbox_error"
    assert not result.passed


def test_python_syntax_error_quoting_permission_text_is_failed_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(code_eval.shutil, "which", lambda name: "/usr/bin/bwrap")
    stderr = _python_stderr('"Operation not permitted" +')
    monkeypatch.setattr(code_eval.subprocess, "run", lambda *args, **kwargs: _completed(stderr))

    result = code_eval.evaluate_humaneval_program("bad", "", "def check(f): pass", "f")

    assert result.status == "failed_tests"
    assert result.stderr == stderr[-1000:].strip()


def test_python_traceback_quoting_bwrap_text_does_not_trigger_fallback_or_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(code_eval.shutil, "which", lambda name: "/usr/bin/bwrap")
    monkeypatch.setenv("CODEX_SANDBOX_NETWORK_DISABLED", "1")
    stderr = _python_stderr('raise RuntimeError("bwrap: NETLINK_ROUTE loopback:")')
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(args[0])
        return _completed(stderr)

    monkeypatch.setattr(code_eval.subprocess, "run", fake_run)

    result = code_eval.evaluate_humaneval_program("def f():\n    raise RuntimeError('x')", "", "def check(f): f()", "f")

    assert len(calls) == 1
    assert result.status == "failed_tests"


def test_candidate_permission_error_is_failed_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(code_eval.shutil, "which", lambda name: "/usr/bin/bwrap")
    stderr = _python_stderr('raise PermissionError(1, "Operation not permitted")')
    monkeypatch.setattr(code_eval.subprocess, "run", lambda *args, **kwargs: _completed(stderr))

    result = code_eval.evaluate_humaneval_program("def f():\n    raise PermissionError(1, 'Operation not permitted')", "", "def check(f): f()", "f")

    assert result.status == "failed_tests"


def test_launcher_detection_uses_untruncated_diagnostic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(code_eval.shutil, "which", lambda name: "/usr/bin/bwrap")
    stderr = "bwrap: creating new namespace failed\n" + "details" * 300
    monkeypatch.setattr(code_eval.subprocess, "run", lambda *args, **kwargs: _completed(stderr))
    result = code_eval.evaluate_humaneval_program("def f(): pass", "", "def check(f): pass", "f")
    assert result.status == "sandbox_error"
    assert len(result.stderr) <= 1000


def test_bubblewrap_network_launcher_diagnostic_still_uses_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(code_eval.shutil, "which", lambda name: "/usr/bin/bwrap")
    monkeypatch.setenv("CODEX_SANDBOX_NETWORK_DISABLED", "1")
    calls = []

    def fake_command(program: str, unshare_network: bool = True) -> list[str]:
        return ["bwrap", "net" if unshare_network else "fallback", program]

    def fake_run(command, *args, **kwargs):
        calls.append(command[1])
        if command[1] == "net":
            return _completed("bwrap: failed to setup loopback: Operation not permitted\n")
        return _completed("", returncode=0)

    monkeypatch.setattr(code_eval, "_bubblewrap_command", fake_command)
    monkeypatch.setattr(code_eval.subprocess, "run", fake_run)

    result = code_eval.evaluate_humaneval_program("def f():\n    return 1", "", "def check(f): assert f() == 1", "f")

    assert calls == ["net", "fallback"]
    assert result.status == "passed"
    assert result.passed


def test_timeout_status_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(code_eval.shutil, "which", lambda name: "/usr/bin/bwrap")

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], timeout=1)

    monkeypatch.setattr(code_eval.subprocess, "run", fake_run)

    result = code_eval.evaluate_humaneval_program("def f(): pass", "", "def check(f): pass", "f")

    assert result.status == "timeout"


def test_missing_launcher_during_network_retry_stays_infrastructure_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEX_SANDBOX_NETWORK_DISABLED", "1")
    monkeypatch.setattr(code_eval, "_bubblewrap_command", lambda program, unshare_network=True: ["bwrap"] if unshare_network else None)
    monkeypatch.setattr(code_eval.subprocess, "run", lambda *args, **kwargs: _completed("bwrap: NETLINK_ROUTE unavailable\n"))
    result = code_eval.evaluate_humaneval_program("def f(): pass", "", "def check(f): pass", "f")
    assert result.status == "sandbox_unavailable"


def test_malformed_source_line_starting_with_bwrap_is_failed_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(code_eval.shutil, "which", lambda name: "/usr/bin/bwrap")
    stderr = _python_stderr('bwrap:')
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(args[0])
        return _completed(stderr)

    monkeypatch.setattr(code_eval.subprocess, "run", fake_run)
    monkeypatch.setenv("CODEX_SANDBOX_NETWORK_DISABLED", "1")

    result = code_eval.evaluate_humaneval_program("bad", "", "def check(f): pass", "f")

    assert len(calls) == 1
    assert result.status == "failed_tests"
    assert "SyntaxError" in result.stderr


def test_fit116_style_malformed_prompt_is_failed_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(code_eval.shutil, "which", lambda name: "/usr/bin/bwrap")
    malformed = (
        'import os\nimport errno\n\n'
        'def describe_error_code(code: int) -> str:\n'
        '    """import os import errno def describe_error_code(code: int) -> str: \"\"\" '
        'Handles errors. >>> describe_error_code(errno.EPERM) "Error Message: Operation not permitted" '
        '\"\"\""""    try:\n'
        '        return "unreachable"\n'
    )
    stderr = _python_stderr(malformed)
    monkeypatch.setattr(code_eval.subprocess, "run", lambda *args, **kwargs: _completed(stderr))

    result = code_eval.evaluate_humaneval_program("bad", "", "def check(f): pass", "f")

    assert result.status == "failed_tests"
    assert "SyntaxError" in result.stderr
