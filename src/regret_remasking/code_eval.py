from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CodeEvaluation:
    passed: bool
    status: str
    stderr: str = ""


def extract_python_code(generation: str) -> str:
    fenced = re.findall(r"```(?:python)?\s*\n(.*?)```", generation, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced[-1].strip()
    text = generation.strip("\n")
    top_level = re.search(r"^(?:from |import |def |class )", text, flags=re.MULTILINE)
    if top_level is not None:
        return text[top_level.start() :].strip()
    return text


def assemble_humaneval_program(
    generation: str,
    prompt: str,
    test: str,
    entry_point: str,
) -> str:
    candidate = extract_python_code(generation)
    if re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", candidate, flags=re.MULTILINE):
        prompt_definition = re.search(
            rf"^def\s+{re.escape(entry_point)}\s*\(", prompt, flags=re.MULTILINE
        )
        preamble = prompt[: prompt_definition.start()] if prompt_definition is not None else ""
        implementation = preamble + candidate
    else:
        implementation = prompt + candidate
    return "\n".join(
        [
            "import resource",
            "resource.setrlimit(resource.RLIMIT_CPU, (2, 2))",
            "resource.setrlimit(resource.RLIMIT_AS, (1_073_741_824, 1_073_741_824))",
            "resource.setrlimit(resource.RLIMIT_FSIZE, (1_048_576, 1_048_576))",
            implementation,
            test,
            f"check({entry_point})",
        ]
    )


def _bubblewrap_command(program: str, unshare_network: bool = True) -> list[str] | None:
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        return None
    prefix = Path(sys.prefix).resolve()
    interpreter_prefix = Path(sys.executable).resolve().parents[1]
    command = [
        bwrap,
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--die-with-parent",
        "--new-session",
    ]
    if unshare_network:
        command.append("--unshare-net")
    bind_paths = dict.fromkeys(
        (Path("/usr"), Path("/lib"), Path("/lib64"), prefix, interpreter_prefix)
    )
    for path in bind_paths:
        if path.exists():
            command.extend(["--ro-bind", str(path), str(path)])
    command.extend(
        [
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--chdir",
            "/tmp",
            sys.executable,
            "-I",
            "-c",
            program,
        ]
    )
    return command


def evaluate_humaneval_program(
    generation: str,
    prompt: str,
    test: str,
    entry_point: str,
    timeout_s: float = 4.0,
) -> CodeEvaluation:
    program = assemble_humaneval_program(generation, prompt, test, entry_point)
    command = _bubblewrap_command(program, unshare_network=True)
    if command is None:
        return CodeEvaluation(False, "sandbox_unavailable")
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"}
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return CodeEvaluation(False, "timeout")
    nested_network_block = "NETLINK_ROUTE" in completed.stderr or "loopback:" in completed.stderr
    if (
        completed.returncode != 0
        and nested_network_block
        and os.environ.get("CODEX_SANDBOX_NETWORK_DISABLED") == "1"
    ):
        fallback = _bubblewrap_command(program, unshare_network=False)
        try:
            completed = subprocess.run(
                fallback,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_s,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return CodeEvaluation(False, "timeout")
    if completed.returncode == 0:
        return CodeEvaluation(True, "passed")
    stderr = completed.stderr[-1000:].strip()
    if "Operation not permitted" in stderr or "bwrap:" in stderr:
        return CodeEvaluation(False, "sandbox_error", stderr)
    return CodeEvaluation(False, "failed_tests", stderr)
