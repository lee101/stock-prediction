from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Sequence


_ALLOWED_STRICT_HOST_KEY_CHECKING = frozenset(
    {"accept-new", "ask", "yes", "no", "off"}
)


def _resolve_strict_host_key_checking() -> str:
    configured = os.environ.get(
        "RUNPOD_SSH_STRICT_HOST_KEY_CHECKING",
        "accept-new",
    ).strip()
    if configured not in _ALLOWED_STRICT_HOST_KEY_CHECKING:
        allowed = ", ".join(sorted(_ALLOWED_STRICT_HOST_KEY_CHECKING))
        raise RuntimeError(
            "RUNPOD_SSH_STRICT_HOST_KEY_CHECKING must be one of "
            f"{allowed}; received {configured!r}"
        )
    return configured


SSH_OPTIONS: tuple[str, ...] = (
    "-o",
    f"StrictHostKeyChecking={_resolve_strict_host_key_checking()}",
    "-o",
    "BatchMode=yes",
)


def tail_excerpt(text: str | None, *, limit: int = 400) -> str:
    rendered = str(text or "").strip()
    if len(rendered) <= limit:
        return rendered
    return rendered[-limit:]


def render_subprocess_error(
    *,
    description: str,
    cmd: Sequence[str],
    result: object,
    excerpt_limit: int = 400,
) -> RuntimeError:
    returncode = int(getattr(result, "returncode", 1))
    stdout = tail_excerpt(getattr(result, "stdout", None), limit=excerpt_limit)
    stderr = tail_excerpt(getattr(result, "stderr", None), limit=excerpt_limit)

    message = [
        f"{description} (exit {returncode})",
        f"command: {shlex.join(list(cmd))}",
    ]
    if stdout:
        message.append(f"stdout excerpt:\n{stdout}")
    if stderr:
        message.append(f"stderr excerpt:\n{stderr}")
    return RuntimeError("\n".join(message))


def run_checked_subprocess(
    cmd: Sequence[str],
    *,
    description: str,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(cmd),
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise render_subprocess_error(
            description=description,
            cmd=cmd,
            result=result,
        )
    return result


def ssh_cmd(
    *,
    ssh_host: str,
    ssh_port: int,
    remote_cmd: str,
) -> list[str]:
    return [
        "ssh",
        *SSH_OPTIONS,
        "-p",
        str(ssh_port),
        f"root@{ssh_host}",
        remote_cmd,
    ]


def ssh_run(
    *,
    ssh_host: str,
    ssh_port: int,
    remote_cmd: str,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ssh_cmd(
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            remote_cmd=remote_cmd,
        ),
        check=False,
        text=True,
        capture_output=capture_output,
    )
