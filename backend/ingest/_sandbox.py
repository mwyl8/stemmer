"""Shared subprocess sandboxing for yt-dlp/ffmpeg.

No Docker in v1 (decision 7), so this isn't a real OS sandbox — no seccomp,
no namespaces. What it does do: no shell (argv list, never a shell string, so
there's no injection surface), a restricted environment (the child doesn't
inherit whatever secrets happen to be in the parent's env — yt-dlp/ffmpeg
only need PATH/HOME/locale to find each other and write temp files), and a
hard timeout every caller must pass explicitly rather than defaulting to
"wait forever".
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

# Env vars yt-dlp/ffmpeg legitimately need; everything else in the parent's
# environment (API keys, tokens, unrelated config) is dropped for the child.
_ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR")


def restricted_env() -> dict[str, str]:
    return {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ}


def run_subprocess(cmd: list[str], timeout: float, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run `cmd` with no shell, a restricted env, and a hard timeout.

    Raises subprocess.TimeoutExpired on timeout — callers wrap that in their
    own domain exception (FetchError/DecodeError) so the caller-facing API
    doesn't leak the subprocess module's exception types.
    """
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=restricted_env(),
        cwd=str(cwd) if cwd else None,
        shell=False,
    )
