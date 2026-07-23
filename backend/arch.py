"""Host CPU architecture + ISA detection for architecture-aware runtime
selection (config.ARCH_RUNTIME_PROFILES). Detection runs fresh on every call
(cheap: one file read or one sysctl call) rather than being cached at import
time, so tests can monkeypatch `detect_host_arch` per-case without fighting a
module-level cache — and so a single long-lived worker process never needs a
restart to pick up a fact about its own host that never changes anyway.

Every profile this resolves to is a *preference*, never a requirement: ONNX
Runtime silently drops an execution provider that isn't compiled into the
current onnxruntime build from the `providers=[...]` list and falls back to
the next one (demucs_onnx.py always appends CPUExecutionProvider last), so
requesting "OpenVINOExecutionProvider" on a plain pip/uv install that only
ships the CPU EP just runs on CPUExecutionProvider instead of raising. Same
posture for the ISA check itself: an unrecognized /proc/cpuinfo or sysctl
shape just yields the portable default profile rather than failing startup.
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

from backend.config import ARCH_RUNTIME_PROFILES, RuntimeProfile

_VNNI_FLAGS = ("avx512_vnni", "avx512vnni")


@dataclass(frozen=True)
class HostArch:
    arch: str  # normalized: "x86_64" | "arm64" | platform.machine() verbatim otherwise
    has_avx512: bool
    has_vnni: bool

    @property
    def profile_key(self) -> str:
        """Which ARCH_RUNTIME_PROFILES bucket this host falls into."""
        if self.arch == "x86_64" and self.has_vnni:
            return "x86_64_vnni"
        if self.arch == "arm64":
            return "arm64"
        return "default"


def detect_host_arch() -> HostArch:
    machine = _normalize_machine(platform.machine())
    if machine != "x86_64":
        return HostArch(arch=machine, has_avx512=False, has_vnni=False)
    flags = _x86_cpu_flags()
    return HostArch(arch=machine, has_avx512="avx512f" in flags, has_vnni=any(f in flags for f in _VNNI_FLAGS))


def profile_for(host: HostArch | None = None) -> RuntimeProfile:
    host = host or detect_host_arch()
    return ARCH_RUNTIME_PROFILES.get(host.profile_key, ARCH_RUNTIME_PROFILES["default"])


def resolve_default_tier(host: HostArch | None = None) -> str:
    """Which tier to run when the caller doesn't pin one — informed by
    scripts/bench_arch.py's measured real-time-factor per (arch, tier), not
    guessed. An explicit tier in a job request always wins; this is only the
    fallback (app.py's `_validate_mode_tier_stems`)."""
    return profile_for(host).tier


def resolve_providers(host: HostArch | None = None) -> tuple:
    """ONNX Runtime execution providers to try, in preference order, for this
    host — CPUExecutionProvider is always appended last as the universal
    fallback so no arch/provider is ever a hard requirement. Each entry is
    either a bare provider name or an (name, options) pair, whichever
    `onnxruntime.InferenceSession(providers=...)` accepts directly."""
    providers = profile_for(host).providers
    names = {p[0] if isinstance(p, tuple) else p for p in providers}
    if "CPUExecutionProvider" in names:
        return providers
    return (*providers, "CPUExecutionProvider")


def _normalize_machine(machine: str) -> str:
    machine = machine.lower()
    if machine == "amd64":
        return "x86_64"
    if machine == "aarch64":
        return "arm64"
    return machine


def _x86_cpu_flags() -> set[str]:
    """Best-effort lowercased CPU feature flags. Never raises — a host whose
    flags can't be read this way just falls back to the "default" profile."""
    try:
        system = platform.system()
        if system == "Linux":
            return _linux_cpu_flags()
        if system == "Darwin":
            return _darwin_cpu_flags()
    except Exception:
        pass
    return set()


def _linux_cpu_flags() -> set[str]:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("flags"):
            return set(line.split(":", 1)[1].split())
    return set()


def _darwin_cpu_flags() -> set[str]:
    # Old Intel Macs only (this codebase otherwise targets Apple Silicon) —
    # leaf7_features is where AVX512VNNI shows up; features covers the rest.
    out = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.leaf7_features", "machdep.cpu.features"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    return {tok.lower() for tok in out.stdout.split()}
