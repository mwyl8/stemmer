"""Architecture-aware runtime selection: mocks each host arch (via
`HostArch` directly, never real /proc/cpuinfo or sysctl calls) and asserts
the right tier/provider comes out the other end — see config.py's
ARCH_RUNTIME_PROFILES for what "right" means per arch, including why arm64's
CoreML entry carries a "CPUOnly" compute-units pin (a measured fp16/ANE
overflow, not a style choice).
"""

from __future__ import annotations

import platform

import pytest

from backend import arch
from backend.arch import HostArch


def test_x86_64_with_vnni_prefers_fast_int8_and_openvino():
    host = HostArch(arch="x86_64", has_avx512=True, has_vnni=True)
    assert host.profile_key == "x86_64_vnni"
    assert arch.resolve_default_tier(host) == "fast"
    providers = arch.resolve_providers(host)
    assert providers[0] == "OpenVINOExecutionProvider"
    assert providers[-1] == "CPUExecutionProvider"


def test_x86_64_without_vnni_falls_back_to_balanced_fp32_cpu_only():
    host = HostArch(arch="x86_64", has_avx512=True, has_vnni=False)
    assert host.profile_key == "default"
    assert arch.resolve_default_tier(host) == "balanced"
    assert arch.resolve_providers(host) == ("CPUExecutionProvider",)


def test_arm64_prefers_balanced_fp32_on_plain_cpu_not_coreml():
    # CoreML was tried and measured out, not just skipped: its default
    # compute units overflow to inf on this graph (fp16 on the Neural
    # Engine), and pinning it to "CPUOnly" to fix that made it ~16x slower
    # than plain CPUExecutionProvider on a clean, uncontended run — see
    # config.py's ARCH_RUNTIME_PROFILES comment for the full measurement.
    host = HostArch(arch="arm64", has_avx512=False, has_vnni=False)
    assert host.profile_key == "arm64"
    assert arch.resolve_default_tier(host) == "balanced"
    assert arch.resolve_providers(host) == ("CPUExecutionProvider",)


def test_unrecognized_arch_falls_back_to_default_profile():
    host = HostArch(arch="riscv64", has_avx512=False, has_vnni=False)
    assert host.profile_key == "default"
    assert arch.resolve_default_tier(host) == "balanced"
    assert arch.resolve_providers(host) == ("CPUExecutionProvider",)


def test_cpu_execution_provider_never_duplicated_when_profile_already_lists_it(monkeypatch):
    from backend import config

    monkeypatch.setitem(
        config.ARCH_RUNTIME_PROFILES,
        "default",
        config.RuntimeProfile(tier="balanced", providers=("CPUExecutionProvider",)),
    )
    host = HostArch(arch="riscv64", has_avx512=False, has_vnni=False)
    assert arch.resolve_providers(host) == ("CPUExecutionProvider",)


def test_detect_host_arch_normalizes_machine_names(monkeypatch):
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    assert arch.detect_host_arch().arch == "arm64"

    monkeypatch.setattr(platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(arch, "_x86_cpu_flags", lambda: set())
    assert arch.detect_host_arch().arch == "x86_64"


def test_detect_host_arch_reads_vnni_from_x86_flags(monkeypatch):
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(arch, "_x86_cpu_flags", lambda: {"fpu", "avx512f", "avx512_vnni"})
    host = arch.detect_host_arch()
    assert host.arch == "x86_64"
    assert host.has_avx512 is True
    assert host.has_vnni is True
    assert host.profile_key == "x86_64_vnni"


def test_detect_host_arch_x86_without_vnni_flag(monkeypatch):
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(arch, "_x86_cpu_flags", lambda: {"fpu", "sse4_2", "avx2"})
    host = arch.detect_host_arch()
    assert host.has_avx512 is False
    assert host.has_vnni is False
    assert host.profile_key == "default"


def test_cpu_flags_never_raises_on_unreadable_host(monkeypatch):
    def _boom():
        raise OSError("no /proc/cpuinfo here")

    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(arch, "_linux_cpu_flags", _boom)
    assert arch._x86_cpu_flags() == set()


@pytest.mark.parametrize(
    "profile_key,expected_tier",
    [("x86_64_vnni", "fast"), ("arm64", "balanced"), ("default", "balanced")],
)
def test_every_profile_key_resolves_to_a_wired_tier(profile_key, expected_tier):
    from backend.config import ARCH_RUNTIME_PROFILES, TIERS

    profile = ARCH_RUNTIME_PROFILES[profile_key]
    assert profile.tier == expected_tier
    assert profile.tier in TIERS
