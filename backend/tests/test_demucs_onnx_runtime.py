"""Architecture-aware provider selection on DemucsONNXSeparator itself:
providers=None should resolve through backend.arch (mockable via
`host_arch=`), an explicit `providers=` should override that entirely, and
`runtime_info()` should report what onnxruntime actually ended up running,
not just what was requested. Needs an exported htdemucs ONNX model —
skips cleanly if scripts/export_onnx.py hasn't been run yet.
"""

from __future__ import annotations

import pytest

from backend.arch import HostArch
from backend.config import MODELS_DIR
from backend.separators.demucs_onnx import DemucsONNXSeparator

ONNX_PATH = MODELS_DIR / "htdemucs_core.onnx"
METADATA_PATH = MODELS_DIR / "htdemucs_core.json"

pytestmark = pytest.mark.skipif(not ONNX_PATH.exists(), reason=f"{ONNX_PATH} not exported")


def test_default_providers_resolve_through_arch_for_arm64():
    # arm64 resolves to plain CPUExecutionProvider, not CoreML — measured,
    # not assumed: CoreML's default compute units overflow to inf on this
    # graph, and pinning it to "CPUOnly" to fix that made it ~16x slower
    # than plain CPU on a clean run (config.py's ARCH_RUNTIME_PROFILES has
    # the full measurement).
    host = HostArch(arch="arm64", has_avx512=False, has_vnni=False)
    sep = DemucsONNXSeparator(model_path=ONNX_PATH, metadata_path=METADATA_PATH, host_arch=host)
    info = sep.runtime_info()
    assert info["arch"] == "arm64"
    assert info["provider"] == "CPUExecutionProvider"
    assert info["model"] == "htdemucs_core"


def test_explicit_providers_override_arch_resolution():
    host = HostArch(arch="arm64", has_avx512=False, has_vnni=False)
    sep = DemucsONNXSeparator(
        model_path=ONNX_PATH, metadata_path=METADATA_PATH, host_arch=host, providers=("CPUExecutionProvider",)
    )
    assert sep.runtime_info()["provider"] == "CPUExecutionProvider"


def test_unavailable_provider_falls_back_to_cpu_without_raising():
    # OpenVINO isn't compiled into this build's onnxruntime — a x86_64_vnni
    # host's preference should still construct successfully and report
    # whatever onnxruntime actually fell back to (never a hard requirement).
    host = HostArch(arch="x86_64", has_avx512=True, has_vnni=True)
    sep = DemucsONNXSeparator(model_path=ONNX_PATH, metadata_path=METADATA_PATH, host_arch=host)
    info = sep.runtime_info()
    assert info["arch"] == "x86_64_vnni"
    assert info["provider"] in sep.session.get_providers()


def test_runtime_info_none_for_bandit_and_delegated_through_chained_and_singing_separators():
    from backend.separators.base import Separator
    from backend.separators.chained_sep import ChainedSeparator
    from backend.separators.singing_sep import SpeechVsSingingSeparator

    class FakeBandit(Separator):
        def separate(self, audio, on_chunk=None):
            raise NotImplementedError

    demucs = DemucsONNXSeparator(model_path=ONNX_PATH, metadata_path=METADATA_PATH)
    assert FakeBandit().runtime_info() is None

    chained = ChainedSeparator(bandit=FakeBandit(), demucs=demucs)
    assert chained.runtime_info() == demucs.runtime_info()

    singing = SpeechVsSingingSeparator(bandit=FakeBandit(), demucs=demucs)
    assert singing.runtime_info() == demucs.runtime_info()
