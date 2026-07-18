"""Download the Bandit (BandSplitRNN, speech/music/effects) pretrained
checkpoint + config into models/bandit/ (gitignored, pulled on setup — never
committed, matching the same rule as the Demucs ONNX weights).

    uv run --group speech python scripts/fetch_bandit_weights.py

Source: Eddycrack864/Music-Source-Separation-Training on Hugging Face, a
redistribution of ZFTurbo/Music-Source-Separation-Training's MIT-licensed
"model_bandit_plus_dnr_sdr_11.47" checkpoint (BandSplitRNN reimplementation
of BandIt, trained on DnR v3) as a plain state_dict — no PyTorch-Lightning
checkpoint unwrapping needed, unlike the original kwatcharasupat/bandit-v2
release.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import MODELS_DIR

BASE_URL = "https://huggingface.co/Eddycrack864/Music-Source-Separation-Training/resolve/main"
OUTPUT_DIR = MODELS_DIR / "bandit"
FILES = {
    "model_bandit_plus_dnr_sdr_11.47.chpt": 148_891_175,
    "config_dnr_bandit_bsrnn_multi_mus64.yaml": None,  # small text file, size not worth checking
}


def _download(name: str, expected_size: int | None) -> Path:
    dest = OUTPUT_DIR / name
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        print(f"already have {dest} ({dest.stat().st_size} bytes), skipping")
        return dest

    url = f"{BASE_URL}/{name}"
    print(f"downloading {url} -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        while chunk := resp.read(1024 * 1024):
            f.write(chunk)
    tmp.rename(dest)
    print(f"wrote {dest} ({dest.stat().st_size} bytes)")
    return dest


def fetch() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, size in FILES.items():
        _download(name, size)


if __name__ == "__main__":
    fetch()
