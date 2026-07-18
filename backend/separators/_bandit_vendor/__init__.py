"""Vendored BandSplitRNN model code (the "Bandit" cinematic separator:
speech/music/effects, trained on DnR v3).

Source: github.com/ZFTurbo/Music-Source-Separation-Training
        (models/bandit/core/model/), MIT License, Copyright (c) 2024
        Roman Solovyev (ZFTurbo).
Original architecture: Watcharasupat et al., "Cinematic Audio Source
        Separation" / "Zero-shot Cinematic Audio Source Separation"
        (the BandIt / BandIt-v2 papers), github.com/kwatcharasupat/bandit-v2
        (Apache-2.0).

Vendored rather than pip-installed because the upstream repos aren't
published packages (they're research repos meant to be cloned), and the
official kwatcharasupat/bandit-v2 checkpoint loader additionally pulls in a
large, CUDA-pinned, partly-internal (Netflix Ray/Metaflow) dependency tree
that isn't practical to install for CPU inference. This is a much smaller,
self-contained slice: only the plain nn.Module definitions needed to
instantiate the model and load ZFTurbo's redistributed plain state_dict
checkpoint (no Hydra, no PyTorch-Lightning checkpoint wrapping — just
`model.load_state_dict(torch.load(path))`).

Only import-path prefixes were changed from the original
(`models.bandit.core.model` -> `backend.separators._bandit_vendor.model`);
the model code itself is unmodified.

Weights are NOT included here — see scripts/fetch_bandit_weights.py, which
downloads them into models/bandit/ (gitignored, pulled on setup).
"""
