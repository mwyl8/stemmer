"""Vendored MelBandRoformer model code (Lead vs. Backing Vocals mode's
karaoke-split model).

Source: github.com/ZFTurbo/Music-Source-Separation-Training
        (models/bs_roformer/mel_band_roformer.py, attend.py), MIT License,
        Copyright (c) 2024 Roman Solovyev (ZFTurbo). That repo's own
        implementation is itself adapted from lucidrains/BS-RoFormer (MIT).

Only import-path prefixes were changed from the original
(`models.bs_roformer.attend` -> `backend.separators._roformer_vendor.attend`);
the model code itself is unmodified — same vendoring posture as
`_bandit_vendor/` (see that package's docstring for the full rationale:
research-repo-not-a-package, and here specifically we also want the exact
same MelBandRoformer class definition the pretrained checkpoint (below) was
trained against, not a reimplementation that could silently diverge).

Weights are NOT included here — scripts/export_roformer_onnx.py downloads
the checkpoint on demand (via huggingface_hub, cached under
~/.cache/huggingface/, not under this repo's models/) from
becruily/mel-band-roformer-karaoke on Hugging Face (the aufr33/viperx
original mirror this repo used to point at, jarredou/aufr33-viperx-karaoke-
melroformer-model, returned 401/RepositoryNotFoundError as of this writing —
gated or removed — so this vendors the becruily retrain instead, same
architecture and same "Vocals"/"Instrumental" stem convention) and exports
the ONNX-exportable core straight to models/karaoke_core.onnx (gitignored,
pulled on setup — same posture as htdemucs_core.onnx).

Only `MelBandRoformer` + `Attend` are vendored — `bs_roformer.py` (the
non-mel-banded sibling architecture) isn't needed here.
"""
