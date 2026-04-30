"""Tests for ``scripts.build_chronos2_latents``.

The Bolt builder's helpers (``_load_symbols``, ``_load_csv``,
``_build_index_table``, ``_make_context``, atomic writers, validators) are
reused verbatim and already covered by ``tests/test_build_chronos_bolt_latents.py``.
This file targets the Chronos-2 specific surface:

1. ``_embed_batch_chronos2`` correctly hooks the encoder and pools by
   strategy (``reg`` / ``last`` / ``mean``), including dropping the
   trailing future patch.
2. ``_load_chronos2_pipeline`` forwards revision / dtype / device.
3. ``main`` end-to-end with a fake pipeline writes parquet + manifest with
   the expected fields and doesn't violate validators.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scripts import build_chronos2_latents as latents


class _FakeChronos2Model:
    """Minimal stand-in for ``Chronos2Model``.

    Behaviour we need:
    - ``model.chronos_config.use_reg_token`` flag
    - a ``model.encoder`` ``nn.Module`` that we can attach a forward hook to
    - calling ``model(context=..., num_output_patches=1)`` runs the encoder
      and returns *something* (the real model returns Chronos2Output)
    - the encoder's output's ``[0]`` is the captured hidden state

    We rebuild a hand-rolled ``(B, NCP + use_reg + 1, D)`` fixture per call so
    the test can vary batch size, while keeping NCP / D / use_reg fixed.
    """

    class _Encoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._fixture: torch.Tensor | None = None

        def forward(self, **_kwargs):  # type: ignore[override]
            assert self._fixture is not None, "fixture must be set before encoder forward"
            return (self._fixture,)

    def __init__(self, *, use_reg_token: bool, NCP: int, D: int):
        self.chronos_config = types.SimpleNamespace(use_reg_token=use_reg_token)
        self.device = torch.device("cpu")
        self.encoder = self._Encoder()
        self._NCP = NCP
        self._D = D
        self._use_reg_token = use_reg_token

    def __call__(self, *, context: torch.Tensor, num_output_patches: int = 1):
        # Build a fresh fixture sized to the batch we just received; then
        # invoke encoder.forward so the hook captures it.
        self.encoder._fixture = _make_fixture(
            context.shape[0], self._NCP, self._D, with_reg=self._use_reg_token
        )
        _ = self.encoder()
        return types.SimpleNamespace(quantile_preds=torch.zeros(context.shape[0], 1, 1))


class _FakePipe:
    def __init__(self, model: _FakeChronos2Model):
        self.model = model


def _make_fixture(B: int, NCP: int, D: int, *, with_reg: bool) -> torch.Tensor:
    """Hand-rolled hidden-state tensor with distinguishable values per slot.

    Slot k holds a column of float ``base + k`` so we can verify which slots
    each pool strategy is reading.
    """
    seq_len = NCP + (1 if with_reg else 0) + 1  # +1 future patch
    out = torch.zeros((B, seq_len, D), dtype=torch.float32)
    for k in range(seq_len):
        out[:, k, :] = float(k) + 0.5
    # Make the batch axis distinguishable too.
    for b in range(B):
        out[b] += float(b) * 100.0
    return out


def test_embed_batch_chronos2_pool_reg_returns_reg_token_slot() -> None:
    B, NCP, D = 2, 4, 5
    model = _FakeChronos2Model(use_reg_token=True, NCP=NCP, D=D)
    pipe = _FakePipe(model)
    contexts = np.zeros((B, NCP * 4), dtype=np.float32)

    out = latents._embed_batch_chronos2(pipe, contexts, "reg")

    expected = np.empty((B, D), dtype=np.float32)
    for b in range(B):
        expected[b, :] = float(b) * 100.0 + 4.5  # slot 4 = REG
    np.testing.assert_array_equal(out, expected)


def test_embed_batch_chronos2_pool_last_uses_last_context_patch() -> None:
    B, NCP, D = 2, 4, 5
    model = _FakeChronos2Model(use_reg_token=True, NCP=NCP, D=D)
    pipe = _FakePipe(model)
    contexts = np.zeros((B, NCP * 4), dtype=np.float32)

    out = latents._embed_batch_chronos2(pipe, contexts, "last")

    expected = np.empty((B, D), dtype=np.float32)
    for b in range(B):
        expected[b, :] = float(b) * 100.0 + 3.5  # slot 3 = last ctx patch
    np.testing.assert_array_equal(out, expected)


def test_embed_batch_chronos2_pool_mean_excludes_reg_and_future() -> None:
    B, NCP, D = 2, 4, 5
    model = _FakeChronos2Model(use_reg_token=True, NCP=NCP, D=D)
    pipe = _FakePipe(model)
    contexts = np.zeros((B, NCP * 4), dtype=np.float32)

    out = latents._embed_batch_chronos2(pipe, contexts, "mean")

    # Mean of slots 0..3 = (0.5+1.5+2.5+3.5)/4 = 2.0; per-batch offset added.
    expected = np.empty((B, D), dtype=np.float32)
    for b in range(B):
        expected[b, :] = float(b) * 100.0 + 2.0
    np.testing.assert_array_equal(out, expected)


def test_embed_batch_chronos2_pool_reg_without_reg_token_raises() -> None:
    B, NCP, D = 2, 4, 5
    model = _FakeChronos2Model(use_reg_token=False, NCP=NCP, D=D)
    pipe = _FakePipe(model)
    contexts = np.zeros((B, NCP * 4), dtype=np.float32)

    with pytest.raises(ValueError, match="use_reg_token"):
        latents._embed_batch_chronos2(pipe, contexts, "reg")


def test_embed_batch_chronos2_unknown_pool_raises() -> None:
    B, NCP, D = 1, 4, 3
    pipe = _FakePipe(_FakeChronos2Model(use_reg_token=True, NCP=NCP, D=D))
    contexts = np.zeros((B, NCP * 4), dtype=np.float32)

    with pytest.raises(ValueError, match="unknown pool"):
        latents._embed_batch_chronos2(pipe, contexts, "weird")


def test_load_chronos2_pipeline_forwards_revision_and_dtype(monkeypatch) -> None:
    calls = []

    class FakePipeline:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            calls.append((model_id, kwargs))
            return object()

    monkeypatch.setitem(
        sys.modules,
        "chronos",
        types.SimpleNamespace(BaseChronosPipeline=FakePipeline),
    )

    latents._load_chronos2_pipeline(
        model_id="amazon/chronos-2",
        device="cuda",
        torch_dtype="bfloat16",
        model_revision="abc123",
    )

    assert calls == [
        (
            "amazon/chronos-2",
            {
                "device_map": "cuda",
                "dtype": torch.bfloat16,
                "revision": "abc123",
            },
        )
    ]


def test_main_writes_parquet_and_manifest_with_chronos2_fields(
    tmp_path: Path, monkeypatch
) -> None:
    symbols_file = tmp_path / "symbols.txt"
    symbols_file.write_text("AAPL\n", encoding="utf-8")
    data_root = tmp_path / "data"
    data_root.mkdir()
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC"),
            "close": np.arange(100, 160, dtype=float),
        }
    ).to_csv(data_root / "AAPL.csv", index=False)

    # Fake pipeline returning a model whose encoder produces a fixed
    # hidden-state tensor. We use ctx=4 → 1 patch so layout is [ctx0, REG, future].
    NCP, D = 1, 4

    class _ConfigurableFakePipeline:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs):
            model = _FakeChronos2Model(use_reg_token=True, NCP=NCP, D=D)
            model._commit_hash = "resolved-sha-2"
            return _FakePipe(model)

    monkeypatch.setitem(
        sys.modules,
        "chronos",
        types.SimpleNamespace(BaseChronosPipeline=_ConfigurableFakePipeline),
    )

    seen = []

    def fake_to_parquet(self, path, *, index: bool, compression: str) -> None:
        assert index is False
        assert compression == "zstd"
        Path(path).write_text("fake parquet", encoding="utf-8")
        seen.append(self.copy())

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)
    output = tmp_path / "chronos2_latents.parquet"

    rc = latents.main(
        [
            "--symbols-file",
            str(symbols_file),
            "--data-root",
            str(data_root),
            "--start-date",
            "2024-01-04",
            "--end-date",
            "2024-01-06",
            "--context-length",
            "3",
            "--batch-size",
            "2",
            "--n-components",
            "0",
            "--model-id",
            "fake/chronos-2",
            "--model-revision",
            "requested-sha-2",
            "--device",
            "cpu",
            "--torch-dtype",
            "float32",
            "--pool",
            "reg",
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    manifest_path = output.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_id"] == "fake/chronos-2"
    assert manifest["model_revision"] == "requested-sha-2"
    assert manifest["model_commit_hash"] == "resolved-sha-2"
    assert manifest["pool"] == "reg"
    assert manifest["raw_embed_dim"] == D
    assert manifest["n_components_after_pca"] == D  # n_components=0 keeps raw
    assert manifest["n_rows"] == 3
    assert manifest["pca_components_path"] is None
    # We wrote the parquet (intercepted by fake_to_parquet) — sanity check the frame.
    assert seen
    df = seen[0]
    assert list(df.columns)[:2] == ["symbol", "date"]
    assert df.shape == (3, 2 + D)
