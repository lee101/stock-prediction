"""Tests for wide augmented binary builder and per-sym-norm."""
from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


# ─── vol-scale augmentation ──────────────────────────────────────────────────

def test_vol_scale_noop():
    """σ=1.0 must return identical data."""
    from scripts.build_wide_augmented import apply_vol_scale
    feat = np.random.randn(50, 5, 16).astype(np.float32)
    out = apply_vol_scale(feat, 1.0)
    np.testing.assert_array_equal(out, feat)


def test_vol_scale_return1d_clipped():
    """Scaled return_1d must not exceed its clip bound of 0.5."""
    from scripts.build_wide_augmented import apply_vol_scale
    feat = np.full((10, 3, 16), 0.45, dtype=np.float32)  # near clip
    out = apply_vol_scale(feat, 1.5)
    assert out[:, :, 0].max() <= 0.5 + 1e-5, "return_1d clip violated"


def test_vol_scale_rsi_unchanged():
    """Feature 10 (RSI) must never be scaled."""
    from scripts.build_wide_augmented import apply_vol_scale
    feat = np.random.randn(20, 4, 16).astype(np.float32)
    out = apply_vol_scale(feat, 2.0)
    np.testing.assert_array_equal(out[:, :, 10], feat[:, :, 10])


def test_vol_scale_volume_unchanged():
    """Features 14-15 (volume z-score, volume delta) must not be scaled."""
    from scripts.build_wide_augmented import apply_vol_scale
    feat = np.random.randn(20, 4, 16).astype(np.float32)
    out = apply_vol_scale(feat, 2.0)
    np.testing.assert_array_equal(out[:, :, 14], feat[:, :, 14])
    np.testing.assert_array_equal(out[:, :, 15], feat[:, :, 15])


def test_vol_scale_amplifies_returns():
    """σ>1 must amplify return features when not at clip boundary."""
    from scripts.build_wide_augmented import apply_vol_scale
    feat = np.zeros((5, 2, 16), dtype=np.float32)
    feat[:, :, 0] = 0.1   # return_1d
    feat[:, :, 1] = 0.2   # return_5d
    out = apply_vol_scale(feat, 1.3)
    np.testing.assert_allclose(out[:, :, 0], 0.13, atol=1e-5)
    np.testing.assert_allclose(out[:, :, 1], 0.26, atol=1e-5)


def test_vol_scale_attenuates_returns():
    """σ<1 must attenuate return features."""
    from scripts.build_wide_augmented import apply_vol_scale
    feat = np.zeros((5, 2, 16), dtype=np.float32)
    feat[:, :, 0] = 0.2
    out = apply_vol_scale(feat, 0.7)
    np.testing.assert_allclose(out[:, :, 0], 0.14, atol=1e-5)


# ─── Per-sym-norm in TradingPolicy ───────────────────────────────────────────

@pytest.fixture
def policy_psn():
    """TradingPolicy with per_sym_norm=True, 17 symbols, 16 feats."""
    from pufferlib_market.train import TradingPolicy
    obs_size = 17 * 16 + 5 + 17  # 294
    num_actions = 1 + 2 * 17     # 35
    return TradingPolicy(obs_size, num_actions, hidden=128,
                         per_sym_norm=True, features_per_sym=16)


def test_per_sym_norm_forward(policy_psn):
    """Per-sym-norm policy produces correct output shapes."""
    x = torch.randn(4, policy_psn.obs_size)
    logits, value = policy_psn(x)
    assert logits.shape == (4, policy_psn.num_actions)
    assert value.shape == (4,)


def test_per_sym_norm_has_layer(policy_psn):
    assert hasattr(policy_psn, "sym_input_norm"), "sym_input_norm layer missing"
    assert policy_psn._per_sym_norm is True


def test_per_sym_norm_normalizes():
    """Per-sym-norm: each symbol's feature vector should be roughly zero-mean unit-var."""
    from pufferlib_market.train import TradingPolicy
    obs_size = 3 * 16 + 5 + 3  # tiny: 3 symbols
    policy = TradingPolicy(obs_size, 7, hidden=64, per_sym_norm=True, features_per_sym=16)

    # Construct obs where each symbol has very different scales.
    B = 8
    x = torch.zeros(B, obs_size)
    x[:, :16] = torch.randn(B, 16) * 5.0    # sym0: high variance
    x[:, 16:32] = torch.randn(B, 16) * 0.1  # sym1: low variance
    x[:, 32:48] = torch.randn(B, 16)         # sym2: normal

    # Apply just the sym_input_norm manually.
    sym_part = x[:, :3 * 16].view(B, 3, 16)
    normed = policy.sym_input_norm(sym_part)
    # After norm, each (batch_item, symbol) pair should have ~0 mean over 16 feats.
    means = normed.detach().numpy().mean(axis=2)  # (B, 3)
    assert np.abs(means).max() < 0.5, f"Means too large after LayerNorm: {means.max()}"


def test_per_sym_norm_backward_compat():
    """Policy without per_sym_norm works normally (no sym_input_norm)."""
    from pufferlib_market.train import TradingPolicy
    obs_size = 17 * 16 + 5 + 17
    policy = TradingPolicy(obs_size, 35, hidden=128)
    assert not policy._per_sym_norm
    assert not hasattr(policy, "sym_input_norm")
    x = torch.randn(4, obs_size)
    logits, value = policy(x)
    assert logits.shape == (4, 35)


def test_per_sym_norm_checkpoint_metadata():
    """per_sym_norm flag is saved to checkpoint payload."""
    from pufferlib_market.train import TradingPolicy, _checkpoint_payload
    import torch.optim as optim

    obs_size = 17 * 16 + 5 + 17
    policy = TradingPolicy(obs_size, 35, hidden=128, per_sym_norm=True, features_per_sym=16)
    optimizer = optim.Adam(policy.parameters())
    payload = _checkpoint_payload(
        policy, optimizer,
        update=1, global_step=1000,
        best_return=0.1, disable_shorts=True,
        action_meta={"action_allocation_bins": 1, "action_level_bins": 1},
    )
    assert payload.get("per_sym_norm") is True, "per_sym_norm missing from payload"
    assert payload.get("features_per_sym") == 16


def test_per_sym_norm_evaluate_holdout_load(tmp_path):
    """evaluate_holdout.TradingPolicy supports per_sym_norm."""
    from pufferlib_market.evaluate_holdout import TradingPolicy as EvalPolicy

    obs_size = 17 * 16 + 5 + 17
    policy = EvalPolicy(obs_size, 35, hidden=128, per_sym_norm=True, features_per_sym=16)
    assert policy._per_sym_norm is True
    x = torch.randn(2, obs_size)
    logits, value = policy(x)
    assert logits.shape == (2, 35)


# ─── MKTD read/write helpers ─────────────────────────────────────────────────

def _make_dummy_mktd(path: Path, nts: int = 50, nsym: int = 5, nfeat: int = 16):
    """Write a minimal valid MKTD binary."""
    HEADER_SIZE = 64
    feat = np.random.randn(nts, nsym, nfeat).astype(np.float32)
    price = np.random.randn(nts, nsym, 5).astype(np.float32)
    mask = np.ones((nts, nsym), dtype=np.uint8)
    hdr = struct.pack("<4sIIIII40s", b"MKTD", 2, nsym, nts, nfeat, 5, b"\x00" * 40)
    with open(path, "wb") as f:
        f.write(hdr)
        for i in range(nsym):
            name = f"SYM{i}".encode()
            f.write(name + b"\x00" * (16 - len(name)))
        f.write(feat.tobytes())
        f.write(price.tobytes())
        f.write(mask.tobytes())
    return feat, price, mask


def test_read_write_roundtrip(tmp_path):
    """_read_mktd / _write_mktd roundtrip."""
    from scripts.build_wide_augmented import _read_mktd, _write_mktd
    p_in = tmp_path / "test.bin"
    feat_orig, price_orig, mask_orig = _make_dummy_mktd(p_in)
    rec = _read_mktd(p_in)
    p_out = tmp_path / "out.bin"
    _write_mktd(p_out, rec, rec["feat"], rec["price"], rec["mask"])
    rec2 = _read_mktd(p_out)
    np.testing.assert_array_equal(rec2["feat"], rec["feat"])
    np.testing.assert_array_equal(rec2["price"], rec["price"])
    np.testing.assert_array_equal(rec2["mask"], rec["mask"])


def test_concat_mktd_list(tmp_path):
    """concat_mktd_list stacks timesteps correctly."""
    from scripts.build_wide_augmented import _read_mktd, concat_mktd_list
    p1 = tmp_path / "a.bin"
    p2 = tmp_path / "b.bin"
    _make_dummy_mktd(p1, nts=30, nsym=4)
    _make_dummy_mktd(p2, nts=20, nsym=4)
    r1 = _read_mktd(p1)
    r2 = _read_mktd(p2)
    merged = concat_mktd_list([r1, r2])
    assert merged["nts"] == 50
    assert merged["feat"].shape == (50, 4, 16)
