"""Tests for the crypto30 daily data pipeline."""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def read_mktd(path):
    with open(path, "rb") as f:
        hdr = f.read(64)
        magic, ver, nsym, nts, nfeat, nprice = struct.unpack_from("<4sIIIII", hdr)
        sym_table = f.read(nsym * 16)
        syms = [sym_table[i * 16 : (i + 1) * 16].rstrip(b"\x00").decode() for i in range(nsym)]
        feat = np.fromfile(f, dtype=np.float32, count=nts * nsym * nfeat).reshape(nts, nsym, nfeat)
        price = np.fromfile(f, dtype=np.float32, count=nts * nsym * nprice).reshape(nts, nsym, nprice)
        mask = np.frombuffer(f.read(), dtype=np.uint8).reshape(nts, nsym)
    return dict(syms=syms, feat=feat, price=price, mask=mask, nsym=nsym, nts=nts, nfeat=nfeat)


TRAIN_BIN = REPO / "pufferlib_market/data/crypto30_daily_aug_train.bin"
VAL_BIN = REPO / "pufferlib_market/data/crypto30_daily_val.bin"


@pytest.mark.skipif(not TRAIN_BIN.exists(), reason="train binary not built")
class TestTrainBinary:
    def test_shape(self):
        d = read_mktd(TRAIN_BIN)
        assert d["nsym"] == 30
        assert d["nts"] == 6300  # 2100 days x 3 vol-scale
        assert d["nfeat"] == 16

    def test_btc_first(self):
        d = read_mktd(TRAIN_BIN)
        assert d["syms"][0] == "BTCUSDT"

    def test_features_bounded(self):
        d = read_mktd(TRAIN_BIN)
        assert d["feat"][:, :, 0].min() >= -0.5  # return_1d
        assert d["feat"][:, :, 0].max() <= 0.5

    def test_no_nan(self):
        d = read_mktd(TRAIN_BIN)
        assert not np.isnan(d["feat"]).any()
        assert not np.isnan(d["price"]).any()

    def test_tradable_mask_nonzero(self):
        d = read_mktd(TRAIN_BIN)
        # BTC should be tradable on all days
        btc_idx = d["syms"].index("BTCUSDT")
        assert d["mask"][:, btc_idx].sum() == d["nts"]

    def test_vol_scale_augmentation(self):
        d = read_mktd(TRAIN_BIN)
        # With 3x vol-scale, timesteps should be 3x base
        assert d["nts"] % 3 == 0
        base_ts = d["nts"] // 3
        # sigma=1.0 slice (middle third) should have same prices as sigma=0.7 (first third)
        np.testing.assert_array_equal(
            d["price"][:base_ts], d["price"][base_ts : 2 * base_ts]
        )


@pytest.mark.skipif(not VAL_BIN.exists(), reason="val binary not built")
class TestValBinary:
    def test_shape(self):
        d = read_mktd(VAL_BIN)
        assert d["nsym"] == 30
        assert d["nts"] == 192
        assert d["nfeat"] == 16

    def test_matic_not_tradable(self):
        d = read_mktd(VAL_BIN)
        matic_idx = d["syms"].index("MATICUSDT")
        assert d["mask"][:, matic_idx].sum() == 0
