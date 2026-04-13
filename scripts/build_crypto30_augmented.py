#!/usr/bin/env python3
"""Build vol-scale augmented crypto30 daily training binary.

No session-shift (crypto is 24/7). Only volatility scaling: 0.7, 1.0, 1.3.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

VOL_SCALES = [0.7, 1.0, 1.3]

# Feature indices to scale (return/vol/momentum ratios, not RSI/volume).
SCALE_FEAT_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13]
FEAT_CLIPS = {
    0: (-0.5,  0.5), 1: (-1.0,  1.0), 2: (-2.0,  2.0),
    3: ( 0.0,  1.0), 4: ( 0.0,  1.0), 5: (-0.5,  0.5),
    6: (-0.5,  0.5), 7: (-0.5,  0.5), 8: ( 0.0,  0.5),
    9: ( 0.0,  0.5), 11: (-3.0, 3.0), 12: (-1.0, 0.0), 13: (-1.0, 0.0),
}

HEADER_SIZE = 64
SYM_NAME_LEN = 16

IN_TRAIN = Path("pufferlib_market/data/crypto30_daily_train.bin")
OUT_TRAIN = Path("pufferlib_market/data/crypto30_daily_aug_train.bin")


def read_mktd(path: Path):
    with open(path, "rb") as f:
        hdr_raw = f.read(HEADER_SIZE)
        magic, version, nsym, nts, nfeat, nprice = struct.unpack_from("<4sIIIII", hdr_raw)
        sym_table = f.read(nsym * SYM_NAME_LEN)
        feat = np.fromfile(f, dtype=np.float32, count=nts * nsym * nfeat).reshape(nts, nsym, nfeat)
        price = np.fromfile(f, dtype=np.float32, count=nts * nsym * nprice).reshape(nts, nsym, nprice)
        mask_raw = f.read()
        mask = np.frombuffer(mask_raw, dtype=np.uint8).reshape(nts, nsym) if mask_raw else None
    return dict(hdr=hdr_raw, sym_table=sym_table, feat=feat, price=price, mask=mask,
                nsym=nsym, nts=nts, nfeat=nfeat, nprice=nprice, version=version, magic=magic)


def write_mktd(path: Path, ref: dict, feat: np.ndarray, price: np.ndarray, mask):
    nts = feat.shape[0]
    new_hdr = bytearray(ref["hdr"])
    struct.pack_into("<I", new_hdr, 12, nts)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(bytes(new_hdr))
        f.write(ref["sym_table"])
        f.write(feat.astype(np.float32).tobytes(order="C"))
        f.write(price.astype(np.float32).tobytes(order="C"))
        if mask is not None:
            f.write(mask.astype(np.uint8).tobytes(order="C"))
    print(f"  Wrote {path} ({nts} timesteps, {ref['nsym']} symbols, {ref['nfeat']} feats)")


def apply_vol_scale(feat: np.ndarray, sigma: float) -> np.ndarray:
    if abs(sigma - 1.0) < 1e-6:
        return feat
    out = feat.copy()
    for fi in SCALE_FEAT_IDX:
        out[:, :, fi] *= sigma
        if fi in FEAT_CLIPS:
            lo, hi = FEAT_CLIPS[fi]
            out[:, :, fi] = out[:, :, fi].clip(lo, hi)
    return out


def main():
    if not IN_TRAIN.exists():
        print(f"ERROR: {IN_TRAIN} not found. Run export_crypto30_daily.py first.")
        return 1

    base = read_mktd(IN_TRAIN)
    print(f"Base: {base['nsym']} symbols, {base['nts']} timesteps, {base['nfeat']} features")

    all_feat = []
    all_price = []
    all_mask = []
    for sigma in VOL_SCALES:
        scaled = apply_vol_scale(base["feat"], sigma)
        all_feat.append(scaled)
        all_price.append(base["price"])
        if base["mask"] is not None:
            all_mask.append(base["mask"])
        print(f"  sigma={sigma}: {scaled.shape[0]} timesteps")

    combined_feat = np.concatenate(all_feat, axis=0)
    combined_price = np.concatenate(all_price, axis=0)
    combined_mask = np.concatenate(all_mask, axis=0) if all_mask else None
    print(f"Combined: {combined_feat.shape[0]} timesteps ({len(VOL_SCALES)}x augmentation)")

    write_mktd(OUT_TRAIN, base, combined_feat, combined_price, combined_mask)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
