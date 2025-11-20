import torch

from neuraldailytraining.trainer import apply_symbol_dropout


def test_apply_symbol_dropout_masks_features_and_group():
    torch.manual_seed(7)
    features = torch.ones(3, 2, 1)
    group_mask = torch.tensor(
        [
            [True, True, False],
            [True, True, False],
            [False, False, True],
        ]
    )

    out_feats, out_group, drop_mask = apply_symbol_dropout(features, group_mask, 0.5, training=True)

    # Dropped symbols are zeroed and disconnected in group mask
    if drop_mask.any():
        assert torch.all(out_feats[drop_mask] == 0)
        keep = ~drop_mask
        # Ensure no connections to dropped indices
        assert torch.all(out_group[drop_mask] == False)
        assert torch.all(out_group[:, drop_mask] == False)
        # At least one kept entry should retain membership with itself
        kept_idx = int(torch.nonzero(keep, as_tuple=False)[0])
        assert out_group[kept_idx, kept_idx]


def test_apply_symbol_dropout_noop_when_disabled():
    features = torch.randn(2, 3, 4)
    group_mask = torch.ones(2, 2, dtype=torch.bool)
    out_features, out_group, drop_mask = apply_symbol_dropout(features, group_mask, 0.0, training=True)
    assert torch.equal(features, out_features)
    assert torch.equal(group_mask, out_group)
    assert drop_mask.any().item() is False
