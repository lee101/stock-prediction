import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from tototraining.toto_trainer import TotoTrainer
from tototraining.toto_ohlc_dataloader import TotoBatchSample
from toto.data.util.dataset import MaskedTimeseries


def _make_masked_timeseries(batch: int = 2, variates: int = 1, seq_len: int = 4) -> MaskedTimeseries:
    series = torch.arange(batch * variates * seq_len, dtype=torch.float32).view(batch, variates, seq_len)
    padding_mask = torch.ones_like(series, dtype=torch.bool)
    id_mask = torch.zeros_like(series, dtype=torch.long)
    timestamps = torch.arange(seq_len, dtype=torch.long).repeat(batch, variates, 1)
    intervals = torch.ones(batch, variates, dtype=torch.long)
    return MaskedTimeseries(series=series, padding_mask=padding_mask, id_mask=id_mask, timestamp_seconds=timestamps, time_interval_seconds=intervals)


def test_prepare_batch_preserves_toto_batch_sample_targets():
    trainer = object.__new__(TotoTrainer)
    device = torch.device("cpu")

    masked = _make_masked_timeseries(batch=2, variates=3, seq_len=6)
    target_price = torch.full((2, 3), 42.0)
    target_pct = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=torch.float32)
    prev_close = torch.full((2, 1), 21.0)
    batch = TotoBatchSample(timeseries=masked, target_price=target_price, prev_close=prev_close, target_pct=target_pct)

    (
        series,
        padding_mask,
        id_mask,
        prepared_target_price,
        prepared_target_pct,
        prepared_prev_close,
        metadata,
    ) = trainer._prepare_batch(batch, device)

    assert torch.equal(series, masked.series)
    assert torch.equal(padding_mask, masked.padding_mask)
    assert torch.equal(id_mask, masked.id_mask)
    assert torch.equal(prepared_target_price, target_price)
    assert torch.equal(prepared_target_pct, target_pct)
    assert torch.equal(prepared_prev_close, prev_close)
    assert metadata == {}
