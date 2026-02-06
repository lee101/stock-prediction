from __future__ import annotations

import json
from pathlib import Path

from scripts.retrain_chronos2_hourly_loras import update_hourly_hparams


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def test_update_hourly_hparams_uses_proxy_template(tmp_path: Path) -> None:
    # Template for BTCUSD exists; BTCFDUSD should reuse it (proxy -> BTCUSD).
    template = {
        "symbol": "BTCUSD",
        "model": "chronos2",
        "config": {
            "name": "hourly_ctx512_skip1_single",
            "model_id": "old-model",
            "context_length": 512,
            "batch_size": 32,
            "quantile_levels": [0.1, 0.5, 0.9],
            "skip_rates": [1],
            "aggregation_method": "single",
            "use_multivariate": False,
        },
        "metadata": {"source": "template", "generated_at": "2020-01-01T00:00:00Z", "frequency": "hourly"},
    }
    (tmp_path / "BTCUSD.json").write_text(json.dumps(template))

    out_path = update_hourly_hparams(
        symbol="BTCFDUSD",
        finetuned_model_id="new-model",
        hyperparam_dir=tmp_path,
    )

    assert out_path == tmp_path / "BTCFDUSD.json"
    payload = _read(out_path)
    assert payload["symbol"] == "BTCFDUSD"
    assert payload["config"]["model_id"] == "new-model"
    assert payload["metadata"]["source"] == "retrain_chronos2_hourly_loras"
    assert payload["metadata"]["finetuned_model_id"] == "new-model"


def test_update_hourly_hparams_falls_back_to_base_config(tmp_path: Path) -> None:
    out_path = update_hourly_hparams(
        symbol="BNBFDUSD",
        finetuned_model_id="checkpoint-path",
        hyperparam_dir=tmp_path,
    )
    payload = _read(out_path)
    assert payload["symbol"] == "BNBFDUSD"
    assert payload["model"] == "chronos2"
    assert payload["config"]["model_id"] == "checkpoint-path"
    assert payload["config"]["context_length"] == 512
    assert payload["config"]["skip_rates"] == [1]

