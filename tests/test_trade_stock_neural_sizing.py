import json

from trade_stock_e2e import _load_strategytraining_symbols


def test_load_strategytraining_symbols_deduplicates_and_normalizes(tmp_path):
    payload = {
        "symbols": ["aapl", "ETHusd", " ETHusd ", "", None, 123],
    }
    results_path = tmp_path / "fast_results.json"
    results_path.write_text(json.dumps(payload), encoding="utf-8")

    symbols = _load_strategytraining_symbols(results_path)

    assert symbols == ["AAPL", "ETHUSD"]


def test_load_strategytraining_symbols_missing_or_invalid(tmp_path):
    missing_path = tmp_path / "missing.json"
    assert _load_strategytraining_symbols(missing_path) == []

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("{not-json}", encoding="utf-8")
    assert _load_strategytraining_symbols(invalid_path) == []
