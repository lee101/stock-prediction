import json

from marketsimulator import run_trade_loop


def test_resolve_simulation_symbols_prefers_cli_list():
    result = run_trade_loop._resolve_simulation_symbols(["AAPL", "MSFT"])
    assert result == ["AAPL", "MSFT"]


def test_resolve_simulation_symbols_reads_experiment_file(tmp_path):
    data = {"symbols": ["spy", "ethusd", "spy"]}
    results_path = tmp_path / "sizing.json"
    results_path.write_text(json.dumps(data), encoding="utf-8")

    result = run_trade_loop._resolve_simulation_symbols(
        None,
        fast_results_path=results_path,
    )

    assert result == ["SPY", "ETHUSD"]


def test_resolve_simulation_symbols_falls_back(tmp_path):
    missing = tmp_path / "missing.json"
    result = run_trade_loop._resolve_simulation_symbols(
        None,
        fast_results_path=missing,
    )

    assert result == run_trade_loop.DEFAULT_SIM_SYMBOLS
