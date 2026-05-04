from __future__ import annotations

import json
from pathlib import Path

import stock_dashboard.server as dash


def test_safe_repo_path_blocks_path_escape() -> None:
    assert dash.safe_repo_path("../../etc/passwd") is None


def test_artifact_summary_reads_gate_and_trade_count(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    artifact = repo / "analysis/bitbankgo_stock_bridge/sample.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps(
            {
                "strategy": "demo",
                "passes_27pct_gate": False,
                "worst_slippage_cell": {"median_monthly_return": -0.1},
                "selected_configs": [{"horizon": 2}],
                "candidate_trades": [{"symbol": "AAA"}],
                "split_rows": {"train": 10},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(dash, "REPO", repo)

    summary = dash.artifact_summary(artifact)

    assert summary is not None
    assert summary["path"] == "analysis/bitbankgo_stock_bridge/sample.json"
    assert summary["strategy"] == "demo"
    assert summary["candidate_trade_count"] == 1
    assert summary["selected_count"] == 1


def test_symbol_payload_filters_trades_and_loads_bars(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    bars = repo / "trainingdatahourly/stocks/AAA.csv"
    bars.parent.mkdir(parents=True)
    bars.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2026-01-01T14:30:00Z,10,11,9,10.5,1000\n",
        encoding="utf-8",
    )
    artifact = repo / "analysis/result.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps(
            {
                "candidate_trades": [
                    {"symbol": "AAA", "side": 1, "entry_ts": "t1"},
                    {"symbol": "BBB", "side": -1, "entry_ts": "t2"},
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(dash, "REPO", repo)

    payload = dash.symbol_payload("AAA", "analysis/result.json")

    assert payload["symbol"] == "AAA"
    assert len(payload["bars"]) == 1
    assert payload["bars"][0]["close"] == 10.5
    assert payload["trades"] == [{"symbol": "AAA", "side": 1, "entry_ts": "t1"}]


def test_portfolio_payload_builds_multi_symbol_replay(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    artifact = repo / "analysis/result.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps(
            {
                "strategy": "bridge",
                "args": {
                    "max_positions": 2,
                    "leverage": 2.0,
                    "fee_rate": 0.001,
                    "fill_buffer_bps": 5.0,
                    "selection_slippage_bps": 20.0,
                },
                "candidate_trades": [
                    {
                        "symbol": "AAA",
                        "side": 1,
                        "entry_ts": "2026-01-01T14:30:00+00:00",
                        "exit_ts": "2026-01-01T15:30:00+00:00",
                        "entry_open": 100.0,
                        "exit_close": 110.0,
                    },
                    {
                        "symbol": "BBB",
                        "side": -1,
                        "entry_ts": "2026-01-01T15:30:00+00:00",
                        "exit_ts": "2026-01-01T16:30:00+00:00",
                        "entry_open": 100.0,
                        "exit_close": 90.0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(dash, "REPO", repo)

    payload = dash.portfolio_payload("analysis/result.json")

    assert payload["strategy"] == "bridge"
    assert [row["symbol"] for row in payload["symbols"]] == ["AAA", "BBB"]
    assert len(payload["trades"]) == 2
    assert payload["trades"][0]["return_pct"] == 0.1
    assert payload["trades"][1]["return_pct"] == 0.1
    assert payload["times"] == [
        "2026-01-01T14:30:00+00:00",
        "2026-01-01T15:30:00+00:00",
        "2026-01-01T16:30:00+00:00",
    ]
    assert payload["equity_points"][-1]["equity"] > 1.0
