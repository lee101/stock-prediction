from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from newnanoalpacahourlyexp import run_multiphase_crossasset


class _DummyData:
    feature_columns = ("f1", "f2", "f3")


def test_parse_symbol_list_normalizes_and_deduplicates() -> None:
    assert run_multiphase_crossasset.parse_symbol_list(" ethusd, AAPL ; ethusd\nmsft ,, all ") == [
        "ETHUSD",
        "AAPL",
        "MSFT",
    ]


def test_discover_crossasset_symbols_combines_roots(tmp_path) -> None:
    crypto_root = tmp_path / "crypto"
    stock_root = tmp_path / "stocks"
    crypto_root.mkdir()
    stock_root.mkdir()
    (crypto_root / "ethusd.csv").write_text("timestamp,open,high,low,close,volume\n")
    (stock_root / "aapl.csv").write_text("timestamp,open,high,low,close,volume\n")
    (stock_root / "ethusd.csv").write_text("timestamp,open,high,low,close,volume\n")

    symbols = run_multiphase_crossasset.discover_crossasset_symbols(
        crypto_root=crypto_root,
        stock_root=stock_root,
    )

    assert symbols == ["ETHUSD", "AAPL"]


def test_filter_usable_symbols_drops_invalid_entries(monkeypatch) -> None:
    def fake_data_module(cfg):
        if cfg.symbol == "AAPL":
            raise ValueError("Insufficient hourly history")
        return _DummyData()

    monkeypatch.setattr(run_multiphase_crossasset, "AlpacaHourlyDataModule", fake_data_module)
    usable, dropped = run_multiphase_crossasset.filter_usable_symbols(
        ["ETHUSD", "AAPL", "BTCUSD"],
        run_multiphase_crossasset.DatasetConfig(symbol="ETHUSD"),
    )

    assert usable == ["ETHUSD", "BTCUSD"]
    assert dropped == {"AAPL": "Insufficient hourly history"}


def test_main_writes_manifest_for_each_finetune_symbol(tmp_path, monkeypatch) -> None:
    crypto_root = tmp_path / "crypto"
    stock_root = tmp_path / "stocks"
    crypto_root.mkdir()
    stock_root.mkdir()
    for symbol in ("ETHUSD", "BTCUSD"):
        (crypto_root / f"{symbol}.csv").write_text("timestamp,open,high,low,close,volume\n")
    (stock_root / "AAPL.csv").write_text("timestamp,open,high,low,close,volume\n")

    def fake_build_data_module(symbols, dataset_cfg):
        return _DummyData()

    def fake_train_model(data, args, *, device):
        checkpoint = tmp_path / f"{args.run_name}.pt"
        checkpoint.write_text("checkpoint")
        history = [SimpleNamespace(epoch=1, train_loss=0.1, train_score=0.2, train_sortino=0.3, train_return=0.4)]
        return SimpleNamespace(best_checkpoint=checkpoint, history=history)

    def fake_load_model(checkpoint_path: Path, input_dim: int, sequence_length: int):
        return object()

    def fake_evaluate_model(**kwargs):
        output_dir = kwargs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text("{}")
        return SimpleNamespace(total_return=0.12, sortino=1.5)

    monkeypatch.setattr(run_multiphase_crossasset, "_build_data_module", fake_build_data_module)
    monkeypatch.setattr(run_multiphase_crossasset, "train_model", fake_train_model)
    monkeypatch.setattr(run_multiphase_crossasset, "_load_model", fake_load_model)
    monkeypatch.setattr(run_multiphase_crossasset, "evaluate_model", fake_evaluate_model)
    monkeypatch.setattr(run_multiphase_crossasset, "_resolve_device", lambda device_arg, *, symbol: torch.device("cpu"))
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_multiphase_crossasset.py",
            "--output-dir",
            str(tmp_path / "outputs"),
            "--crypto-data-root",
            str(crypto_root),
            "--stock-data-root",
            str(stock_root),
            "--finetune-symbols",
            "ETHUSD,BTCUSD",
            "--epochs-pretrain",
            "1",
            "--epochs-finetune",
            "1",
            "--dry-train-steps-pretrain",
            "1",
            "--dry-train-steps-finetune",
            "1",
            "--no-use-compile",
            "--no-drop-unusable-symbols",
        ],
    )

    run_multiphase_crossasset.main()

    manifest = json.loads((tmp_path / "outputs" / "manifest.json").read_text())
    assert manifest["pretrain_symbols"] == ["BTCUSD", "ETHUSD", "AAPL"]
    assert manifest["finetune_symbols"] == ["ETHUSD", "BTCUSD"]
    assert len(manifest["phases"]) == 2
    assert {entry["symbol"] for entry in manifest["phases"]} == {"ETHUSD", "BTCUSD"}
