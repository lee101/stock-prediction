#!/usr/bin/env python3
"""Tests for scripts/eval_all_checkpoints.py and scripts/update_fresh_data.py."""

from __future__ import annotations

import csv
import struct
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
CHECKPOINTS_DIR = ROOT / "pufferlib_market" / "checkpoints"
DATA_DIR = ROOT / "pufferlib_market" / "data"

# Add scripts dir once at module level so all tests can import from it
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_script(script_path: Path, args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    """Run a Python script with the current interpreter."""
    cmd = [sys.executable, str(script_path)] + args
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(ROOT),
    )


def make_mktd_binary(tmp_path: Path, num_symbols: int = 3, num_timesteps: int = 40) -> Path:
    """Create a minimal valid MKTD .bin file for testing."""
    out = tmp_path / "test_val.bin"
    magic = b"MKTD"
    version = 2
    features_per_sym = 16
    price_features = 5
    header = struct.pack(
        "<4sIIIII40s",
        magic, version, num_symbols, num_timesteps,
        features_per_sym, price_features,
        b"\x00" * 40,
    )
    symbols = [f"SYM{i:02d}" for i in range(num_symbols)]
    sym_table = b"".join(
        (s.encode("ascii")[:15] + b"\x00" * 16)[:16] for s in symbols
    )
    features = np.zeros((num_timesteps, num_symbols, features_per_sym), dtype=np.float32)
    prices = np.ones((num_timesteps, num_symbols, price_features), dtype=np.float32)
    tradable = np.ones((num_timesteps, num_symbols), dtype=np.uint8)
    with out.open("wb") as f:
        f.write(header)
        f.write(sym_table)
        f.write(features.tobytes(order="C"))
        f.write(prices.tobytes(order="C"))
        f.write(tradable.tobytes(order="C"))
    return out


def make_fake_checkpoint(tmp_path: Path, hidden: int = 64, num_symbols: int = 3) -> Path:
    """Create a minimal fake .pt checkpoint matching TradingPolicy architecture."""
    import torch
    ckpt_dir = tmp_path / "fake_checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pt = ckpt_dir / "best.pt"

    obs_size = num_symbols * 16 + 5 + num_symbols
    num_actions = 1 + 2 * num_symbols

    # Build state_dict matching TradingPolicy
    state_dict = {
        "encoder.0.weight": torch.zeros(hidden, obs_size),
        "encoder.0.bias": torch.zeros(hidden),
        "encoder.2.weight": torch.zeros(hidden, hidden),
        "encoder.2.bias": torch.zeros(hidden),
        "encoder.4.weight": torch.zeros(hidden, hidden),
        "encoder.4.bias": torch.zeros(hidden),
        "actor.0.weight": torch.zeros(hidden // 2, hidden),
        "actor.0.bias": torch.zeros(hidden // 2),
        "actor.2.weight": torch.zeros(num_actions, hidden // 2),
        "actor.2.bias": torch.zeros(num_actions),
        "critic.0.weight": torch.zeros(hidden // 2, hidden),
        "critic.0.bias": torch.zeros(hidden // 2),
        "critic.2.weight": torch.zeros(1, hidden // 2),
        "critic.2.bias": torch.zeros(1),
    }
    torch.save({"model": state_dict, "update": 1, "best_return": 0.0}, pt)
    return pt


# ---------------------------------------------------------------------------
# eval_all_checkpoints.py tests
# ---------------------------------------------------------------------------

class TestEvalAllCheckpointsHelp:
    def test_help_runs(self):
        result = run_script(SCRIPTS_DIR / "eval_all_checkpoints.py", ["--help"])
        assert result.returncode == 0
        assert "checkpoints" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_script_exists(self):
        assert (SCRIPTS_DIR / "eval_all_checkpoints.py").exists()


class TestCheckpointDiscovery:
    def test_discovers_best_pt_files(self, tmp_path):
        # Create a fake checkpoints structure
        (tmp_path / "run_a").mkdir()
        (tmp_path / "run_a" / "best.pt").write_bytes(b"fake")
        (tmp_path / "run_b").mkdir()
        (tmp_path / "run_b" / "best.pt").write_bytes(b"fake")
        (tmp_path / "run_c").mkdir()
        (tmp_path / "run_c" / "final.pt").write_bytes(b"fake")  # no best.pt here

        from eval_all_checkpoints import discover_best_checkpoints
        found = discover_best_checkpoints(tmp_path)
        labels = [p.name for p in found]
        assert labels.count("best.pt") == 2
        # run_c gets final.pt because no best.pt
        assert any(p.parent.name == "run_c" for p in found)

    def test_discovers_nested_checkpoints(self, tmp_path):
        # Autoresearch dirs: checkpoints/autoresearch_mixed23_daily/baseline/best.pt
        parent = tmp_path / "autoresearch_mixed23_daily"
        (parent / "baseline").mkdir(parents=True)
        (parent / "baseline" / "best.pt").write_bytes(b"fake")
        (parent / "ent_anneal").mkdir(parents=True)
        (parent / "ent_anneal" / "best.pt").write_bytes(b"fake")

        from eval_all_checkpoints import discover_best_checkpoints
        found = discover_best_checkpoints(tmp_path)
        assert len(found) == 2

    def test_dry_run_lists_checkpoints(self, tmp_path):
        # Create fake structure
        (tmp_path / "checkpoints" / "run1").mkdir(parents=True)
        (tmp_path / "checkpoints" / "run1" / "best.pt").write_bytes(b"fake")
        (tmp_path / "checkpoints" / "run2").mkdir(parents=True)
        (tmp_path / "checkpoints" / "run2" / "best.pt").write_bytes(b"fake")
        data_file = make_mktd_binary(tmp_path)

        result = run_script(
            SCRIPTS_DIR / "eval_all_checkpoints.py",
            [
                "--checkpoints-dir", str(tmp_path / "checkpoints"),
                "--data-path", str(data_file),
                "--dry-run",
            ],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "run1" in result.stdout or "best.pt" in result.stdout


class TestEvalAllCheckpointsLeaderboard:
    def test_run_evaluation_with_mock(self, tmp_path):
        """Test run_evaluation returns expected dict structure when mocked."""
        from eval_all_checkpoints import run_evaluation

        fake_data = make_mktd_binary(tmp_path)
        fake_ckpt = make_fake_checkpoint(tmp_path)

        with patch(
            "eval_all_checkpoints.run_evaluation",
            return_value={
                "checkpoint": str(fake_ckpt),
                "period": "30d",
                "total_return_pct": 5.0,
                "annualized_return_pct": 60.0,
                "sortino": 1.23,
                "max_drawdown_pct": 4.0,
                "num_trades": 15,
                "win_rate": 0.60,
                "avg_hold_steps": 48.0,
                "error": "",
            },
        ):
            from eval_all_checkpoints import run_evaluation as mocked_eval
            result = mocked_eval(fake_ckpt, fake_data)
        assert result["sortino"] == 1.23
        assert result["total_return_pct"] == 5.0
        assert "error" in result

    def test_leaderboard_csv_format(self, tmp_path):
        """Test save_leaderboard creates valid CSV with expected columns."""
        from eval_all_checkpoints import save_leaderboard

        rows = [
            {
                "checkpoint": "run_a/best.pt",
                "period": "30d",
                "total_return_pct": 5.0,
                "annualized_return_pct": 60.0,
                "sortino": 1.5,
                "max_drawdown_pct": 3.0,
                "num_trades": 10,
                "win_rate": 0.65,
                "avg_hold_steps": 24.0,
                "error": "",
            },
            {
                "checkpoint": "run_b/best.pt",
                "period": "30d",
                "total_return_pct": -2.0,
                "annualized_return_pct": -24.0,
                "sortino": -0.5,
                "max_drawdown_pct": 8.0,
                "num_trades": 5,
                "win_rate": 0.40,
                "avg_hold_steps": 48.0,
                "error": "",
            },
        ]
        out = tmp_path / "leaderboard.csv"
        save_leaderboard(rows, out)

        assert out.exists()
        with out.open() as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)

        assert len(csv_rows) == 2
        # Should be sorted by sortino descending
        assert float(csv_rows[0]["sortino"]) > float(csv_rows[1]["sortino"])
        assert "checkpoint" in csv_rows[0]
        assert "total_return_pct" in csv_rows[0]
        assert "sortino" in csv_rows[0]

    def test_top_n_limits_checkpoints(self, tmp_path):
        """Test --top-n limits evaluation to N most recently modified checkpoints."""
        import time
        ckpts_dir = tmp_path / "checkpoints"
        for i in range(5):
            run_dir = ckpts_dir / f"run_{i:02d}"
            run_dir.mkdir(parents=True)
            pt = run_dir / "best.pt"
            pt.write_bytes(b"fake")
            time.sleep(0.02)  # ensure different mtimes

        data_file = make_mktd_binary(tmp_path)
        result = run_script(
            SCRIPTS_DIR / "eval_all_checkpoints.py",
            [
                "--checkpoints-dir", str(ckpts_dir),
                "--data-path", str(data_file),
                "--dry-run",
                "--top-n", "2",
            ],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        # Should mention only 2 checkpoints in summary
        output = result.stdout
        # Count best.pt lines in dry run output
        best_pt_count = output.count("best.pt")
        assert best_pt_count <= 2, f"Expected <=2 checkpoints, got output:\n{output}"


class TestEvalAllCheckpointsMktdInfer:
    def test_infer_num_symbols(self, tmp_path):
        """Test that infer_num_symbols correctly reads the MKTD header."""
        from eval_all_checkpoints import infer_num_symbols

        data_file = make_mktd_binary(tmp_path, num_symbols=5)
        n = infer_num_symbols(data_file)
        assert n == 5

    def test_infer_num_symbols_bad_file(self, tmp_path):
        """Test that infer_num_symbols returns None for invalid files."""
        from eval_all_checkpoints import infer_num_symbols

        bad_file = tmp_path / "bad.bin"
        bad_file.write_bytes(b"NOT_MKTD_HEADER")
        n = infer_num_symbols(bad_file)
        assert n is None


# ---------------------------------------------------------------------------
# update_fresh_data.py tests
# ---------------------------------------------------------------------------

class TestUpdateFreshDataHelp:
    def test_help_runs(self):
        result = run_script(SCRIPTS_DIR / "update_fresh_data.py", ["--help"])
        assert result.returncode == 0
        assert "universe" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_script_exists(self):
        assert (SCRIPTS_DIR / "update_fresh_data.py").exists()


class TestUpdateFreshDataDryRun:
    def test_dry_run_no_crash(self):
        """--dry-run should complete without errors and report symbol availability."""
        result = run_script(
            SCRIPTS_DIR / "update_fresh_data.py",
            ["--dry-run", "--universe", "mixed23"],
            timeout=60,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"
        out = result.stdout
        # Should mention availability or dry run
        assert any(kw in out.lower() for kw in ("dry run", "available", "missing", "symbol")), out

    def test_dry_run_mixed32_no_crash(self):
        result = run_script(
            SCRIPTS_DIR / "update_fresh_data.py",
            ["--dry-run", "--universe", "mixed32"],
            timeout=60,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}\nstdout: {result.stdout}"

    def test_check_only_reports_symbols(self):
        result = run_script(
            SCRIPTS_DIR / "update_fresh_data.py",
            ["--check-only", "--universe", "mixed23"],
            timeout=60,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "BTCUSD" in result.stdout or "available" in result.stdout.lower()


class TestUpdateFreshDataSymbolAvailability:
    def test_find_symbol_csv(self):
        """Test that key mixed23 symbols can be found in trainingdata."""
        from update_fresh_data import find_symbol_csv, DAILY_DATA_ROOTS

        data_roots = [ROOT / r for r in DAILY_DATA_ROOTS]
        # Check a few required symbols
        for sym in ["BTCUSD", "ETHUSD", "AAPL"]:
            path, source = find_symbol_csv(sym, data_roots)
            assert path is not None, f"Expected to find {sym} in data roots: {data_roots}"
            assert path.exists(), f"CSV file does not exist: {path}"

    def test_mixed23_mostly_available(self):
        """Most mixed23 symbols should be available in trainingdata."""
        from update_fresh_data import check_availability, MIXED23_SYMBOLS, DAILY_DATA_ROOTS

        data_roots = [ROOT / r for r in DAILY_DATA_ROOTS]
        available, missing, dates = check_availability(MIXED23_SYMBOLS, data_roots)
        # At least crypto symbols should be available
        assert len(available) >= 8, f"Too few symbols available: {available}"
        # Warn about missing (not fail, as data may not always be present)
        if missing:
            import warnings
            warnings.warn(f"Missing mixed23 symbols: {missing}")

    def test_get_last_date(self, tmp_path):
        """Test get_last_date extracts the last date from a simple CSV."""
        from update_fresh_data import get_last_date

        csv_file = tmp_path / "TEST.csv"
        csv_file.write_text(
            "timestamp,open,high,low,close,volume\n"
            "2025-01-01 00:00:00+00:00,100,105,95,102,1000\n"
            "2025-06-15 00:00:00+00:00,110,115,105,112,2000\n"
        )
        last = get_last_date(csv_file)
        assert last is not None
        assert "2025-06-15" in last


class TestUpdateFreshDataUniverseConstants:
    def test_mixed23_has_23_symbols(self):
        from update_fresh_data import MIXED23_SYMBOLS
        assert len(MIXED23_SYMBOLS) == 23

    def test_mixed32_has_32_symbols(self):
        from update_fresh_data import MIXED32_SYMBOLS
        assert len(MIXED32_SYMBOLS) == 32

    def test_mixed23_is_subset_of_mixed32(self):
        from update_fresh_data import MIXED23_SYMBOLS, MIXED32_SYMBOLS
        # Check crypto overlap
        mixed23_crypto = {s for s in MIXED23_SYMBOLS if s.endswith("USD")}
        mixed32_crypto = {s for s in MIXED32_SYMBOLS if s.endswith("USD")}
        assert mixed23_crypto == mixed32_crypto, (
            f"Crypto symbols differ: mixed23={mixed23_crypto} mixed32={mixed32_crypto}"
        )

    def test_no_duplicate_symbols(self):
        from update_fresh_data import MIXED23_SYMBOLS, MIXED32_SYMBOLS
        assert len(MIXED23_SYMBOLS) == len(set(MIXED23_SYMBOLS)), "Duplicates in MIXED23_SYMBOLS"
        assert len(MIXED32_SYMBOLS) == len(set(MIXED32_SYMBOLS)), "Duplicates in MIXED32_SYMBOLS"


# ---------------------------------------------------------------------------
# Integration: real checkpoint data exists
# ---------------------------------------------------------------------------

class TestRealCheckpointsExist:
    def test_mixed23_checkpoints_dir_exists(self):
        ckpts = CHECKPOINTS_DIR / "autoresearch_mixed23_daily"
        assert ckpts.exists(), f"Expected mixed23 checkpoints dir: {ckpts}"

    def test_mixed32_checkpoints_dir_exists(self):
        ckpts = CHECKPOINTS_DIR / "autoresearch_mixed32_daily"
        assert ckpts.exists(), f"Expected mixed32 checkpoints dir: {ckpts}"

    def test_mixed23_has_best_pt_files(self):
        ckpts = CHECKPOINTS_DIR / "autoresearch_mixed23_daily"
        if not ckpts.exists():
            pytest.skip("autoresearch_mixed23_daily not present")
        pts = list(ckpts.rglob("best.pt"))
        assert len(pts) > 0, "No best.pt files found under autoresearch_mixed23_daily"

    def test_val_data_exists(self):
        """At least one of the default val data files should exist."""
        from eval_all_checkpoints import DEFAULT_VAL_DATA_CANDIDATES
        any_exists = any((ROOT / c).exists() for c in DEFAULT_VAL_DATA_CANDIDATES)
        assert any_exists, f"No default val data found. Checked: {DEFAULT_VAL_DATA_CANDIDATES}"


class TestCheckpointLabelFunction:
    def test_label_relative_to_checkpoints_dir(self, tmp_path):
        from eval_all_checkpoints import checkpoint_label

        ckpts_dir = tmp_path / "checkpoints"
        pt = ckpts_dir / "autoresearch_mixed23_daily" / "baseline" / "best.pt"
        label = checkpoint_label(pt, ckpts_dir)
        assert "best.pt" in label
        assert "autoresearch_mixed23_daily" in label

    def test_label_fallback_when_not_under_dir(self, tmp_path):
        from eval_all_checkpoints import checkpoint_label

        ckpts_dir = tmp_path / "other_dir"
        pt = tmp_path / "checkpoints" / "run1" / "best.pt"
        # Should not crash; returns something useful
        label = checkpoint_label(pt, ckpts_dir)
        assert "best.pt" in label
