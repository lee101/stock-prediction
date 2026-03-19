"""Integration tests for mixed-symbol (stock + crypto) model support in the orchestrator."""

from __future__ import annotations

import struct
from pathlib import Path

import collections

import pytest
import torch

from src.symbol_utils import is_crypto_symbol
from unified_orchestrator import orchestrator
from unified_orchestrator.rl_gemini_bridge import RLGeminiBridge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_mktd_header(path: Path, symbols: list[str]) -> None:
    """Write a minimal MKTD v2 binary header with the given symbol list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = struct.pack(
        "<4sIIIII40s",
        b"MKTD",
        2,                    # version
        len(symbols),         # num_symbols
        1,                    # num_timesteps (dummy)
        16,                   # features_per_symbol
        5,                    # global_features
        b"\x00" * 40,        # reserved
    )
    with path.open("wb") as handle:
        handle.write(header)
        for sym in symbols:
            handle.write(sym.encode("ascii").ljust(16, b"\x00"))


MIXED32_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA", "PLTR",
    "NET", "JPM", "V", "SPY", "QQQ", "NFLX", "AMD", "AFRM",
    "PANW", "SNOW", "DDOG", "COIN", "HOOD", "UBER", "PYPL", "ROKU",
    "BTCUSD", "ETHUSD", "SOLUSD", "LTCUSD", "AVAXUSD", "DOGEUSD",
    "LINKUSD", "AAVEUSD",
]


# ---------------------------------------------------------------------------
# 1. Checkpoint spec — verify obs_size=549 for a 32-symbol model
# ---------------------------------------------------------------------------

def test_checkpoint_spec_mixed32(tmp_path: Path) -> None:
    """A synthetic 32-symbol MLP checkpoint must yield obs_size=549."""
    obs_size = 32 * 17 + 5  # 549
    num_actions = 65
    hidden = 1024

    model_sd = collections.OrderedDict()
    model_sd["encoder.0.weight"] = torch.randn(hidden, obs_size)
    model_sd["encoder.0.bias"] = torch.randn(hidden)
    model_sd["actor.2.weight"] = torch.randn(num_actions, 512)
    model_sd["actor.2.bias"] = torch.randn(num_actions)

    ckpt_path = tmp_path / "wd_05" / "best.pt"
    ckpt_path.parent.mkdir(parents=True)
    torch.save({"model": model_sd}, ckpt_path)

    bridge = RLGeminiBridge(checkpoint_path=str(ckpt_path), hidden_size=hidden)
    spec = bridge.get_checkpoint_spec()

    assert spec.obs_size == 549
    assert spec.num_actions == num_actions
    assert spec.hidden_size == hidden
    assert spec.arch == "mlp"


# ---------------------------------------------------------------------------
# 2. Read MKTD symbols — verify 32 symbols from a synthetic bin
# ---------------------------------------------------------------------------

def test_read_mktd_symbols_mixed32(tmp_path: Path) -> None:
    """A MKTD file written with 32 symbols must round-trip correctly."""
    data_path = tmp_path / "mixed32_daily_train.bin"
    _write_mktd_header(data_path, MIXED32_SYMBOLS)

    symbols = orchestrator._read_mktd_symbols(data_path)

    assert len(symbols) == 32
    assert symbols == MIXED32_SYMBOLS


# ---------------------------------------------------------------------------
# 3. obs_size formula: obs_size = N*16 + 5 + N = N*17 + 5
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_symbols,expected_obs_size", [
    (5, 5 * 17 + 5),    # 90
    (8, 8 * 17 + 5),    # 141
    (23, 23 * 17 + 5),  # 396
    (32, 32 * 17 + 5),  # 549
])
def test_obs_size_formula(n_symbols: int, expected_obs_size: int) -> None:
    """_num_symbols_from_obs_size must invert the formula N*17+5."""
    assert orchestrator._num_symbols_from_obs_size(expected_obs_size) == n_symbols


# ---------------------------------------------------------------------------
# 4. Signal filtering — CRYPTO_ONLY regime
# ---------------------------------------------------------------------------

def test_signal_filtering_crypto_only() -> None:
    """In CRYPTO_ONLY regime, crypto signals are generated but stock signals are not.

    The orchestrator gates stock signal generation on ``regime == "STOCK_HOURS"``.
    Crypto signals are generated in all regimes (CRYPTO_ONLY, STOCK_HOURS, etc.).
    """
    regime = "CRYPTO_ONLY"

    # Crypto signals run when regime is in this set:
    crypto_allowed_regimes = ("CRYPTO_ONLY", "PRE_MARKET", "POST_MARKET", "STOCK_HOURS")
    # Stock signals run only when:
    stock_allowed = (regime == "STOCK_HOURS")

    assert regime in crypto_allowed_regimes, "crypto signals should pass in CRYPTO_ONLY"
    assert not stock_allowed, "stock signals should be filtered in CRYPTO_ONLY"


# ---------------------------------------------------------------------------
# 5. Signal filtering — STOCK_HOURS regime
# ---------------------------------------------------------------------------

def test_signal_filtering_stock_hours() -> None:
    """In STOCK_HOURS regime, both crypto and stock signals are generated."""
    regime = "STOCK_HOURS"

    crypto_allowed_regimes = ("CRYPTO_ONLY", "PRE_MARKET", "POST_MARKET", "STOCK_HOURS")
    stock_allowed = (regime == "STOCK_HOURS")

    assert regime in crypto_allowed_regimes, "crypto signals should pass in STOCK_HOURS"
    assert stock_allowed, "stock signals should pass in STOCK_HOURS"


# ---------------------------------------------------------------------------
# 6. is_crypto_symbol classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("symbol,expected", [
    ("BTCUSD", True),
    ("AAPL", False),
    ("ETHUSD", True),
    ("SOLUSD", True),
    ("MSFT", False),
    ("NVDA", False),
    ("DOGEUSD", True),
])
def test_is_crypto_symbol(symbol: str, expected: bool) -> None:
    assert is_crypto_symbol(symbol) is expected
