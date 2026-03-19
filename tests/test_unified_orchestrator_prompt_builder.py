from __future__ import annotations

from unified_orchestrator.prompt_builder import build_unified_prompt
from unified_orchestrator.state import UnifiedPortfolioSnapshot


def test_stock_prompt_uses_stock_direction_examples():
    prompt = build_unified_prompt(
        symbol="NVDA",
        history_rows=[
            {
                "timestamp": f"2026-03-12T{hour:02d}:00",
                "open": 100.0 + hour,
                "high": 100.5 + hour,
                "low": 99.5 + hour,
                "close": 100.2 + hour,
                "volume": 1_000 + hour,
            }
            for hour in range(24)
        ],
        current_price=123.45,
        snapshot=UnifiedPortfolioSnapshot(regime="STOCK_HOURS"),
        asset_class="stock",
    )

    assert "SYMBOL: NVDA (margin equity" in prompt
    assert '"direction": "long", "short", or "hold"' in prompt
