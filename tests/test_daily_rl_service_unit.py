from __future__ import annotations

from pathlib import Path

import trade_daily_stock_prod as daily_stock


REPO = Path(__file__).resolve().parents[1]
UNIT_PATH = REPO / "systemd" / "daily-rl-trader.service"
SETUP_SCRIPT = REPO / "scripts" / "setup_systemd_daily_rl_trader.sh"


def test_daily_rl_service_unit_tracks_live_prod_entrypoint() -> None:
    text = UNIT_PATH.read_text(encoding="utf-8")

    assert "trade_daily_stock_prod.py" in text
    assert "--daemon" in text
    assert "--live" in text
    assert f"--allocation-pct {daily_stock.DEFAULT_ALLOCATION_PCT:g}" in text
    assert "trade_mixed_daily.py" not in text
    assert "--paper" not in text


def test_daily_rl_service_unit_sources_secrets_without_committing_keys() -> None:
    text = UNIT_PATH.read_text(encoding="utf-8")

    assert "source ~/.secretbashrc" in text
    assert "ALLOW_ALPACA_LIVE_TRADING=1" in text
    assert "ALP_KEY_ID_PROD=" not in text
    assert "ALP_SECRET_KEY_PROD=" not in text


def test_daily_rl_setup_script_is_safe_by_default() -> None:
    text = SETUP_SCRIPT.read_text(encoding="utf-8")

    assert "systemctl daemon-reload" in text
    assert 'sudo systemctl enable "${SERVICE_NAME}"' in text
    assert "--restart" in text
    assert "did not start it" in text
    assert "enable --now" not in text


def test_daily_rl_setup_script_runs_preflight_before_install() -> None:
    text = SETUP_SCRIPT.read_text(encoding="utf-8")

    assert "--check-config" in text
    assert "--check-config-text" in text
    assert "systemd-analyze verify" in text
    assert "Running ${SERVICE_NAME} preflight" in text
    assert "--skip-preflight" in text


def test_daily_rl_setup_script_supports_check_only_validation() -> None:
    text = SETUP_SCRIPT.read_text(encoding="utf-8")

    assert "--check-only" in text
    assert "Preflight completed; not installing" in text
    assert 'echo "  ${0} --check-only"' in text
