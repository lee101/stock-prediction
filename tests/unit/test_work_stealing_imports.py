"""Test that all work stealing imports resolve correctly."""


def test_process_utils_imports():
    """Verify process_utils can import all work stealing helpers."""
    from src.process_utils import spawn_open_position_at_maxdiff_takeprofit
    from src.work_stealing_config import (
        CRYPTO_SYMBOLS,
        get_entry_tolerance_for_symbol,
        is_crypto_out_of_hours,
        should_force_immediate_crypto,
    )

    assert callable(spawn_open_position_at_maxdiff_takeprofit)
    assert callable(get_entry_tolerance_for_symbol)
    assert callable(is_crypto_out_of_hours)
    assert callable(should_force_immediate_crypto)
    assert isinstance(CRYPTO_SYMBOLS, frozenset)


def test_maxdiff_cli_imports():
    """Verify maxdiff_cli can import work stealing coordinator."""
    from src.work_stealing_coordinator import get_coordinator

    coordinator = get_coordinator()
    assert coordinator is not None
    assert hasattr(coordinator, "attempt_steal")


def test_coordinator_imports():
    """Verify coordinator has all required dependencies."""
    from src.work_stealing_config import (
        WORK_STEALING_COOLDOWN_SECONDS,
        WORK_STEALING_ENTRY_TOLERANCE_PCT,
        WORK_STEALING_FIGHT_COOLDOWN_SECONDS,
        WORK_STEALING_FIGHT_THRESHOLD,
        WORK_STEALING_PROTECTION_PCT,
    )

    assert isinstance(WORK_STEALING_COOLDOWN_SECONDS, int)
    assert isinstance(WORK_STEALING_ENTRY_TOLERANCE_PCT, float)
    assert isinstance(WORK_STEALING_FIGHT_COOLDOWN_SECONDS, int)
    assert isinstance(WORK_STEALING_FIGHT_THRESHOLD, int)
    assert isinstance(WORK_STEALING_PROTECTION_PCT, float)


def test_trade_e2e_crypto_rank_logic():
    """Verify trade_stock_e2e can call process_utils with crypto_rank."""
    import inspect

    from src.process_utils import spawn_open_position_at_maxdiff_takeprofit

    sig = inspect.signature(spawn_open_position_at_maxdiff_takeprofit)
    assert "crypto_rank" in sig.parameters
