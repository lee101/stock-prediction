import pytest


@pytest.mark.unit
def test_is_bagsfm_trading_disabled_default_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BAGSFM_TRADING_DISABLED", raising=False)
    monkeypatch.delenv("DISABLE_BAGSFM_TRADING", raising=False)

    from bagsfm.config import is_bagsfm_trading_disabled

    assert is_bagsfm_trading_disabled() is False


@pytest.mark.unit
@pytest.mark.parametrize(
    ("env_var", "value"),
    [
        ("BAGSFM_TRADING_DISABLED", "1"),
        ("BAGSFM_TRADING_DISABLED", "true"),
        ("BAGSFM_TRADING_DISABLED", "YES"),
        ("DISABLE_BAGSFM_TRADING", "1"),
        ("DISABLE_BAGSFM_TRADING", "on"),
        ("DISABLE_BAGSFM_TRADING", "anything-nonempty"),
    ],
)
def test_is_bagsfm_trading_disabled_truthy(
    monkeypatch: pytest.MonkeyPatch, env_var: str, value: str
) -> None:
    monkeypatch.delenv("BAGSFM_TRADING_DISABLED", raising=False)
    monkeypatch.delenv("DISABLE_BAGSFM_TRADING", raising=False)
    monkeypatch.setenv(env_var, value)

    from bagsfm.config import is_bagsfm_trading_disabled

    assert is_bagsfm_trading_disabled() is True


@pytest.mark.unit
@pytest.mark.parametrize(
    ("env_var", "value"),
    [
        ("BAGSFM_TRADING_DISABLED", ""),
        ("BAGSFM_TRADING_DISABLED", "0"),
        ("BAGSFM_TRADING_DISABLED", "false"),
        ("DISABLE_BAGSFM_TRADING", ""),
        ("DISABLE_BAGSFM_TRADING", "0"),
        ("DISABLE_BAGSFM_TRADING", "no"),
    ],
)
def test_is_bagsfm_trading_disabled_falsey(
    monkeypatch: pytest.MonkeyPatch, env_var: str, value: str
) -> None:
    monkeypatch.delenv("BAGSFM_TRADING_DISABLED", raising=False)
    monkeypatch.delenv("DISABLE_BAGSFM_TRADING", raising=False)
    monkeypatch.setenv(env_var, value)

    from bagsfm.config import is_bagsfm_trading_disabled

    assert is_bagsfm_trading_disabled() is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sign_and_send_blocked_when_bagsfm_trading_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BAGSFM_TRADING_DISABLED", "1")

    from bagsfm.bags_api import SolanaTransactionExecutor, SwapTransaction
    from bagsfm.config import BagsConfig

    executor = SolanaTransactionExecutor(BagsConfig())
    swap_tx = SwapTransaction(
        swap_transaction="dummy",
        compute_unit_limit=0,
        prioritization_fee_lamports=0,
        last_valid_block_height=0,
    )

    result = await executor.sign_and_send(swap_tx)

    assert result.success is False
    assert result.error is not None
    assert "disabled" in result.error.lower()

