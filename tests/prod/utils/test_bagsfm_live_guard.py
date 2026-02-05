import pytest


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on", "Y"])
def test_is_truthy_env_truthy(monkeypatch, value: str) -> None:
    from bagsfm.utils import is_truthy_env

    monkeypatch.setenv("BAGSFM_ENABLE_LIVE_TRADING", value)
    assert is_truthy_env("BAGSFM_ENABLE_LIVE_TRADING") is True


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "", "no", "off", "random"])
def test_is_truthy_env_falsy(monkeypatch, value: str) -> None:
    from bagsfm.utils import is_truthy_env

    monkeypatch.setenv("BAGSFM_ENABLE_LIVE_TRADING", value)
    assert is_truthy_env("BAGSFM_ENABLE_LIVE_TRADING") is False


def test_is_truthy_env_default(monkeypatch) -> None:
    from bagsfm.utils import is_truthy_env

    monkeypatch.delenv("BAGSFM_ENABLE_LIVE_TRADING", raising=False)
    assert is_truthy_env("BAGSFM_ENABLE_LIVE_TRADING") is False
    assert is_truthy_env("BAGSFM_ENABLE_LIVE_TRADING", default=True) is True


def test_require_live_trading_enabled_raises(monkeypatch) -> None:
    from bagsfm.utils import require_live_trading_enabled

    monkeypatch.delenv("BAGSFM_ENABLE_LIVE_TRADING", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        require_live_trading_enabled()
    assert "BAGSFM_ENABLE_LIVE_TRADING" in str(excinfo.value)


def test_require_live_trading_enabled_allows(monkeypatch) -> None:
    from bagsfm.utils import require_live_trading_enabled

    monkeypatch.setenv("BAGSFM_ENABLE_LIVE_TRADING", "1")
    require_live_trading_enabled()

