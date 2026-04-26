from __future__ import annotations

import pytest

import unified_hourly_experiment.train_unified_policy as train_unified_policy


@pytest.mark.parametrize(
    ("argv", "expected_message"),
    [
        (["prog", "--symbols", ""], "At least one symbol is required."),
        (["prog", "--symbols", "../../etc/passwd"], "Unsupported symbol"),
        (["prog", "--crypto-symbols", "../../etc/passwd"], "Unsupported symbol"),
        (["prog", "--forecast-horizons", " , "], "At least one forecast horizon is required."),
    ],
)
def test_train_unified_policy_rejects_invalid_plan_inputs_before_data_loading(
    monkeypatch,
    argv: list[str],
    expected_message: str,
) -> None:
    def _unexpected(*args, **kwargs):
        raise AssertionError("Invalid plan input should fail before touching the data module")

    monkeypatch.setattr(train_unified_policy.sys, "argv", argv)
    monkeypatch.setattr(train_unified_policy, "MultiSymbolDataModule", _unexpected)

    with pytest.raises(SystemExit, match=r"^Plan error:") as excinfo:
        train_unified_policy.main()
    assert expected_message in str(excinfo.value)
