import numpy as np
import pandas as pd
import torch

from strategytrainingneural.data import augment_metrics, build_dataset, split_dataset_by_date
from strategytrainingneural.trainer import train_sortino_policy, train_xgboost_policy


def _make_mock_dataframe(num_days: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=num_days, freq="D")
    strategies = ["Alpha", "Beta"]
    rows = []
    for idx, date in enumerate(dates):
        for strategy in strategies:
            base = 0.01 if strategy == "Alpha" else 0.005
            sign = 1.0 if idx % 2 == 0 else -0.7
            daily_return = base * sign
            rows.append(
                {
                    "date": date,
                    "strategy": strategy,
                    "rolling_sharpe": base * 10,
                    "rolling_sortino": base * 20,
                    "rolling_ann_return": base * 15,
                    "capital": 100_000 + idx * 1000,
                    "daily_return": daily_return,
                    "mode": "normal" if idx % 5 else "probe",
                    "gate_config": "-" if idx % 7 else "StockDir",
                    "day_class": "stock",
                }
            )
    df = pd.DataFrame(rows).sort_values(["strategy", "date"]).reset_index(drop=True)
    return augment_metrics(df)


def test_trainers_run_end_to_end():
    torch.manual_seed(0)
    df = _make_mock_dataframe()
    dataset = build_dataset(df)
    train_ds, val_ds = split_dataset_by_date(dataset, validation_fraction=0.3)

    result = train_sortino_policy(
        train_ds,
        validation_dataset=val_ds,
        epochs=5,
        learning_rate=5e-3,
        return_weight=0.1,
        max_weight=1.5,
        device="cpu",
    )
    assert len(result.history) == 5
    assert np.isfinite(result.final_metrics["score"])
    assert result.final_metrics["best_epoch"] >= 1
    assert "best_val_score" in result.final_metrics

    xgb_result = train_xgboost_policy(
        train_ds,
        evaluation_dataset=val_ds,
        num_rounds=5,
        temperature_grid=(0.1, 0.5),
        return_weight=0.1,
    )
    assert np.isfinite(xgb_result.score)


def test_train_sortino_policy_warm_start_changes_metrics():
    torch.manual_seed(0)
    df = _make_mock_dataframe(num_days=20)
    dataset = build_dataset(df)
    train_ds, val_ds = split_dataset_by_date(dataset, validation_fraction=0.25)

    base_result = train_sortino_policy(
        train_ds,
        validation_dataset=val_ds,
        epochs=3,
        learning_rate=1e-2,
        max_weight=1.5,
        device="cpu",
    )
    base_state = {k: v.detach().cpu().clone() for k, v in base_result.model.state_dict().items()}

    torch.manual_seed(0)
    cold_start = train_sortino_policy(
        train_ds,
        validation_dataset=val_ds,
        epochs=1,
        learning_rate=1e-2,
        max_weight=1.5,
        device="cpu",
    )

    torch.manual_seed(0)
    warm_start = train_sortino_policy(
        train_ds,
        validation_dataset=val_ds,
        epochs=1,
        learning_rate=1e-2,
        max_weight=1.5,
        device="cpu",
        initial_state_dict=base_state,
    )

    assert not np.isclose(cold_start.history[0].train_score, warm_start.history[0].train_score)


def test_build_dataset_accepts_existing_feature_spec():
    df = _make_mock_dataframe(num_days=15)
    base_dataset = build_dataset(df)
    subset = df.iloc[:20].copy()
    resumed_dataset = build_dataset(subset, feature_spec=base_dataset.feature_spec)

    assert resumed_dataset.features.shape[1] == base_dataset.features.shape[1]
    assert resumed_dataset.feature_spec.feature_names == base_dataset.feature_spec.feature_names


def test_best_epoch_tracking_matches_history():
    torch.manual_seed(1)
    df = _make_mock_dataframe(num_days=24)
    dataset = build_dataset(df)
    train_ds, val_ds = split_dataset_by_date(dataset, validation_fraction=0.25)
    result = train_sortino_policy(
        train_ds,
        validation_dataset=val_ds,
        epochs=6,
        learning_rate=5e-3,
        return_weight=0.1,
        max_weight=1.25,
        device="cpu",
    )
    val_history = [entry for entry in result.history if entry.val_score is not None]
    best_entry = max(val_history, key=lambda entry: entry.val_score)
    assert result.final_metrics["best_epoch"] == best_entry.epoch
    assert np.isclose(result.final_metrics["score"], best_entry.val_score, atol=1e-4)
    assert np.isclose(result.final_metrics["best_val_score"], best_entry.val_score, atol=1e-6)
