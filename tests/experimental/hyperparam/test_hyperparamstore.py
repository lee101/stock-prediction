from hyperparamstore import (
    HyperparamStore,
    load_best_config,
    load_model_selection,
    save_best_config,
    save_model_selection,
)


def test_save_and_load_hyperparams(tmp_path):
    store = HyperparamStore(tmp_path)
    windows = {"val_window": 10, "test_window": 5, "forecast_horizon": 1}

    path = save_best_config(
        model="toto",
        symbol="TEST",
        config={"name": "demo", "num_samples": 123},
        validation={"price_mae": 1.0, "pct_return_mae": 0.1, "latency_s": 0.5},
        test={"price_mae": 2.0, "pct_return_mae": 0.2, "latency_s": 0.6},
        windows=windows,
        metadata={"source": "unit_test"},
        store=store,
    )

    record = load_best_config("toto", "TEST", store=store)
    assert record is not None
    assert record.config["num_samples"] == 123
    assert record.validation["price_mae"] == 1.0
    assert record.test["pct_return_mae"] == 0.2
    assert record.metadata["source"] == "unit_test"
    selection_path = save_model_selection(
        symbol="TEST",
        model="toto",
        config={"name": "demo", "num_samples": 123},
        validation={"price_mae": 1.0},
        test={"price_mae": 2.0},
        windows=windows,
        metadata={"extra": "info"},
        config_path=str(path),
        store=store,
    )
    assert selection_path.exists()
    selection = load_model_selection("TEST", store=store)
    assert selection is not None
    assert selection["model"] == "toto"
    assert selection["config"]["num_samples"] == 123
    assert selection["validation"]["price_mae"] == 1.0
    assert selection["windows"]["val_window"] == 10
    assert selection["metadata"]["extra"] == "info"


def test_load_missing_config(tmp_path):
    store = HyperparamStore(tmp_path)
    assert load_best_config("toto", "UNKNOWN", store=store) is None
    assert load_model_selection("UNKNOWN", store=store) is None
