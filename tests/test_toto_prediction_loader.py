import pandas as pd

from hftraining.data_utils import load_toto_prediction_history


def test_load_toto_prediction_history_parses_basic_csv(tmp_path):
    csv_content = "\n".join(
        [
            "instrument,close_last_price,close_predicted_price,close_predicted_price_value,takeprofit_profit,close_trade_values",
            "AAPL-2024-01-02 07:10:00,150.0,151.5,152.25,0.42,\"tensor([1, -1])\"",
        ]
    )
    csv_path = tmp_path / "predictions.csv"
    csv_path.write_text(csv_content)

    frames, columns = load_toto_prediction_history(tmp_path)

    assert "AAPL" in frames
    frame = frames["AAPL"]
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert len(frame) == 1
    assert "toto_pred_close_last_price" in frame.columns
    assert "toto_pred_takeprofit_profit" in frame.columns
    assert all(col.startswith("toto_pred_") for col in frame.columns)
    expected_index = pd.DatetimeIndex(pd.to_datetime(["2024-01-02"]), name="prediction_date")
    pd.testing.assert_index_equal(frame.index, expected_index)
    assert frame.loc[expected_index[0], "toto_pred_close_last_price"] == 150.0
    assert frame.loc[expected_index[0], "toto_pred_takeprofit_profit"] == 0.42
    assert "toto_pred_close_trade_values" not in frame.columns
    assert columns == list(frame.columns)
