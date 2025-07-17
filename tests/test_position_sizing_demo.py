import pandas as pd
from scripts.position_sizing_demo import generate_demo_data, run_demo


def test_generate_demo_data_shapes():
    csv = ["WIKI-AAPL.csv"]
    actual, predicted = generate_demo_data(num_assets=3, num_days=50, csv_files=csv, ema_span=3)
    assert isinstance(actual, pd.DataFrame)
    assert isinstance(predicted, pd.DataFrame)
    assert actual.shape == (50, 3)
    assert predicted.shape == (50, 3)


def test_run_demo_returns_dataframe(tmp_path):
    csv = ["WIKI-AAPL.csv"]
    out = tmp_path / "chart.png"
    df = run_demo(n_values=[1], leverage_values=[1.0], num_assets=2, num_days=30, csv_files=csv, output=str(out), show_plot=False)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
