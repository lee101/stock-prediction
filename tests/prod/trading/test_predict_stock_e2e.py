import pandas as pd
import pytest


@pytest.mark.integration
async def test_make_trade_suggestions():
    save_file_name_min = 'results/predictions-2023-06-12_19-51-02.csv'
    save_file_name = 'results/predictions-2023-06-12_19-58-30.csv'
    from pathlib import Path

    if not Path(save_file_name_min).exists() or not Path(save_file_name).exists():
        pytest.skip("historic prediction fixtures not available")

    from predict_stock_e2e import make_trade_suggestions

    minutedf = pd.read_csv(save_file_name_min)
    dailydf = pd.read_csv(save_file_name)
    make_trade_suggestions(dailydf, minutedf)
