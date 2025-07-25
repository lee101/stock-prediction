import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.position_sizing_optimizer import (
    top_n_expected_return_sizing,
    backtest_position_sizing_series,
    sharpe_ratio,
)


def generate_demo_data(
    num_assets: int = 5,
    num_days: int = 200,
    csv_files: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    ema_span: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load real returns and naive predictions from CSV price data."""
    if not csv_files:
        csv_files = [str(ROOT / "WIKI-AAPL.csv")]

    frames = []
    for path in csv_files:
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        if start_date or end_date:
            df = df.loc[start_date:end_date]
        returns = df["Close"].pct_change().dropna()
        if num_days:
            returns = returns.iloc[:num_days]
        frames.append(returns)

    base = pd.concat(frames, axis=1)
    base.columns = [f"base_{i}" for i in range(len(frames))]

    assets = []
    for i in range(num_assets):
        series = base.iloc[:, i % base.shape[1]]
        assets.append(series)
    actual = pd.concat(assets, axis=1)
    actual.columns = [f"asset_{i}" for i in range(num_assets)]

    if ema_span:
        predicted = actual.ewm(span=ema_span, adjust=False).mean().shift(1).fillna(0)
    else:
        predicted = actual.shift(1).fillna(0)
    return actual, predicted


def run_demo(
    n_values: list[int] | None = None,
    leverage_values: list[float] | None = None,
    num_assets: int = 5,
    num_days: int = 200,
    csv_files: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    ema_span: int | None = None,
    output: str = "pnl_demo.png",
    save_csv: str | None = None,
    risk_free_rate: float = 0.0,
    show_plot: bool = False,
) -> pd.DataFrame:
    n_values = n_values or [1, 3]
    leverage_values = leverage_values or [0.5, 1.0, 2.0]

    actual, predicted = generate_demo_data(
        num_assets=num_assets,
        num_days=num_days,
        csv_files=csv_files,
        start_date=start_date,
        end_date=end_date,
        ema_span=ema_span,
    )
    pnl_curves = {}
    for n in n_values:
        for lev in leverage_values:
            sizes = top_n_expected_return_sizing(predicted, n=n, leverage=lev)
            pnl_series = backtest_position_sizing_series(
                actual, predicted, lambda _: sizes
            )
            pnl_series = pnl_series - risk_free_rate / 252
            pnl_curves[f"n{n}_lev{lev}"] = pnl_series.cumsum()

    df_curves = pd.DataFrame(pnl_curves)
    df_curves.plot(title="Cumulative pnl by sizing parameters")
    plt.xlabel("Day")
    plt.ylabel("PnL")
    plt.tight_layout()
    plt.savefig(output)
    if show_plot:
        plt.show()
    print(f"Chart saved to {output}")
    if save_csv:
        df_curves.to_csv(save_csv, index=False)
        print(f"PnL data saved to {save_csv}")

    for col in df_curves.columns:
        pnl_total = df_curves[col].iloc[-1]
        sharpe = sharpe_ratio(df_curves[col].diff().fillna(0), risk_free_rate=risk_free_rate)
        print(f"{col}: total pnl={pnl_total:.4f} sharpe={sharpe:.3f}")

    return df_curves


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run position sizing demo")
    parser.add_argument("--n", nargs="*", type=int, default=[1, 3], help="n values")
    parser.add_argument("--lev", nargs="*", type=float, default=[0.5, 1.0, 2.0] , help="leverage values")
    parser.add_argument("--assets", type=int, default=5, help="number of assets")
    parser.add_argument("--days", type=int, default=200, help="number of days")
    parser.add_argument("--csv", nargs="*", help="CSV files for historical data")
    parser.add_argument("--start", help="start date YYYY-MM-DD")
    parser.add_argument("--end", help="end date YYYY-MM-DD")
    parser.add_argument("--ema-span", type=int, help="EMA span for predictions")
    parser.add_argument("--output", default="pnl_demo.png", help="output chart file")
    parser.add_argument("--save-csv", help="optional csv for pnl data")
    parser.add_argument("--rf", type=float, default=0.0, help="annual risk free rate")
    parser.add_argument("--show", action="store_true", help="display chart interactively")
    args = parser.parse_args()
    run_demo(
        args.n,
        args.lev,
        args.assets,
        args.days,
        args.csv,
        start_date=args.start,
        end_date=args.end,
        ema_span=args.ema_span,
        output=args.output,
        save_csv=args.save_csv,
        risk_free_rate=args.rf,
        show_plot=args.show,
    )
