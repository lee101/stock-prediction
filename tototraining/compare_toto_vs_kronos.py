#!/usr/bin/env python3
"""
Compare retrained Toto models against Kronos baseline

Runs systematic comparisons using test_kronos_vs_toto.py framework
and tracks which model performs better on each stock.
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ComparisonResult:
    """Results of comparing Toto vs Kronos on a stock"""
    symbol: str
    toto_mae: Optional[float]
    kronos_mae: Optional[float]
    winner: str  # "toto", "kronos", or "tie"
    improvement_pct: float
    toto_latency: Optional[float]
    kronos_latency: Optional[float]
    forecast_horizon: int
    timestamp: str
    error: Optional[str] = None


class TotoKronosComparer:
    """Compares Toto vs Kronos models systematically"""

    def __init__(
        self,
        hyperparam_root: Path = Path("hyperparams"),
        results_root: Path = Path("comparison_results")
    ):
        self.hyperparam_root = hyperparam_root
        self.results_root = results_root
        self.results_root.mkdir(parents=True, exist_ok=True)

        self.toto_dir = hyperparam_root / "toto"
        self.kronos_dir = hyperparam_root / "kronos"

    def get_available_toto_models(self) -> List[str]:
        """Get list of stocks with trained Toto models"""
        if not self.toto_dir.exists():
            return []

        models = []
        for config_file in self.toto_dir.glob("*.json"):
            symbol = config_file.stem
            models.append(symbol)

        return sorted(models)

    def get_available_kronos_configs(self) -> List[str]:
        """Get list of stocks with Kronos configs"""
        if not self.kronos_dir.exists():
            return []

        configs = []
        for config_file in self.kronos_dir.glob("*.json"):
            symbol = config_file.stem
            configs.append(symbol)

        return sorted(configs)

    def compare_single_stock(
        self,
        symbol: str,
        forecast_horizon: int = 64,
        use_stored_hyperparams: bool = True
    ) -> ComparisonResult:
        """Compare Toto vs Kronos on a single stock"""

        print(f"\n{'='*100}")
        print(f"Comparing {symbol}: Toto vs Kronos (horizon={forecast_horizon})")
        print(f"{'='*100}")

        # Check if configs exist
        toto_config = self.toto_dir / f"{symbol}.json"
        kronos_config = self.kronos_dir / f"{symbol}.json"

        has_toto = toto_config.exists()
        has_kronos = kronos_config.exists()

        print(f"Toto config: {'✅' if has_toto else '❌'} {toto_config}")
        print(f"Kronos config: {'✅' if has_kronos else '❌'} {kronos_config}")

        if not has_toto and not has_kronos:
            return ComparisonResult(
                symbol=symbol,
                toto_mae=None,
                kronos_mae=None,
                winner="none",
                improvement_pct=0.0,
                toto_latency=None,
                kronos_latency=None,
                forecast_horizon=forecast_horizon,
                timestamp=datetime.now().isoformat(),
                error="No configs found for either model"
            )

        # Build comparison command
        cmd = [
            "uv", "run", "python", "test_kronos_vs_toto.py",
            "--symbol", symbol,
        ]

        # Set environment for forecast horizon
        env = {"FORECAST_HORIZON": str(forecast_horizon)}

        if use_stored_hyperparams:
            env["USE_STORED_HYPERPARAMS"] = "1"

        print(f"\nRunning comparison...")
        print(f"Command: {' '.join(cmd)}")
        print(f"Environment: {env}\n")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 min timeout
                cwd=PROJECT_ROOT,
                env={**subprocess.os.environ, **env}
            )

            # Save output
            output_file = self.results_root / f"{symbol}_comparison.txt"
            with open(output_file, 'w') as f:
                f.write(result.stdout)
                f.write("\n" + "="*80 + "\n")
                f.write(result.stderr)

            # Parse results
            comparison = self._parse_comparison_output(
                result.stdout + result.stderr,
                symbol,
                forecast_horizon
            )

            # Print results
            self._print_comparison(comparison)

            return comparison

        except subprocess.TimeoutExpired:
            print("⏱️  Comparison timed out!")
            return ComparisonResult(
                symbol=symbol,
                toto_mae=None,
                kronos_mae=None,
                winner="timeout",
                improvement_pct=0.0,
                toto_latency=None,
                kronos_latency=None,
                forecast_horizon=forecast_horizon,
                timestamp=datetime.now().isoformat(),
                error="Timeout"
            )
        except Exception as e:
            print(f"❌ Comparison error: {e}")
            return ComparisonResult(
                symbol=symbol,
                toto_mae=None,
                kronos_mae=None,
                winner="error",
                improvement_pct=0.0,
                toto_latency=None,
                kronos_latency=None,
                forecast_horizon=forecast_horizon,
                timestamp=datetime.now().isoformat(),
                error=str(e)
            )

    def _parse_comparison_output(
        self,
        output: str,
        symbol: str,
        forecast_horizon: int
    ) -> ComparisonResult:
        """Parse comparison results from output"""

        toto_mae = None
        kronos_mae = None
        toto_latency = None
        kronos_latency = None

        # Look for MAE metrics in output
        for line in output.split('\n'):
            # Toto metrics
            if 'toto' in line.lower() and 'mae' in line.lower():
                try:
                    # Try to extract MAE value
                    if 'price_mae' in line.lower():
                        parts = line.split(':')
                        if len(parts) > 1:
                            toto_mae = float(parts[1].strip().split()[0])
                except:
                    pass

            if 'toto' in line.lower() and 'latency' in line.lower():
                try:
                    parts = line.split(':')
                    if len(parts) > 1:
                        toto_latency = float(parts[1].strip().split()[0].rstrip('s'))
                except:
                    pass

            # Kronos metrics
            if 'kronos' in line.lower() and 'mae' in line.lower():
                try:
                    if 'price_mae' in line.lower():
                        parts = line.split(':')
                        if len(parts) > 1:
                            kronos_mae = float(parts[1].strip().split()[0])
                except:
                    pass

            if 'kronos' in line.lower() and 'latency' in line.lower():
                try:
                    parts = line.split(':')
                    if len(parts) > 1:
                        kronos_latency = float(parts[1].strip().split()[0].rstrip('s'))
                except:
                    pass

        # Determine winner
        winner = "tie"
        improvement_pct = 0.0

        if toto_mae is not None and kronos_mae is not None:
            if toto_mae < kronos_mae:
                winner = "toto"
                improvement_pct = ((kronos_mae - toto_mae) / kronos_mae) * 100
            elif kronos_mae < toto_mae:
                winner = "kronos"
                improvement_pct = -((toto_mae - kronos_mae) / toto_mae) * 100
        elif toto_mae is not None:
            winner = "toto"
        elif kronos_mae is not None:
            winner = "kronos"

        return ComparisonResult(
            symbol=symbol,
            toto_mae=toto_mae,
            kronos_mae=kronos_mae,
            winner=winner,
            improvement_pct=improvement_pct,
            toto_latency=toto_latency,
            kronos_latency=kronos_latency,
            forecast_horizon=forecast_horizon,
            timestamp=datetime.now().isoformat()
        )

    def _print_comparison(self, result: ComparisonResult):
        """Print comparison results"""
        print(f"\n{'='*100}")
        print(f"RESULTS: {result.symbol}")
        print(f"{'='*100}")

        if result.error:
            print(f"❌ Error: {result.error}")
        elif result.toto_mae is None and result.kronos_mae is None:
            print("⚠️  No metrics available")
        else:
            if result.toto_mae:
                print(f"Toto MAE:   {result.toto_mae:.4f}")
            if result.kronos_mae:
                print(f"Kronos MAE: {result.kronos_mae:.4f}")

            if result.winner == "toto":
                print(f"\n✅ WINNER: Toto (improved by {result.improvement_pct:.1f}%)")
            elif result.winner == "kronos":
                print(f"\n❌ WINNER: Kronos (Toto worse by {abs(result.improvement_pct):.1f}%)")
            else:
                print(f"\n🤝 TIE")

            if result.toto_latency and result.kronos_latency:
                print(f"\nLatency - Toto: {result.toto_latency:.2f}s, Kronos: {result.kronos_latency:.2f}s")

        print(f"{'='*100}\n")

    def compare_all_stocks(
        self,
        stocks: Optional[List[str]] = None,
        forecast_horizon: int = 64
    ) -> Dict[str, ComparisonResult]:
        """Compare all available stocks"""

        # Get stocks to compare
        if stocks is None:
            toto_stocks = set(self.get_available_toto_models())
            kronos_stocks = set(self.get_available_kronos_configs())
            stocks = sorted(toto_stocks | kronos_stocks)

        if not stocks:
            print("No stocks found to compare!")
            return {}

        print(f"\n{'='*100}")
        print(f"COMPARING {len(stocks)} STOCKS: TOTO VS KRONOS")
        print(f"{'='*100}\n")

        results = {}

        for symbol in stocks:
            result = self.compare_single_stock(symbol, forecast_horizon)
            results[symbol] = result

        # Save results
        self._save_comparison_summary(results, forecast_horizon)

        # Print summary
        self._print_summary(results)

        return results

    def _save_comparison_summary(
        self,
        results: Dict[str, ComparisonResult],
        forecast_horizon: int
    ):
        """Save comparison summary"""

        summary = {
            "forecast_horizon": forecast_horizon,
            "timestamp": datetime.now().isoformat(),
            "total_stocks": len(results),
            "results": {
                symbol: {
                    "toto_mae": r.toto_mae,
                    "kronos_mae": r.kronos_mae,
                    "winner": r.winner,
                    "improvement_pct": r.improvement_pct,
                    "toto_latency": r.toto_latency,
                    "kronos_latency": r.kronos_latency,
                    "error": r.error,
                }
                for symbol, r in results.items()
            }
        }

        # Save JSON
        summary_file = self.results_root / f"comparison_summary_h{forecast_horizon}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to: {summary_file}")

    def _print_summary(self, results: Dict[str, ComparisonResult]):
        """Print overall summary"""

        toto_wins = sum(1 for r in results.values() if r.winner == "toto")
        kronos_wins = sum(1 for r in results.values() if r.winner == "kronos")
        ties = sum(1 for r in results.values() if r.winner == "tie")
        errors = sum(1 for r in results.values() if r.error is not None)

        valid_results = [r for r in results.values() if r.toto_mae and r.kronos_mae]
        avg_improvement = (
            sum(r.improvement_pct for r in valid_results) / len(valid_results)
            if valid_results else 0.0
        )

        print(f"\n{'='*100}")
        print("OVERALL SUMMARY")
        print(f"{'='*100}")
        print(f"Total stocks compared: {len(results)}")
        print(f"Toto wins: {toto_wins}")
        print(f"Kronos wins: {kronos_wins}")
        print(f"Ties: {ties}")
        print(f"Errors: {errors}")

        if valid_results:
            print(f"\nAverage improvement (Toto over Kronos): {avg_improvement:+.1f}%")

        # Top improvements
        sorted_results = sorted(
            valid_results,
            key=lambda r: r.improvement_pct,
            reverse=True
        )

        if sorted_results:
            print(f"\nTop 5 Toto Improvements:")
            for r in sorted_results[:5]:
                print(f"  {r.symbol}: {r.improvement_pct:+.1f}% (Toto: {r.toto_mae:.4f} vs Kronos: {r.kronos_mae:.4f})")

            print(f"\nTop 5 Kronos Advantages:")
            for r in reversed(sorted_results[-5:]):
                print(f"  {r.symbol}: {r.improvement_pct:+.1f}% (Toto: {r.toto_mae:.4f} vs Kronos: {r.kronos_mae:.4f})")

        print(f"{'='*100}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare Toto vs Kronos models")
    parser.add_argument("--symbol", type=str, help="Single stock to compare")
    parser.add_argument("--stocks", nargs="+", help="Multiple stocks to compare")
    parser.add_argument("--all", action="store_true", help="Compare all available stocks")
    parser.add_argument("--forecast-horizon", type=int, default=64,
                       help="Forecast horizon for comparison")
    parser.add_argument("--hyperparam-root", type=Path, default=Path("hyperparams"),
                       help="Root directory for hyperparameter configs")
    parser.add_argument("--results-dir", type=Path, default=Path("comparison_results"),
                       help="Directory to save comparison results")

    args = parser.parse_args()

    # Create comparer
    comparer = TotoKronosComparer(
        hyperparam_root=args.hyperparam_root,
        results_root=args.results_dir
    )

    # Run comparisons
    if args.symbol:
        # Single stock comparison
        result = comparer.compare_single_stock(args.symbol, args.forecast_horizon)
    elif args.stocks:
        # Multiple specific stocks
        results = comparer.compare_all_stocks(args.stocks, args.forecast_horizon)
    elif args.all:
        # All available stocks
        results = comparer.compare_all_stocks(forecast_horizon=args.forecast_horizon)
    else:
        print("Please specify --symbol, --stocks, or --all")
        sys.exit(1)


if __name__ == "__main__":
    main()
