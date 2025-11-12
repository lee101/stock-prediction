#!/usr/bin/env python3
"""
Unified Hyperparameter Tracking System

Tracks hyperparameter sweeps across all models (Toto, Kronos, Chronos2) with
consistent metrics (especially pct_mae) for easy comparison and selection.

Usage:
    from hparams_tracker import HyperparamTracker

    tracker = HyperparamTracker("model_sweeps.json")

    # During training:
    tracker.log_run(
        model_name="toto",
        hyperparams={
            "patch_size": 32,
            "learning_rate": 0.0003,
            "context_length": 512,
        },
        metrics={
            "val_pct_mae": 0.45,
            "val_price_mae": 0.08,
            "val_r2": 0.35,
            "test_pct_mae": 0.48,
        },
        checkpoint_path="path/to/checkpoint.pt"
    )

    # At inference time:
    best_model = tracker.get_best_model(metric="val_pct_mae", model_type="toto")
    print(f"Best model: {best_model['checkpoint_path']}")
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class HyperparamRun:
    """Single hyperparameter sweep run"""
    run_id: str
    model_name: str  # "toto", "kronos", "chronos2"
    timestamp: str
    hyperparams: Dict[str, Any]
    metrics: Dict[str, float]
    checkpoint_path: Optional[str] = None
    training_time_seconds: Optional[float] = None
    notes: Optional[str] = None
    git_commit: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HyperparamTracker:
    """Track and compare hyperparameter sweeps across models"""

    def __init__(self, db_path: str = "hyperparams/sweep_results.json"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.runs: List[HyperparamRun] = []
        self.load()

    def load(self):
        """Load existing runs from disk"""
        if self.db_path.exists():
            with open(self.db_path, 'r') as f:
                data = json.load(f)
                self.runs = [
                    HyperparamRun(**run) for run in data.get('runs', [])
                ]

    def save(self):
        """Save runs to disk"""
        data = {
            'runs': [run.to_dict() for run in self.runs],
            'last_updated': datetime.utcnow().isoformat()
        }
        with open(self.db_path, 'w') as f:
            json.dump(data, f, indent=2)

    def log_run(
        self,
        model_name: str,
        hyperparams: Dict[str, Any],
        metrics: Dict[str, float],
        checkpoint_path: Optional[str] = None,
        training_time_seconds: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> str:
        """
        Log a hyperparameter sweep run

        Args:
            model_name: "toto", "kronos", "chronos2", etc.
            hyperparams: Dict of hyperparameters
            metrics: Dict of metrics (should include val_pct_mae, test_pct_mae)
            checkpoint_path: Path to saved model checkpoint
            training_time_seconds: Training duration
            notes: Optional notes about the run

        Returns:
            run_id: Unique identifier for this run
        """
        # Generate run ID
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_id = f"{model_name}_{timestamp}"

        # Try to get git commit
        git_commit = None
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', '--short', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                git_commit = result.stdout.strip()
        except Exception:
            pass

        run = HyperparamRun(
            run_id=run_id,
            model_name=model_name,
            timestamp=datetime.utcnow().isoformat(),
            hyperparams=hyperparams,
            metrics=metrics,
            checkpoint_path=checkpoint_path,
            training_time_seconds=training_time_seconds,
            notes=notes,
            git_commit=git_commit
        )

        self.runs.append(run)
        self.save()

        return run_id

    def get_runs(
        self,
        model_name: Optional[str] = None,
        min_metric_value: Optional[Dict[str, float]] = None,
        max_metric_value: Optional[Dict[str, float]] = None,
    ) -> List[HyperparamRun]:
        """
        Get runs with optional filtering

        Args:
            model_name: Filter by model type (e.g., "toto")
            min_metric_value: Dict of {metric_name: min_value}
            max_metric_value: Dict of {metric_name: max_value}
        """
        runs = self.runs

        if model_name:
            runs = [r for r in runs if r.model_name == model_name]

        if min_metric_value:
            for metric, min_val in min_metric_value.items():
                runs = [r for r in runs if r.metrics.get(metric, float('inf')) >= min_val]

        if max_metric_value:
            for metric, max_val in max_metric_value.items():
                runs = [r for r in runs if r.metrics.get(metric, float('inf')) <= max_val]

        return runs

    def get_best_model(
        self,
        metric: str = "val_pct_mae",
        model_name: Optional[str] = None,
        minimize: bool = True,
        require_checkpoint: bool = True,
    ) -> Optional[HyperparamRun]:
        """
        Get the best model based on a metric

        Args:
            metric: Metric to optimize (e.g., "val_pct_mae", "test_r2")
            model_name: Filter by model type
            minimize: True to minimize metric (e.g., MAE), False to maximize (e.g., R2)
            require_checkpoint: Only consider runs with checkpoints

        Returns:
            Best run, or None if no runs found
        """
        runs = self.get_runs(model_name=model_name)

        if require_checkpoint:
            runs = [r for r in runs if r.checkpoint_path is not None]

        # Filter runs that have the metric
        runs = [r for r in runs if metric in r.metrics]

        if not runs:
            return None

        if minimize:
            return min(runs, key=lambda r: r.metrics[metric])
        else:
            return max(runs, key=lambda r: r.metrics[metric])

    def get_top_k_models(
        self,
        k: int,
        metric: str = "val_pct_mae",
        model_name: Optional[str] = None,
        minimize: bool = True,
    ) -> List[HyperparamRun]:
        """Get top K models by metric"""
        runs = self.get_runs(model_name=model_name)
        runs = [r for r in runs if metric in r.metrics]

        if not runs:
            return []

        runs_sorted = sorted(runs, key=lambda r: r.metrics[metric], reverse=not minimize)
        return runs_sorted[:k]

    def compare_models(
        self,
        metrics: List[str] = None,
        model_names: List[str] = None,
    ) -> pd.DataFrame:
        """
        Create comparison table across models

        Args:
            metrics: List of metrics to compare (default: common ones)
            model_names: List of models to include (default: all)

        Returns:
            DataFrame with comparison
        """
        if metrics is None:
            metrics = [
                "val_pct_mae",
                "test_pct_mae",
                "val_price_mae",
                "test_price_mae",
                "val_r2",
                "test_r2"
            ]

        runs = self.runs
        if model_names:
            runs = [r for r in runs if r.model_name in model_names]

        data = []
        for run in runs:
            row = {
                'run_id': run.run_id,
                'model': run.model_name,
                'timestamp': run.timestamp[:10],  # Just date
            }
            # Add metrics
            for metric in metrics:
                row[metric] = run.metrics.get(metric, None)
            # Add key hyperparams
            row['checkpoint'] = run.checkpoint_path is not None
            data.append(row)

        return pd.DataFrame(data)

    def get_hyperp aram_impact(
        self,
        model_name: str,
        hyperparam: str,
        metric: str = "val_pct_mae"
    ) -> pd.DataFrame:
        """
        Analyze impact of a specific hyperparameter on performance

        Returns DataFrame with hyperparam value vs metric
        """
        runs = self.get_runs(model_name=model_name)
        runs = [r for r in runs if hyperparam in r.hyperparams and metric in r.metrics]

        data = []
        for run in runs:
            data.append({
                'run_id': run.run_id,
                hyperparam: run.hyperparams[hyperparam],
                metric: run.metrics[metric]
            })

        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values(by=hyperparam)
        return df

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate markdown report of all sweeps"""
        lines = [
            "# Hyperparameter Sweep Report",
            f"\nGenerated: {datetime.utcnow().isoformat()}",
            f"\nTotal runs: {len(self.runs)}",
            ""
        ]

        # Best models per type
        lines.append("## Best Models by Type (val_pct_mae)")
        lines.append("")
        for model_name in ["toto", "kronos", "chronos2"]:
            best = self.get_best_model(metric="val_pct_mae", model_name=model_name)
            if best:
                lines.append(f"### {model_name.upper()}")
                lines.append(f"- Run ID: {best.run_id}")
                lines.append(f"- Val pct_MAE: {best.metrics.get('val_pct_mae', 'N/A'):.4f}")
                lines.append(f"- Test pct_MAE: {best.metrics.get('test_pct_mae', 'N/A'):.4f}")
                lines.append(f"- Val R²: {best.metrics.get('val_r2', 'N/A'):.4f}")
                lines.append(f"- Checkpoint: {best.checkpoint_path}")
                lines.append(f"- Hyperparams: {json.dumps(best.hyperparams, indent=2)}")
                lines.append("")

        # Comparison table
        lines.append("## Model Comparison")
        lines.append("")
        df = self.compare_models()
        if not df.empty:
            lines.append(df.to_markdown(index=False))
        else:
            lines.append("No runs recorded yet.")
        lines.append("")

        report = "\n".join(lines)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)

        return report


def main():
    """Example usage"""
    tracker = HyperparamTracker("hyperparams/sweep_results.json")

    # Example: Log a run
    tracker.log_run(
        model_name="toto",
        hyperparams={
            "patch_size": 32,
            "learning_rate": 0.0003,
            "context_length": 512,
            "epochs": 30
        },
        metrics={
            "val_pct_mae": 0.45,
            "val_price_mae": 0.08,
            "val_r2": 0.35,
            "test_pct_mae": 0.48,
            "test_price_mae": 0.09,
            "test_r2": 0.32
        },
        checkpoint_path="tototraining/checkpoints/best_model.pt",
        notes="Improved hyperparameters aligned with Toto paper"
    )

    # Get best model
    best = tracker.get_best_model(metric="val_pct_mae", model_name="toto")
    if best:
        print(f"Best Toto model: {best.checkpoint_path}")
        print(f"Val pct_MAE: {best.metrics['val_pct_mae']:.4f}")

    # Generate report
    report = tracker.generate_report("hyperparams/sweep_report.md")
    print(report)


if __name__ == "__main__":
    main()
