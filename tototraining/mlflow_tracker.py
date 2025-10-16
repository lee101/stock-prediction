#!/usr/bin/env python3
"""
MLflow Experiment Tracking for Toto Training Pipeline
Provides comprehensive experiment tracking with hyperparameters, metrics, artifacts, and model versioning.
"""

import os
import json
import pickle
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import numpy as np

try:
    import mlflow
    import mlflow.pytorch
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None
    MlflowClient = None

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class MLflowTracker:
    """
    MLflow experiment tracking system for Toto training pipeline.
    Handles experiment creation, metric logging, hyperparameter tracking, and model versioning.
    """
    
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str = "mlruns",
        registry_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
        auto_log_model: bool = True,
        log_system_metrics: bool = True
    ):
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow not available. Install with: uv pip install mlflow")
        
        self.experiment_name = experiment_name
        self.auto_log_model = auto_log_model
        self._log_system_metrics_enabled = log_system_metrics
        
        # Setup MLflow tracking
        mlflow.set_tracking_uri(tracking_uri)
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)
        
        # Create or get experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=artifact_location
                )
            else:
                experiment_id = experiment.experiment_id
        except Exception as e:
            print(f"Warning: Could not create/get experiment: {e}")
            experiment_id = None
        
        self.experiment_id = experiment_id
        self.client = MlflowClient()
        
        # Run management
        self.active_run = None
        self.run_id = None
        
        # Metrics storage for batch operations
        self.metrics_buffer = {}
        self.step_counter = 0
        
        print(f"MLflow tracker initialized for experiment: {experiment_name}")
        print(f"Tracking URI: {tracking_uri}")
        if self.experiment_id:
            print(f"Experiment ID: {self.experiment_id}")
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False
    ) -> str:
        """Start a new MLflow run"""
        if self.active_run is not None:
            print("Warning: A run is already active. Ending previous run.")
            self.end_run()
        
        # Create run name with timestamp if not provided
        if run_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_name = f"toto_training_{timestamp}"
        
        # Default tags
        default_tags = {
            "training_framework": "pytorch",
            "model_type": "toto",
            "experiment_type": "time_series_forecasting",
            "created_by": "toto_training_pipeline"
        }
        
        if tags:
            default_tags.update(tags)
        
        self.active_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            nested=nested,
            tags=default_tags
        )
        
        self.run_id = self.active_run.info.run_id
        print(f"Started MLflow run: {run_name} (ID: {self.run_id})")
        
        return self.run_id
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters"""
        if self.active_run is None:
            print("Warning: No active run. Start a run first.")
            return
        
        # Convert complex objects to strings
        processed_params = {}
        for key, value in params.items():
            if isinstance(value, (str, int, float, bool)):
                processed_params[key] = value
            elif isinstance(value, (list, tuple)):
                processed_params[key] = str(value)
            elif hasattr(value, '__dict__'):  # Objects with attributes
                processed_params[key] = str(value)
            else:
                processed_params[key] = str(value)
        
        mlflow.log_params(processed_params)
        print(f"Logged {len(processed_params)} hyperparameters")
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric"""
        if self.active_run is None:
            print("Warning: No active run. Start a run first.")
            return
        
        if step is None:
            step = self.step_counter
            self.step_counter += 1
        
        mlflow.log_metric(key, value, step)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics"""
        if self.active_run is None:
            print("Warning: No active run. Start a run first.")
            return
        
        if step is None:
            step = self.step_counter
            self.step_counter += 1
        
        # Filter out non-numeric values
        numeric_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
                numeric_metrics[key] = value
            else:
                print(f"Warning: Skipping non-numeric metric {key}: {value}")
        
        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics, step)
    
    def log_training_metrics(
        self,
        epoch: int,
        batch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        train_accuracy: Optional[float] = None,
        val_accuracy: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        additional_metrics: Optional[Dict[str, float]] = None
    ):
        """Log training metrics with automatic step management"""
        metrics = {
            'train_loss': train_loss,
            'epoch': epoch,
            'batch': batch
        }
        
        if val_loss is not None:
            metrics['val_loss'] = val_loss
        if learning_rate is not None:
            metrics['learning_rate'] = learning_rate
        if train_accuracy is not None:
            metrics['train_accuracy'] = train_accuracy
        if val_accuracy is not None:
            metrics['val_accuracy'] = val_accuracy
        if gradient_norm is not None:
            metrics['gradient_norm'] = gradient_norm
        
        if additional_metrics:
            metrics.update(additional_metrics)
        
        global_step = epoch * 1000 + batch  # Create unique step
        self.log_metrics(metrics, step=global_step)
    
    def log_epoch_summary(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        train_accuracy: Optional[float] = None,
        val_accuracy: Optional[float] = None,
        epoch_time: Optional[float] = None,
        additional_metrics: Optional[Dict[str, float]] = None
    ):
        """Log epoch-level summary metrics"""
        metrics = {
            'epoch_train_loss': train_loss,
            'epoch': epoch
        }
        
        if val_loss is not None:
            metrics['epoch_val_loss'] = val_loss
        if train_accuracy is not None:
            metrics['epoch_train_accuracy'] = train_accuracy
        if val_accuracy is not None:
            metrics['epoch_val_accuracy'] = val_accuracy
        if epoch_time is not None:
            metrics['epoch_time_seconds'] = epoch_time
        
        if additional_metrics:
            metrics.update(additional_metrics)
        
        self.log_metrics(metrics, step=epoch)
    
    def log_system_metrics(
        self,
        cpu_percent: float,
        memory_percent: float,
        memory_used_gb: float,
        gpu_utilization: Optional[float] = None,
        gpu_memory_percent: Optional[float] = None,
        gpu_temperature: Optional[float] = None,
        step: Optional[int] = None
    ):
        """Log system performance metrics"""
        if not self._log_system_metrics_enabled:
            return
        
        metrics = {
            'system_cpu_percent': cpu_percent,
            'system_memory_percent': memory_percent,
            'system_memory_used_gb': memory_used_gb
        }
        
        if gpu_utilization is not None:
            metrics['system_gpu_utilization'] = gpu_utilization
        if gpu_memory_percent is not None:
            metrics['system_gpu_memory_percent'] = gpu_memory_percent
        if gpu_temperature is not None:
            metrics['system_gpu_temperature'] = gpu_temperature
        
        self.log_metrics(metrics, step)
    
    def log_model_checkpoint(
        self,
        model,
        checkpoint_path: str,
        epoch: int,
        metrics: Dict[str, float],
        model_name: Optional[str] = None
    ):
        """Log model checkpoint"""
        if not TORCH_AVAILABLE:
            print("Warning: PyTorch not available. Cannot log model.")
            return
        
        try:
            # Log the model
            if self.auto_log_model:
                model_name = model_name or f"toto_model_epoch_{epoch}"
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path=f"models/{model_name}",
                    registered_model_name=f"{self.experiment_name}_model"
                )
            
            # Log checkpoint file as artifact
            mlflow.log_artifact(checkpoint_path, "checkpoints")
            
            # Log checkpoint metrics
            checkpoint_metrics = {f"checkpoint_{k}": v for k, v in metrics.items()}
            self.log_metrics(checkpoint_metrics, step=epoch)
            
            print(f"Logged model checkpoint for epoch {epoch}")
            
        except Exception as e:
            print(f"Warning: Could not log model checkpoint: {e}")
    
    def log_best_model(
        self,
        model,
        model_path: str,
        best_metric_name: str,
        best_metric_value: float,
        epoch: int
    ):
        """Log best model with special tags"""
        if not TORCH_AVAILABLE:
            print("Warning: PyTorch not available. Cannot log best model.")
            return
        
        try:
            # Log as best model
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="models/best_model",
                registered_model_name=f"{self.experiment_name}_best_model"
            )
            
            # Log artifact
            mlflow.log_artifact(model_path, "best_model")
            
            # Log best model metrics
            mlflow.log_metrics({
                f"best_{best_metric_name}": best_metric_value,
                "best_model_epoch": epoch
            })
            
            # Tag as best model
            mlflow.set_tag("is_best_model", "true")
            mlflow.set_tag("best_metric", best_metric_name)
            
            print(f"Logged best model: {best_metric_name}={best_metric_value:.6f} at epoch {epoch}")
            
        except Exception as e:
            print(f"Warning: Could not log best model: {e}")
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log an artifact (file or directory)"""
        if self.active_run is None:
            print("Warning: No active run. Start a run first.")
            return
        
        try:
            mlflow.log_artifact(local_path, artifact_path)
            print(f"Logged artifact: {local_path}")
        except Exception as e:
            print(f"Warning: Could not log artifact {local_path}: {e}")
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """Log multiple artifacts from a directory"""
        if self.active_run is None:
            print("Warning: No active run. Start a run first.")
            return
        
        try:
            mlflow.log_artifacts(local_dir, artifact_path)
            print(f"Logged artifacts from: {local_dir}")
        except Exception as e:
            print(f"Warning: Could not log artifacts from {local_dir}: {e}")
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration as both parameters and artifact"""
        # Log as parameters
        self.log_hyperparameters(config)
        
        # Save and log as artifact
        config_path = Path("temp_config.json")
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            
            self.log_artifact(str(config_path), "config")
            config_path.unlink()  # Clean up temp file
            
        except Exception as e:
            print(f"Warning: Could not log config artifact: {e}")
    
    def log_predictions(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        step: int,
        prefix: str = "predictions"
    ):
        """Log prediction vs actual analysis"""
        try:
            # Calculate metrics
            mse = np.mean((predictions - actuals) ** 2)
            mae = np.mean(np.abs(predictions - actuals))
            rmse = np.sqrt(mse)
            
            # Correlation
            if len(predictions) > 1:
                correlation = np.corrcoef(predictions, actuals)[0, 1]
                r_squared = correlation ** 2
            else:
                correlation = 0.0
                r_squared = 0.0
            
            # Log metrics
            prediction_metrics = {
                f"{prefix}_mse": mse,
                f"{prefix}_mae": mae,
                f"{prefix}_rmse": rmse,
                f"{prefix}_correlation": correlation,
                f"{prefix}_r_squared": r_squared
            }
            
            self.log_metrics(prediction_metrics, step)
            
            # Save predictions as artifact
            predictions_data = {
                'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
                'actuals': actuals.tolist() if isinstance(actuals, np.ndarray) else actuals,
                'step': step,
                'metrics': prediction_metrics
            }
            
            temp_path = Path(f"temp_predictions_{step}.json")
            with open(temp_path, 'w') as f:
                json.dump(predictions_data, f, indent=2)
            
            self.log_artifact(str(temp_path), "predictions")
            temp_path.unlink()
            
        except Exception as e:
            print(f"Warning: Could not log predictions: {e}")
    
    def log_feature_importance(self, feature_names: List[str], importances: np.ndarray, step: int):
        """Log feature importance"""
        try:
            # Create importance dictionary
            importance_dict = dict(zip(feature_names, importances))
            
            # Log as metrics
            for name, importance in importance_dict.items():
                self.log_metric(f"feature_importance_{name}", importance, step)
            
            # Save as artifact
            temp_path = Path(f"temp_feature_importance_{step}.json")
            with open(temp_path, 'w') as f:
                json.dump({
                    'feature_names': feature_names,
                    'importances': importances.tolist(),
                    'step': step
                }, f, indent=2)
            
            self.log_artifact(str(temp_path), "feature_importance")
            temp_path.unlink()
            
        except Exception as e:
            print(f"Warning: Could not log feature importance: {e}")
    
    def set_tag(self, key: str, value: str):
        """Set a tag for the current run"""
        if self.active_run is None:
            print("Warning: No active run. Start a run first.")
            return
        
        mlflow.set_tag(key, value)
    
    def set_tags(self, tags: Dict[str, str]):
        """Set multiple tags"""
        for key, value in tags.items():
            self.set_tag(key, value)
    
    def end_run(self, status: str = "FINISHED"):
        """End the current MLflow run"""
        if self.active_run is not None:
            mlflow.end_run(status=status)
            print(f"Ended MLflow run: {self.run_id}")
            self.active_run = None
            self.run_id = None
        else:
            print("Warning: No active run to end.")
    
    def get_run_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current run"""
        if self.run_id is None:
            return None
        
        run = self.client.get_run(self.run_id)
        return {
            'run_id': run.info.run_id,
            'experiment_id': run.info.experiment_id,
            'status': run.info.status,
            'start_time': run.info.start_time,
            'end_time': run.info.end_time,
            'artifact_uri': run.info.artifact_uri,
            'lifecycle_stage': run.info.lifecycle_stage
        }
    
    def get_run_metrics(self) -> Dict[str, float]:
        """Get all metrics for the current run"""
        if self.run_id is None:
            return {}
        
        run = self.client.get_run(self.run_id)
        return run.data.metrics
    
    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple runs"""
        comparison = {
            'runs': {},
            'common_metrics': set(),
            'common_params': set()
        }
        
        for run_id in run_ids:
            try:
                run = self.client.get_run(run_id)
                comparison['runs'][run_id] = {
                    'metrics': run.data.metrics,
                    'params': run.data.params,
                    'tags': run.data.tags
                }
                
                if not comparison['common_metrics']:
                    comparison['common_metrics'] = set(run.data.metrics.keys())
                    comparison['common_params'] = set(run.data.params.keys())
                else:
                    comparison['common_metrics'] &= set(run.data.metrics.keys())
                    comparison['common_params'] &= set(run.data.params.keys())
                    
            except Exception as e:
                print(f"Warning: Could not retrieve run {run_id}: {e}")
        
        return comparison
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        status = "FAILED" if exc_type is not None else "FINISHED"
        self.end_run(status)


# Convenience function for quick MLflow setup
def create_mlflow_tracker(
    experiment_name: str,
    tracking_uri: str = "mlruns",
    **kwargs
) -> MLflowTracker:
    """Create an MLflow tracker with sensible defaults"""
    return MLflowTracker(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    if MLFLOW_AVAILABLE:
        with create_mlflow_tracker("test_experiment") as tracker:
            tracker.start_run("test_run")
            
            # Log configuration
            config = {
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 10,
                "model_type": "toto"
            }
            tracker.log_config(config)
            
            # Simulate training
            for epoch in range(3):
                train_loss = 1.0 - epoch * 0.1
                val_loss = train_loss + 0.1
                
                tracker.log_training_metrics(
                    epoch=epoch,
                    batch=0,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    learning_rate=0.001
                )
            
            print("Example MLflow logging completed!")
    else:
        print("MLflow not available for example")