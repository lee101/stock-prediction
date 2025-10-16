#!/usr/bin/env python3
"""
Demo of the Toto Training Logging System
Demonstrates the complete logging and monitoring system with a simple training simulation.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

# Import our logging components
from training_logger import create_training_logger
from checkpoint_manager import create_checkpoint_manager
from training_callbacks import (
    CallbackManager, CallbackState, EarlyStopping, 
    ReduceLROnPlateau, MetricTracker
)

try:
    from tensorboard_monitor import create_tensorboard_monitor
    TENSORBOARD_AVAILABLE = True
except:
    TENSORBOARD_AVAILABLE = False

try:
    from mlflow_tracker import create_mlflow_tracker
    MLFLOW_AVAILABLE = True
except:
    MLFLOW_AVAILABLE = False

from dashboard_config import create_dashboard_generator


class SimpleModel(nn.Module):
    """Simple model for demonstration"""
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )
    
    def forward(self, x):
        return self.layers(x)


def generate_fake_data(batch_size=32):
    """Generate fake training data"""
    x = torch.randn(batch_size, 10)
    # Create target with some pattern
    y = (x[:, 0] * 0.5 + x[:, 1] * 0.3 - x[:, 2] * 0.2 + torch.randn(batch_size) * 0.1).unsqueeze(1)
    return x, y


def simulate_training():
    """Simulate a complete training process with all logging components"""
    
    print("🚀 Starting Toto Training Logging System Demo")
    print("=" * 60)
    
    # Configuration
    experiment_name = f"demo_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config = {
        "learning_rate": 0.01,
        "batch_size": 32,
        "epochs": 20,
        "model_type": "simple_mlp",
        "hidden_layers": [50, 20],
        "dropout": 0.2
    }
    
    print(f"📝 Experiment: {experiment_name}")
    print(f"📋 Config: {config}")
    
    # Initialize model
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.MSELoss()
    
    # Initialize logging systems
    print("\n🔧 Initializing Logging Systems...")
    
    # 1. Structured Logger
    training_logger = create_training_logger(experiment_name, "logs")
    training_logger.log_training_start(config)
    
    # 2. TensorBoard (if available)
    tensorboard_monitor = None
    if TENSORBOARD_AVAILABLE:
        try:
            tensorboard_monitor = create_tensorboard_monitor(experiment_name, "tensorboard_logs")
            # Create sample input for model graph
            sample_input = torch.randn(1, 10)
            tensorboard_monitor.set_model(model, sample_input)
            print("✅ TensorBoard Monitor initialized")
        except Exception as e:
            print(f"⚠️  TensorBoard Monitor failed: {e}")
    
    # 3. MLflow (if available)
    mlflow_tracker = None
    if MLFLOW_AVAILABLE:
        try:
            mlflow_tracker = create_mlflow_tracker(experiment_name, "mlruns")
            run_id = mlflow_tracker.start_run(f"{experiment_name}_run")
            mlflow_tracker.log_config(config)
            print("✅ MLflow Tracker initialized")
        except Exception as e:
            print(f"⚠️  MLflow Tracker failed: {e}")
    
    # 4. Checkpoint Manager
    checkpoint_manager = create_checkpoint_manager(
        "checkpoints", monitor_metric="val_loss", mode="min"
    )
    print("✅ Checkpoint Manager initialized")
    
    # 5. Training Callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, verbose=True),
        ReduceLROnPlateau(optimizer, monitor="val_loss", patience=3, factor=0.7, verbose=True),
        MetricTracker(['train_loss', 'val_loss', 'learning_rate'])
    ]
    callback_manager = CallbackManager(callbacks)
    callback_manager.on_training_start()
    print("✅ Training Callbacks initialized")
    
    # 6. Dashboard Generator
    dashboard_generator = create_dashboard_generator(experiment_name)
    dashboard_config = dashboard_generator.create_training_dashboard()
    dashboard_generator.save_configurations(dashboard_config)
    dashboard_generator.save_html_dashboard(dashboard_config)
    print("✅ Dashboard Configuration generated")
    
    print(f"\n🎯 Starting Training Loop...")
    print("-" * 40)
    
    training_start_time = time.time()
    best_val_loss = float('inf')
    
    try:
        for epoch in range(config["epochs"]):
            epoch_start_time = time.time()
            
            # Training phase
            model.train()
            train_losses = []
            gradient_norms = []
            
            # Simulate multiple batches
            num_batches = 10
            for batch_idx in range(num_batches):
                x_batch, y_batch = generate_fake_data(config["batch_size"])
                
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                
                # Calculate gradient norm
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                gradient_norms.append(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
                
                optimizer.step()
                train_losses.append(loss.item())
                
                # Log batch metrics occasionally
                if tensorboard_monitor and batch_idx % 3 == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    tensorboard_monitor.log_training_metrics(
                        epoch, batch_idx, loss.item(), current_lr
                    )
                
                if mlflow_tracker and batch_idx % 5 == 0:
                    mlflow_tracker.log_training_metrics(
                        epoch, batch_idx, loss.item(),
                        learning_rate=optimizer.param_groups[0]['lr'],
                        gradient_norm=gradient_norms[-1]
                    )
            
            train_loss = np.mean(train_losses)
            avg_grad_norm = np.mean(gradient_norms)
            
            # Validation phase
            model.eval()
            val_losses = []
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for _ in range(3):  # 3 validation batches
                    x_val, y_val = generate_fake_data(config["batch_size"])
                    outputs = model(x_val)
                    val_loss = criterion(outputs, y_val)
                    val_losses.append(val_loss.item())
                    
                    all_predictions.extend(outputs.cpu().numpy().flatten())
                    all_targets.extend(y_val.cpu().numpy().flatten())
            
            val_loss = np.mean(val_losses)
            
            # Calculate additional metrics
            predictions_array = np.array(all_predictions)
            targets_array = np.array(all_targets)
            mae = np.mean(np.abs(predictions_array - targets_array))
            correlation = np.corrcoef(predictions_array, targets_array)[0, 1] if len(predictions_array) > 1 else 0
            
            epoch_time = time.time() - epoch_start_time
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log to all systems
            training_logger.log_training_metrics(
                epoch=epoch,
                batch=num_batches-1,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=current_lr,
                gradient_norm=avg_grad_norm,
                additional_metrics={'mae': mae, 'correlation': correlation}
            )
            
            if tensorboard_monitor:
                tensorboard_monitor.log_validation_metrics(epoch, val_loss, additional_metrics={'mae': mae})
                tensorboard_monitor.log_gradients()
                tensorboard_monitor.log_model_weights()
                
                # Log system metrics
                sys_metrics = training_logger.get_system_metrics()
                tensorboard_monitor.log_system_metrics(
                    sys_metrics.cpu_percent,
                    sys_metrics.memory_percent,
                    sys_metrics.gpu_utilization,
                    sys_metrics.gpu_memory_used_gb / sys_metrics.gpu_memory_total_gb * 100 if sys_metrics.gpu_memory_total_gb else None,
                    sys_metrics.gpu_temperature
                )
            
            if mlflow_tracker:
                mlflow_tracker.log_epoch_summary(
                    epoch, train_loss, val_loss,
                    epoch_time=epoch_time,
                    additional_metrics={'mae': mae, 'correlation': correlation}
                )
                
                # Log predictions occasionally
                if epoch % 5 == 0:
                    mlflow_tracker.log_predictions(
                        predictions_array, targets_array, epoch, "validation"
                    )
            
            # Save checkpoint
            metrics_for_checkpoint = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'mae': mae,
                'correlation': correlation,
                'learning_rate': current_lr
            }
            
            checkpoint_info = checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=epoch * num_batches,
                metrics=metrics_for_checkpoint,
                tags={'experiment': experiment_name}
            )
            
            # Check for best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                training_logger.log_best_model(
                    checkpoint_info.path if checkpoint_info else "unknown",
                    "val_loss",
                    val_loss
                )
                
                if mlflow_tracker:
                    mlflow_tracker.log_best_model(
                        model, checkpoint_info.path if checkpoint_info else "",
                        "val_loss", val_loss, epoch
                    )
            
            # Callback processing
            callback_state = CallbackState(
                epoch=epoch,
                step=epoch * num_batches,
                train_loss=train_loss,
                val_loss=val_loss,
                train_metrics={'mae': mae, 'gradient_norm': avg_grad_norm},
                val_metrics={'mae': mae, 'correlation': correlation},
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict()
            )
            
            should_stop = callback_manager.on_epoch_end(callback_state)
            
            # Log epoch summary
            samples_per_sec = (num_batches * config["batch_size"]) / epoch_time
            training_logger.log_epoch_summary(
                epoch, train_loss, val_loss, epoch_time, samples_per_sec
            )
            
            # Print progress
            print(f"Epoch {epoch+1:2d}/{config['epochs']:2d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {epoch_time:.1f}s")
            
            if should_stop:
                training_logger.log_early_stopping(epoch, 5, "val_loss", best_val_loss)
                print(f"⏹️  Early stopping triggered at epoch {epoch}")
                break
    
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        training_logger.log_error(e, "training loop")
    
    finally:
        # End training
        total_time = time.time() - training_start_time
        
        callback_manager.on_training_end()
        
        final_metrics = {'best_val_loss': best_val_loss, 'total_epochs': epoch + 1}
        training_logger.log_training_complete(epoch + 1, total_time, final_metrics)
        
        if mlflow_tracker:
            final_metrics.update({
                'final_train_loss': train_loss,
                'final_val_loss': val_loss,
                'total_training_time_hours': total_time / 3600
            })
            mlflow_tracker.log_hyperparameters(config)
            for metric_name, metric_value in final_metrics.items():
                mlflow_tracker.log_metric(metric_name, metric_value)
            mlflow_tracker.end_run()
        
        if tensorboard_monitor:
            tensorboard_monitor.close()
        
        training_logger.stop_system_monitoring()
        training_logger.save_training_summary()
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TRAINING SUMMARY")
        print("=" * 60)
        print(f"✅ Total Epochs: {epoch + 1}")
        print(f"⏱️  Total Time: {total_time:.2f}s ({total_time/60:.1f}m)")
        print(f"🏆 Best Val Loss: {best_val_loss:.6f}")
        print(f"📈 Final Train Loss: {train_loss:.6f}")
        print(f"📉 Final Val Loss: {val_loss:.6f}")
        
        # Show where to find results
        print(f"\n🎯 MONITORING RESULTS")
        print("-" * 40)
        print(f"📁 Logs: logs/{experiment_name}_*")
        print(f"💾 Checkpoints: checkpoints/")
        print(f"🎛️  Dashboard: dashboard_configs/{experiment_name}_dashboard.html")
        
        if TENSORBOARD_AVAILABLE:
            print(f"📊 TensorBoard: tensorboard --logdir tensorboard_logs")
        
        if MLFLOW_AVAILABLE:
            print(f"🧪 MLflow: mlflow ui --backend-store-uri mlruns")
        
        print(f"🐳 Full Stack: docker-compose up -d (in dashboard_configs/)")
        
        checkpoint_summary = checkpoint_manager.get_checkpoint_summary()
        print(f"💽 Checkpoints: {checkpoint_summary['total_checkpoints']} regular, {checkpoint_summary['best_checkpoints']} best")
        
        print(f"\n🎉 Demo completed successfully!")


if __name__ == "__main__":
    simulate_training()