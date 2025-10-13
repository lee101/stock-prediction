# Toto Training Logging and Monitoring System

A comprehensive, production-ready logging and monitoring system for the Toto retraining pipeline. This system provides structured logging, real-time monitoring, experiment tracking, and automated model management.

## 🚀 Features

### Core Logging Components

1. **Structured Training Logger** (`training_logger.py`)
   - Comprehensive logging for training metrics, loss curves, validation scores
   - System resource monitoring (CPU, memory, GPU)
   - Automatic log rotation and structured output
   - Thread-safe background monitoring

2. **TensorBoard Integration** (`tensorboard_monitor.py`)
   - Real-time visualization of loss, accuracy, gradients
   - Model weight and gradient histograms
   - System metrics dashboards
   - Prediction vs actual scatter plots
   - Learning rate schedule tracking

3. **MLflow Experiment Tracking** (`mlflow_tracker.py`)
   - Hyperparameter and metric tracking across runs
   - Model versioning and artifact storage
   - Run comparison and analysis
   - Integration with model registry

4. **Checkpoint Management** (`checkpoint_manager.py`)
   - Automatic saving of best models
   - Checkpoint rotation and cleanup
   - Model recovery and resuming
   - Integrity verification and backup

5. **Training Callbacks** (`training_callbacks.py`)
   - Early stopping with patience
   - Learning rate scheduling
   - Plateau detection and warnings
   - Metric trend analysis

6. **Dashboard Configuration** (`dashboard_config.py`)
   - Grafana dashboard templates
   - Prometheus monitoring setup
   - Docker Compose monitoring stack
   - Custom HTML dashboards

## 📁 File Structure

```
tototraining/
├── training_logger.py          # Core structured logging
├── tensorboard_monitor.py      # TensorBoard integration
├── mlflow_tracker.py          # MLflow experiment tracking
├── checkpoint_manager.py       # Model checkpoint management
├── training_callbacks.py       # Training callbacks (early stopping, LR scheduling)
├── dashboard_config.py         # Dashboard configuration generator
├── enhanced_trainer.py         # Complete trainer with all logging
├── test_logging_integration.py # Integration tests
└── LOGGING_README.md           # This documentation
```

## 🔧 Installation

### Required Dependencies

```bash
# Core dependencies
uv pip install torch pandas numpy psutil

# Optional but recommended
uv pip install tensorboard mlflow matplotlib GPUtil pyyaml
```

### Quick Start

1. **Run Integration Tests:**
```bash
python test_logging_integration.py
```

2. **Start Enhanced Training:**
```bash
python enhanced_trainer.py
```

3. **Monitor Training:**
```bash
# TensorBoard
tensorboard --logdir tensorboard_logs

# MLflow UI
mlflow ui --backend-store-uri mlruns

# Monitoring Stack (Docker)
cd dashboard_configs
docker-compose up -d
```

## 📊 Usage Examples

### Basic Structured Logging

```python
from training_logger import create_training_logger

with create_training_logger("my_experiment") as logger:
    logger.log_training_start({"learning_rate": 0.001, "batch_size": 32})
    
    for epoch in range(10):
        # Your training code here
        train_loss = train_model()
        val_loss = validate_model()
        
        logger.log_training_metrics(
            epoch=epoch,
            batch=0,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=0.001
        )
        
        logger.log_epoch_summary(epoch, train_loss, val_loss)
    
    logger.log_training_complete(10, 3600.0, {"best_val_loss": 0.5})
```

### TensorBoard Monitoring

```python
from tensorboard_monitor import create_tensorboard_monitor

with create_tensorboard_monitor("my_experiment") as tb:
    # Set model for graph logging
    tb.set_model(model, sample_input)
    
    for epoch in range(10):
        for batch, (x, y) in enumerate(dataloader):
            # Training step
            loss = train_step(x, y)
            
            # Log metrics
            tb.log_training_metrics(epoch, batch, loss, learning_rate=0.001)
            
            # Log gradients and weights
            tb.log_gradients()
            tb.log_model_weights()
        
        # Validation
        val_loss = validate()
        tb.log_validation_metrics(epoch, val_loss)
```

### MLflow Experiment Tracking

```python
from mlflow_tracker import create_mlflow_tracker

with create_mlflow_tracker("my_experiment") as tracker:
    # Start run
    tracker.start_run("training_run_1")
    
    # Log configuration
    config = {"learning_rate": 0.001, "batch_size": 32, "epochs": 100}
    tracker.log_config(config)
    
    for epoch in range(100):
        # Training
        train_loss, val_loss = train_epoch()
        
        # Log metrics
        tracker.log_training_metrics(
            epoch, 0, train_loss, val_loss, learning_rate=0.001
        )
        
        # Log best model
        if val_loss < best_loss:
            tracker.log_best_model(model, "model.pth", "val_loss", val_loss, epoch)
```

### Checkpoint Management

```python
from checkpoint_manager import create_checkpoint_manager

manager = create_checkpoint_manager(
    checkpoint_dir="checkpoints",
    monitor_metric="val_loss",
    mode="min"
)

for epoch in range(100):
    train_loss, val_loss = train_epoch()
    
    # Save checkpoint
    checkpoint_info = manager.save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        step=epoch * len(dataloader),
        metrics={"train_loss": train_loss, "val_loss": val_loss}
    )
    
    if checkpoint_info and checkpoint_info.is_best:
        print(f"New best model at epoch {epoch}!")

# Load best checkpoint
manager.load_best_checkpoint(model, optimizer)
```

### Training Callbacks

```python
from training_callbacks import (
    CallbackManager, EarlyStopping, ReduceLROnPlateau, MetricTracker
)

# Create callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", patience=10),
    ReduceLROnPlateau(optimizer, monitor="val_loss", patience=5, factor=0.5),
    MetricTracker(["train_loss", "val_loss"])
]

manager = CallbackManager(callbacks)
manager.on_training_start()

for epoch in range(100):
    train_loss, val_loss = train_epoch()
    
    # Check callbacks
    state = CallbackState(
        epoch=epoch, step=epoch*100, 
        train_loss=train_loss, val_loss=val_loss
    )
    
    should_stop = manager.on_epoch_end(state)
    if should_stop:
        print("Training stopped by callbacks")
        break

manager.on_training_end()
```

### Complete Enhanced Training

```python
from enhanced_trainer import EnhancedTotoTrainer
from toto_ohlc_trainer import TotoOHLCConfig

config = TotoOHLCConfig(
    patch_size=12, stride=6, embed_dim=128,
    num_layers=4, num_heads=8, dropout=0.1
)

with EnhancedTotoTrainer(
    config=config,
    experiment_name="my_experiment",
    enable_tensorboard=True,
    enable_mlflow=True
) as trainer:
    trainer.train(num_epochs=100)
```

## 📈 Monitoring Dashboards

### TensorBoard
- **URL:** http://localhost:6006
- **Features:** Real-time loss curves, gradient histograms, model graphs
- **Usage:** `tensorboard --logdir tensorboard_logs`

### MLflow UI
- **URL:** http://localhost:5000
- **Features:** Experiment comparison, model registry, artifact storage
- **Usage:** `mlflow ui --backend-store-uri mlruns`

### Grafana Dashboard
- **URL:** http://localhost:3000 (admin/admin)
- **Features:** System metrics, alerting, custom dashboards
- **Setup:** `docker-compose up -d` in `dashboard_configs/`

### Custom HTML Dashboard
- **Location:** `dashboard_configs/{experiment_name}_dashboard.html`
- **Features:** Simple monitoring without external dependencies

## 🔧 Configuration

### Environment Variables

```bash
# Optional: Customize directories
export TOTO_LOG_DIR="./custom_logs"
export TOTO_CHECKPOINT_DIR="./custom_checkpoints"
export TOTO_TENSORBOARD_DIR="./custom_tensorboard"
export TOTO_MLFLOW_URI="./custom_mlruns"
```

### Training Logger Configuration

```python
logger = TotoTrainingLogger(
    experiment_name="my_experiment",
    log_dir="logs",
    log_level=logging.INFO,
    enable_system_monitoring=True,
    system_monitor_interval=30.0,  # seconds
    metrics_buffer_size=1000
)
```

### Checkpoint Manager Configuration

```python
manager = CheckpointManager(
    checkpoint_dir="checkpoints",
    max_checkpoints=5,           # Keep last 5 checkpoints
    save_best_k=3,              # Keep top 3 best models
    monitor_metric="val_loss",
    mode="min",
    save_frequency=1,           # Save every epoch
    compress_checkpoints=True
)
```

### TensorBoard Configuration

```python
tb_monitor = TensorBoardMonitor(
    experiment_name="my_experiment",
    log_dir="tensorboard_logs",
    enable_model_graph=True,
    enable_weight_histograms=True,
    enable_gradient_histograms=True,
    histogram_freq=100,         # Log histograms every 100 batches
    image_freq=500             # Log images every 500 batches
)
```

## 🚨 Alerting and Monitoring

### Prometheus Alerts

The system generates Prometheus alerting rules for:
- Training stalled (no progress)
- High GPU temperature (>85°C)
- Low GPU utilization (<30%)
- High memory usage (>90%)
- Increasing training loss

### Custom Alerts

Add custom alerts in `dashboard_configs/toto_training_alerts.yml`:

```yaml
- alert: CustomAlert
  expr: your_metric > threshold
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Your alert description"
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors:**
   ```bash
   # Install missing dependencies
   uv pip install missing_package
   ```

2. **Permission Issues:**
   ```bash
   # Ensure write permissions for log directories
   chmod 755 logs/ checkpoints/ tensorboard_logs/
   ```

3. **GPU Monitoring Issues:**
   ```bash
   # Install GPU utilities
   uv pip install GPUtil nvidia-ml-py
   ```

4. **Port Conflicts:**
   ```bash
   # Check port usage
   netstat -tlnp | grep :6006  # TensorBoard
   netstat -tlnp | grep :5000  # MLflow
   netstat -tlnp | grep :3000  # Grafana
   ```

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Log Locations

- **Structured Logs:** `logs/{experiment_name}_{timestamp}/`
- **TensorBoard:** `tensorboard_logs/{experiment_name}_{timestamp}/`
- **MLflow:** `mlruns/{experiment_id}/{run_id}/`
- **Checkpoints:** `checkpoints/`
- **Dashboard Configs:** `dashboard_configs/`

## 📝 Best Practices

1. **Experiment Naming:** Use descriptive names with timestamps
2. **Log Levels:** Use appropriate log levels (DEBUG for development, INFO for production)
3. **Disk Space:** Monitor disk usage, especially for large models
4. **Backup:** Regularly backup best models and important experiments
5. **Resource Monitoring:** Keep an eye on system resources during training
6. **Clean Up:** Periodically clean old checkpoints and logs

## 🤝 Contributing

To extend the logging system:

1. **New Logger:** Inherit from `BaseCallback` for training events
2. **New Monitor:** Follow the pattern of existing monitors
3. **New Dashboard:** Add panels to `dashboard_config.py`
4. **Testing:** Add tests to `test_logging_integration.py`

## 📄 License

This logging system is part of the Toto training pipeline and follows the same license terms.

## 🙋 Support

For issues and questions:

1. Check the troubleshooting section
2. Run integration tests: `python test_logging_integration.py`
3. Check log files for detailed error messages
4. Review configuration settings

---

**Happy Training! 🚀**