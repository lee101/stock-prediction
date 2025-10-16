#!/usr/bin/env python3
"""
Dashboard Configuration for Toto Training Pipeline
Provides configuration and setup for monitoring dashboards (Grafana, custom web dashboard, etc.).
"""

import os
import json
import yaml
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict


@dataclass
class DashboardPanel:
    """Configuration for a dashboard panel"""
    title: str
    type: str  # 'graph', 'stat', 'table', 'heatmap', etc.
    metrics: List[str]
    width: int = 12
    height: int = 8
    refresh: str = "5s"
    time_range: str = "1h"
    aggregation: str = "mean"
    description: Optional[str] = None
    thresholds: Optional[Dict[str, float]] = None
    colors: Optional[List[str]] = None


@dataclass
class DashboardRow:
    """Configuration for a dashboard row"""
    title: str
    panels: List[DashboardPanel]
    collapsed: bool = False


@dataclass
class DashboardConfig:
    """Complete dashboard configuration"""
    title: str
    description: str
    rows: List[DashboardRow]
    refresh_interval: str = "5s"
    time_range: str = "1h"
    timezone: str = "browser"
    theme: str = "dark"
    tags: Optional[List[str]] = None


class DashboardGenerator:
    """
    Generates dashboard configurations for various monitoring systems.
    Supports Grafana, custom web dashboards, and configuration exports.
    """
    
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.config_dir = Path("dashboard_configs")
        self.config_dir.mkdir(exist_ok=True)
        
    def create_training_dashboard(self) -> DashboardConfig:
        """Create a comprehensive training monitoring dashboard"""
        
        # Training Metrics Row
        training_panels = [
            DashboardPanel(
                title="Training & Validation Loss",
                type="graph",
                metrics=["train_loss", "val_loss"],
                width=6,
                height=6,
                description="Training and validation loss curves over time",
                colors=["#1f77b4", "#ff7f0e"]
            ),
            DashboardPanel(
                title="Learning Rate",
                type="graph",
                metrics=["learning_rate"],
                width=6,
                height=6,
                description="Learning rate schedule over time",
                colors=["#2ca02c"]
            ),
            DashboardPanel(
                title="Current Epoch",
                type="stat",
                metrics=["epoch"],
                width=3,
                height=4,
                description="Current training epoch"
            ),
            DashboardPanel(
                title="Training Speed",
                type="stat",
                metrics=["samples_per_sec"],
                width=3,
                height=4,
                description="Training throughput (samples/second)",
                thresholds={"warning": 100, "critical": 50}
            ),
            DashboardPanel(
                title="Best Validation Loss",
                type="stat",
                metrics=["best_val_loss"],
                width=3,
                height=4,
                description="Best validation loss achieved",
                colors=["#d62728"]
            ),
            DashboardPanel(
                title="Patience Counter",
                type="stat",
                metrics=["early_stopping_patience"],
                width=3,
                height=4,
                description="Early stopping patience counter",
                thresholds={"warning": 5, "critical": 8}
            )
        ]
        
        # Model Metrics Row
        model_panels = [
            DashboardPanel(
                title="Gradient Norm",
                type="graph",
                metrics=["gradient_norm"],
                width=6,
                height=6,
                description="Gradient norm over time (gradient clipping indicator)",
                thresholds={"warning": 1.0, "critical": 10.0}
            ),
            DashboardPanel(
                title="Model Accuracy",
                type="graph",
                metrics=["train_accuracy", "val_accuracy"],
                width=6,
                height=6,
                description="Training and validation accuracy",
                colors=["#1f77b4", "#ff7f0e"]
            ),
            DashboardPanel(
                title="Weight Statistics",
                type="table",
                metrics=["weight_mean", "weight_std", "weight_norm"],
                width=12,
                height=6,
                description="Model weight statistics by layer"
            )
        ]
        
        # System Metrics Row
        system_panels = [
            DashboardPanel(
                title="CPU Usage",
                type="graph",
                metrics=["system_cpu_percent"],
                width=3,
                height=6,
                description="CPU utilization percentage",
                thresholds={"warning": 80, "critical": 95},
                colors=["#2ca02c"]
            ),
            DashboardPanel(
                title="Memory Usage",
                type="graph",
                metrics=["system_memory_percent"],
                width=3,
                height=6,
                description="Memory utilization percentage",
                thresholds={"warning": 80, "critical": 95},
                colors=["#ff7f0e"]
            ),
            DashboardPanel(
                title="GPU Utilization",
                type="graph",
                metrics=["system_gpu_utilization"],
                width=3,
                height=6,
                description="GPU utilization percentage",
                thresholds={"warning": 50, "critical": 30},
                colors=["#d62728"]
            ),
            DashboardPanel(
                title="GPU Memory",
                type="graph",
                metrics=["system_gpu_memory_percent"],
                width=3,
                height=6,
                description="GPU memory usage percentage",
                thresholds={"warning": 80, "critical": 95},
                colors=["#9467bd"]
            ),
            DashboardPanel(
                title="GPU Temperature",
                type="stat",
                metrics=["system_gpu_temperature"],
                width=4,
                height=4,
                description="GPU temperature (°C)",
                thresholds={"warning": 75, "critical": 85}
            ),
            DashboardPanel(
                title="Disk Usage",
                type="stat",
                metrics=["system_disk_used_gb"],
                width=4,
                height=4,
                description="Disk space used (GB)"
            ),
            DashboardPanel(
                title="Training Time",
                type="stat",
                metrics=["training_time_hours"],
                width=4,
                height=4,
                description="Total training time (hours)"
            )
        ]
        
        # Loss Analysis Row
        analysis_panels = [
            DashboardPanel(
                title="Loss Comparison",
                type="graph",
                metrics=["train_loss", "val_loss", "loss_gap"],
                width=8,
                height=6,
                description="Training vs validation loss with gap analysis",
                colors=["#1f77b4", "#ff7f0e", "#2ca02c"]
            ),
            DashboardPanel(
                title="Overfitting Indicator",
                type="stat",
                metrics=["overfitting_score"],
                width=4,
                height=6,
                description="Overfitting risk score",
                thresholds={"warning": 0.3, "critical": 0.5}
            ),
            DashboardPanel(
                title="Training Progress",
                type="graph",
                metrics=["progress_percent"],
                width=6,
                height=4,
                description="Training progress percentage"
            ),
            DashboardPanel(
                title="ETA",
                type="stat",
                metrics=["estimated_time_remaining"],
                width=6,
                height=4,
                description="Estimated time remaining"
            )
        ]
        
        # Create dashboard rows
        rows = [
            DashboardRow(
                title="Training Metrics",
                panels=training_panels
            ),
            DashboardRow(
                title="Model Performance",
                panels=model_panels
            ),
            DashboardRow(
                title="System Resources",
                panels=system_panels
            ),
            DashboardRow(
                title="Training Analysis",
                panels=analysis_panels
            )
        ]
        
        # Create complete dashboard config
        dashboard = DashboardConfig(
            title=f"Toto Training Dashboard - {self.experiment_name}",
            description="Comprehensive monitoring dashboard for Toto model training",
            rows=rows,
            refresh_interval="5s",
            time_range="1h",
            tags=["toto", "training", "ml", "monitoring"]
        )
        
        return dashboard
    
    def generate_grafana_config(self, dashboard_config: DashboardConfig) -> Dict[str, Any]:
        """Generate Grafana dashboard JSON configuration"""
        
        grafana_dashboard = {
            "dashboard": {
                "id": None,
                "title": dashboard_config.title,
                "description": dashboard_config.description,
                "tags": dashboard_config.tags or [],
                "timezone": dashboard_config.timezone,
                "refresh": dashboard_config.refresh_interval,
                "time": {
                    "from": f"now-{dashboard_config.time_range}",
                    "to": "now"
                },
                "timepicker": {
                    "refresh_intervals": ["5s", "10s", "30s", "1m", "5m", "15m", "30m", "1h", "2h", "1d"]
                },
                "panels": [],
                "schemaVersion": 27,
                "version": 1
            }
        }
        
        panel_id = 1
        grid_y = 0
        
        for row in dashboard_config.rows:
            # Add row panel
            row_panel = {
                "collapsed": row.collapsed,
                "gridPos": {"h": 1, "w": 24, "x": 0, "y": grid_y},
                "id": panel_id,
                "panels": [],
                "title": row.title,
                "type": "row"
            }
            
            grafana_dashboard["dashboard"]["panels"].append(row_panel)
            panel_id += 1
            grid_y += 1
            
            grid_x = 0
            max_height = 0
            
            # Add panels in this row
            for panel in row.panels:
                grafana_panel = self._create_grafana_panel(panel, panel_id, grid_x, grid_y)
                grafana_dashboard["dashboard"]["panels"].append(grafana_panel)
                
                panel_id += 1
                grid_x += panel.width
                max_height = max(max_height, panel.height)
                
                # Start new row if needed
                if grid_x >= 24:
                    grid_x = 0
                    grid_y += max_height
                    max_height = 0
            
            # Move to next row
            if grid_x > 0:
                grid_y += max_height
        
        return grafana_dashboard
    
    def _create_grafana_panel(self, panel: DashboardPanel, panel_id: int, x: int, y: int) -> Dict[str, Any]:
        """Create a Grafana panel configuration"""
        
        base_panel = {
            "id": panel_id,
            "title": panel.title,
            "type": panel.type,
            "gridPos": {
                "h": panel.height,
                "w": panel.width,
                "x": x,
                "y": y
            },
            "options": {},
            "fieldConfig": {
                "defaults": {},
                "overrides": []
            },
            "targets": []
        }
        
        # Add description if provided
        if panel.description:
            base_panel["description"] = panel.description
        
        # Add thresholds if provided
        if panel.thresholds:
            base_panel["fieldConfig"]["defaults"]["thresholds"] = {
                "mode": "absolute",
                "steps": [
                    {"color": "green", "value": None},
                    {"color": "yellow", "value": panel.thresholds.get("warning", 0)},
                    {"color": "red", "value": panel.thresholds.get("critical", 0)}
                ]
            }
        
        # Add colors if provided
        if panel.colors:
            base_panel["fieldConfig"]["overrides"] = [
                {
                    "matcher": {"id": "byName", "options": metric},
                    "properties": [{"id": "color", "value": {"mode": "fixed", "fixedColor": color}}]
                }
                for metric, color in zip(panel.metrics, panel.colors)
            ]
        
        # Configure targets (metrics)
        for i, metric in enumerate(panel.metrics):
            target = {
                "expr": f'{metric}{{job="toto-training"}}',
                "interval": "",
                "legendFormat": metric.replace("_", " ").title(),
                "refId": chr(65 + i)  # A, B, C, etc.
            }
            base_panel["targets"].append(target)
        
        # Panel-specific configuration
        if panel.type == "graph":
            base_panel["options"] = {
                "legend": {"displayMode": "visible", "placement": "bottom"},
                "tooltip": {"mode": "multi"}
            }
            base_panel["fieldConfig"]["defaults"]["custom"] = {
                "drawStyle": "line",
                "lineInterpolation": "linear",
                "lineWidth": 2,
                "fillOpacity": 10,
                "gradientMode": "none",
                "spanNulls": False,
                "insertNulls": False,
                "showPoints": "never",
                "pointSize": 5,
                "stacking": {"mode": "none", "group": "A"},
                "axisPlacement": "auto",
                "axisLabel": "",
                "scaleDistribution": {"type": "linear"},
                "hideFrom": {"legend": False, "tooltip": False, "vis": False},
                "thresholdsStyle": {"mode": "off"}
            }
        
        elif panel.type == "stat":
            base_panel["options"] = {
                "reduceOptions": {
                    "values": False,
                    "calcs": ["lastNotNull"],
                    "fields": ""
                },
                "orientation": "auto",
                "textMode": "auto",
                "colorMode": "value",
                "graphMode": "area",
                "justifyMode": "auto"
            }
        
        elif panel.type == "table":
            base_panel["options"] = {
                "showHeader": True
            }
            base_panel["fieldConfig"]["defaults"]["custom"] = {
                "align": "auto",
                "displayMode": "auto"
            }
        
        return base_panel
    
    def generate_prometheus_config(self) -> Dict[str, Any]:
        """Generate Prometheus scrape configuration"""
        
        prometheus_config = {
            "global": {
                "scrape_interval": "15s",
                "evaluation_interval": "15s"
            },
            "scrape_configs": [
                {
                    "job_name": "toto-training",
                    "scrape_interval": "5s",
                    "static_configs": [
                        {
                            "targets": ["localhost:8000"]
                        }
                    ],
                    "metrics_path": "/metrics",
                    "scrape_timeout": "5s"
                }
            ],
            "rule_files": ["toto_training_alerts.yml"]
        }
        
        return prometheus_config
    
    def generate_alerting_rules(self) -> Dict[str, Any]:
        """Generate Prometheus alerting rules"""
        
        alerting_rules = {
            "groups": [
                {
                    "name": "toto_training_alerts",
                    "rules": [
                        {
                            "alert": "TrainingStalled",
                            "expr": "increase(epoch[10m]) == 0",
                            "for": "10m",
                            "labels": {"severity": "warning"},
                            "annotations": {
                                "summary": "Training appears to be stalled",
                                "description": "No progress in epochs for the last 10 minutes"
                            }
                        },
                        {
                            "alert": "HighGPUTemperature",
                            "expr": "system_gpu_temperature > 85",
                            "for": "2m",
                            "labels": {"severity": "critical"},
                            "annotations": {
                                "summary": "GPU temperature is critically high",
                                "description": "GPU temperature is {{ $value }}°C"
                            }
                        },
                        {
                            "alert": "LowGPUUtilization",
                            "expr": "system_gpu_utilization < 30",
                            "for": "5m",
                            "labels": {"severity": "warning"},
                            "annotations": {
                                "summary": "Low GPU utilization detected",
                                "description": "GPU utilization is {{ $value }}%"
                            }
                        },
                        {
                            "alert": "HighMemoryUsage",
                            "expr": "system_memory_percent > 90",
                            "for": "5m",
                            "labels": {"severity": "warning"},
                            "annotations": {
                                "summary": "High memory usage detected",
                                "description": "Memory usage is {{ $value }}%"
                            }
                        },
                        {
                            "alert": "TrainingLossIncreasing",
                            "expr": "increase(train_loss[30m]) > 0",
                            "for": "30m",
                            "labels": {"severity": "warning"},
                            "annotations": {
                                "summary": "Training loss is increasing",
                                "description": "Training loss has been increasing for 30 minutes"
                            }
                        }
                    ]
                }
            ]
        }
        
        return alerting_rules
    
    def generate_docker_compose(self) -> str:
        """Generate Docker Compose configuration for monitoring stack"""
        
        docker_compose = """
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: toto-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - ./toto_training_alerts.yml:/etc/prometheus/toto_training_alerts.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:latest
    container_name: toto-grafana
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning
      - ./grafana/dashboards:/etc/grafana/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    networks:
      - monitoring
    depends_on:
      - prometheus

  node-exporter:
    image: prom/node-exporter:latest
    container_name: toto-node-exporter
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    networks:
      - monitoring

networks:
  monitoring:
    driver: bridge

volumes:
  prometheus_data:
  grafana_data:
"""
        return docker_compose.strip()
    
    def save_configurations(self, dashboard_config: DashboardConfig):
        """Save all dashboard configurations to files"""
        
        # Save dashboard config as JSON
        dashboard_file = self.config_dir / f"{self.experiment_name}_dashboard_config.json"
        with open(dashboard_file, 'w') as f:
            json.dump(asdict(dashboard_config), f, indent=2, default=str)
        
        # Generate and save Grafana config
        grafana_config = self.generate_grafana_config(dashboard_config)
        grafana_file = self.config_dir / f"{self.experiment_name}_grafana_dashboard.json"
        with open(grafana_file, 'w') as f:
            json.dump(grafana_config, f, indent=2)
        
        # Generate and save Prometheus config
        prometheus_config = self.generate_prometheus_config()
        prometheus_file = self.config_dir / "prometheus.yml"
        with open(prometheus_file, 'w') as f:
            yaml.dump(prometheus_config, f, default_flow_style=False)
        
        # Generate and save alerting rules
        alerting_rules = self.generate_alerting_rules()
        alerts_file = self.config_dir / "toto_training_alerts.yml"
        with open(alerts_file, 'w') as f:
            yaml.dump(alerting_rules, f, default_flow_style=False)
        
        # Generate and save Docker Compose
        docker_compose = self.generate_docker_compose()
        compose_file = self.config_dir / "docker-compose.yml"
        with open(compose_file, 'w') as f:
            f.write(docker_compose)
        
        # Create Grafana provisioning configs
        grafana_dir = self.config_dir / "grafana"
        provisioning_dir = grafana_dir / "provisioning"
        dashboards_dir = provisioning_dir / "dashboards"
        datasources_dir = provisioning_dir / "datasources"
        
        for dir_path in [grafana_dir, provisioning_dir, dashboards_dir, datasources_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Datasource provisioning
        datasource_config = {
            "apiVersion": 1,
            "datasources": [
                {
                    "name": "Prometheus",
                    "type": "prometheus",
                    "access": "proxy",
                    "url": "http://prometheus:9090",
                    "isDefault": True
                }
            ]
        }
        
        with open(datasources_dir / "prometheus.yml", 'w') as f:
            yaml.dump(datasource_config, f, default_flow_style=False)
        
        # Dashboard provisioning
        dashboard_provisioning = {
            "apiVersion": 1,
            "providers": [
                {
                    "name": "toto-dashboards",
                    "orgId": 1,
                    "folder": "",
                    "type": "file",
                    "disableDeletion": False,
                    "updateIntervalSeconds": 10,
                    "allowUiUpdates": True,
                    "options": {
                        "path": "/etc/grafana/dashboards"
                    }
                }
            ]
        }
        
        with open(dashboards_dir / "dashboard.yml", 'w') as f:
            yaml.dump(dashboard_provisioning, f, default_flow_style=False)
        
        # Copy Grafana dashboard JSON to dashboards directory
        grafana_dashboards_dir = grafana_dir / "dashboards"
        grafana_dashboards_dir.mkdir(parents=True, exist_ok=True)
        
        dashboard_dest = grafana_dashboards_dir / f"{self.experiment_name}_dashboard.json"
        if grafana_file.exists():
            shutil.copy2(grafana_file, dashboard_dest)
        
        print(f"Dashboard configurations saved to {self.config_dir}")
        print("To start monitoring stack: docker-compose up -d")
        print("Grafana will be available at: http://localhost:3000 (admin/admin)")
        print("Prometheus will be available at: http://localhost:9090")
    
    def generate_simple_html_dashboard(self, dashboard_config: DashboardConfig) -> str:
        """Generate a simple HTML dashboard for basic monitoring"""
        
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #1a1a1a;
            color: #ffffff;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }}
        .row {{
            margin-bottom: 30px;
        }}
        .row-title {{
            font-size: 1.5em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #4CAF50;
        }}
        .panel-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
        }}
        .panel {{
            background-color: #2d2d2d;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            flex: 1;
            min-width: 300px;
        }}
        .panel h3 {{
            margin-top: 0;
            color: #ffffff;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #4CAF50;
            text-align: center;
        }}
        .metric-label {{
            text-align: center;
            color: #cccccc;
            margin-top: 5px;
        }}
        .plot {{
            width: 100%;
            height: 300px;
        }}
        .status-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        .status-good {{ background-color: #4CAF50; }}
        .status-warning {{ background-color: #FF9800; }}
        .status-critical {{ background-color: #F44336; }}
        .refresh-info {{
            position: fixed;
            top: 10px;
            right: 10px;
            background-color: rgba(0, 0, 0, 0.7);
            padding: 10px;
            border-radius: 5px;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="refresh-info">
        <span class="status-indicator status-good"></span>
        Auto-refresh: {refresh_interval}
    </div>
    
    <div class="header">
        <h1>{title}</h1>
        <p>{description}</p>
        <p>Last updated: <span id="last-update"></span></p>
    </div>
    
    {content}
    
    <script>
        // Auto-refresh functionality
        function updateTimestamp() {{
            document.getElementById('last-update').textContent = new Date().toLocaleString();
        }}
        
        // Simulate real-time data updates
        function refreshData() {{
            // In a real implementation, this would fetch data from your training logger
            updateTimestamp();
            // Add your data fetching and chart updating logic here
        }}
        
        // Initialize
        updateTimestamp();
        setInterval(refreshData, 5000); // Refresh every 5 seconds
        
        // Sample data for demonstration
        const sampleData = {{
            train_loss: [1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
            val_loss: [1.1, 1.0, 0.9, 0.8, 0.7, 0.65],
            epochs: [1, 2, 3, 4, 5, 6]
        }};
        
        // Create sample plots
        function createSamplePlots() {{
            // Loss curve
            const lossTrace1 = {{
                x: sampleData.epochs,
                y: sampleData.train_loss,
                mode: 'lines+markers',
                name: 'Training Loss',
                line: {{ color: '#1f77b4' }}
            }};
            
            const lossTrace2 = {{
                x: sampleData.epochs,
                y: sampleData.val_loss,
                mode: 'lines+markers',
                name: 'Validation Loss',
                line: {{ color: '#ff7f0e' }}
            }};
            
            const lossLayout = {{
                title: '',
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: {{ color: '#ffffff' }},
                xaxis: {{ title: 'Epoch', gridcolor: '#444444' }},
                yaxis: {{ title: 'Loss', gridcolor: '#444444' }}
            }};
            
            Plotly.newPlot('loss-plot', [lossTrace1, lossTrace2], lossLayout);
        }}
        
        // Initialize plots when page loads
        window.onload = function() {{
            createSamplePlots();
        }};
    </script>
</body>
</html>
"""
        
        # Generate content for each row
        content_sections = []
        
        for row in dashboard_config.rows:
            row_content = f'<div class="row"><div class="row-title">{row.title}</div><div class="panel-container">'
            
            for panel in row.panels:
                if panel.type == 'stat':
                    panel_content = f'''
                    <div class="panel" style="flex: 0 1 {panel.width/12*100}%;">
                        <h3>{panel.title}</h3>
                        <div class="metric-value" id="{panel.title.lower().replace(' ', '-')}-value">--</div>
                        <div class="metric-label">{panel.description or panel.title}</div>
                    </div>
                    '''
                elif panel.type == 'graph':
                    panel_content = f'''
                    <div class="panel" style="flex: 0 1 {panel.width/12*100}%;">
                        <h3>{panel.title}</h3>
                        <div id="{panel.title.lower().replace(' ', '-').replace('&', 'and')}-plot" class="plot"></div>
                    </div>
                    '''
                else:
                    panel_content = f'''
                    <div class="panel" style="flex: 0 1 {panel.width/12*100}%;">
                        <h3>{panel.title}</h3>
                        <p>{panel.description or "Data visualization panel"}</p>
                    </div>
                    '''
                
                row_content += panel_content
            
            row_content += '</div></div>'
            content_sections.append(row_content)
        
        # Fill template
        html_content = html_template.format(
            title=dashboard_config.title,
            description=dashboard_config.description,
            refresh_interval=dashboard_config.refresh_interval,
            content='\n'.join(content_sections)
        )
        
        return html_content
    
    def save_html_dashboard(self, dashboard_config: DashboardConfig):
        """Save HTML dashboard to file"""
        html_content = self.generate_simple_html_dashboard(dashboard_config)
        html_file = self.config_dir / f"{self.experiment_name}_dashboard.html"
        
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        print(f"HTML dashboard saved to: {html_file}")
        print(f"Open in browser: file://{html_file.absolute()}")


# Convenience function
def create_dashboard_generator(experiment_name: str) -> DashboardGenerator:
    """Create a dashboard generator with sensible defaults"""
    return DashboardGenerator(experiment_name=experiment_name)


if __name__ == "__main__":
    # Example usage
    generator = create_dashboard_generator("toto_training_experiment")
    
    # Create dashboard configuration
    dashboard_config = generator.create_training_dashboard()
    
    # Save all configurations
    generator.save_configurations(dashboard_config)
    
    # Save HTML dashboard
    generator.save_html_dashboard(dashboard_config)
    
    print("Dashboard configurations generated successfully!")
    print("Available dashboards:")
    print("  - Grafana: Use docker-compose.yml to start monitoring stack")
    print("  - HTML: Open the generated HTML file in a browser")
    print("  - Prometheus: Configuration files ready for custom setup")