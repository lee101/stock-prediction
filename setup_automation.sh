#!/bin/bash
# DISABLED - Moving to hourly trading
# Setup automated daily workflows
# This script is currently commented out and not in use

exit 0  # Script disabled

echo "Setting up automated neural trading workflows"
echo "=============================================="

# Get the current directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo ""
echo "Option 1: Cron Jobs (Recommended)"
echo "=================================="
echo ""
echo "Add these lines to your crontab (crontab -e):"
echo ""
echo "# Daily data update and model retraining (6 AM)"
echo "0 6 * * * cd $SCRIPT_DIR && ./daily_retrain_and_trade.sh >> logs/daily_workflow.log 2>&1"
echo ""
echo "# Weekly full hyperparameter search (Sunday 2 AM)"
echo "0 2 * * 0 cd $SCRIPT_DIR && ./launch_training.sh phase1 >> logs/weekly_phase1.log 2>&1"
echo ""

echo "Option 2: Systemd Services (More robust)"
echo "========================================="
echo ""

# Create systemd service for daily workflow
cat > /tmp/neural-daily.service << 'EOF'
[Unit]
Description=Neural Trading Daily Workflow
After=network.target

[Service]
Type=oneshot
User=$(whoami)
WorkingDirectory=$SCRIPT_DIR
ExecStart=$SCRIPT_DIR/daily_retrain_and_trade.sh
StandardOutput=append:$SCRIPT_DIR/logs/daily_workflow.log
StandardError=append:$SCRIPT_DIR/logs/daily_workflow.log

[Install]
WantedBy=multi-user.target
EOF

# Create systemd timer for daily workflow
cat > /tmp/neural-daily.timer << EOF
[Unit]
Description=Run Neural Trading Daily Workflow
Requires=neural-daily.service

[Timer]
OnCalendar=daily
OnCalendar=*-*-* 06:00:00
Persistent=true

[Install]
WantedBy=timers.target
EOF

# Create weekly Phase 1 service
cat > /tmp/neural-weekly-phase1.service << 'EOF'
[Unit]
Description=Neural Trading Weekly Phase 1 Training
After=network.target

[Service]
Type=oneshot
User=$(whoami)
WorkingDirectory=$SCRIPT_DIR
ExecStart=$SCRIPT_DIR/launch_training.sh phase1
StandardOutput=append:$SCRIPT_DIR/logs/weekly_phase1.log
StandardError=append:$SCRIPT_DIR/logs/weekly_phase1.log

[Install]
WantedBy=multi-user.target
EOF

# Create weekly timer
cat > /tmp/neural-weekly-phase1.timer << EOF
[Unit]
Description=Run Neural Trading Weekly Phase 1
Requires=neural-weekly-phase1.service

[Timer]
OnCalendar=Sun
OnCalendar=Sun *-*-* 02:00:00
Persistent=true

[Install]
WantedBy=timers.target
EOF

echo "Systemd service files created in /tmp/"
echo ""
echo "To install systemd services (requires sudo):"
echo "  sudo cp /tmp/neural-*.service /etc/systemd/system/"
echo "  sudo cp /tmp/neural-*.timer /etc/systemd/system/"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl enable neural-daily.timer"
echo "  sudo systemctl enable neural-weekly-phase1.timer"
echo "  sudo systemctl start neural-daily.timer"
echo "  sudo systemctl start neural-weekly-phase1.timer"
echo ""

echo "Option 3: Manual Commands"
echo "========================="
echo ""
echo "Run daily workflow manually:"
echo "  ./daily_retrain_and_trade.sh"
echo ""
echo "Run weekly Phase 1 search:"
echo "  ./launch_training.sh phase1"
echo ""

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"

echo "✅ Setup complete!"
echo ""
echo "Recommended: Use cron jobs (simpler) or systemd timers (more robust)"
