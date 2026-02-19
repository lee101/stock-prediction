#!/bin/bash
# Wait for sweep to finish, then run covariate forecast experiment
set -e
cd /home/lee/code/stock

echo "Waiting for sweep_improvements.py to finish..."
while pgrep -f "sweep_improvements.py" > /dev/null 2>&1; do
    sleep 60
done
echo "Sweep done at $(date)"

echo "Starting covariate forecast experiment..."
CUDA_VISIBLE_DEVICES=0 .venv313/bin/python binanceleveragesui/eval_covariate_forecasts.py 2>&1
echo "Covariate experiment done at $(date)"
