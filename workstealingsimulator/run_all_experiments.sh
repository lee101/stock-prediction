#!/bin/bash
# Run all steal hyperparameter experiments sequentially

echo "================================================================"
echo "STEAL HYPERPARAMETER OPTIMIZATION EXPERIMENTS"
echo "================================================================"
echo "Baseline to beat: \$19.1M PnL, 12 steals, 0 blocks, Score: 9.5M"
echo "================================================================"
echo ""

echo "Waiting for current optimization to complete..."
while ps aux | grep -q "[p]ython3.*simulator.py"; do
    sleep 10
done
echo "✓ Current optimization complete"
echo ""

echo "================================================================"
echo "EXPERIMENT 1: Narrow Protection Range (0.12-0.22%)"
echo "================================================================"
cd /home/lee/code/stock/workstealingsimulator
python3 experiment1_narrow_protection.py 2>&1 | tee exp1_narrow_protection.log
echo "✓ Experiment 1 complete"
echo ""

echo "================================================================"
echo "EXPERIMENT 2: Disable Crypto Fighting"
echo "================================================================"
python3 experiment2_no_fighting.py 2>&1 | tee exp2_no_fighting.log
echo "✓ Experiment 2 complete"
echo ""

echo "================================================================"
echo "EXPERIMENT 3: Narrow Cooldown (120-200s)"
echo "================================================================"
python3 experiment3_narrow_cooldown.py 2>&1 | tee exp3_narrow_cooldown.log
echo "✓ Experiment 3 complete"
echo ""

echo "================================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "================================================================"
echo "Results saved in:"
echo "  - exp1_narrow_protection.log"
echo "  - exp2_no_fighting.log"
echo "  - exp3_narrow_cooldown.log"
echo ""
echo "Extracting best results from each..."
grep "FINAL PERFORMANCE" exp*.log -A 10
