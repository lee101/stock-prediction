#!/bin/bash
# DISABLED - Moving to hourly trading
# Complete training workflow launcher
# This script is currently commented out and not in use

exit 0  # Script disabled

set -e

echo "=================================="
echo "Neural Daily Training Launcher"
echo "=================================="
echo ""

# Function to show usage
usage() {
    echo "Usage: $0 [quick|phase1|phase2|full]"
    echo ""
    echo "  quick   - Run 3 test experiments (~2 hours)"
    echo "  phase1  - Full hyperparameter search (~30-50 hours)"
    echo "  phase2  - Final fit on all data (~1-2 hours)"
    echo "  full    - Complete workflow: phase1 then phase2"
    echo ""
    exit 1
}

# Parse command
MODE=${1:-quick}

case $MODE in
    quick)
        echo "Running QUICK TEST mode (3 experiments)"
        echo "Estimated time: ~2 hours"
        echo ""
        source .venv313/bin/activate
        python auto_improve_loop.py --max-experiments 3
        ;;

    phase1)
        echo "Running PHASE 1: Full hyperparameter search"
        echo "Estimated time: 30-50 hours"
        echo "Results will be saved to improvement_results.jsonl"
        echo ""
        source .venv313/bin/activate
        nohup python auto_improve_loop.py > phase1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
        PID=$!
        echo "Phase 1 launched in background (PID: $PID)"
        echo "Monitor with: tail -f phase1_*.log"
        echo "Check results: tail -f improvement_results.jsonl"
        ;;

    phase2)
        echo "Running PHASE 2: Final fit on all data"
        echo "This will use the best config from improvement_results.jsonl"
        echo "Training on 100% of data (validation_days=0)"
        echo ""
        if [ ! -f improvement_results.jsonl ]; then
            echo "ERROR: improvement_results.jsonl not found!"
            echo "You must run phase1 first to find the best config"
            exit 1
        fi
        source .venv313/bin/activate
        python final_fit_all_data.py --output-name production_$(date +%Y%m%d)
        ;;

    full)
        echo "Running FULL WORKFLOW"
        echo "This will run Phase 1 then automatically run Phase 2"
        echo ""
        source .venv313/bin/activate

        # Phase 1
        echo "Starting Phase 1..."
        python auto_improve_loop.py > phase1_full_$(date +%Y%m%d_%H%M%S).log 2>&1

        # Phase 2
        echo "Phase 1 complete! Starting Phase 2..."
        python final_fit_all_data.py --output-name production_$(date +%Y%m%d)

        echo ""
        echo "✅ Complete workflow finished!"
        ;;

    *)
        usage
        ;;
esac

echo ""
echo "Done!"
