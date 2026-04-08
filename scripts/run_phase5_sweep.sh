#!/usr/bin/env bash
# Phase 5 canonical sweep: PPO/SAC/QR-PPO x constrained both x 5 seeds x 2M steps.
# Smoke first to validate plumbing, then the real sweep, then leaderboard.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "[phase5] smoke validation"
python fp4/bench/sweep.py --smoke --algos ppo,sac,qr_ppo --constrained both

echo "[phase5] real sweep (no timeouts)"
python fp4/bench/sweep.py --algos ppo,sac,qr_ppo --constrained both \
    --seeds 0,1,2,3,4 --steps 2000000

echo "[phase5] leaderboard"
python fp4/bench/make_leaderboard.py
