# ETH PPO Remote Iteration Plan (10 Steps)

1. Sync code to remote (`git pull --rebase`) and confirm active training PID.
2. Run `iter_explore_1` with `120000` timesteps over the default `run_iteration_batch.sh` grid (currently 12 variants).
3. Rank variants by `robust_score` and sanity-check `long_return_5_10bp` plus trade activity to avoid near-zero-turnover traps.
4. Re-run top risk profile family with longer horizon (`iter_explore_2`, `200000` timesteps) using the same 10-variant grid.
5. Compare `24h`, `7d`, and `30d` windows and require consistent positive mean return at `5` and `10` bps fills.
6. Tighten realism checks by comparing top checkpoints against live fills with `compare_live_vs_sim_eth.py`.
7. Run `iter_refine_1` (`300000` timesteps) and bias selection toward lowest drawdown under 10 bps fill buffer.
8. Validate winner checkpoints across multi-period stress windows (short, medium, long) with `evaluate_checkpoint_windows.py`.
9. Promote candidate only if metrics remain positive in both 5 bps and 10 bps modes and live-vs-sim divergence stays stable.
10. Archive leaderboard and summary JSON outputs under `analysis/eth_risk_ppo/` and update deployment recommendation notes.

## Remote Queue Examples

Queue one iteration behind an active trainer:

```bash
WAIT_FOR_PID=<active_trainer_pid> \
REMOTE_SSH="sshpass -p '$SSHPASS' ssh -o StrictHostKeyChecking=no" \
bash fastalgorithms/eth_risk_ppo/run_iteration_batch_remote.sh iter_explore_1 120000
```

Start the next two iterations after the first batch PID exits:

```bash
# launch iter_explore_2 after iter_explore_1 PID completes
WAIT_FOR_PID=<iter_explore_1_pid> \
REMOTE_SSH="sshpass -p '$SSHPASS' ssh -o StrictHostKeyChecking=no" \
bash fastalgorithms/eth_risk_ppo/run_iteration_batch_remote.sh iter_explore_2 200000

# launch iter_refine_1 after iter_explore_2 PID completes
WAIT_FOR_PID=<iter_explore_2_pid> \
REMOTE_SSH="sshpass -p '$SSHPASS' ssh -o StrictHostKeyChecking=no" \
bash fastalgorithms/eth_risk_ppo/run_iteration_batch_remote.sh iter_refine_1 300000
```
