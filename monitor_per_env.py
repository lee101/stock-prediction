#!/usr/bin/env python3
"""
Monitor per_env autoresearch leaderboards and auto-run 50-window deep eval
for any candidate that escapes the degenerate state (score > -50).
Compares candidate to current ensemble (2201+8597 softmax_avg).
"""
import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

LEADERBOARDS = [
    "autoresearch_stocks12_ext_per_env1_leaderboard.csv",
    "autoresearch_stocks12_ext_per_env2_leaderboard.csv",
    "autoresearch_stocks12_ext_focused1_leaderboard.csv",
    "autoresearch_stocks12_ext_focused2_leaderboard.csv",
]

CHECKPOINT_ROOTS = {
    "ext_per_env1": "pufferlib_market/checkpoints/autoresearch_stocks12_ext_per_env1",
    "ext_per_env2": "pufferlib_market/checkpoints/autoresearch_stocks12_ext_per_env2",
    "ext_focused1": "pufferlib_market/checkpoints/autoresearch_stocks12_ext_focused1",
    "ext_focused2": "pufferlib_market/checkpoints/autoresearch_stocks12_ext_focused2",
}

# Current production: tp05_s123 standalone: med=16.52%, p10=10.45%, worst=5.62%, 0/50 neg
# (Previous 3-model ensemble 2201+8597+5526 was med=14.94%, p10=7.64% — tp05 beats it solo)
ENSEMBLE_CKPTS = [
    "pufferlib_market/checkpoints/stocks12_v2_sweep/stock_trade_pen_05_s123/best.pt",
]

VAL_DATA = "pufferlib_market/data/stocks12_daily_val.bin"
ESCAPE_THRESHOLD = -50.0  # score > -50 means escaped degenerate state
EVAL_SCRIPT = """
import sys, json
sys.path.insert(0, '.')
from pathlib import Path
from pufferlib_market.hourly_replay import read_mktd, simulate_daily_policy, MktdData
from ensemble_inference import EnsembleTrader
from pufferlib_market.evaluate_multiperiod import load_policy
import numpy as np
import torch
import torch.nn.functional as F

val_data = read_mktd('{val_data}')
n_windows = 50
window_size = 90
rng = np.random.default_rng(42)
max_start = val_data.num_timesteps - window_size
starts = sorted(rng.choice(max_start + 1, size=n_windows, replace=False))

def eval_policy(policy_fn, label):
    rets = []
    for start in starts:
        sliced = MktdData(
            version=val_data.version, symbols=val_data.symbols,
            features=val_data.features[start:start+window_size+1],
            prices=val_data.prices[start:start+window_size+1],
            tradable=val_data.tradable[start:start+window_size+1] if val_data.tradable is not None else None,
        )
        try:
            sim = simulate_daily_policy(
                sliced, policy_fn, max_steps=window_size,
                fee_rate=0.001, fill_buffer_bps=5.0,
                periods_per_year=252.0, enable_drawdown_profit_early_exit=False,
            )
            rets.append(sim.total_return)
        except:
            rets.append(0.0)
    rets = np.array(rets)
    neg = int((rets < 0).sum())
    return {{
        'label': label,
        'med': float(np.median(rets) * 100),
        'p10': float(np.percentile(rets, 10) * 100),
        'worst': float(rets.min() * 100),
        'neg': neg,
        'n': len(rets),
    }}

# Load new candidate
new_ckpt = '{checkpoint}'
policy_new, _, _ = load_policy(new_ckpt, val_data.num_symbols)
policy_new.eval()
def new_fn(obs):
    import numpy as np
    obs_t = torch.from_numpy(obs.astype(np.float32, copy=False)).unsqueeze(0)
    with torch.no_grad():
        logits, _ = policy_new(obs_t)
    return int(logits.argmax(-1).item())

# Current ensemble
ensemble = EnsembleTrader({ensemble_ckpts!r}, num_symbols=val_data.num_symbols, device='cpu', mode='softmax_avg')
ensemble_fn = ensemble.get_policy_fn(deterministic=True)

# 3-model ensemble
ensemble3 = EnsembleTrader({ensemble_ckpts!r} + [new_ckpt], num_symbols=val_data.num_symbols, device='cpu', mode='softmax_avg')
ensemble3_fn = ensemble3.get_policy_fn(deterministic=True)

results = {{}}
results['current_ensemble'] = eval_policy(ensemble_fn, 'tp05_s123')
results['candidate'] = eval_policy(new_fn, 'new_candidate')
results['ensemble4'] = eval_policy(ensemble3_fn, 'tp05_s123+new')
print(json.dumps(results, indent=2))
"""


def get_all_evaluated(leaderboard_path: str) -> set:
    """Return set of descriptions already evaluated (deep eval done)."""
    deep_eval_path = leaderboard_path.replace('.csv', '_deep_eval.json')
    if not os.path.exists(deep_eval_path):
        return set()
    with open(deep_eval_path) as f:
        data = json.load(f)
    return set(data.keys())


def save_deep_eval(leaderboard_path: str, description: str, result: dict):
    deep_eval_path = leaderboard_path.replace('.csv', '_deep_eval.json')
    existing = {}
    if os.path.exists(deep_eval_path):
        with open(deep_eval_path) as f:
            existing = json.load(f)
    existing[description] = result
    with open(deep_eval_path, 'w') as f:
        json.dump(existing, f, indent=2)
    print(f"  Saved deep eval to {deep_eval_path}")


def find_checkpoint(root: str, description: str) -> str | None:
    ckpt = os.path.join(root, description, 'best.pt')
    return ckpt if os.path.exists(ckpt) else None


def run_deep_eval(checkpoint: str) -> dict | None:
    script = EVAL_SCRIPT.format(
        checkpoint=checkpoint,
        val_data=VAL_DATA,
        ensemble_ckpts=ENSEMBLE_CKPTS,
    )
    try:
        result = subprocess.run(
            ['python', '-c', script],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            print(f"  Deep eval failed: {result.stderr[-500:]}")
            return None
        # Find JSON in output
        lines = result.stdout.strip().split('\n')
        json_start = next((i for i, l in enumerate(lines) if l.strip().startswith('{')), None)
        if json_start is None:
            print(f"  No JSON in output: {result.stdout[-200:]}")
            return None
        return json.loads('\n'.join(lines[json_start:]))
    except subprocess.TimeoutExpired:
        print("  Deep eval timed out")
        return None
    except Exception as e:
        print(f"  Deep eval error: {e}")
        return None


def check_leaderboard(lb_path: str, root: str, already_evaluated: set) -> list:
    """Return list of new candidates to evaluate."""
    if not os.path.exists(lb_path):
        return []
    import pandas as pd
    df = pd.read_csv(lb_path)
    escaped = df[df['holdout_robust_score'] > ESCAPE_THRESHOLD]
    candidates = []
    for _, row in escaped.iterrows():
        desc = row['description']
        if desc not in already_evaluated:
            candidates.append({
                'description': desc,
                'score': float(row['holdout_robust_score']),
                'med_20win': float(row.get('holdout_median_return_pct', 0)),
            })
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=int, default=120, help='Check interval in seconds')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    args = parser.parse_args()

    print("Monitoring per_env autoresearch leaderboards...")
    print(f"Escape threshold: score > {ESCAPE_THRESHOLD}")
    print(f"Leaderboards: {LEADERBOARDS}")
    print()

    roots = [
        "pufferlib_market/checkpoints/autoresearch_stocks12_ext_per_env1",
        "pufferlib_market/checkpoints/autoresearch_stocks12_ext_per_env2",
        "pufferlib_market/checkpoints/autoresearch_stocks12_ext_focused1",
        "pufferlib_market/checkpoints/autoresearch_stocks12_ext_focused2",
    ]

    while True:
        for lb_path, root in zip(LEADERBOARDS, roots):
            already_evaluated = get_all_evaluated(lb_path)
            candidates = check_leaderboard(lb_path, root, already_evaluated)
            for cand in candidates:
                desc = cand['description']
                score = cand['score']
                print(f"\n[{time.strftime('%H:%M:%S')}] CANDIDATE FOUND: {desc} (score={score:.2f}, 20-win med={cand['med_20win']:.2f}%)")
                ckpt = find_checkpoint(root, desc)
                if ckpt is None:
                    print(f"  Checkpoint not found at {root}/{desc}/best.pt — skipping")
                    save_deep_eval(lb_path, desc, {'error': 'checkpoint not found'})
                    continue
                print(f"  Running 50-window deep eval + ensemble test...")
                result = run_deep_eval(ckpt)
                if result is not None:
                    ens2 = result['current_ensemble']
                    cand_r = result['candidate']
                    ens3 = result['ensemble4']
                    print(f"  Current (tp05_s123 alone):    med={ens2['med']:.2f}% p10={ens2['p10']:.2f}% neg={ens2['neg']}/50")
                    print(f"  Candidate alone:              med={cand_r['med']:.2f}% p10={cand_r['p10']:.2f}% neg={cand_r['neg']}/50")
                    print(f"  2-model (tp05+new):           med={ens3['med']:.2f}% p10={ens3['p10']:.2f}% neg={ens3['neg']}/50")
                    improvement = ens3['neg'] <= ens2['neg'] and ens3['p10'] >= ens2['p10'] - 0.5
                    if improvement:
                        print(f"  *** IMPROVES BASELINE! Add {desc}/best.pt alongside tp05_s123 ***")
                    else:
                        print(f"  Does not improve baseline (neg: {ens2['neg']}→{ens3['neg']}, p10: {ens2['p10']:.1f}→{ens3['p10']:.1f})")
                    result['checkpoint'] = ckpt
                    result['raw_score_20win'] = score
                    save_deep_eval(lb_path, desc, result)
        if args.once:
            break
        print(f"\r[{time.strftime('%H:%M:%S')}] Waiting {args.interval}s...", end='', flush=True)
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
