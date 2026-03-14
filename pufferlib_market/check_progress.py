"""Quick progress checker for all PufferLib training/research."""
import csv
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

def check_processes():
    """Check running training processes."""
    result = subprocess.run(
        ["ps", "aux"], capture_output=True, text=True
    )
    lines = [l for l in result.stdout.split("\n")
             if "pufferlib_market" in l and "grep" not in l and "check_progress" not in l]
    print("=== Running Processes ===")
    if not lines:
        print("  None")
    for l in lines:
        # Extract the command
        parts = l.split()
        pid = parts[1]
        cmd = " ".join(parts[10:])
        # Shorten
        if "autoresearch" in cmd:
            print(f"  PID {pid}: autoresearch_rl")
        elif "train" in cmd:
            # Find checkpoint dir
            if "--checkpoint-dir" in cmd:
                idx = cmd.index("--checkpoint-dir") + len("--checkpoint-dir") + 1
                ckpt = cmd[idx:].split()[0].split("/")[-1]
            else:
                ckpt = "?"
            print(f"  PID {pid}: training → {ckpt}")
        elif "evaluate" in cmd:
            print(f"  PID {pid}: evaluate")
        else:
            print(f"  PID {pid}: {cmd[:80]}")

def check_training_logs():
    """Check latest training log stats."""
    print("\n=== Training Logs ===")
    logs = sorted(REPO.glob("pufferlib_market/training_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    for log in logs[:5]:
        name = log.stem.replace("training_", "")
        lines = log.read_text().strip().split("\n")
        # Find last line with ret=
        last_ret = None
        for line in reversed(lines):
            if "ret=" in line:
                last_ret = line.strip()
                break
        if "Training complete" in lines[-1]:
            # Find best return
            for line in reversed(lines):
                if "Best return:" in line:
                    print(f"  {name}: DONE - {line.strip()}")
                    break
        elif last_ret:
            # Extract key stats
            parts = {}
            for token in last_ret.split():
                if "=" in token:
                    k, v = token.split("=", 1)
                    parts[k] = v.rstrip(",")
            step = parts.get("step", "?")
            ret = parts.get("ret", "?")
            wr = parts.get("wr", "?")
            sortino = parts.get("sortino", "?")
            print(f"  {name}: step={step} ret={ret} wr={wr} sortino={sortino}")

def check_leaderboard():
    """Check autoresearch leaderboard."""
    lb_path = REPO / "pufferlib_market" / "autoresearch_leaderboard.csv"
    if not lb_path.exists():
        return
    print("\n=== Autoresearch Leaderboard (top 10) ===")
    with open(lb_path) as f:
        rows = list(csv.DictReader(f))
    valid = [r for r in rows if r.get("val_return") and r["val_return"] != "None"]
    valid.sort(key=lambda r: float(r["val_return"]), reverse=True)
    print(f"  {len(rows)} trials completed")
    for r in valid[:10]:
        vr = float(r["val_return"])
        desc = r["description"]
        sort = r["val_sortino"]
        prof = r["val_profitable_pct"]
        marker = " ***" if vr > 0 else ""
        print(f"  {desc:30s} val={vr:+.4f} sortino={sort:>8s} prof={prof:>5s}%{marker}")

def check_trading():
    """Check trading processes."""
    print("\n=== Trading ===")
    result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    stock = [l for l in result.stdout.split("\n") if "trade_unified" in l and "grep" not in l]
    crypto = [l for l in result.stdout.split("\n")
              if ("trade_binance" in l or "orchestrator" in l) and "grep" not in l]
    print(f"  Stock trader (Alpaca): {'RUNNING' if stock else 'NOT RUNNING'}")
    print(f"  Crypto trader (Binance): {'RUNNING' if crypto else 'NOT RUNNING'}")

if __name__ == "__main__":
    check_processes()
    check_training_logs()
    check_leaderboard()
    check_trading()
