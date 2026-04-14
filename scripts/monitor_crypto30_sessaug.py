#!/usr/bin/env python3
"""Monitor crypto30 sessaug training: periodically evaluate new checkpoints."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SESSAUG_DIR = REPO / "pufferlib_market/checkpoints/crypto30_sessaug"
EVAL_MARKER = "_eval90d.json"


def get_training_progress(subdir: Path) -> str:
    log = subdir / "train.log"
    if not log.exists():
        return "no log"
    lines = log.read_text().strip().split("\n")
    if not lines:
        return "empty log"
    last = lines[-1]
    # Extract step info like [  49/1831]
    if "[" in last and "/" in last:
        bracket = last[last.index("["):last.index("]") + 1]
        return bracket.strip()
    return "?"


def find_unevaluated(subdir: Path) -> list[Path]:
    unevaluated = []
    for pt_name in ["val_best.pt", "best.pt", "final.pt"]:
        ckpt = subdir / pt_name
        eval_file = subdir / f"{pt_name.replace('.pt', '')}{EVAL_MARKER}"
        if ckpt.exists() and not eval_file.exists():
            unevaluated.append(ckpt)
    return unevaluated


def evaluate_checkpoint(ckpt: Path, val_bin: str) -> dict | None:
    import subprocess
    out_file = ckpt.parent / f"{ckpt.stem}{EVAL_MARKER}"
    cmd = [
        sys.executable, "-m", "pufferlib_market.evaluate_holdout",
        "--checkpoint", str(ckpt),
        "--data-path", val_bin,
        "--eval-hours", "90", "--n-windows", "50",
        "--fee-rate", "0.001", "--fill-buffer-bps", "5.0",
        "--decision-lag", "2", "--deterministic", "--no-early-stop",
        "--out", str(out_file),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if out_file.exists():
            return json.loads(out_file.read_text())
    except Exception as e:
        print(f"  eval failed: {e}")
    return None


def main():
    val_bin = str(REPO / "pufferlib_market/data/crypto30_daily_val.bin")
    print(f"Monitoring {SESSAUG_DIR}")
    print(f"Checking for new checkpoints to evaluate...\n")

    while True:
        all_done = True
        for subdir in sorted(SESSAUG_DIR.iterdir()):
            if not subdir.is_dir():
                continue
            tag = subdir.name
            progress = get_training_progress(subdir)
            unevaluated = find_unevaluated(subdir)

            has_final = (subdir / "final.pt").exists()
            if not has_final:
                all_done = False

            if unevaluated:
                for ckpt in unevaluated:
                    if ckpt.name == "final.pt" or (ckpt.name == "best.pt" and has_final):
                        print(f"[{tag}] {progress} evaluating {ckpt.name}...")
                        result = evaluate_checkpoint(ckpt, val_bin)
                        if result:
                            med = result.get("median_total_return", 0) * 100
                            sort = result.get("median_sortino", 0)
                            p10 = result.get("p10_total_return", 0) * 100
                            neg = result.get("num_negative_windows", "?")
                            print(f"  -> med={med:+.2f}% sort={sort:.2f} p10={p10:+.2f}% neg={neg}/50")
                        else:
                            print(f"  -> eval failed")
            else:
                status = "DONE" if has_final else f"training {progress}"
                # Show existing eval results
                evals = list(subdir.glob(f"*{EVAL_MARKER}"))
                if evals:
                    for ef in evals:
                        try:
                            d = json.loads(ef.read_text())
                            med = d.get("median_total_return", 0) * 100
                            sort = d.get("median_sortino", 0)
                            print(f"  [{tag}] {status} | {ef.stem}: med={med:+.2f}% sort={sort:.2f}")
                        except Exception:
                            pass
                else:
                    print(f"  [{tag}] {status} (no evals yet)")

        if all_done:
            print("\nAll training runs complete. Final summary:")
            for subdir in sorted(SESSAUG_DIR.iterdir()):
                if not subdir.is_dir():
                    continue
                for ef in sorted(subdir.glob(f"*{EVAL_MARKER}")):
                    try:
                        d = json.loads(ef.read_text())
                        med = d.get("median_total_return", 0) * 100
                        sort = d.get("median_sortino", 0)
                        p10 = d.get("p10_total_return", 0) * 100
                        neg = d.get("num_negative_windows", "?")
                        print(f"  {subdir.name}/{ef.stem}: med={med:+.2f}% sort={sort:.2f} p10={p10:+.2f}% neg={neg}")
                    except Exception:
                        pass
            break

        print(f"\n--- Sleeping 30 minutes ---\n")
        time.sleep(1800)


if __name__ == "__main__":
    main()
