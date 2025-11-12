#!/usr/bin/env python3
"""
Apply improved hyperparameter configurations for crypto assets.

Based on analysis of current configs and best practices, this script
generates improved configurations to test.
"""
import json
from pathlib import Path

OUTPUT_DIR = Path("hyperparams/crypto_improved")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def create_improved_ethusd_config():
    """
    ETHUSD currently: 128 samples, trimmed_mean_20, 3.75% MAE
    BTCUSD (best): 1024 samples, trimmed_mean_5, 1.95% MAE

    Strategy: Increase samples and reduce trimming to match BTCUSD approach
    """
    configs = []

    # Config 1: Moderate increase
    configs.append({
        "symbol": "ETHUSD",
        "model": "toto",
        "config": {
            "name": "toto_improved_512_trimmed_mean_10",
            "num_samples": 512,
            "aggregate": "trimmed_mean_10",
            "samples_per_batch": 64
        },
        "notes": "4x samples, less aggressive trimming"
    })

    # Config 2: Match BTCUSD approach
    configs.append({
        "symbol": "ETHUSD",
        "model": "toto",
        "config": {
            "name": "toto_improved_1024_trimmed_mean_5",
            "num_samples": 1024,
            "aggregate": "trimmed_mean_5",
            "samples_per_batch": 128
        },
        "notes": "Match BTCUSD config (best performer)"
    })

    # Config 3: High sample count
    configs.append({
        "symbol": "ETHUSD",
        "model": "toto",
        "config": {
            "name": "toto_improved_2048_trimmed_mean_5",
            "num_samples": 2048,
            "aggregate": "trimmed_mean_5",
            "samples_per_batch": 128
        },
        "notes": "Max samples for best accuracy"
    })

    return configs


def create_improved_btcusd_config():
    """
    BTCUSD currently: 1024 samples, trimmed_mean_5, 1.95% MAE
    Already good, but let's try to push even lower
    """
    configs = []

    # Config 1: More samples
    configs.append({
        "symbol": "BTCUSD",
        "model": "toto",
        "config": {
            "name": "toto_improved_2048_trimmed_mean_5",
            "num_samples": 2048,
            "aggregate": "trimmed_mean_5",
            "samples_per_batch": 128
        },
        "notes": "Double samples for potentially better accuracy"
    })

    # Config 2: Try quantile instead of trimmed mean
    configs.append({
        "symbol": "BTCUSD",
        "model": "toto",
        "config": {
            "name": "toto_improved_1024_quantile_0.50",
            "num_samples": 1024,
            "aggregate": "quantile_0.50",
            "samples_per_batch": 128
        },
        "notes": "Median aggregation for robustness"
    })

    return configs


def create_improved_uniusd_config():
    """
    UNIUSD currently: Kronos model, 320 samples, 2.85% MAE
    Try Toto model which performs better on BTCUSD/ETHUSD
    """
    configs = []

    # Config 1: Switch to Toto with BTCUSD-like config
    configs.append({
        "symbol": "UNIUSD",
        "model": "toto",
        "config": {
            "name": "toto_improved_1024_trimmed_mean_5",
            "num_samples": 1024,
            "aggregate": "trimmed_mean_5",
            "samples_per_batch": 128
        },
        "notes": "Switch to Toto model with proven BTCUSD config"
    })

    # Config 2: Higher samples
    configs.append({
        "symbol": "UNIUSD",
        "model": "toto",
        "config": {
            "name": "toto_improved_2048_trimmed_mean_5",
            "num_samples": 2048,
            "aggregate": "trimmed_mean_5",
            "samples_per_batch": 128
        },
        "notes": "Max samples for volatile asset"
    })

    # Config 3: Keep Kronos but optimize
    configs.append({
        "symbol": "UNIUSD",
        "model": "kronos",
        "config": {
            "name": "kronos_improved_temp0.200_p0.85_s512_k32_clip1.80_ctx288",
            "temperature": 0.20,
            "top_p": 0.85,
            "top_k": 32,
            "sample_count": 512,
            "max_context": 288,
            "clip": 1.8
        },
        "notes": "Improved Kronos: higher samples, better sampling params"
    })

    return configs


def main():
    """Generate improved configs for all crypto assets."""
    print("Generating improved crypto forecasting configs...")
    print(f"Output directory: {OUTPUT_DIR}")

    all_configs = {
        "ETHUSD": create_improved_ethusd_config(),
        "BTCUSD": create_improved_btcusd_config(),
        "UNIUSD": create_improved_uniusd_config()
    }

    for symbol, configs in all_configs.items():
        print(f"\n{symbol}:")
        for i, config in enumerate(configs, 1):
            filename = OUTPUT_DIR / f"{symbol}_config{i}.json"
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"  {i}. {config['config']['name']}")
            print(f"     {config['notes']}")
            print(f"     Saved to: {filename}")

    print(f"\n{'='*60}")
    print("✓ Generated improved configs")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("1. Run test_hyperparameters_extended.py with these configs")
    print("2. Or manually test with evaluate script")
    print("3. Compare results against current best configs")
    print("4. Update hyperparams/best/ with winners")


if __name__ == "__main__":
    main()
