"""Architecture sweep configurations.

Each config is a dict of TrainingConfig overrides. The sweep runner
trains each one, then evaluates on a robust multi-scenario market simulator.
"""
from __future__ import annotations

SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD"]
VALIDATION_DAYS = 90
EPOCHS_QUICK = 8
EPOCHS_FULL = 20

BASE = dict(
    batch_size=64,
    sequence_length=96,
    learning_rate=3e-4,
    weight_decay=1e-4,
    grad_clip=1.0,
    maker_fee=0.001,
    initial_cash=1.0,
    loss_type="multiwindow_dd",
    multiwindow_fractions="0.25,0.5,0.75,1.0",
    multiwindow_aggregation="softmin",
    dd_penalty=2.0,
    return_weight=0.1,
    fill_temperature=0.01,
    fill_buffer_pct=0.0005,
    decision_lag_bars=1,
    decision_lag_range="0,1,2",
    validation_use_binary_fills=True,
    use_compile=True,
    use_tf32=True,
    use_flash_attention=True,
    use_flex_attention=True,
    use_amp=False,
    warmup_steps=50,
    lr_schedule="cosine",
    lr_min_ratio=0.05,
    feature_noise_std=0.02,
    price_offset_pct=0.0003,
    min_price_gap_pct=0.0003,
    trade_amount_scale=100.0,
    num_outputs=4,
    max_hold_hours=24.0,
    seed=42,
    epochs=EPOCHS_QUICK,
    dataset=dict(
        validation_days=VALIDATION_DAYS,
        sequence_length=96,
        forecast_horizons=(1, 24),
        cache_only=True,
    ),
)


def _merge(base: dict, overrides: dict) -> dict:
    out = {}
    for k, v in base.items():
        if k in overrides and isinstance(v, dict) and isinstance(overrides[k], dict):
            out[k] = {**v, **overrides[k]}
        else:
            out[k] = overrides.get(k, v)
    for k, v in overrides.items():
        if k not in out:
            out[k] = v
    return out


# -- Nano transformer variants (LLM-style, RoPE + GQA + RMSNorm) --

NANO_TINY = _merge(BASE, dict(
    model_arch="nano", transformer_dim=128, transformer_layers=4, transformer_heads=4,
    num_kv_heads=2, mlp_ratio=4.0, transformer_dropout=0.05,
    use_qk_norm=True, logits_softcap=12.0,
))

NANO_SMALL = _merge(BASE, dict(
    model_arch="nano", transformer_dim=192, transformer_layers=6, transformer_heads=6,
    num_kv_heads=3, mlp_ratio=4.0, transformer_dropout=0.05,
    use_qk_norm=True, logits_softcap=12.0,
))

NANO_MEDIUM = _merge(BASE, dict(
    model_arch="nano", transformer_dim=256, transformer_layers=8, transformer_heads=8,
    num_kv_heads=4, mlp_ratio=4.0, transformer_dropout=0.05,
    use_qk_norm=True, logits_softcap=12.0,
))

NANO_LARGE = _merge(BASE, dict(
    model_arch="nano", transformer_dim=384, transformer_layers=10, transformer_heads=8,
    num_kv_heads=4, mlp_ratio=3.5, transformer_dropout=0.05,
    use_qk_norm=True, logits_softcap=12.0,
))

NANO_WIDE = _merge(BASE, dict(
    model_arch="nano", transformer_dim=512, transformer_layers=6, transformer_heads=8,
    num_kv_heads=4, mlp_ratio=3.0, transformer_dropout=0.08,
    use_qk_norm=True, logits_softcap=12.0,
    learning_rate=2e-4,
))

# -- Nano with memory tokens (global context) --

NANO_MEM8 = _merge(NANO_MEDIUM, dict(
    num_memory_tokens=8,
))

NANO_MEM16_DILATED = _merge(NANO_MEDIUM, dict(
    num_memory_tokens=16,
    dilated_strides="1,4,24",
))

# -- Nano with residual scalars (DeepNet-style) --

NANO_DEEP_RESSCALE = _merge(BASE, dict(
    model_arch="nano", transformer_dim=256, transformer_layers=12, transformer_heads=8,
    num_kv_heads=4, mlp_ratio=4.0, transformer_dropout=0.05,
    use_qk_norm=True, logits_softcap=12.0,
    use_residual_scalars=True, residual_scale_init=1.0, skip_scale_init=0.1,
))

# -- Mamba SSM variants --

MAMBA_SMALL = _merge(BASE, dict(
    model_arch="mamba", transformer_dim=192, transformer_layers=6,
    transformer_dropout=0.05,
))

MAMBA_MEDIUM = _merge(BASE, dict(
    model_arch="mamba", transformer_dim=256, transformer_layers=8,
    transformer_dropout=0.05,
))

MAMBA_LARGE = _merge(BASE, dict(
    model_arch="mamba", transformer_dim=384, transformer_layers=8,
    transformer_dropout=0.05,
))

# -- Classic transformer (baseline) --

CLASSIC_MEDIUM = _merge(BASE, dict(
    model_arch="classic", transformer_dim=256, transformer_layers=4, transformer_heads=8,
    transformer_dropout=0.1,
))

# -- Optimizer variants on best arch --

NANO_MUON = _merge(NANO_MEDIUM, dict(
    optimizer_name="muon",
    muon_lr=0.02,
    muon_momentum=0.95,
    muon_nesterov=True,
    muon_ns_steps=5,
    embed_lr_mult=0.3,
    head_lr_mult=1.0,
))

# -- Loss variants --

NANO_SORTINO_DD = _merge(NANO_MEDIUM, dict(
    loss_type="sortino_dd",
    dd_penalty=3.0,
))

NANO_CALMAR = _merge(NANO_MEDIUM, dict(
    loss_type="calmar",
))

# -- Longer training on promising archs --

NANO_MEDIUM_LONG = _merge(NANO_MEDIUM, dict(
    epochs=EPOCHS_FULL,
    lr_schedule="cosine",
    cooldown_fraction=0.1,
))

NANO_LARGE_LONG = _merge(NANO_LARGE, dict(
    epochs=EPOCHS_FULL,
    lr_schedule="cosine",
    cooldown_fraction=0.1,
))


# -- Proven-style configs (matching best existing model approach) --

PROVEN_BASE = dict(
    batch_size=32,
    sequence_length=96,
    learning_rate=3e-4,
    weight_decay=1e-4,
    grad_clip=1.0,
    maker_fee=0.0,
    initial_cash=1.0,
    loss_type="sortino",
    return_weight=0.08,
    fill_temperature=0.01,
    fill_buffer_pct=0.0,
    decision_lag_bars=0,
    decision_lag_range="",
    validation_use_binary_fills=True,
    use_compile=False,
    use_tf32=True,
    use_flash_attention=True,
    use_flex_attention=True,
    use_amp=False,
    warmup_steps=100,
    lr_schedule="none",
    lr_min_ratio=0.05,
    feature_noise_std=0.0,
    price_offset_pct=0.0003,
    min_price_gap_pct=0.0003,
    trade_amount_scale=100.0,
    num_outputs=4,
    max_hold_hours=24.0,
    seed=1337,
    epochs=5,
    transformer_dropout=0.1,
    dataset=dict(
        validation_days=70,
        sequence_length=96,
        forecast_horizons=(1, 24),
        cache_only=True,
    ),
)

PROVEN_CLASSIC = _merge(PROVEN_BASE, dict(
    model_arch="classic", transformer_dim=256, transformer_layers=4, transformer_heads=8,
))

PROVEN_NANO = _merge(PROVEN_BASE, dict(
    model_arch="nano", transformer_dim=256, transformer_layers=4, transformer_heads=8,
    num_kv_heads=4, use_qk_norm=True, logits_softcap=12.0, mlp_ratio=4.0,
    transformer_dropout=0.1,
))

PROVEN_NANO_DEEP = _merge(PROVEN_BASE, dict(
    model_arch="nano", transformer_dim=256, transformer_layers=8, transformer_heads=8,
    num_kv_heads=4, use_qk_norm=True, logits_softcap=12.0, mlp_ratio=4.0,
    transformer_dropout=0.1,
))

PROVEN_NANO_WIDE = _merge(PROVEN_BASE, dict(
    model_arch="nano", transformer_dim=384, transformer_layers=4, transformer_heads=8,
    num_kv_heads=4, use_qk_norm=True, logits_softcap=12.0, mlp_ratio=3.5,
    transformer_dropout=0.1,
))

PROVEN_NANO_FEE = _merge(PROVEN_BASE, dict(
    model_arch="nano", transformer_dim=256, transformer_layers=4, transformer_heads=8,
    num_kv_heads=4, use_qk_norm=True, logits_softcap=12.0, mlp_ratio=4.0,
    maker_fee=0.001,
    fill_buffer_pct=0.0005,
    decision_lag_bars=1,
))

PROVEN_NANO_MULTIWINDOW = _merge(PROVEN_BASE, dict(
    model_arch="nano", transformer_dim=256, transformer_layers=4, transformer_heads=8,
    num_kv_heads=4, use_qk_norm=True, logits_softcap=12.0, mlp_ratio=4.0,
    loss_type="multiwindow",
    multiwindow_fractions="0.5,1.0",
    multiwindow_aggregation="softmin",
))

PROVEN_NANO_MEM = _merge(PROVEN_BASE, dict(
    model_arch="nano", transformer_dim=256, transformer_layers=4, transformer_heads=8,
    num_kv_heads=4, use_qk_norm=True, logits_softcap=12.0, mlp_ratio=4.0,
    num_memory_tokens=8,
))


ALL_CONFIGS = {
    "nano_tiny": NANO_TINY,
    "nano_small": NANO_SMALL,
    "nano_medium": NANO_MEDIUM,
    "nano_large": NANO_LARGE,
    "nano_wide": NANO_WIDE,
    "nano_mem8": NANO_MEM8,
    "nano_mem16_dilated": NANO_MEM16_DILATED,
    "nano_deep_resscale": NANO_DEEP_RESSCALE,
    "mamba_small": MAMBA_SMALL,
    "mamba_medium": MAMBA_MEDIUM,
    "mamba_large": MAMBA_LARGE,
    "classic_medium": CLASSIC_MEDIUM,
    "nano_muon": NANO_MUON,
    "nano_sortino_dd": NANO_SORTINO_DD,
    "nano_calmar": NANO_CALMAR,
    "nano_medium_long": NANO_MEDIUM_LONG,
    "nano_large_long": NANO_LARGE_LONG,
    "proven_classic": PROVEN_CLASSIC,
    "proven_nano": PROVEN_NANO,
    "proven_nano_deep": PROVEN_NANO_DEEP,
    "proven_nano_wide": PROVEN_NANO_WIDE,
    "proven_nano_fee": PROVEN_NANO_FEE,
    "proven_nano_multiwindow": PROVEN_NANO_MULTIWINDOW,
    "proven_nano_mem": PROVEN_NANO_MEM,
}

# Quick subset for fast iteration
QUICK_CONFIGS = {k: ALL_CONFIGS[k] for k in [
    "nano_tiny", "nano_small", "nano_medium", "nano_large",
    "mamba_medium", "classic_medium",
]}
