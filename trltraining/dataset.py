"""Dataset wiring for TRL trading-plan training."""

from __future__ import annotations

from dataclasses import dataclass

from qwen_rl_trading.data_prompt import SYMBOLS_30, MarketSnapshot, PromptDataset, build_chat_messages

from .config import TRLTradingConfig


@dataclass(slots=True)
class DatasetBundle:
    train_prompts: list[dict[str, object]]
    val_snapshots: list[tuple[str, MarketSnapshot]]
    snapshot_map: dict[str, MarketSnapshot]
    symbols: list[str]


def build_dataset_bundle(config: TRLTradingConfig) -> DatasetBundle:
    """Build train prompts, validation snapshots, and reward lookup map."""
    config.validate()
    symbols = SYMBOLS_30[: config.n_symbols]

    train_ds = PromptDataset(
        data_dir=config.data_dir,
        symbols=symbols,
        forecast_cache_dir=config.forecast_cache_dir,
        lookback=config.lookback_hours,
        eval_horizon=config.eval_horizon_hours,
        stride=config.stride,
        prompt_variant=config.prompt_variant,
        val_fraction=0.15,
        val_mode=False,
    )
    val_ds = PromptDataset(
        data_dir=config.data_dir,
        symbols=symbols,
        forecast_cache_dir=config.forecast_cache_dir,
        lookback=config.lookback_hours,
        eval_horizon=config.eval_horizon_hours,
        stride=config.stride,
        prompt_variant=config.prompt_variant,
        val_fraction=0.15,
        val_mode=True,
    )

    snapshot_map: dict[str, MarketSnapshot] = {}
    train_prompts: list[dict[str, object]] = []
    for idx in range(len(train_ds)):
        prompt_text, snapshot = train_ds[idx]
        snapshot_map[snapshot.window_id] = snapshot
        train_prompts.append({"prompt": build_chat_messages(prompt_text)})

    val_snapshots: list[tuple[str, MarketSnapshot]] = []
    for idx in range(len(val_ds)):
        prompt_text, snapshot = val_ds[idx]
        snapshot_map[snapshot.window_id] = snapshot
        val_snapshots.append((prompt_text, snapshot))

    return DatasetBundle(
        train_prompts=train_prompts,
        val_snapshots=val_snapshots,
        snapshot_map=snapshot_map,
        symbols=symbols,
    )
