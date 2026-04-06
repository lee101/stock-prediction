from __future__ import annotations

from dataclasses import dataclass

import pytest
from trltraining.config import RECOMMENDED_MODEL_PRESET, TRLTradingConfig
from trltraining.dataset import build_dataset_bundle
from trltraining.methods import recommend_trainer
from trltraining.train_grpo import build_grpo_kwargs


def test_recommend_trainer_prefers_grpo_with_colocated_vllm() -> None:
    recommendation = recommend_trainer()

    assert recommendation.trainer_type == "grpo"
    assert recommendation.use_vllm is True
    assert recommendation.vllm_mode == "colocate"
    assert "scalar reward" in recommendation.rationale


def test_build_grpo_kwargs_includes_vllm_fields_when_enabled() -> None:
    config = TRLTradingConfig(
        model_preset=RECOMMENDED_MODEL_PRESET,
        use_vllm=True,
        vllm_mode="colocate",
        group_size=6,
        output_dir="trltraining/checkpoints/test",
    )

    kwargs = build_grpo_kwargs(config)

    assert kwargs["num_generations"] == 6
    assert kwargs["use_vllm"] is True
    assert kwargs["vllm_mode"] == "colocate"
    assert kwargs["max_prompt_length"] == config.max_prompt_length
    assert kwargs["max_completion_length"] == config.max_completion_length


def test_build_grpo_kwargs_omits_vllm_fields_when_disabled() -> None:
    config = TRLTradingConfig(
        model_preset=RECOMMENDED_MODEL_PRESET,
        use_vllm=False,
        output_dir="trltraining/checkpoints/test",
    )

    kwargs = build_grpo_kwargs(config)

    assert "use_vllm" not in kwargs
    assert "vllm_mode" not in kwargs


def test_config_validate_rejects_bad_vllm_mode() -> None:
    config = TRLTradingConfig(
        model_preset=RECOMMENDED_MODEL_PRESET,
        vllm_mode="bad-mode",
        output_dir="trltraining/checkpoints/test",
    )

    with pytest.raises(ValueError, match="unsupported vllm_mode"):
        config.validate()


def test_build_dataset_bundle_wraps_prompt_dataset(monkeypatch) -> None:
    @dataclass(slots=True)
    class _Snapshot:
        window_id: str

    class _FakePromptDataset:
        def __init__(self, *args, val_mode: bool = False, **kwargs):
            self._items = (
                [("train prompt", _Snapshot("train-window"))]
                if not val_mode
                else [("val prompt", _Snapshot("val-window"))]
            )

        def __len__(self) -> int:
            return len(self._items)

        def __getitem__(self, idx: int):
            return self._items[idx]

    monkeypatch.setattr("trltraining.dataset.PromptDataset", _FakePromptDataset)
    monkeypatch.setattr("trltraining.dataset.build_chat_messages", lambda prompt: [{"role": "user", "content": prompt}])

    bundle = build_dataset_bundle(TRLTradingConfig(output_dir="trltraining/checkpoints/test"))

    assert bundle.train_prompts == [{"prompt": [{"role": "user", "content": "train prompt"}]}]
    assert bundle.val_snapshots[0][0] == "val prompt"
    assert set(bundle.snapshot_map) == {"train-window", "val-window"}
