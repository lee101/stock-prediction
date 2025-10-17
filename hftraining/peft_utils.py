from __future__ import annotations

from typing import Iterable, Optional

try:  # pragma: no cover - optional dependency
    from peft import LoraConfig, get_peft_model  # type: ignore
except ImportError:  # pragma: no cover
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore


def apply_lora_or_dora(
    hf_model,
    *,
    task_type: str = "SEQ_2_SEQ_LM",
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[Iterable[str]] = ("q", "k", "v", "o", "wi", "wo"),
    use_dora: bool = False,
):
    """
    if LoraConfig is None or get_peft_model is None:
        raise ImportError(
            "peft is required for LoRA/DoRA fine-tuning. "
            "Install with `uv pip install peft`."
        )
    Wrap a Hugging Face model with LoRA or DoRA adapters.

    Args:
        hf_model: The base transformers.PreTrainedModel instance.
        task_type: One of peft.TaskType values; Chronos-Bolt is seq2seq.
        r: Rank of the adapter matrices.
        alpha: Scaling factor applied to the adapters.
        dropout: Adapter dropout probability.
        target_modules: Iterable of module name fragments to target. ``None`` to
            let PEFT match defaults for the architecture.
        use_dora: If True, activates DoRA weight decomposition (LoRA variant).
    """
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=task_type,
        target_modules=list(target_modules) if target_modules else None,
        use_dora=use_dora,
    )
    peft_model = get_peft_model(hf_model, config)
    try:
        peft_model.print_trainable_parameters()  # helpful logging during setup
    except AttributeError:
        pass
    return peft_model
