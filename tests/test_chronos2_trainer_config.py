"""Verify Chronos2 HF Trainer default training_kwargs are set correctly."""
from __future__ import annotations

import pytest


@pytest.mark.unit
def test_training_kwargs_defaults():
    """_fit_pipeline builds training_kwargs with cosine scheduler, warmup, and tf32."""
    import ast
    from pathlib import Path

    source = Path("chronos2_trainer.py").read_text()
    tree = ast.parse(source)

    value = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "training_kwargs":
                    value = node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "training_kwargs":
                value = node.value

    assert value is not None, "training_kwargs assignment not found in chronos2_trainer.py"
    assert isinstance(value, ast.Call)
    kw_map = {}
    for kw in value.keywords:
        if isinstance(kw.value, ast.Constant):
            kw_map[kw.arg] = kw.value.value
    assert kw_map["lr_scheduler_type"] == "cosine", "scheduler should be cosine"
    assert kw_map["warmup_ratio"] == 0.05, "warmup_ratio should be 0.05"
    assert kw_map["tf32"] is True, "tf32 should be True"
    assert kw_map["bf16"] is False, "bf16 should remain False"
