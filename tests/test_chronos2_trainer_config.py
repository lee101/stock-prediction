"""Verify Chronos2 HF Trainer default training_kwargs are set correctly."""
from __future__ import annotations

import pytest


@pytest.mark.unit
def test_training_kwargs_defaults():
    """_fit_pipeline wires scheduler/warmup from TrainerConfig defaults."""
    import ast
    from pathlib import Path
    from chronos2_trainer import TrainerConfig

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
        kw_map[kw.arg] = kw.value
    assert isinstance(kw_map["lr_scheduler_type"], ast.Call)
    assert isinstance(kw_map["lr_scheduler_type"].args[0], ast.Attribute)
    assert kw_map["lr_scheduler_type"].args[0].attr == "lr_scheduler_type"
    assert isinstance(kw_map["warmup_ratio"], ast.Call)
    assert isinstance(kw_map["warmup_ratio"].args[0], ast.Attribute)
    assert kw_map["warmup_ratio"].args[0].attr == "warmup_ratio"
    assert TrainerConfig.__dataclass_fields__["lr_scheduler_type"].default == "cosine"
    assert TrainerConfig.__dataclass_fields__["warmup_ratio"].default == 0.05
    assert isinstance(kw_map["tf32"], ast.Constant)
    assert isinstance(kw_map["bf16"], ast.Constant)
    assert kw_map["tf32"].value is True, "tf32 should be True"
    assert kw_map["bf16"].value is False, "bf16 should remain False"
