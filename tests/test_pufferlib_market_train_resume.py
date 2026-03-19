from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from pufferlib_market.train import ResumeState, _checkpoint_payload, _load_resume_checkpoint
except (ImportError, ModuleNotFoundError):
    pytest.skip("Required module pufferlib_market.train not available", allow_module_level=True)


def _make_model_and_optimizer() -> tuple[nn.Module, optim.Optimizer]:
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    optimizer = optim.Adam(model.parameters(), lr=1e-3, eps=1e-5)
    return model, optimizer


def _seed_optimizer_state(model: nn.Module, optimizer: optim.Optimizer) -> None:
    torch.manual_seed(7)
    loss = model(torch.randn(5, 4)).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def test_load_resume_checkpoint_restores_model_and_optimizer(tmp_path) -> None:
    source_model, source_optimizer = _make_model_and_optimizer()
    _seed_optimizer_state(source_model, source_optimizer)
    action_meta = {
        "action_allocation_bins": 1,
        "action_level_bins": 1,
        "action_max_offset_bps": 0.0,
    }
    checkpoint_path = tmp_path / "resume.pt"
    torch.save(
        _checkpoint_payload(
            source_model,
            source_optimizer,
            update=7,
            global_step=1_536,
            best_return=1.25,
            disable_shorts=True,
            action_meta=action_meta,
        ),
        checkpoint_path,
    )

    target_model, target_optimizer = _make_model_and_optimizer()
    resume_state = _load_resume_checkpoint(
        checkpoint_path,
        policy=target_model,
        optimizer=target_optimizer,
        device=torch.device("cpu"),
        disable_shorts=True,
        action_meta=action_meta,
    )

    assert resume_state == ResumeState(update=7, global_step=1_536, best_return=1.25)
    for key, value in source_model.state_dict().items():
        assert torch.equal(target_model.state_dict()[key], value)
    assert target_optimizer.state_dict()["state"]


def test_load_resume_checkpoint_rejects_action_grid_mismatch(tmp_path) -> None:
    model, optimizer = _make_model_and_optimizer()
    _seed_optimizer_state(model, optimizer)
    checkpoint_path = tmp_path / "resume.pt"
    torch.save(
        _checkpoint_payload(
            model,
            optimizer,
            update=1,
            global_step=256,
            best_return=0.5,
            disable_shorts=False,
            action_meta={
                "action_allocation_bins": 5,
                "action_level_bins": 1,
                "action_max_offset_bps": 0.0,
            },
        ),
        checkpoint_path,
    )

    target_model, target_optimizer = _make_model_and_optimizer()
    with pytest.raises(ValueError, match="action_allocation_bins"):
        _load_resume_checkpoint(
            checkpoint_path,
            policy=target_model,
            optimizer=target_optimizer,
            device=torch.device("cpu"),
            disable_shorts=False,
            action_meta={
                "action_allocation_bins": 1,
                "action_level_bins": 1,
                "action_max_offset_bps": 0.0,
            },
        )


def test_load_resume_checkpoint_rejects_unsupported_payload(tmp_path) -> None:
    checkpoint_path = tmp_path / "resume.pt"
    torch.save({"unexpected": "format"}, checkpoint_path)
    target_model, target_optimizer = _make_model_and_optimizer()

    with pytest.raises(ValueError, match="Unsupported resume checkpoint format"):
        _load_resume_checkpoint(
            checkpoint_path,
            policy=target_model,
            optimizer=target_optimizer,
            device=torch.device("cpu"),
            disable_shorts=False,
            action_meta={
                "action_allocation_bins": 1,
                "action_level_bins": 1,
                "action_max_offset_bps": 0.0,
            },
        )


def test_load_resume_checkpoint_allows_short_mask_change(tmp_path) -> None:
    model, optimizer = _make_model_and_optimizer()
    _seed_optimizer_state(model, optimizer)
    checkpoint_path = tmp_path / "resume.pt"
    torch.save(
        _checkpoint_payload(
            model,
            optimizer,
            update=3,
            global_step=512,
            best_return=0.75,
            disable_shorts=True,
            action_meta={
                "action_allocation_bins": 1,
                "action_level_bins": 1,
                "action_max_offset_bps": 0.0,
            },
        ),
        checkpoint_path,
    )

    target_model, target_optimizer = _make_model_and_optimizer()
    resume_state = _load_resume_checkpoint(
        checkpoint_path,
        policy=target_model,
        optimizer=target_optimizer,
        device=torch.device("cpu"),
        disable_shorts=False,
        action_meta={
            "action_allocation_bins": 1,
            "action_level_bins": 1,
            "action_max_offset_bps": 0.0,
        },
    )

    assert resume_state == ResumeState(update=3, global_step=512, best_return=0.75)
