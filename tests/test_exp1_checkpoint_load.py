import torch

from binanceexp1.run_experiment import _infer_max_len as infer_max_len_run
from binanceexp1.sweep import _infer_max_len as infer_max_len_sweep
from binanceneural.config import TrainingConfig


def test_infer_max_len_prefers_pos_encoding_shape():
    cfg = TrainingConfig(sequence_length=96)
    state_dict = {"pos_encoding.pe": torch.zeros(96, 256)}
    assert infer_max_len_run(state_dict, cfg) == 96
    assert infer_max_len_sweep(state_dict, cfg) == 96


def test_infer_max_len_falls_back_to_sequence_length():
    cfg = TrainingConfig(sequence_length=128)
    state_dict = {}
    assert infer_max_len_run(state_dict, cfg) == 128
    assert infer_max_len_sweep(state_dict, cfg) == 128
