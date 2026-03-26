from sharpnessadjusteddailypolicy2.config import build_experiment, experiment_names



def test_muon_small_build_applies_expected_overrides():
    spec, dataset_cfg, train_cfg = build_experiment(
        "muon_small_stable",
        seed=17,
        dataset_overrides={"symbols": ("AAA", "BBB")},
        training_overrides={"batch_size": 8, "max_train_batches": 4},
    )
    assert spec.name == "muon_small_stable"
    assert train_cfg.optimizer_name == "muon"
    assert train_cfg.transformer_dim == 192
    assert train_cfg.transformer_layers == 3
    assert train_cfg.transformer_heads == 6
    assert train_cfg.ema_decay == 0.995
    assert train_cfg.batch_size == 8
    assert train_cfg.max_train_batches == 4
    assert train_cfg.seed == 17
    assert dataset_cfg.symbols == ("AAA", "BBB")
    assert train_cfg.dataset.symbols == ("AAA", "BBB")



def test_adamw_variant_and_registry_are_exposed():
    assert "adamw_small_stable" in experiment_names()
    _, _, train_cfg = build_experiment("adamw_small_stable")
    assert train_cfg.optimizer_name == "adamw"
    assert train_cfg.weight_decay > 0
    assert train_cfg.ema_decay > 0
