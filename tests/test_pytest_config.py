from __future__ import annotations

from pathlib import Path


def test_pytest_ini_does_not_force_shared_repo_basetemp() -> None:
    text = (Path(__file__).resolve().parents[1] / "pytest.ini").read_text(encoding="utf-8")

    assert "--basetemp" not in text
    assert ".pytest_tmp" in text


def test_compile_and_model_stress_tests_are_opt_in() -> None:
    root = Path(__file__).resolve().parents[1]

    for relative_path, required_markers in {
        "tests/test_chronos2_compile_fuzzing.py": (
            "pytest.mark.slow",
            "pytest.mark.model_required",
            "pytest.mark.cuda_required",
        ),
        "tests/test_chronos_compile_accuracy.py": (
            "pytest.mark.slow",
            "pytest.mark.model_required",
            "pytest.mark.cuda_required",
        ),
        "tests/test_chronos2_e2e_compile.py": (
            "pytest.mark.slow",
            "pytest.mark.model_required",
        ),
        "tests/test_cuda_graph_quick_check.py": (
            "pytest.mark.slow",
            "pytest.mark.model_required",
            "pytest.mark.cuda_required",
        ),
        "tests/test_cudagraph_mutation_warning.py": (
            "pytest.mark.slow",
            "pytest.mark.cuda_required",
        ),
        "tests/test_fastforecaster_trainer_smoke.py": (
            "pytest.mark.slow",
            "pytest.mark.model_required",
        ),
        "tests/test_autoresearch_stock_train_smoke.py": (
            "pytest.mark.slow",
            "pytest.mark.cuda_required",
        ),
        "tests/test_fastmarketsim_env.py": ("pytest.mark.slow",),
        "tests/test_fastmarketsim_parity.py": ("pytest.mark.slow",),
    }.items():
        text = (root / relative_path).read_text(encoding="utf-8")
        for marker in required_markers:
            assert marker in text
