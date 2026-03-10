# Autoresearch Stock Experiments

This package is reserved for isolated experiment variants used by the
autoresearch stock planner.

Guidelines:

- Put new experiment code in `src/autoresearch_stock/experiments/<experiment_slug>/`.
- Keep each experiment self-contained and deterministic.
- Add a short `README.md` inside each experiment directory with the hypothesis and exact replay command.
- Keep `src/autoresearch_stock/train.py` minimal when wiring an experiment in or out.
- Do not duplicate the fixed evaluation harness from `prepare.py`.
