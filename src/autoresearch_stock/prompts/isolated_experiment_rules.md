Use isolated, replayable experiment structure.

Rules:

- If you need more than a tiny `train.py` tweak, create a new self-contained module under `src/autoresearch_stock/experiments/<experiment_slug>/`.
- Keep `src/autoresearch_stock/train.py` as a thin dispatcher or wiring layer when possible.
- Write a short `README.md` inside the experiment directory with:
  - hypothesis,
  - exact files changed,
  - exact benchmark command,
  - borrowed repo ideas,
  - why the change should help the realistic simulator.
- Keep all new behavior deterministic:
  - set seeds if you add any,
  - avoid wall-clock-dependent branching,
  - avoid network access,
  - keep symbol lists, windows, and thresholds explicit.
- Mirror notes and exact replay commands into the scheduler experiment bundle directory for the turn.
- Prefer auditable functions and plain data structures over framework-heavy abstractions.

Success criteria:

- one coherent experiment,
- exact replay path,
- realistic simulator unchanged,
- no hidden dependencies.
