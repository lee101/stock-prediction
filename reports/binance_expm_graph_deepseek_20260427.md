# Binance Expm / Graph / DeepSeek Experiments

Date: 2026-04-27

## Goal

Prototype three research lanes for beating the current Binance baselines under
realistic sequential evaluation:

- Differentiable matrix-exponential SSM: learn a continuous-time generator
  `Q_t`, use exact `torch.matrix_exp(Q_t)`, and optimize portfolio weights
  without Monte Carlo sampling.
- Transition graph policy: encode stateful rules like rotate from a winner into
  a dumped asset, go cash in bad regimes, and optionally short weak assets.
- DeepSeek graph planner: ask `deepseek-v4-pro` for graph-policy configs, then
  walk them through the same local simulator hour by hour.

Promotion target remains unchanged: worst-slippage monthly return at least
`27%`, max drawdown at most `20%`, decision lag at least `2`.

## New Experiment Code

- `experiments/binance_expm_ssm/`
- `experiments/binance_transition_graph/`
- `experiments/binance_deepseek_planner/`
- `experiments/binance_openai_planner/`

The graph simulator is shared by the hand/random graph search and DeepSeek
planner. It evaluates hourly Binance CSVs sequentially with decision lag,
fees, slippage, leverage, and borrow cost. It is research-only; any winner
still needs the pufferlib C binary-fill gate before production.

## Runs

### Transition Graph Search

Command:

```bash
python experiments/binance_transition_graph/run_graph_experiment.py \
  --symbols BTCUSDT ETHUSDT SOLUSDT DOGEUSDT AAVEUSDT LINKUSDT \
  --train-hours 2160 \
  --eval-hours 720 \
  --trials 128 \
  --top-k 8 \
  --max-leverage 2 \
  --allow-short \
  --output-dir experiments/binance_transition_graph/results
```

Artifacts:

- `experiments/binance_transition_graph/results/graph_experiment_20260427_122113.json`
- `experiments/binance_transition_graph/results/graph_experiment_20260427_122113.csv`
- `experiments/binance_transition_graph/results/graph_experiment_20260427_122113.md`

Result:

- Best holdout candidate was `graph_cash`: `+0.00%` worst-slippage monthly,
  `0.00%` drawdown.
- Best non-cash holdout row was `rand_graph_1337_0083`: `-5.68%` worst-slippage
  monthly, `11.44%` drawdown.
- Several train leaders had positive train monthly returns (`+3%` to `+5%`),
  but all non-cash rows failed the 30-day unseen holdout.

Verdict: useful harness, no production candidate.

### Differentiable Expm SSM

Command:

```bash
python experiments/binance_expm_ssm/expm_ssm.py \
  --symbols BTCUSDT ETHUSDT SOLUSDT DOGEUSDT AAVEUSDT LINKUSDT \
  --train-hours 720 \
  --eval-hours 168 \
  --epochs 10 \
  --hidden-dim 32 \
  --num-states 8 \
  --output-dir experiments/binance_expm_ssm/results
```

Artifact:

- `experiments/binance_expm_ssm/results/expm_ssm_20260427_122119.json`

Result:

- Train monthly: `-0.08%`, train max drawdown: `14.84%`.
- Eval monthly: `-22.58%`, eval max drawdown: `9.59%`.
- Final eval weights stayed diversified and low-turnover, so the model mostly
  learned a smooth static allocation rather than a profitable transition policy.

Verdict: differentiable exact-transition path works technically, but this small
objective/model is not yet useful.

### DeepSeek `deepseek-v4-pro` Graph Planner

Command:

```bash
.venv313/bin/python experiments/binance_deepseek_planner/deepseek_graph_planner.py \
  --load-env-real \
  --model deepseek-v4-pro \
  --symbols BTCUSDT ETHUSDT SOLUSDT DOGEUSDT AAVEUSDT LINKUSDT \
  --train-hours 720 \
  --eval-hours 240 \
  --max-leverage 2 \
  --allow-short \
  --candidates 1 \
  --max-tokens 8192 \
  --output-dir experiments/binance_deepseek_planner/results
```

Artifacts:

- `experiments/binance_deepseek_planner/results/deepseek_eval_20260427_122810.json`
- `experiments/binance_deepseek_planner/results/deepseek_eval_20260427_122810.csv`
- `experiments/binance_deepseek_planner/results/deepseek_raw_20260427_122810.txt`

Result:

- Candidate: `Adaptive Risk-Aware Rotation`.
- Worst-slippage monthly: `-27.21%`.
- Worst-slippage max drawdown: `14.04%`.
- The first `deepseek-v4-pro` attempt with three candidates returned truncated
  JSON; raw artifact: `experiments/binance_deepseek_planner/results/deepseek_raw_20260427_122539.txt`.
- The runner now requests JSON-object formatting when supported and records
  parse failures as JSON artifacts instead of losing the run.

Verdict: `deepseek-v4-pro` integration works, but the first parsed candidate is
not profitable.

### DeepSeek Simulator-In-The-Loop Daily Replanner

Command:

```bash
.venv313/bin/python experiments/binance_deepseek_planner/deepseek_iterative_replanner.py \
  --load-env-real \
  --provider deepseek \
  --model deepseek-v4-pro \
  --days 7 \
  --history-hours 168 \
  --calibration-days 4 \
  --candidates 1 \
  --symbols BTCUSDT ETHUSDT SOLUSDT DOGEUSDT AAVEUSDT LINKUSDT \
  --output-dir experiments/binance_deepseek_planner/results \
  --max-tokens 4096 \
  --temperature 0.2 \
  --api-timeout-s 180 \
  --retries 3
```

Artifacts:

- Guarded run: `experiments/binance_deepseek_planner/results/deepseek_iter_replanner_20260427_133224.json`
- Guarded report: `experiments/binance_deepseek_planner/results/deepseek_iter_replanner_20260427_133224.md`
- Active-only replay: `experiments/binance_deepseek_planner/results/deepseek_iter_replanner_20260427_134452.json`
- Active-only report: `experiments/binance_deepseek_planner/results/deepseek_iter_replanner_20260427_134452.md`

Method:

- DeepSeek proposes daily plan templates rather than absolute one-shot price
  plans.
- Local code mutates allocation and stop/take-profit variants.
- Candidate selection scores only trailing calibration days; the selected
  template is then executed on the next unseen 24-hour window.
- A `--prod-inference` flag flips the same template interface to OpenAI
  `gpt-5.5` / `xhigh` structured output for production-time planning, but this
  run used DeepSeek for research.

Results:

- Guarded selector with cash guard: `$10,000.00` final equity, `+0.00%` total,
  `0.00%` drawdown, `0` trades. It rejected every active candidate because the
  trailing score was negative after drawdown/tail/trade penalties.
- Active-only replay using the same DeepSeek raw responses: `$10,008.23` final
  equity, `+0.08%` total, `+0.36%` monthly-equivalent, `0.38%` max drawdown,
  Sortino `1.36`, `18` trades.
- DeepSeek provider reliability was mixed: one full run needed a retry for an
  empty provider response, and earlier attempts exposed truncated JSON. The
  runner now has API timeouts, flushed day-level progress, parse retries, and
  raw-response replay.

Verdict: simulator-in-the-loop selection improved the one-shot GPT-5.5 result
from `-3.31%` to flat/slightly positive on the same week, but it is still far
below the production gate. The promising part is the workflow, not the current
edge size.

### Prompt/Reasoning Ablations

Commands:

```bash
.venv313/bin/python experiments/binance_openai_planner/gpt55_week_replanner.py \
  --provider codex \
  --model gpt-5.5 \
  --reasoning-effort low \
  --prompt-mode standard \
  --feature-style summary \
  --days 7 \
  --history-hours 168 \
  --symbols BTCUSDT ETHUSDT SOLUSDT DOGEUSDT AAVEUSDT LINKUSDT \
  --output-dir experiments/binance_openai_planner/results \
  --codex-timeout-s 360
```

```bash
.venv313/bin/python experiments/binance_openai_planner/gpt55_week_replanner.py \
  --provider codex \
  --model gpt-5.5 \
  --reasoning-effort low \
  --prompt-mode minimal \
  --feature-style ohlc_mid \
  --line-bars 48 \
  --days 7 \
  --history-hours 168 \
  --symbols BTCUSDT ETHUSDT SOLUSDT DOGEUSDT AAVEUSDT LINKUSDT \
  --output-dir experiments/binance_openai_planner/results \
  --codex-timeout-s 360
```

```bash
.venv313/bin/python experiments/binance_openai_planner/gpt55_week_replanner.py \
  --provider codex \
  --model gpt-5.5 \
  --reasoning-effort low \
  --prompt-mode minimal \
  --feature-style line_pct \
  --line-bars 48 \
  --days 7 \
  --history-hours 168 \
  --symbols BTCUSDT ETHUSDT SOLUSDT DOGEUSDT AAVEUSDT LINKUSDT \
  --output-dir experiments/binance_openai_planner/results \
  --codex-timeout-s 360
```

```bash
.venv313/bin/python experiments/binance_deepseek_planner/deepseek_iterative_replanner.py \
  --load-env-real \
  --provider deepseek \
  --model deepseek-v4-pro \
  --prompt-mode minimal \
  --feature-style ohlc_mid \
  --line-bars 48 \
  --days 7 \
  --history-hours 168 \
  --calibration-days 4 \
  --candidates 1 \
  --disable-cash-guard \
  --symbols BTCUSDT ETHUSDT SOLUSDT DOGEUSDT AAVEUSDT LINKUSDT \
  --output-dir experiments/binance_deepseek_planner/results \
  --max-tokens 4096 \
  --temperature 0.2 \
  --api-timeout-s 180 \
  --retries 3
```

Results:

| Run | Prompt / features | Total | Monthly eq. | Max DD | Trades | Artifact |
|-----|-------------------|-------|-------------|--------|--------|----------|
| GPT-5.5 xhigh one-shot baseline | standard summary | `-3.31%` | `-13.50%` | `4.56%` | `62` | `experiments/binance_openai_planner/results/gpt55_week_replanner_20260427_124238.md` |
| GPT-5.5 low | standard summary | `-2.25%` | `-9.33%` | `2.75%` | `46` | `experiments/binance_openai_planner/results/gpt55_week_replanner_20260427_134931.md` |
| GPT-5.5 low | minimal + OHLC midpoint line tokens | `-0.66%` | `-2.81%` | `1.34%` | `38` | `experiments/binance_openai_planner/results/gpt55_week_replanner_20260427_135129.md` |
| GPT-5.5 low | no-think + OHLC midpoint line tokens | `-1.75%` | `-7.34%` | `2.61%` | `32` | `experiments/binance_openai_planner/results/gpt55_week_replanner_20260427_150247.md` |
| GPT-5.5 low | minimal + OHLC previous-close pct tokens | `-1.14%` | `-4.81%` | `2.50%` | `32` | `experiments/binance_openai_planner/results/gpt55_week_replanner_20260427_141925.md` |
| DeepSeek Pro | minimal + OHLC midpoint, active only | `-1.08%` | `-4.56%` | `1.40%` | `18` | `experiments/binance_deepseek_planner/results/deepseek_iter_replanner_20260427_135326.md` |

Notes:

- The user's hypothesis is directionally supported on GPT-5.5: lower reasoning
  was less bad than xhigh on the same week, and compact line-token price action
  was better than feature-summary-only prompting.
- Midpoint-normalized OHLC tokens beat previous-close OHLC tokens for GPT-5.5
  on this week. The likely reason is that midpoint encoding gives the model a
  direct intrabar range/position representation, which is closer to limit and
  stop placement than raw OHLC moves from previous close.
- DeepSeek did not benefit from the minimal/OHLC-midpoint prompt in this run
  and remained provider-unstable: several days required retries for empty or
  malformed responses.

Verdict: use low-reasoning/minimal prompts and compact midpoint OHLC line tokens
for the next LLM planning experiments. This still is not profitable enough; it
only reduces the one-shot failure from large negative to small negative.

### DeepSeek Energy Refit Agent

Command:

```bash
.venv313/bin/python experiments/binance_deepseek_planner/deepseek_energy_refit_agent.py \
  --load-env-real \
  --model deepseek-v4-flash \
  --prompt-mode no_think \
  --feature-style ohlc_mid \
  --line-bars 48 \
  --days 7 \
  --history-hours 168 \
  --calibration-days 4 \
  --candidates 1 \
  --population 72 \
  --boost-pool 192 \
  --boost-take 16 \
  --neural-pool 192 \
  --neural-take 16 \
  --deepseek-refit \
  --symbols BTCUSDT ETHUSDT SOLUSDT DOGEUSDT AAVEUSDT LINKUSDT \
  --max-tokens 2048 \
  --temperature 0.1 \
  --api-timeout-s 120 \
  --retries 3 \
  --output-dir experiments/binance_deepseek_planner/results
```

Method:

- DeepSeek `deepseek-v4-flash` seeds compact JSON templates with `no_think`
  prompts and no tool/web/code instructions.
- Local code mutates template vectors over side/allocation/entry/take-profit/stop.
- A boosted-tree scorer refits on trailing simulation scores. `xgboost` and
  `catboost` were not installed in `.venv313`, so this run used sklearn's
  HistGradientBoosting fallback.
- Optional DeepSeek refit asks for smaller constrained mutations.
- A small Torch MLP scorer does the final lower-aggression mutation pass.

Results:

| Run | Total | Monthly eq. | Max DD | Trades | Artifact |
|-----|-------|-------------|--------|--------|----------|
| Flash no-think energy refit | `-1.50%` | `-6.30%` | `1.56%` | `12` | `experiments/binance_deepseek_planner/results/deepseek_energy_refit_20260427_144204.md` |
| Same flash seeds, risk clamps | `-0.87%` | `-3.70%` | `1.03%` | `10` | `experiments/binance_deepseek_planner/results/deepseek_energy_refit_20260427_150120.md` |
| Risk clamps + cash guard | `-0.45%` | `-1.91%` | `0.45%` | `10` | `experiments/binance_deepseek_planner/results/deepseek_energy_refit_20260427_150144.md` |
| Risk clamps + cash guard + score hurdle `0.0025` | `+0.00%` | `+0.00%` | `0.00%` | `0` | `experiments/binance_deepseek_planner/results/deepseek_energy_refit_20260427_150229.md` |
| Gradient + CEM/RL active replay | `-1.24%` | `-5.24%` | `1.24%` | `8` | `experiments/binance_deepseek_planner/results/deepseek_energy_refit_20260427_164305.md` |
| Gradient + CEM/RL robust 4d gates | `-0.74%` | `-3.14%` | `0.76%` | `8` | `experiments/binance_deepseek_planner/results/deepseek_energy_refit_20260427_164508.md` |
| Gradient + CEM/RL robust 10d gates + score hurdle | `+0.00%` | `+0.00%` | `0.00%` | `0` | `experiments/binance_deepseek_planner/results/deepseek_energy_refit_20260427_164546.md` |

Findings:

- The staged optimizer overfit the trailing calibration days. It often found
  small positive calibration scores, then lost on the next unseen day.
- Risk clamps helped, but did not create edge.
- A minimum calibration-score hurdle correctly rejected all weak active trades,
  preserving cash at `0.00%`.
- Direct Torch surrogate-gradient refinement and CEM/evolution-strategy RL
  refinement increased apparent calibration scores, but still failed on the
  next unseen day. The 4-day robust gate reduced losses; the 10-day robust gate
  rejected every active trade.
- DeepSeek `deepseek-v4-pro` under `no_think` JSON prompting was less reliable
  than flash in this session: repeated empty responses and malformed JSON
  forced early termination of the Pro run.
- "No thinking" was worse than "minimal low thinking" for GPT-5.5 on this week:
  `-1.75%` versus `-0.66%`.

Verdict: the advanced staged agent is useful infrastructure, but the current
scoring objective is not strong enough. More optimizer iterations are not
automatically better: they exploit the calibration slice unless there are
strong rolling-window robustness gates. The next change should train the local
refitters on many rolling windows and evaluate held-out windows before trusting
boosted/neural/gradient/RL predicted score.

### Rolling GPU Energy Trainer

Implemented `experiments/binance_deepseek_planner/rolling_energy_trainer.py` as
the next version of the energy-agent path. It generates random daily plan
templates over many rolling historical days, simulates each next-day outcome,
then trains a Torch MLP scorer on market features plus the template vector. At
inference it scores large candidate batches on GPU, gradient-refines top
candidates through the scorer, and only trades if the predicted score clears a
held-out validation top-day threshold.

Command for the best 7-day run:

```bash
.venv313/bin/python experiments/binance_deepseek_planner/rolling_energy_trainer.py \
  --train-days 240 \
  --val-days 45 \
  --eval-days 7 \
  --history-hours 240 \
  --templates-per-day 48 \
  --eval-candidates 8192 \
  --gradient-top 128 \
  --gradient-steps 60 \
  --epochs 60 \
  --batch-size 2048 \
  --device cuda \
  --symbols BTCUSDT ETHUSDT SOLUSDT DOGEUSDT AAVEUSDT LINKUSDT
```

Results:

| Variant | Total return | Monthly equiv. | Max DD | Trades | Notes |
|---|---:|---:|---:|---:|---|
| 2048 candidates, fixed `0.0005` hurdle | `-0.13%` | `-0.56%` | `0.50%` | `18` | active but one marginal final-day loss |
| 2048 candidates, manual `0.0010` hurdle | `+0.14%` | `+0.62%` | `0.25%` | `14` | diagnostic only; threshold chosen after inspection |
| 2048 candidates, validation-calibrated hurdle | `+0.14%` | `+0.62%` | `0.25%` | `14` | learned threshold `0.000914` from 16/45 validation top-days |
| 8192 candidates, validation-calibrated hurdle | `+0.18%` | `+0.77%` | `0.08%` | `12` | wider GPU inference search improved return and DD |

Artifacts:

- `experiments/binance_deepseek_planner/results/rolling_energy_trainer_20260427_193720.md`
- `experiments/binance_deepseek_planner/results/rolling_energy_trainer_20260427_193907.md`
- `experiments/binance_deepseek_planner/results/rolling_energy_trainer_20260427_194218.md`
- `experiments/binance_deepseek_planner/results/rolling_energy_trainer_20260427_194511.md`

Verdict: this does not approach the production target, but it is a better
research primitive than the single-window refitters. The validation-calibrated
threshold avoids the most obvious weak active days without eval-week peeking,
and the higher inference-compute run improved both return and drawdown. This is
the first evidence in these experiments that scaling candidate search and
gradient fixup can help when the scorer is trained on many rolling windows
instead of a tiny trailing calibration slice.

Follow-up variants:

| Variant | Total return | Monthly equiv. | Max DD | Trades | Result |
|---|---:|---:|---:|---:|---|
| Frontier candidate generator, conservative gate | `+0.00%` | `+0.00%` | `0.00%` | `0` | validation rejected all active trades |
| Frontier + return-weight/high-gross forced active | `-1.02%` | `-4.32%` | `1.21%` | `16` | bad; generator not ready |
| Legacy + 4d top-k trailing rerank | `+0.13%` | `+0.57%` | `0.06%` | `12` | lower DD, lower return than best |
| Legacy + raw-return label target | `-0.17%` | `-0.73%` | `0.37%` | `4` | worse than score label |
| Legacy + 2-member scorer ensemble | `+0.05%` | `+0.22%` | `0.15%` | `8` | higher val corr, worse PnL |

New knobs added to the trainer:

- `--candidate-style legacy|frontier|mixed`
- `--label-target score|total_return`
- `--return-weight`, `--drawdown-penalty`, `--trade-penalty`, `--loss-penalty`
- `--rerank-calibration-days`, `--rerank-top`, `--rerank-calibration-weight`, `--rerank-worst-weight`
- `--ensemble-size`, `--uncertainty-penalty`

The best result is still the earlier 8192-candidate legacy run:
`rolling_energy_trainer_20260427_194511.md`, `+0.18%` total,
`+0.77%` monthly-equivalent, `0.08%` max DD, 12 trades. The follow-up runs are
useful negative controls: more complex candidate priors, raw-return labels,
ensembles, and trailing reranking did not improve PnL on this week. The next
valid improvement target is a multi-window sweep that accepts a knob only if it
improves median and p10 unseen weekly/monthly return.

### OpenAI Strict Structured-Output Planner

Commands:

```bash
python experiments/binance_openai_planner/openai_graph_planner.py \
  --offline \
  --symbols BTCUSDT ETHUSDT SOLUSDT DOGEUSDT AAVEUSDT LINKUSDT \
  --train-hours 240 \
  --eval-hours 120 \
  --max-leverage 2 \
  --allow-short \
  --candidates 1 \
  --output-dir experiments/binance_openai_planner/results_smoke
```

```bash
.venv313/bin/python experiments/binance_openai_planner/openai_graph_planner.py \
  --load-env-real \
  --model gpt-5.4-nano \
  --reasoning-effort low \
  --symbols BTCUSDT ETHUSDT SOLUSDT DOGEUSDT AAVEUSDT LINKUSDT \
  --train-hours 720 \
  --eval-hours 240 \
  --max-leverage 2 \
  --allow-short \
  --candidates 1 \
  --output-dir experiments/binance_openai_planner/results
```

Artifacts:

- `experiments/binance_openai_planner/results_smoke/openai_eval_20260427_123457.json`
- `experiments/binance_openai_planner/results/openai_eval_20260427_123457.json`

Result:

- The offline smoke verified the OpenAI planner path and production gate
  metadata. It uses Responses API Structured Outputs with `json_schema` and
  `strict: true`.
- Offline smoke best row: `graph_regime_short`, `+16.18%` worst-slippage
  monthly, `13.18%` drawdown. This is below the `27%` monthly gate, so
  `prod_route=False`.
- The live `gpt-5.4-nano` structured-output call reached OpenAI but failed with
  `insufficient_quota` (HTTP 429). No model fallback was used.
- If a future candidate passes `monthly_target >= 0.27` and
  `max_drawdown <= 0.20`, the artifact records production routing to
  `gpt-5.5` with reasoning effort `xhigh`. The production model does not need
  to be called during the cheap research smoke.

Verdict: strict OpenAI structured-output wiring is implemented and offline
verified; live API verification is blocked by account quota.

### GPT-5.5/xhigh Daily Portfolio Replanner

Command:

```bash
.venv313/bin/python experiments/binance_openai_planner/gpt55_week_replanner.py \
  --provider codex \
  --model gpt-5.5 \
  --reasoning-effort xhigh \
  --days 7 \
  --history-hours 168 \
  --symbols BTCUSDT ETHUSDT SOLUSDT DOGEUSDT AAVEUSDT LINKUSDT \
  --output-dir experiments/binance_openai_planner/results \
  --codex-timeout-s 360
```

Artifacts:

- `experiments/binance_openai_planner/results/gpt55_week_replanner_20260427_124238.json`
- `experiments/binance_openai_planner/results/gpt55_week_replanner_20260427_124238.md`

Result:

- Provider/model: Codex CLI, `gpt-5.5`, reasoning `xhigh`, schema-constrained output.
- Window: 7 sequential daily plans, one model call per day, then 24 hourly Binance bars per plan.
- Symbols: `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `DOGEUSDT`, `AAVEUSDT`, `LINKUSDT`.
- Execution assumptions: 10 bps fee, 5 bps slippage, two-hour decision lag, one entry per symbol per day, daily forced flatten.
- Initial cash: `$10,000.00`.
- Final equity: `$9,669.20`.
- Total return: `-3.31%`.
- Monthly-equivalent return: `-13.50%`.
- Max drawdown: `4.56%`.
- Sortino: `-8.02`.
- Trades: `62`.

Failure notes:

- The model produced coherent stockagent-style plans, but most days were still
  long correlated crypto beta. The short legs were small and intermittent, so
  they did not hedge the selloff days enough.
- Tight stops limited drawdown, but after 15 bps all-in friction they also
  turned normal hourly noise into repeated small realized losses.
- This looks less like a parsing/planning failure and more like an objective
  mismatch: the LLM is generating plausible discretionary plans, not plans
  optimized against the simulator's exact fill, lag, and fee mechanics.

Verdict: the big-model daily replanner does not clear the research gate on
this week. It is useful as a constrained plan generator, but not a production
allocator without a simulator-in-the-loop mutation or RL/black-box optimizer.

## Takeaways

- The graph simulator is the best immediate foundation: it gives a fast,
  path-dependent way to compare human rules, random search, and LLM-generated
  plans before spending pufferlib evaluation time.
- Recent data punished almost every active graph policy; cash winning the
  90d/30d screen is a useful negative result and matches the broader Binance33
  observation that fragile recent edges do not generalize cleanly.
- The expm SSM is promising as infrastructure for exact differentiable regime
  dynamics, but the next version needs a better objective: monthly-return gate,
  explicit cash state, drawdown-constrained allocation, and perhaps supervised
  transition labels from known strong policies.
- DeepSeek should be used as a candidate generator, not as a trusted optimizer.
  The useful loop is: ask for compact JSON configs, simulate sequentially,
  feed back metrics, and ask for targeted mutations.
- DeepSeek-first simulator feedback is materially better than one-shot LLM
  plans, but the first daily-template loop only reached `+0.08%` over a week in
  active-only mode. This is research signal, not a deployable strategy.
- GPT-5.5 low reasoning and compact OHLC midpoint line tokens were materially
  less bad than xhigh/summary prompting, supporting the idea that extra thinking
  and broad feature prose can distract from the actual price-action tokens.
- OpenAI planner should remain strict-schema only. Cheap `gpt-5.4-nano` can be
  used for plumbing tests once quota exists; production routing should switch to
  `gpt-5.5` / `xhigh` only after the offline PnL/DD gate passes.
- The direct GPT-5.5/xhigh stockagent-style replanner was not enough by itself:
  the plans were coherent, but they still lost `3.31%` over the 7-day window.
  The likely missing piece is simulator feedback during plan search, not a
  larger one-shot planner.
- The rolling GPU energy scorer is the best current agent-refit primitive:
  it trains on many simulated windows, uses CUDA for scoring/training, and the
  8192-candidate inference run improved the week to `+0.18%` with `0.08%` max
  DD. This is still research-only, not production.

## Next Experiments

- Add rolling multi-window evaluation to the graph harness so configs are
  selected by median monthly and p10 monthly across many unseen windows, not one
  final 30-day slice.
- Add a DeepSeek feedback loop that sends the failed metrics back to
  `deepseek-v4-pro` and requests constrained mutations of the top two configs.
- Add simulator-in-the-loop mutation for the GPT-5.5 daily plans: ask for a
  small batch of candidate plans per day, simulate all, then carry forward only
  the best risk-adjusted plan into the next day.
- Improve the DeepSeek daily-template objective: reduce the cash guard's
  conservatism with a minimum-active exploration mode, add rolling multi-window
  replay, and ask DeepSeek to mutate only the top active templates instead of
  re-planning from scratch each day.
- Extend the expm SSM with a cash state and max-drawdown barrier loss before
  trying larger hidden sizes or GPU training.
- Turn the rolling energy trainer into a multi-window scaling sweep: train on
  rolling histories, evaluate candidate counts/gradient steps across many
  unseen weeks, and accept extra inference compute only if median and p10
  monthly returns improve without increasing max drawdown.
