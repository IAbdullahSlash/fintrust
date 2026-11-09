# Addendum 2 (2026-11-09): label-free behaviour monitors

Written before any monitor below was computed. The LLM answers already exist (experiments 1-3, `outputs/`), and the
loss-based results of those experiments are known, so this is an analysis of existing data, not a new prospective run.
What is fixed here, before computing anything, is every monitor, threshold rule, fault and metric.

## Question

Can monitors that look at an LLM agent's *behaviour* (its answers and its responses to fixed probe prompts) catch faults
that a loss-based monitor misses or catches late, and at what false-alarm cost?

## Data

Per asset (the six NSE instruments), daily LLM answers p_t in [0, 1] and the binary label y_t from
`../fair/nse_vol_forecasts.csv`. The label of day t is known on day t+1. Reference period: 2024-01-01 to 2024-06-30
(R1). Calibration period: 2024-07-01 to 2024-12-31 (R2). Evaluation: 2025, with every fault starting on 2025-01-01.

Configurations (main model, prompts): E1 = `llama31_8b`, experiment-1 prompts; E2 = `llama31_8b`, `feat_` prompts;
E3 = `phi3_medium`, experiment-1 prompts.

Faults from 2025-01-01: alarming headline, calming headline, persona drift, silent swap to `llama32_1b` (same prompt
family), and the three output faults of the paper applied to the LLM answers with `engine.corrupt_matrix`
(inversion, stuck at 0.95, overconfident). Clean 2025 is used to count false alarms.

## Monitors

* **L1, loss CUSUM (one-sided).** Daily Brier loss b_t = (p_t - y_t)^2, mean mu and s.d. sigma on R1 for this agent and
  asset. S_t = max(0, S_{t-1} + (b_t - mu)/sigma - 0.5); alarm when S_t > h. Uses the label one day late.
  This is the per-agent reference suggested in the paper's discussion and is the strongest loss monitor we consider.
* **L2, loss CUSUM (two-sided).** As L1 plus the mirror statistic for a *fall* in loss; alarm when either exceeds h.
  Included so that a silent swap that lowers loss is not missed only because the monitor looks one way.
* **B1, answer-distribution drift (PSI).** Population stability index between the answers of the last 20 trading days
  and the R1 answers of the same agent and asset, on ten equal bins of [0, 1] (the last bin includes 1), with 0.5
  pseudo-counts per bin. Needs no labels.
* **B2, probe prompts (fingerprint).** A fixed set of K probe prompts with recorded answers from the trusted
  configuration is re-asked every day; alarm when the mean absolute difference from the recorded answers exceeds 0.10
  or any probe answer fails to parse. Because the faulty configurations were only queried on 2025 prompts, probes are
  drawn from the 2025 prompts of the asset (their content is irrelevant to the check). We report the share of 1,000
  random probe sets (seeded) that alarm, for K = 5, 10 and 20. Headline injection is assumed to enter through the live
  input feed and is therefore not applied to probe prompts; persona drift and model swap change the system prompt or
  the model and are seen by probes; output faults are applied to probe answers as a post-processing fault would be.
  With greedy decoding a clean probe set reproduces its recorded answers, so B2 has no false alarms here; under
  non-deterministic serving the 0.10 tolerance is what absorbs noise, which these data cannot test.
* **B3, format check.** Alarm when more than 5% of the last 20 answers failed to parse.

## Thresholds

For L1, L2 and B1, h is the largest statistic reached on R2 over all six assets of the configuration (statistics
restarted at the start of R2), so each monitor has no alarm on calibration data by construction. For B1 we also
report the conventional PSI rule (alarm above 0.25). No threshold is changed after seeing 2025.

## Metrics

Per configuration, fault and monitor: number of assets (of six) with an alarm in 2025 and median trading days from
2025-01-01 to the first alarm. For clean 2025: number of assets with any alarm and days to the first one. B2 reports
the share of probe sets that alarm (delay one trading day when it alarms).

## Limitations stated in advance

None of the three models had forecasting skill, so detection of faults in a useful LLM is not tested. Thresholds set
to zero false alarms on six months may be loose or tight; the clean 2025 count measures this.
