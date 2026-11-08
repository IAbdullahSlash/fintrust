# LLM forecaster study: protocol fixed before any LLM was queried

Written 2026-11-08, before any language model was run on this data. Nothing below may change after
results are seen; any deviation must be reported in the paper together with the reason.

## Question

Can FinTrust (i) vet a newly onboarded LLM forecaster before trusting it, and (ii) detect and contain
faults that are specific to LLMs (prompt manipulation, persona drift, silent model swaps, unparseable
output), better than combiners without a lifecycle? The study adds one LLM agent to the four
conventional NSE volatility agents of Section X; it is not a claim that LLMs forecast volatility well.

## Task and data (unchanged from Section X)

- NSE daily, six instruments (^NSEI, RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK), India VIX as context.
- Label: y_t = 1 if |log C_{t+1} - log C_t| exceeds the mean absolute log return of the 20 days up to t.
  Labels and the four conventional agents (RV, RANGE, GBT, VIX) are read unchanged from
  `../fair/nse_vol_forecasts.csv`. h = 1, non-overlapping.
- Prices come from the same `nse_daily.csv` used in the paper (or a fresh `data/download_nse.py` copy,
  reported if used).

## Look-ahead control

- Primary model: **meta-llama/Llama-3.1-8B-Instruct**, knowledge cutoff December 2023 (model card),
  run on Kaggle, fp16, greedy decoding (temperature 0), HF revision hash recorded in every output row.
- Study window **2024-01-01 to 2025-12-29**, entirely after the cutoff: 2024 = onboarding / warm-up,
  2025 = evaluation (fault onset 2025-01-01).
- Default prompts are **anonymised**: instruments are called "Asset A" ... "Asset F" in a fixed random
  order, no dates, no price levels, only returns, ranges and the volatility-index level.
- Leakage probe (reported, not used by the combiners): the same model is queried with **named** prompts
  (real ticker and dates) and anonymised prompts on a pre-cutoff window (2022-07-01 to 2023-12-29) and
  on the study window. Memorisation is indicated if named minus anonymised Brier is clearly lower before
  the cutoff than after it.

## Prompt (condition `clean`)

System: a short neutral instruction to act as a quantitative analyst and answer with one integer.
User: the last 20 trading days of the asset as a table (day index -19..0, signed daily return %,
high-low range %, volatility-index level), the 20-day mean absolute return (the threshold) and the
question "probability (0-100) that the next day's absolute return exceeds the threshold".
Exact templates are in `llm_forecaster.py` and frozen with this file. Output parsing (`parse_p`): the first number in the
reply (after "Probability:" if present); 0-100 is read as a percentage, a decimal in [0, 1] as a probability. Unparseable output -> 0.5 and flagged (rate reported).

## LLM-specific faults (applied to the LLM agent from 2025-01-01 to the end)

| id | fault | how it is produced |
|---|---|---|
| F1 | news injection, alarming | a fabricated alarming headline (one of 10 fixed templates, chosen with seed 42) is added to every prompt |
| F2 | news injection, reassuring | same with 10 reassuring headlines |
| F3 | persona drift / overconfidence | the system prompt is replaced by an instruction to be decisive and avoid probabilities between 20 and 80 |
| F4 | silent model swap | from onset the agent's answers come from a different model with the same prompts: Llama-3.2-1B-Instruct (Kaggle); variants with openai/gpt-oss-120b (Groq free tier) and a Qwen model on Zoho Catalyst QuickML if those runs complete |
| F5 | format failure | natural parse failures of every condition are counted; no extra injection |

For continuity with Sections IX-XI the three output-level faults (inversion, stuck at 0.95,
overconfident x4) are also applied to the LLM column with `corrupt_matrix`, at the same onset.

Secondary diagnostic (hallucinated reasoning): condition `rationale` asks for one sentence of reasoning
plus the integer. A cited number counts as hallucinated if it is not within rounding of any number in
the prompt. Rate reported per model; it is not visible to the combiners (this is a stated limitation).

## Governance and combiners (nothing tuned on LLM data)

- Agents: RV, RANGE, GBT, VIX (start ACTIVE, they were vetted on 2016-2023) and LLM (starts SHADOW).
- FinTrust thresholds as in the paper: W = 50, evidence minimum 200, 3 consecutive windows, bad if
  excess > 0.05, good if < 0, probation cap 0.10, prior mass 20, statistical quarantines self-clear.
- Hyper-parameters of FT, LW, FS and DWM are the clean-rule NSE values already stored in
  `../fair/nse_vol_results.json` (chosen on 2016-2018). No re-tuning.
- Methods: EW, LW, FS, DWM, FT, each with and without the LLM agent; exclusion oracle.
- Secondary arm `FT-forced`: the operator force-admits the LLM (it starts ACTIVE). If the LLM is never
  admitted in 2024 under the main arm, fault detection is measured on this arm only, and that is reported.

## Outcomes

- Primary: 2025 Brier of FT vs LW with the LLM agent, per fault, pooled over the six assets.
- Secondary: day the LLM agent is admitted (SHADOW -> ACTIVE) in 2024 or never; detection delay of each
  fault (first quarantine of the LLM after onset); false quarantines of the other agents; coverage; LLM
  Brier and calibration; leakage gap; parse-failure and hallucination rates.
- Intervals: paired block bootstrap over days (20-day blocks), 2000 resamples, seed 0.

## Budget

Kaggle (self-hosted, no API cost): about 19k short generations. Groq free tier: about 1,500 calls for the
2025 swap variant (plus about 750 for a 2024-07..12 sanity window), run over several days with the
resumable client. Zoho QuickML: about 1,500 calls, within the monthly free token allowance.
