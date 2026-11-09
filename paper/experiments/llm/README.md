# LLM forecaster study (option D, self-hosted model in the cloud)

The protocol is fixed in [`PREREGISTRATION.md`](PREREGISTRATION.md) and was written before any model was queried.

| File | What it does |
|---|---|
| `llm_forecaster.py` | builds the prompts (`build`) and runs them through a backend (`run`); resumable |
| `api_backends.py` | Groq and Zoho Catalyst free-tier clients (standard library only) |
| `kaggle_llm_forecaster.ipynb` | the self-hosted run on Kaggle's free T4 GPUs (Llama 3.1 8B + Llama 3.2 1B) |
| `kaggle_llm_experiments2_3.ipynb` | follow-up run (addendum): experiment 2 (signals + reasoning) and experiment 3 (Phi-3-medium, cutoff Oct 2023) |
| `kaggle_llm_experiment3.ipynb` | experiment 3 alone in a fresh session (the combined run ran out of GPU memory on Phi-3-medium) |
| `run_llm.py` | adds the LLM agent to the NSE agents, builds every fault scenario, runs the combiners, writes `llm_results.json` |
| `outputs/` | model answers, one JSONL per model and condition (committed, like the other forecast files) |
| `requests/` | prompts built from the price file (not committed: derived from Yahoo Finance data) |

## 1. Self-hosted model on Kaggle (main run, free)

Follow the steps at the top of `kaggle_llm_forecaster.ipynb`: accept the two Llama licences on Hugging Face, add an
`HF_TOKEN` secret, pick **GPU T4 x2**, turn internet on, then *Save & Run All (Commit)*. About 19k short generations;
roughly 1 to 3 hours (estimate). Download `llm_outputs.zip` and unzip it into `outputs/`.

## 2. Free API models (optional extra models for the swap fault)

These need no GPU, so any machine works, including a Kaggle or Colab CPU notebook.

```bash
python llm_forecaster.py build --prices <path to nse_daily.csv>          # writes requests/

# Groq free plan: about 150-300 calls/day for gpt-oss-120b, so 1,482 calls take about a week.
# Run the same command once a day; it stops cleanly at the daily limit and resumes next time.
export GROQ_API_KEY=...                                                  # console.groq.com > API Keys
python llm_forecaster.py run --backend groq --model openai/gpt-oss-120b --tag groq_gptoss120b \
       --requests requests/clean.jsonl --start 2025-01-01

# Zoho Catalyst QuickML (free tier about 1M input + 500K output tokens a month; enough for 1,482 calls)
export ZOHO_CLIENT_ID=... ZOHO_CLIENT_SECRET=... ZOHO_REFRESH_TOKEN=... ZOHO_ORG_ID=... ZOHO_ENDPOINT_URL=...
export ZOHO_BODY_TEMPLATE=zoho_body.json      # paste the JSON sample from the endpoint's test page, with {system} {user} {max_tokens}
python llm_forecaster.py run --backend zoho --tag zoho_qwen --requests requests/clean.jsonl --start 2025-01-01 --limit 5   # try 5 first
```

The Zoho client has not been tested against a live endpoint yet; check the first five answers before a full run.

## 3. Results

```bash
python run_llm.py                                                    # experiment 1 -> llm_results.json
python run_llm.py --prefix feat_                                     # experiment 2 -> llm_results_feat.json
python run_llm.py --main phi3_medium --swap llama32_1b --out llm_results_phi3.json   # experiment 3
```

Scenarios without the outputs they need are skipped and listed in the result file.

## Offline check

`--backend stub` replaces the model with a simple rule, so the whole pipeline can be checked without a GPU or key:

```bash
python llm_forecaster.py build --prices <nse_daily.csv> --out /tmp/req
for c in clean clean_pre named named_pre news_alarm news_calm persona rationale; do
  python llm_forecaster.py run --backend stub --tag stub --requests /tmp/req/$c.jsonl --out /tmp/out; done
python run_llm.py --outputs /tmp/out --requests /tmp/req --main stub --swap --out /tmp/stub_results.json
```
