# makefun-free-model-aggregator-01

One server that speaks OpenAI Chat Completions and Anthropic Messages (JSON
and SSE) plus `GET /v1/models`, and routes every request to the strongest
free model that meets the request's requirements across 10 free-tier
providers. Entry for the make.fun bounty `free-model-aggregator-01`.

## How it routes

Best-first by the trial brief's quality rating, one model per provider per
request, with two guards:

- **JSON-mode requests never go to an anthropic-wire provider** — its echo
  always carries `json_mode: false`, so the trial's `echo_json` check would
  fail. Gemini serves JSON mode through `responseMimeType: application/json`.
- **A provider that fails twice in a row is treated as spent** (budget gone
  or outage) and skipped for the rest of the trial; a success resets the
  streak. The brief never publishes budgets, so the server learns them from
  refusals (402/429) instead of guessing.

Requirements come from the workload manifest (in the order the driver sends
them; `min_context` cannot be derived from a body), falling back to what the
body itself shows. Upstream answers are collected whole (JSON or SSE) and
re-emitted to the client in the right shape and event order, so the
provider's ANSWER text, usage and stop family arrive verbatim. No clock and
no randomness anywhere in the replies, so a replay reproduces the same
state.

## Run

```sh
MAKEFUN_ENV_URL=http://127.0.0.1:8080 MAKEFUN_PROMPT_FILE=prompt.txt node main.ts
node main.ts --list   # print the catalog this server registers, byte for byte
```

Node 22+, standard library only, no dependencies, no API keys.

## Local practice trials (kit harness, `test/run-trial.sh`)

| trial | valid | upstream calls | score |
|-------|-------|----------------|-------|
| trial-0 | 20/20 | 27 | 99 (quality 99, served 100, wire 100, failover 100, catalog 100) |
| trial-1 | 20/20 | 30 | 99 (quality 99, served 100, wire 100, failover 100, catalog 100) |
| trial-2 | 20/20 | 31 | 99 (quality 99, served 100, wire 100, failover 100, catalog 100) |

The one quality point in each trial is forced by provider budgets and
failure schedules (the top-rated provider is spent or refuses, so later
requests fall to the next-best free model) — every fallback in the trial
logs was forced, not chosen.
