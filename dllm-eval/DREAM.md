# DREAM integration

DREAM support is experimental. Use the existing environment and cached
`Dream-org/Dream-v0-Instruct-7B` weights; generation stays offline. The LLaDA
implementation, historical results and mathematical/code scorers are unchanged.
Reference answers and executable tests are read only after generation is saved.

## Relay and controls

Relay uses a 32-token confidence-parallel sliding frontier, threshold 0.90,
four MASK windows, eight-layer history stripes and history budget 128. DREAM's
logit for position `i` comes from hidden row `i-1`: both live queries and their
readout predecessors are recomputed, as are changed token identities. Other
positions retain full-context KV. Embedding boundaries always use current token
identities, including positions decoded after cache initialization. Consumed
hidden rows alone enter the output projection. There is no draft verification.

Both control backends use the same Instruct chat template as Relay:

| `--dream-backend` | Sampling | Cache |
| --- | --- | --- |
| `native` | Official DREAM entropy sampler, temperature 0.1, top-p 0.9, 256/512 steps | None |
| `uncached` | Same sliding frontier and greedy confidence rule as Relay | None |
| `relay` | Same sliding frontier and greedy confidence rule | Relay stripes |

The `uncached` control separates the sampling change from approximate caching.
Relay changes the native trajectory and is **not lossless**. A corrected
integration or a passing smoke test does not establish benchmark accuracy.

## External baselines

d²Cache and Elastic-Cache load their own pinned DREAM model, cache and sampler
modules. The evaluation wrapper does not substitute Relay components or patch
upstream source. Each external method runs in a fresh process.

| Method | Native profile |
| --- | --- |
| Fast-dLLM v1 | Official PrefixCache, block 32, confidence threshold 0.90; dual cache off |
| d²Cache | Official semi-AR/parallel example: block 32, threshold 0.90, generation sigma 0; cache rollout 0.1, current-k 32, sigma 10, inflate 4; DREAM top-p 0.9 |
| Elastic-Cache | Official GSM8K profile: window 32, gamma 0.90, threshold 0.90, track 1; HumanEval script: window 16, gamma 0.98 |

d²Cache's parallel example explicitly disables the generation certainty prior;
its default full-canvas certainty-prior configuration is a different profile.
Elastic has no supplied DREAM MATH/MBPP task scripts in the pinned checkout; those
tasks use the GSM8K window/cache defaults and must be reported as such.

External scripts use a BOS-prefixed completion prompt. For DREAM Instruct,
d²Cache's HumanEval input follows its supplied `humaneval_instruct` task wrapper
and generation prefix. Relay uses the Instruct chat template. All share the same
checkpoint and underlying prepared questions, but **native-protocol results are
not a matched-prompt ablation**. Record these prompt/sampler differences rather
than claiming identical evaluation protocols or quietly changing a baseline.
Final response truncation and the common task scorers remain unchanged.

## Small checks

Use an idle physical GPU, explicit cache/data paths and a fresh output directory.
The package can be installed in the existing environment with
`pip install -e . --no-deps`.

```bash
python -m dllm_eval.run --model dream --dream-backend relay --gpu 7 \
  --data-root /path/to/prepared-data --hf-home /path/to/hf-home \
  --hf-hub-cache /path/to/hub --limit 4 --offset 64 \
  --output runs/dream-relay-check

python -m dllm_eval.baseline_run --model dream --method d2cache \
  --baseline-source /path/to/d2Cache --gpu 7 \
  --data-root /path/to/prepared-data --hf-home /path/to/hf-home \
  --hf-hub-cache /path/to/hub --limit 4 --offset 64 \
  --output runs/dream-d2-check
```

Use `uncached` or `native` for the Relay controls; use `elastic_cache` with its
official source checkout for Elastic. Each source checkout must match the pinned
registry revision. Dataset paths can be overridden with `--dataset TASK=PATH`.
Resume requires an unchanged source/configuration/data manifest; incompatible
old outputs must retain their own directory and must not be mixed into a new run.

Each algorithm worker performs two excluded startup requests, then generates
each benchmark request once. Request timing excludes model loading, tokenization,
warm-up, output processing, scoring and file I/O, and includes model execution,
cache handling and decoding decisions. Audit replay is separate from timing.
Each invocation writes a block-progress log under `log/`, per-task tables,
durable records and a summary. GPU identity, occupancy checks, exclusive leases,
source checks and duplicate-run protection remain enabled.

The full-campaign controller supports six simultaneous GPU leases and one log per
method, in the order Relay, d²Cache, Elastic and v1 without FlashAttention.
Its existing completion barrier and exact-resume requirements remain active.
Do not resume the cancelled DREAM campaign with changed code. Validate the new
integration and quality before creating another full campaign.

Official sources: [DREAM](https://github.com/DreamLM/Dream),
[Fast-dLLM](https://github.com/NVlabs/Fast-dLLM),
[d²Cache](https://github.com/Kamichanw/d2Cache),
[Elastic-Cache](https://github.com/VILA-Lab/Elastic-Cache).
External source retains its original licenses and attribution.
