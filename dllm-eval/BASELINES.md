# External LLaDA baselines

The adapters call the pinned official [d2Cache](https://github.com/Kamichanw/d2Cache)
and [Elastic-Cache](https://github.com/VILA-Lab/Elastic-Cache) model, cache and generation code.
Upstream source files are not patched. Runtime checkouts and manifests are ignored.

Both use the historical LLaDA-8B-Instruct revision, BF16, seed 51713, batch size one,
lengths 256/512, and existing GSM8K, MATH, HumanEval and MBPP prompts and scoring.
This compares the methods on our fixed model, hardware and datasets rather than
reproducing the papers' other models and hardware. Native LLaDA, v1 and Relay are not regenerated.

## Official configurations

d2Cache uses threshold 0.90, block length 32, `maskgit_plus`, temperature zero,
generation sigma zero, and at least one committed position per call. Cache parameters
are rollout_p=0.1, current_k=32, sigma=10, inflate_w=4. Generation sigma and cache
sigma serve different purposes. This follows the
[official parallel-decoding example](https://github.com/Kamichanw/d2Cache/blob/216b4557f4baf318a246763af805f414cbfb21a4/docs/kv_caching.md).
Official eager attention and per-step CPU recording are retained; no EOS early stop is added.
The pre-refactor official commit `216b455` is pinned because the subsequent cache
refactor fails its initial cache-state assertion in this environment.

Elastic-Cache uses the unmodified `generate_with_elastic_cache`: window length 16,
threshold 0.90, gamma 0.90, track_num=1, block_caching=True, tokens_per_iter=1,
and official EOS stopping/filling. Attention tracking and float64 confidence
calculation remain unchanged. These follow the four upstream LLaDA task scripts
at commit `1960d8f`. Those scripts default to LLaDA-1.5; our adapter loads the fixed
LLaDA-8B-Instruct checkpoint with strict parameter-loading checks. No new weights
or model environment are installed. Batch size is the official single-request size.

Actual model calls and maximum simultaneous commit counts are recorded.
Elastic-Cache's reported layer-refresh fraction is only a diagnostic, not a speedup.

## Evaluation and timing

An excluded complete warm request precedes each timed request. Tokens, processed
text and NFE must agree. Timing includes generation, cache management and each
method's scheduling/recording. Loading, prompt preparation, warm-up, final output
reconstruction/cleanup, scoring, I/O and logging are excluded. GPU identity,
occupancy, exclusive locking, frozen source/configuration/data, prompt whitelists
and duplicate/resume checks remain active.

The final-expression-v3 mathematical scorer and official code-execution scoring
are unchanged. References and tests are read only after generation is persisted.

Elastic-Cache first passes 16 smoke requests on GPU0 across every task/length cell.
Its six-GPU full evaluation waits until all six existing d2Cache workers finish.
Each method has 13,966 full requests. Each algorithm has its own log containing
all of its GPU workers, block progress bars and completed-cell tables. The
cancelled dLLM-Cache experiment is excluded.

```bash
PYTHONPATH=.:dllm-eval python -m dllm_eval.baseline_batch \
  --gpus 1 3 4 5 6 7 --output runs/cache-baselines \
  --upstream-root /path/to/upstream --data-root /path/to/benchmarks \
  --hf-home /path/to/hf-home --hf-hub-cache /path/to/hf-hub --log-dir log
```
