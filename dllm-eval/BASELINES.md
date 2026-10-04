# External LLaDA baselines

The evaluation adapters run the official [dLLM-cache](https://github.com/maomaocun/dLLM-cache)
and [d2Cache](https://github.com/Kamichanw/d2Cache) generation, model, and cache functions
without changing their source files. Official checkouts live in ignored runtime directories.

Both methods use the historical LLaDA-8B-Instruct revision, BF16, batch size one,
seed 51713, lengths 256/512, and the same prepared prompts and scoring as Relay.
This is a controlled comparison with our benchmark table, not a reproduction of
each paper's different hardware, few-shot prompts, or batch size.

## Fixed official profiles

dLLM-cache uses the **first cached command** in each official LLaDA-Instruct task script:

| Task | Block length | Prompt refresh interval | Response refresh interval |
|---|---:|---:|---:|
| GSM8K | 8 | 50 | 7 |
| HumanEval | 32 | 50 | 8 |
| MBPP | 32 | 100 | 5 |
| MATH | 256 | 50 | 1 |

Transfer ratio is 0.25; steps equal generation length; temperature and CFG scale
are zero. HumanEval also has a second upstream profile (25/5); it is not selected
after looking at scores. The same chosen profile is used for both lengths.

d2Cache is pinned to its official pre-refactor commit `216b455` because the later
`79fb5f6` refactor fails its first prefill/cache-state assertion. No upstream code
is patched. d2Cache uses `configs/gen_args.py`'s `d2cache` defaults: rollout_p=0.1,
current_k=32, sigma=10, inflate_w=0, full generation-length block,
one transfer per call, `maskgit_plus`, temperature zero, and no added parallel
threshold. Its official **eager attention** is required for attention rollout;
we do not substitute a different attention algorithm. No early EOS stop is added.

The fixed sources are recorded in each ignored run manifest. Before every timed
request, an excluded complete warm-up request is made; token sequences, output
text, and NFE must agree. Time includes official generation and cache bookkeeping.
Model loading, prompt preparation, warm-up, final output reconstruction/cleanup,
scoring, and file I/O are excluded. The official d2Cache generator's per-step CPU
record transfers remain part of generation time.

Its optional Hydra/OmegaConf CLI dependencies can reside in a private `deps/`
directory beside `upstream/`. They are loaded only during upstream import;
the existing mathematical scorer's ANTLR runtime is restored afterward. No base
environment package is upgraded or downgraded.

## Run

Use the existing environment and data cache. Each authorized card has an exclusive
lease, UUID check, and fresh occupancy check. The campaign freezes its own code,
rejects modified upstream files, first checks 32 smoke requests, then evaluates
27,932 requests across two methods and eight cells per method. It produces one
combined log with block progress bars and result tables. LLaDA, v1, and Relay
are not regenerated.

```bash
PYTHONPATH=.:dllm-eval python -m dllm_eval.baseline_batch \
  --gpus 1 3 4 5 6 7 --output runs/cache-baselines \
  --upstream-root /path/to/upstream --data-root /path/to/benchmarks \
  --hf-home /path/to/hf-home --hf-hub-cache /path/to/hf-hub --log-dir log
```
