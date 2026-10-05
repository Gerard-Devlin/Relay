# DREAM evaluation

Use the existing environment and the downloaded
`Dream-org/Dream-v0-Instruct-7B` checkpoint. Generation is offline.
Every method receives the same prepared task prompt with DREAM's official BOS
prefix, without the LLaDA chat template. Reference answers and code tests enter
only the scorer, after generation has been saved.

## Methods

| Method | Official generation path |
| --- | --- |
| Relay | Native DREAM entropy sampler with Relay stripe refresh |
| Fast-dLLM v1 | Prefix cache + confidence-threshold parallel decoding, dual cache off |
| d2Cache | Official shift-aware generator, eager attention and dual adaptive cache, parallel threshold 0.90 |
| Elastic-Cache | Official DREAM elastic generator, window 32, threshold/gamma 0.90, track 1 |

The external cache and sampler files are not rewritten. Their stopping schedules
remain distinct: Elastic stops at the first decoded EOS; the other profiles
finish their configured canvas. All outputs are truncated at the first EOS and
task stop strings before the same four task scorers. The DREAM profile is separate
from the historical LLaDA profile and results must not be merged across models.

## Commands

Replace the paths and GPU with the intended data/cache/output locations and an
idle physical GPU. Install the package with `pip install -e . --no-deps` in the
existing environment if it is not already installed.

```bash
python -m dllm_eval.run --model dream --gpu 7 \
  --data-root /path/to/prepared-data --hf-home /path/to/hf-home \
  --hf-hub-cache /path/to/hub --limit 2 --output runs/dream-relay-smoke

python -m dllm_eval.baseline_run --model dream --method fast_dllm_v1 \
  --baseline-source /path/to/Fast-dLLM --gpu 7 \
  --data-root /path/to/prepared-data --hf-home /path/to/hf-home \
  --hf-hub-cache /path/to/hub --limit 2 --output runs/dream-v1-smoke

python -m dllm_eval.baseline_run --model dream --method d2cache \
  --baseline-source /path/to/d2Cache --gpu 7 \
  --data-root /path/to/prepared-data --hf-home /path/to/hf-home \
  --hf-hub-cache /path/to/hub --limit 2 --output runs/dream-d2-smoke

python -m dllm_eval.baseline_run --model dream --method elastic_cache \
  --baseline-source /path/to/Elastic-Cache --gpu 7 \
  --data-root /path/to/prepared-data --hf-home /path/to/hf-home \
  --hf-hub-cache /path/to/hub --limit 2 --output runs/dream-elastic-smoke
```

The external checkouts must match the pinned revisions in the adapter's source
registries. Pin them with `git checkout` after cloning the official repositories.
Run methods serially in fresh processes. Remove `--limit 2` for full evaluation;
all four tasks and both 256/512 lengths are selected by default. A nondefault
dataset location can be supplied with `--dataset TASK=PATH`. Output directories
cannot be reused without an exact `--resume` manifest match.

Each invocation writes one block-progress transcript under `log/`, per-task
tables, durable generation records, grades and a summary. Every prompt receives
an excluded warm request before its timed replay. Reported request time excludes
model loading, tokenization, warm-up, output processing, scoring and file I/O.
CPU integration checks are not evidence of checkpoint accuracy or GPU speed;
publish measurements only after the intended hardware run completes.

The initial checkpoint smoke covered all four tasks at 256 and 512 tokens for
each method: 32 requests, each with an excluded warm replay. Tokens, text and NFE
matched the replay. Relay's full-row operator control also matched native DREAM
logits in four real states. These checks establish integration only: its stripe
cache changed a native-correct GSM8K answer in the smoke, so DREAM quality and
speed remain unvalidated. Relay currently uses the native entropy schedule,
whereas the external profiles use parallel block/window schedules.

Official implementations: [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM),
[d2Cache](https://github.com/Kamichanw/d2Cache),
[Elastic-Cache](https://github.com/VILA-Lab/Elastic-Cache),
[DREAM](https://github.com/DreamLM/Dream). External source retains its original
licenses and attribution.
