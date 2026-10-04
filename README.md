# RelayCache

**Stripe-refresh caching for diffusion language models.**

In the measured LLaDA profile, RelayCache keeps active MASK positions and newly changed token identities on the full 32-layer path.
Stable history is refreshed in rotating 8-layer stripes, using previously paid boundary states between stripes.
Mandatory dirty history is refreshed before optional saliency-selected history, with a budget of 128 positions.
All positions still provide cached attention keys and values. This reduces repeated history computation;
it is an **approximate cache**, not lossless execution of native LLaDA.

The cache policy lives in `relay_cache/cache.py`; its execution adapters live in
`relay_cache/execution.py`. The model execution and scoring components are pinned and
credited in [Acknowledgement](#acknowledgement). Speculative verification is
disabled in the published configuration.

## Layout

```text
relay_cache/
  cache.py, execution.py  shared historical Relay cache policy and execution utilities
  llada/                 LLaDA loading, generation and model execution
  dream/                 DREAM loading and model-specific stripe adapter
  __main__.py            single-request CLI
dllm-eval/     preparation, evaluation, scoring, recovery, tests and published summaries
```

## Installation

Use Linux, a CUDA GPU with at least 24 GB available, and the original environment recorded in
[`environment.json`](dllm-eval/configs/environment.json). The measured environment uses Python 3.11.15,
PyTorch 2.7.1+cu128, Triton 3.3.1, Transformers 4.57.3 and NumPy 2.4.6.
Install into that existing environment without upgrading dependencies:

```bash
python -m pip install -e . --no-deps --no-build-isolation
```

The editable source checkout is required: execution files and evaluation configuration
are resolved from the repository. Existing cached weights must contain
`GSAI-ML/LLaDA-8B-Instruct@08b83a6feb34df1a6011b80c3c00c7563e963b07`.
The loader works offline and refuses an incomplete checkpoint. No training or weight conversion is performed.

## Generate one answer

```bash
python -m relay_cache --gpu 2 --length 256 --task gsm8k \
  --hf-home /path/to/hf-home --hf-hub-cache /path/to/hub \
  --prompt "What is 17 multiplied by 23?" --output runs/example.json
```

Use `--prompt-file` for a UTF-8 file, and `--preformatted` for an already formatted paper prompt.
The command performs one excluded warm request before its timed replay. It refuses occupied GPUs and existing output files.
GPU UUID binding and POSIX flock leases are checked before loading weights. No other processes are killed or altered.

## DREAM integration

The model adapters are separated into `relay_cache/llada/` and `relay_cache/dream/`,
following the model-per-directory organization used by Fast-dLLM. The historical LLaDA
cache math, kernels, configuration, scorer and result files are unchanged.

DREAM uses `Dream-org/Dream-v0-Instruct-7B` at revision
`05334cb9faaf763692dcf9d8737c642be2b2a6ae`. Download once into the existing HF cache:

```bash
python -m relay_cache.dream.loading --cache-dir /path/to/hub
# If required by your network, append --endpoint https://hf-mirror.com
```

Generation is offline. No second environment or model conversion is needed.
DREAM retains its official full-canvas diffusion sampler (entropy ranking,
temperature0, alg_temp0, 256/512 steps respectively). This is a model-specific sampling
profile, separate from the published LLaDA block32/threshold0.90 profile. Relay refreshes
stable history in 8-layer stripes; the last stripe contains four of DREAM's 28 layers.
All MASK positions, the preceding positions required by DREAM's shifted logits,
and changed token identities traverse all layers. Every position still provides KV;
GQA retains four KV heads rather than treating them as 28 independent heads.
The official sampler, absolute RoPE positions and model weights are not rewritten.
`--dream-backend native` provides an uncached control using the same checkpoint and sampler.

```bash
python -m relay_cache --model dream --gpu 2 --length 256 --task gsm8k \
  --hf-home /path/to/hf-home --hf-hub-cache /path/to/hub \
  --prompt "What is 17 multiplied by 23?" --output runs/dream-example.json

python -m dllm_eval.run --model dream --gpu 2 --data-root /path/to/prepared-data \
  --hf-home /path/to/hf-home --hf-hub-cache /path/to/hub \
  --limit 4 --output runs/dream-check
```

The evaluation entrypoint uses the same prompt whitelist, four task scorers,
GPU leases, warm replay checks, source guards and resume rules. DREAM responses are
truncated at the tokenizer EOS and then the prescribed task stop strings before scoring;
all raw generated tokens remain in the run record. DREAM manifests and
outputs identify their own model, revision, sampler and backend and cannot resume a
LLaDA run. Batch-one, unpadded full-attention requests are supported initially;
AR append-cache and padded/multi-request inputs are rejected.

This integration has CPU reference and lifecycle checks. **DREAM GPU correctness,
accuracy and timing are pending**; the historical table below contains only LLaDA
Relay results and must not be read as DREAM measurements. The stripe cache is approximate.

## Reproduce the four tasks

The fixed configuration is [`reproduction.json`](dllm-eval/configs/reproduction.json): BF16,
block32, threshold0.90, gamma0.80, track4/mask4, stripe8, history128, shuffle seed51713,
length256/512 with speculative verification disabled. Runtime rejects settings that differ from this reproduction profile.

Use the original prepared datasets, including the prescribed few-shot prompts:

| Task | Prepared filename | Samples |
|---|---|---:|
| GSM8K | `gsm8k_paper_5shot.json` | 1,319 |
| MATH | `math_paper_4shot_v2.json` | 5,000 |
| HumanEval | `humaneval_test.json` | 164 |
| MBPP | `mbpp_paper_3shot.json` | 500 |

The runner checks the expected counts, unique selected IDs and legal prompt fields, and records the
input identities privately with each run for safe resume. Preparation utilities are available as
`python -m dllm_eval.prepare_gsm8k`, `python -m dllm_eval.prepare_tasks --task math` (or `--task mbpp`), and
`python -m dllm_eval.prepare_humaneval`; use the original dataset cache and prescribed formatting.
MBPP's public task tests and fixed few-shot examples are part of its prescribed legitimate prompt;
scoring references, solutions and hidden tests are never inserted by the generation pipeline.

```bash
python -m dllm_eval.run --gpu 2 --data-root /path/to/prepared-data \
  --hf-home /path/to/hf-home --hf-hub-cache /path/to/hub \
  --output runs/relay
```

Override individual locations with `--dataset gsm8k=/path/to/gsm8k_paper_5shot.json`.
For a small check add `--limit 4`; default coverage is all eight cells.
For disjoint multi-GPU workers, start one process per GPU with `--rank 0 --world-size 6`,
then rank1 through5, each with a separate output directory. Assignment is deterministic and shared IDs never overlap.
No LLaDA or Fast-dLLM baseline is regenerated by this entrypoint.

### Logs and progress

Each invocation writes one UTF-8 transcript under `log/`, including startup messages, warnings,
progress snapshots, per-cell result tables, the final table, and failure tracebacks. The terminal
shows a live block-style (`█`) `tqdm` progress bar for each dataset/length cell. Redirected output uses readable
line-by-line progress snapshots. Use `--log-dir /path/to/log` to change the directory; resumed
invocations get a separate transcript so previous logs are preserved. Sharded workers each have
their own transcript and report their local shard's metrics.

The bar's elapsed time and ETA include the full evaluation loop. The `Request (s)` table column
always uses the unchanged warmed generation interval stored in each record; it is not the
progress bar's wall time. Logging does not add model calls or change grading, sample selection,
resume validation, or request timing boundaries.

For a persistent server session:

```bash
tmux new -s relay-full
# Inside tmux, from the repository and with the original environment active:
python -u -m dllm_eval.run --gpu 2 --data-root /path/to/prepared-data \
  --hf-home /path/to/hf-home --hf-hub-cache /path/to/hub \
  --output runs/relay-full
```

Detach with `Ctrl-b`, then `d`; reattach with `tmux attach -t relay-full`.
The runner prints its exact log filename at startup. Results and durable records stay under
`--output`; `log/` contains only human-readable transcripts and is ignored by Git.

Generation is persisted before grading. An interrupted run can be continued with the same command plus `--resume`.
Saved generations are not regenerated; missing grades are recovered. A changed source, data, configuration,
environment or run manifest refuses resume. Durable queues and pure coverage/recovery utilities are in `dllm_eval`.

## Scoring

GSM8K and MATH use the frozen **final-expression-v3** policy. The final expression is selected independently
of the reference; reasoning is never searched for a matching answer. Ambiguous, malformed, missing or
unparseable answers remain failures. MATH uses Math-Verify **0.1.0** with SymPy **1.14.0**.
The narrow historical Union(Boolean) compatibility repair requires an exact real-set certificate and agreement
with the original comparator; unsupported cases still raise errors. HumanEval and MBPP use the same sanitation,
isolated official test execution, resource limits and 6-second timeout as the historical run.

The legacy Minerva extraction score is a different metric; it is not substituted for the uniformly audited scores below.
The existing LLaDA/v1 audit is published without rescoring or regenerating those methods.

## Historical full results

These are the completed **13,966 requests / eight cells**, measured on the original RTX5090 server.
They are retained historical results, not measurements of the repository migration.

| Dataset | Length | Correct / total | Accuracy (%) | Prepared request (s) | NFE |
|---|---:|---:|---:|---:|---:|
| GSM8K (5-shot) | 256 | 1023/1319 | 77.56 | 2.276 | 68.25 |
| GSM8K (5-shot) | 512 | 990/1319 | 75.06 | 2.746 | 79.73 |
| MATH (4-shot) | 256 | 1901/5000 | 38.02 | 3.050 | 94.36 |
| MATH (4-shot) | 512 | 2099/5000 | 41.98 | 4.389 | 133.54 |
| HumanEval (0-shot) | 256 | 68/164 | 41.46 | 1.752 | 56.77 |
| HumanEval (0-shot) | 512 | 74/164 | 45.12 | 4.600 | 146.60 |
| MBPP (3-shot) | 256 | 193/500 | 38.60 | 0.957 | 27.62 |
| MBPP (3-shot) | 512 | 197/500 | 39.40 | 1.840 | 53.93 |

Time is the synchronized, **warmed prepared-prompt request** interval, including the generator,
cache/stripe work, readout, sampling and generation-loop CPU overhead. It excludes model loading,
tokenization, host-to-device input preparation, the immediately preceding full warm request,
subsequent benchmark text postprocessing, JSON I/O and scoring. Raw decoder readback inside generation
is included. These values are not cold-start application latency or kernel-only timings.

The run had an interrupted scoring phase followed by disjoint recovery; prior output files were preserved.
The full metrics, 128-prompt screening summary and baseline correctness comparison are in
[`dllm-eval/results`](dllm-eval/results). Screening and retrospective paired intervals
do not establish statistical losslessness or superiority over every method.

## Validation

The CPU suite includes the original **35 Relay tests** plus checks for source
mismatch, GPU binding and occupation, prompt whitelisting, exception cleanup, locks and
duplicate/resumed runs.

```bash
python -m unittest discover -s dllm-eval/tests -v
```

The standalone package was compared with the previously evaluated Relay implementation
on **32 same-GPU paired cases**: four fixed prompts per task, at both 256 and 512 tokens.
All cases matched in raw tokens, every commit action, NFE, iteration count, processed text,
scoring and cache-work counters. Clean timing requests and instrumented audits were run
separately. Installation and imports were also checked outside the original repository.

Those paired checks used already-seen development prompts; they are not an independent accuracy study
or proof of equivalence to native LLaDA. Historical full metrics were preserved without regeneration or rescoring.

## Acknowledgement

RelayCache builds on execution components from [Flash-dLLM](https://github.com/VILA-Lab/Flash-dLLM),
[LLaDA](https://github.com/ML-GSAI/LLaDA),
[DREAM](https://github.com/DreamLM/Dream), [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM),
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness),
[Hugging Face Transformers](https://github.com/huggingface/transformers), and
[Math-Verify](https://github.com/huggingface/Math-Verify). We thank their authors for the open-source implementations.

The pinned model, generator and Triton execution files in `relay_cache/llada/model/` retain the original
Flash-dLLM implementation (Copyright 2026 VILA Lab, Apache-2.0). The HumanEval sanitizer retains
the Fast-dLLM implementation and its original
Copyright 2025 NVIDIA CORPORATION & AFFILIATES header, under Apache-2.0, modified upstream from
HKUNLP/Dream. These upstream components are distinct from Relay's stripe-refresh cache policy.
The repository [LICENSE](LICENSE) contains the complete Apache-2.0 terms.

Minerva-MATH and MBPP evaluation utilities come from lm-eval 0.4.8 under the MIT license reproduced below.
Only the original whitelisted definitions are loaded during evaluation. DREAM model and sampler code are loaded from the pinned official checkpoint cache
(Copyright the Dream team, HKUNLP Group and Hugging Face, Apache-2.0); they are not redistributed.
Math-Verify is an external dependency; model weights and benchmark datasets are not redistributed and retain their own licenses.

<details>
<summary>MIT license for bundled lm-evaluation-harness utilities</summary>

```text
MIT License

Copyright (c) 2020 EleutherAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

</details>
