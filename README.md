# RelayCache

RelayCache refreshes selected history tokens through a rotating layer stripe, retaining boundary hidden states between denoising steps. Active prediction tokens and tokens whose identities changed still traverse the full network. It is a training-free, approximate cache method; it does not claim equivalence to uncached native inference. Flash-Verify is disabled.

## Repository

```text
relay_cache/
  llada/     # Original language-only Relay implementation
  dream/     # DREAM integration with shifted readout and grouped-head KV
  lladav/    # Historical, full-evaluated LLaDA-V integration
dllm-eval/
  configs/   # Fixed reproduction profiles and environment
  dllm_eval/ # Data preparation, evaluation, saved-output scoring
  tests/
  results/   # Aggregate historical results and migration checks
```

The LLaDA-V release preserves the previous full-evaluation profile, including native visual cache priming, image preprocessing, token restrictions and stopping. Later parameter searches and ablations are not included. The existing LLaDA and DREAM implementation files are unchanged by this release.

## Installation

Use Linux, an NVIDIA GPU with sufficient free memory, and the captured Python 3.11.15 environment. Reuse that environment when available:

```bash
git clone https://github.com/Gerard-Devlin/Relay.git
cd Relay
pip install -e . --no-deps
```

For a fresh Python environment:

```bash
pip install -r dllm-eval/configs/requirements-reproduction.txt --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e . --no-deps
```

Runs check the captured dependency versions. Model weights and datasets are not distributed here.

| Model | Checkpoint revision | Profile |
|---|---|---|
| GSAI-ML/LLaDA-8B-Instruct | `08b83a6feb34df1a6011b80c3c00c7563e963b07` | `dllm-eval/configs/reproduction.json` |
| Dream-org/Dream-v0-Instruct-7B | `05334cb9faaf763692dcf9d8737c642be2b2a6ae` | `dllm-eval/configs/dream.json` |
| GSAI-ML/LLaDA-V | `a10ba1790083e10e07cfcab6e7be93e1823f5792` | `dllm-eval/configs/lladav.json` |

LLaDA-V also requires `google/siglip2-so400m-patch14-384` at `e8e487298228002f3d8a82e0cd5c8ea9c567f57f`. Its checkpoint-tuned vision weights are loaded from LLaDA-V. Language parameters are shared with the Relay execution view; no extra LLaDA checkpoint download is required for visual generation.

All profiles use BF16 weights, block size 32, confidence threshold 0.90, track/mask counts 4, stripe width 8, history budget 128, seed 51713, and generation lengths 256/512. LLaDA-V retains FP32 RoPE frequencies and the original `llava_llada` conversation with the ` /think` suffix. Configuration changes are rejected by the reproduction entry points.

## Generate

`GPU` is an idle physical GPU index or UUID; `HF_CACHE` is the Hugging Face hub cache directory containing the pinned snapshots.

```bash
python -m relay_cache --model llada --gpu "$GPU" --hf-hub-cache "$HF_CACHE" \
  --prompt 'What is 12 multiplied by 13?' --length 256 --output runs/llada.json

python -m relay_cache --model dream --gpu "$GPU" --hf-hub-cache "$HF_CACHE" \
  --prompt 'What is 12 multiplied by 13?' --length 256 --output runs/dream.json

python -m relay_cache.lladav --gpu "$GPU" --hf-hub-cache "$HF_CACHE" \
  --image example.png --prompt 'Solve the problem in this image.' --length 256 --output runs/lladav.json
```

## Reproduce the evaluations

For LLaDA and DREAM, export the four text datasets with `dllm_eval.prepare_gsm8k`, `dllm_eval.prepare_humaneval`, and `dllm_eval.prepare_tasks` (see each command's `--help`). The fixed profiles expect GSM8K 5-shot, MATH 4-shot, HumanEval 0-shot and MBPP 3-shot, with 1,319/5,000/164/500 questions respectively.

```bash
python -m dllm_eval.run --model llada --data-root "$TEXT_DATA" \
  --hf-hub-cache "$HF_CACHE" --gpu "$GPU" --output runs/llada
python -m dllm_eval.run --model dream --data-root "$TEXT_DATA" \
  --hf-hub-cache "$HF_CACHE" --gpu "$GPU" --output runs/dream
```

For LLaDA-V, `--download` fetches the pinned model, vision tower and dataset splits. Omit it when the snapshots are already cached:

```bash
python -m dllm_eval.visual_data --download --hf-hub-cache "$HF_CACHE" --output data/visual
python -m dllm_eval.visual_run --data-root data/visual --hf-hub-cache "$HF_CACHE" \
  --gpu "$GPU" --output runs/lladav
export DEEPSEEK_API_KEY=...  # Supply privately; never commit credentials.
python -m dllm_eval.visual_score --run runs/lladav --data-root data/visual --output runs/lladav_scores
```

Visual evaluation uses MathVista `testmini` (1,000 questions) and the three visual MathVerse `testmini` variants (2,364 questions), both zero-shot. Generation receives only the prepared question and RGB image. References are accessed during the separate scoring pass. Raw outputs are retained, so changing a judge does not require generating again.

Evaluation workers perform two excluded startup warmups and one generation per formal question. Request time includes decoding and cache priming, and excludes model loading, prompt/image preparation, vision encoding, scoring and file I/O. Image preparation time, NFE and peak allocated/reserved memory are recorded separately. Each invocation writes a transcript with block progress bars under `log/`; `--resume` validates the manifest before reusing records. Independent shards can use `--rank` and `--world-size` with distinct output directories.

Text scoring retains the existing GSM8K extraction, `final-expression-v3` / Math-Verify 0.1.0 mathematics policy, and code execution wrappers. Visual scores use the historical **DeepSeek Flash proxy**: MathVista extraction plus original normalization/exact matching; MathVerse last-30-space-separated-word extraction plus equivalence judging. These are **not official GPT-4o scores**. API errors remain pending, and identical requests share cached judge responses. API model changes can affect newly judged scores.

## Historical visual results

These are the saved full-evaluation results, not a new full run during migration. Times are seconds per prepared-prompt request, measured after warmup on RTX 5090.

| Dataset | Length | Correct / total | Acc (%) | Time (s) |
|---|---:|---:|---:|---:|
| MathVista | 256 | 540 / 1000 | 54.000 | 6.216 |
| MathVista | 512 | 511 / 1000 | 51.100 | 9.823 |
| MathVerse | 256 | 541 / 2364 | 22.885 | 6.381 |
| MathVerse | 512 | 480 / 2364 | 20.305 | 9.968 |

Aggregate records are in [`dllm-eval/results/historical/`](dllm-eval/results/historical/). Migration checks are reported separately under `dllm-eval/results/migration/`; finite checks do not establish equivalence for every input.

## Acknowledgments

We build on [LLaDA](https://github.com/ML-GSAI/LLaDA), [DREAM](https://github.com/DreamLM/Dream), [LLaDA-V](https://github.com/ML-GSAI/LLaDA-V), [LLaVA](https://github.com/haotian-liu/LLaVA), [Flash-dLLM](https://github.com/VILA-Lab/Flash-dLLM), [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM), Transformers and the MathVista/MathVerse evaluation implementations. Adapted execution/model code retains its attribution. See [NOTICE](NOTICE) and [LICENSE](LICENSE) for details.
