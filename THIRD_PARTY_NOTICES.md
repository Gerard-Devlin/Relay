# Third-party attribution

RelayCache includes a minimal execution substrate from [Flash-dLLM](https://github.com/VILA-Lab/Flash-dLLM),
commit `7437a550fd3d1a0752edcbf58bd015ad69083068`, under Apache-2.0.
The original model, generator and Triton kernel files live in `relay_cache/model/`.
These files retain their original bytes; they are upstream components, not new Relay algorithms.
Their license is in `third_party/flash_dllm/LICENSE`; individual file SHA-256 hashes are in `third_party/sources.json`.

The HumanEval sanitizer in `dllm-eval/dllm_eval/vendor/sanitize.py` is from NVlabs/Fast-dLLM,
copyright 2025 NVIDIA CORPORATION & AFFILIATES, Apache-2.0, modified upstream from HKUNLP/Dream.
Its original header and implementation are retained. The repository LICENSE supplies its license text.

Frozen Minerva-MATH and MBPP utilities come from the original lm-eval 0.4.8 installation.
They are retained for exact reference extraction and fixed few-shot prompts, under the MIT license
in `third_party/lm_eval_LICENSE.md`. Their optional module-level imports are not executed during scoring;
the original selected definitions are loaded through an AST whitelist.

Model weights and benchmark datasets are not redistributed. Respect their own licenses.
Math-Verify 0.1.0 is an external scoring dependency and is not vendored.
The original source mappings and hashes are in `dllm-eval/provenance/migration.json`.
