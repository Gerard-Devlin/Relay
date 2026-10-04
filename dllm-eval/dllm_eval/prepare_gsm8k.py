"""Export the exact lm-eval GSM8K 5-shot contexts used by Fast-dLLM v1.

The installed lm-eval task still names the historical ``gsm8k`` Hub repo.
The server cache was populated through its current ``openai/gsm8k`` name, so
we change only the dataset identifier while leaving the task template,
few-shot sampler, generation arguments, and seed untouched.
"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--dataset-path", default="openai/gsm8k")
    return parser.parse_args()


def main():
    args = parse_args()
    import lm_eval
    import yaml
    from lm_eval.api.task import ConfigurableTask

    task_yaml = Path(lm_eval.__file__).parent / "tasks" / "gsm8k" / "gsm8k.yaml"
    config = yaml.safe_load(task_yaml.read_text(encoding="utf-8"))
    if config["task"] != "gsm8k" or config["num_fewshot"] != 5:
        raise RuntimeError("Installed lm-eval GSM8K task no longer matches the v1 protocol")
    config["dataset_path"] = args.dataset_path
    task = ConfigurableTask(config=config)
    task.set_fewshot_seed(args.seed)
    task.build_all_requests(cache_requests=False)

    rows = []
    for index, instance in enumerate(task.instances):
        prompt, generation_kwargs = instance.arguments
        doc = instance.doc
        if not prompt.endswith("Answer:"):
            raise RuntimeError(f"Unexpected lm-eval prompt at test row {index}")
        rows.append({
            "id": f"test:{index}",
            "question": doc["question"],
            "answer": doc["answer"],
            "paper_prompt": prompt,
            "generation_kwargs": generation_kwargs,
        })

    if len(rows) != 1319:
        raise RuntimeError(f"Expected 1319 GSM8K test rows, found {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "rows": len(rows),
        "fewshot": config["num_fewshot"],
        "seed": args.seed,
        "output": str(args.output),
        "first_prompt_chars": len(rows[0]["paper_prompt"]),
    }, indent=2))


if __name__ == "__main__":
    main()
