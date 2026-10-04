"""Apply the installed lm-eval GSM8K filters and exact-match metric verbatim."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from types import SimpleNamespace


def score(samples, records):
    import lm_eval
    import yaml
    from lm_eval.filters import build_filter_ensemble
    from lm_eval.api.metrics import exact_match_fn
    path = Path(lm_eval.__file__).parent / "tasks" / "gsm8k" / "gsm8k.yaml"
    config = yaml.safe_load(path.read_text())
    options = {key: value for key, value in config["metric_list"][0].items()
               if key not in ("metric", "aggregation", "higher_is_better")}
    methods = [key for key, value in records[0].items()
               if isinstance(value, dict) and "text" in value and "seconds" in value]
    details = [dict(id=row["id"], methods={}) for row in records]
    results = {}
    for method in methods:
        instances = [SimpleNamespace(resps=[row[method]["text"]],
                                    doc=samples[str(row["id"])], filtered_resps={}) for row in records]
        for pipeline in config["filter_list"]:
            components = [(part["function"], {key: value for key, value in part.items() if key != "function"})
                          for part in pipeline["filter"]]
            build_filter_ensemble(pipeline["name"], components).apply(instances)
        scores = {pipeline["name"]: [] for pipeline in config["filter_list"]}
        for instance, detail in zip(instances, details):
            result = {}
            for name, response in instance.filtered_resps.items():
                correct = exact_match_fn(references=[instance.doc["answer"]], predictions=[response], **options)["exact_match"]
                result[name] = bool(correct)
                scores[name].append(float(correct))
            detail["methods"][method] = result
        results[method] = {name: sum(values)/len(values) for name, values in scores.items()}
    return dict(examples=len(records), results=results, details=details,
                provenance=dict(lm_eval_version=version("lm_eval"),
                                task_yaml_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))


def main():
    parser = argparse.ArgumentParser()
    for name in ("dataset", "results", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    from .queue import load_completed_records
    sample_list = json.loads(args.dataset.read_text())
    records = load_completed_records(sorted(args.results.glob("rank_*.jsonl")), expected_count=len(sample_list))
    samples = {str(row["id"]): row for row in sample_list}
    for index, row in records.items():
        if row["id"] != sample_list[index]["id"]:
            raise ValueError("GSM8K ID/index mismatch")
    result = score(samples, [records[index] for index in sorted(records)])
    result["provenance"]["dataset_sha256"] = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result["results"], indent=2), flush=True)


if __name__ == "__main__":
    main()
