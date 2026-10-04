"""Execute official HumanEval tests for already-generated completions."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

def clean_completion(prompt, generated, entry_point):
    from .vendor.sanitize import sanitize
    body = generated.split("```python\n", 1)[-1].split("```", 1)[0]
    return sanitize(prompt + "\n" + body, entry_point)


def check(program, test, entry_point, timeout):
    wrapper = ("import resource\n"
               "resource.setrlimit(resource.RLIMIT_CPU, (4, 4))\n"
               "resource.setrlimit(resource.RLIMIT_AS, (2147483648, 2147483648))\n" +
               program + "\n" + test + f"\ncheck({entry_point})\n")
    with tempfile.TemporaryDirectory(prefix="relay-humaneval-") as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(wrapper, encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"}
        try:
            result = subprocess.run([sys.executable, "-I", str(path)], cwd=directory, env=env,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    timeout=timeout, check=False)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=6.0)
    args = parser.parse_args()
    samples = {x["task_id"]: x for x in json.loads(args.dataset.read_text(encoding="utf-8"))}
    records = []
    for path in sorted(args.results.glob("rank_*.jsonl")):
        records += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records.sort(key=lambda x: x["index"])
    metadata = {"index", "id", "target"}
    methods = [key for key, value in records[0].items()
               if key not in metadata and isinstance(value, dict) and "text" in value]
    details, scores = [], {method: 0 for method in methods}
    for record in records:
        sample = samples[record["id"]]
        row = {"task_id": record["id"]}
        for method in methods:
            code = clean_completion(sample["prompt"], record[method]["text"], sample["entry_point"])
            passed = check(code, sample["test"], sample["entry_point"], args.timeout)
            row[method] = passed
            scores[method] += int(passed)
        details.append(row)
    output = {"examples": len(records), "pass@1": {k: v / len(records) for k, v in scores.items()},
              "details": details}
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"examples": output["examples"], "pass@1": output["pass@1"]}, indent=2))


if __name__ == "__main__":
    main()
