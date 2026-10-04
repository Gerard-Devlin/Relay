"""Run the MBPP tests locally for generated programs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def clean_completion(text):
    if "```python" in text:
        text = text.split("```python", 1)[1]
    elif "```" in text:
        text = text.split("```", 1)[1]
    return text.split("```", 1)[0].split("[DONE]", 1)[0].strip()


def check(code, tests, timeout):
    wrapper = ("import resource\n"
               "resource.setrlimit(resource.RLIMIT_CPU, (4, 4))\n"
               "resource.setrlimit(resource.RLIMIT_AS, (2147483648, 2147483648))\n" +
               code + "\n" + "\n".join(tests) + "\n")
    with tempfile.TemporaryDirectory(prefix="relay-mbpp-") as directory:
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
    samples = {str(row["task_id"]): row for row in
               json.loads(args.dataset.read_text(encoding="utf-8"))}
    records = []
    for path in sorted(args.results.glob("rank_*.jsonl")):
        records.extend(json.loads(line) for line in
                       path.read_text(encoding="utf-8").splitlines())
    records.sort(key=lambda row: row["index"])
    metadata = {"index", "id", "target"}
    methods = [key for key, value in records[0].items()
               if key not in metadata and isinstance(value, dict) and "text" in value]
    scores = {method: 0 for method in methods}
    details = []
    for record in records:
        sample = samples[str(record["id"])]
        row = {"task_id": record["id"]}
        for method in methods:
            passed = check(clean_completion(record[method]["text"]),
                           sample["test_list"], args.timeout)
            row[method] = passed
            scores[method] += int(passed)
        details.append(row)
    output = {"examples": len(records),
              "pass@1": {key: value / len(records) for key, value in scores.items()},
              "details": details}
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"examples": len(records), "pass@1": output["pass@1"]}, indent=2))


if __name__ == "__main__":
    main()
