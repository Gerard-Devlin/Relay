"""Export the remaining Fast-dLLM v1 paper benchmark prompts.

The prompts reproduce lm-eval 0.4.8's fixed Minerva-MATH four-shot and MBPP
three-shot templates while avoiding imports of optional scoring packages during
data preparation.  The underlying benchmark rows are loaded from the ordinary
Hugging Face datasets cache.
"""

import argparse
import ast
import json
from pathlib import Path
import re


MATH_CONFIGS = (
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
)


def literal_function_result(path, function_name):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            for statement in node.body:
                if isinstance(statement, ast.Return):
                    return ast.literal_eval(statement.value)
    raise RuntimeError(f"Cannot find literal return from {function_name} in {path}")


def last_boxed(text):
    index = max(text.rfind(r"\boxed"), text.rfind(r"\fbox"))
    if index < 0:
        raise ValueError("MATH solution has no boxed answer")
    if text.startswith(r"\boxed ", index):
        return text[index + len(r"\boxed "):].split("$", 1)[0]
    left = text.find("{", index)
    depth = 0
    for pos in range(left, len(text)):
        depth += text[pos] == "{"
        depth -= text[pos] == "}"
        if depth == 0:
            return text[left + 1:pos]
    raise ValueError("Unbalanced boxed MATH answer")


def normalize_math(value):
    value = value.split("=")[-1]
    substitutions = (("an ", ""), ("a ", ""), (".$", "$"), (r"\$", ""),
                     (r"\ ", ""), (" ", ""), ("mbox", "text"),
                     (",\\text{and}", ","), ("\\text{and}", ","))
    for old, new in substitutions:
        value = value.replace(old, new)
    value = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", r"$\3$", value)
    value = re.sub(r"(\\text\{|\\textbf\{|\\overline\{)(.*?)(\})", r"\2", value)
    value = re.sub(r"(\\boxed\{)(.*)(\})", r"\2", value)
    value = re.sub(r"(frac)([^{])(.)", r"frac{\2}{\3}", value)
    value = re.sub(r"(sqrt)([^{])", r"sqrt{\2}", value).replace("$", "")
    if value.replace(",", "").isdigit():
        value = value.replace(",", "")
    return value


def math_prompt(doc):
    return "Problem:\n" + doc["problem"] + "\n\nSolution:"


def mbpp_prompt(doc):
    tests = doc["test_list"]
    return ("You are an expert Python programmer, and here is your task: " + doc["text"] +
            " Your code should pass these tests:\n\n" + tests[0] + "\n" + tests[1] +
            "\n" + tests[2] + "\n[BEGIN]\n")


def export_math(task_root, output):
    from datasets import load_dataset
    fewshots = literal_function_result(task_root / "minerva_math_utils.py",
                                       "list_fewshot_samples")
    # lm-eval joins doc_to_text and doc_to_target with target_delimiter=" ".
    prefix = "\n\n".join(math_prompt(row) + " " + row["solution"] for row in fewshots) + "\n\n"
    rows = []
    for config in MATH_CONFIGS:
        dataset = load_dataset("EleutherAI/hendrycks_math", config,
                               split="test", trust_remote_code=True)
        for index, doc in enumerate(dataset):
            rows.append({
                "id": f"{config}:test:{index}", "category": config,
                "problem": doc["problem"], "solution": doc["solution"],
                "answer": normalize_math(last_boxed(doc["solution"])),
                "paper_prompt": prefix + math_prompt(doc),
                "generation_kwargs": {"until": ["Problem:"], "do_sample": False,
                                      "temperature": 0},
            })
    if len(rows) != 5000:
        raise RuntimeError(f"Expected 5000 MATH test rows, found {len(rows)}")
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(rows)


def export_mbpp(task_root, output):
    from datasets import load_dataset
    fewshots = literal_function_result(task_root / "mbpp_utils.py",
                                       "list_fewshot_samples")
    prefix = "\n\n".join(mbpp_prompt(row) + row["code"] + "\n[DONE]"
                           for row in fewshots) + "\n\n"
    rows = []
    for doc in load_dataset("google-research-datasets/mbpp", "full", split="test"):
        rows.append({
            "task_id": doc["task_id"], "text": doc["text"], "code": doc["code"],
            "test_list": doc["test_list"], "test_setup_code": doc.get("test_setup_code", ""),
            "challenge_test_list": doc.get("challenge_test_list", []),
            "paper_prompt": prefix + mbpp_prompt(doc),
            "generation_kwargs": {"until": ["[DONE]"], "do_sample": False},
        })
    if len(rows) != 500:
        raise RuntimeError(f"Expected 500 MBPP test rows, found {len(rows)}")
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("math", "mbpp"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    task_root = Path(__file__).resolve().parent / "vendor"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = (export_math(task_root, args.output) if args.task == "math" else
             export_mbpp(task_root, args.output))
    print(json.dumps({"task": args.task, "rows": count, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
