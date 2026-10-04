"""Rescore saved MATH generations with the installed Minerva primary metric.

This never generates text or edits generation shards/summary.json.  We load
only the exact_match dependencies from lm-eval's installed utils.py so its
optional, separately reported math_verify metric need not be installed.
"""

import argparse
import ast
from concurrent.futures import ProcessPoolExecutor
import hashlib
from importlib import metadata, util
import json
import logging
import os
from pathlib import Path
import re
import signal
import time
from typing import Optional


FUNCTIONS = {
    "last_boxed_only_string", "remove_boxed", "is_equiv",
    "get_unnormalized_answer", "normalize_final_answer",
}
CONSTANTS = {"SUBSTITUTIONS", "REMOVED_EXPRESSIONS"}
_METRIC = None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def installed_utils():
    path = Path(__file__).resolve().parent / "vendor" / "minerva_math_utils.py"
    if sha256(path) != "7d8b058afbb950487cf7c72daedabbeef29f298fda94eadfd386d9943be067b6":
        raise RuntimeError("Frozen original Minerva utility changed")
    return path


def extract_metric_ast(source, filename="minerva_math/utils.py"):
    """Select definitions, never execute module imports or top-level effects."""
    tree = ast.parse(source, filename=filename)
    definitions = []
    found = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
            if node.decorator_list:
                raise ValueError(f"Unexpected decorated official function: {node.name}")
            key = node.name
        elif isinstance(node, ast.ClassDef) and node.name == "timeout":
            if node.decorator_list or node.bases or node.keywords:
                raise ValueError("Unexpected official timeout declaration")
            key = node.name
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id in CONSTANTS):
            ast.literal_eval(node.value)  # reject executable initializers
            key = node.targets[0].id
        else:
            continue
        if key in found:
            raise ValueError(f"Duplicate official definition: {key}")
        found.add(key)
        definitions.append(node)
    missing = (FUNCTIONS | CONSTANTS | {"timeout"}) - found
    if missing:
        raise ValueError(f"Unsupported lm_eval primary metric; missing {sorted(missing)}")
    return ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))


def load_metric(path, include_math_verify=False):
    if not hasattr(signal, "SIGALRM"):
        raise RuntimeError("Official Minerva symbolic timeout requires Linux/POSIX")
    try:
        import sympy
        from sympy.parsing.latex import parse_latex
        antlr_version = metadata.version("antlr4-python3-runtime")
    except (ImportError, metadata.PackageNotFoundError) as error:
        raise RuntimeError("MATH exact_match requires sympy and antlr4-python3-runtime==4.11.*") from error
    if not antlr_version.startswith("4.11"):
        raise RuntimeError(f"Official scorer requires antlr4 4.11.*, found {antlr_version}")
    namespace = {
        "re": re, "signal": signal, "Optional": Optional,
        "sympy": sympy, "parse_latex": parse_latex,
        "eval_logger": logging.getLogger("dllm_eval.score_math"),
    }
    source = Path(path).read_text(encoding="utf-8")
    tree = extract_metric_ast(source, str(path))
    exec(compile(tree, str(path), "exec"), namespace)
    if include_math_verify:
        from math_verify import parse, verify
        namespace["math_verify_functions"] = (parse, verify)
    return namespace


def score_record(record, sample, metric):
    # The preparer's abbreviated normalization is not an official gold label.
    # Rebuild the target from the original solution exactly as process_docs.
    boxed = metric["last_boxed_only_string"](sample["solution"])
    gold = metric["normalize_final_answer"](metric["remove_boxed"](boxed))
    methods = {}
    for method, result in record.items():
        if not isinstance(result, dict) or "text" not in result:
            continue
        extracted = metric["get_unnormalized_answer"](result["text"])
        prediction = metric["normalize_final_answer"](extracted)
        methods[method] = {
            "prediction": prediction,
            "exact_match": bool(metric["is_equiv"](prediction, gold)),
            "extraction_valid": extracted != "[invalidanswer]",
            "previous_correct": result.get("correct"),
        }
        if "math_verify_functions" in metric:
            parse, verify = metric["math_verify_functions"]
            # Identical secondary metric to installed process_results. Keep it
            # separate: boxed-answer recognition is not the primary extractor.
            methods[method]["math_verify"] = bool(verify(parse(gold), parse(result["text"])))
    if not methods:
        raise ValueError(f"No generated methods in record {record.get('id')}")
    return {"index": record["index"], "id": record["id"], "gold": gold, "methods": methods}


def read_records(results, samples):
    files = sorted(Path(results).glob("rank_*.jsonl"))
    if not files:
        raise ValueError(f"No generation shards at {results}")
    rows, ids, indices = [], set(), set()
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            ident = str(record["id"])
            if ident in ids or record["index"] in indices:
                raise ValueError(f"Duplicate record id/index: {ident}/{record['index']}")
            if ident not in samples:
                raise ValueError(f"Unknown MATH id: {ident}")
            ids.add(ident)
            indices.add(record["index"])
            rows.append(record)
    if not rows:
        raise ValueError("Generation shards are empty")
    rows.sort(key=lambda row: row["index"])
    return rows, files


def summarize(details):
    methods = set(details[0]["methods"])
    if any(set(row["methods"]) != methods for row in details):
        raise ValueError("Methods differ across records; do not report unequal sample sets")
    result = {
        method: {
            "examples": len(details),
            "exact_match": sum(row["methods"][method]["exact_match"] for row in details) / len(details),
            "valid_extraction_rate": sum(row["methods"][method]["extraction_valid"] for row in details) / len(details),
            "changed_from_provisional": sum(
                row["methods"][method]["previous_correct"] is not None
                and row["methods"][method]["exact_match"] != row["methods"][method]["previous_correct"]
                for row in details),
        } for method in sorted(methods)
    }
    for method in methods:
        if all("math_verify" in row["methods"][method] for row in details):
            result[method]["math_verify"] = sum(
                row["methods"][method]["math_verify"] for row in details) / len(details)
    return result


def _init_worker(path, include_math_verify=False):
    global _METRIC
    _METRIC = load_metric(path, include_math_verify)


def _worker(pair):
    return score_record(pair[0], pair[1], _METRIC)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--official-utils", type=Path, default=None)
    parser.add_argument("--math-verify", action="store_true",
                        help="Also report the installed official math_verify secondary metric")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing scoring artifact: {args.output}")
    if args.output.name == "summary.json" or re.fullmatch(r"rank_.*\.jsonl", args.output.name):
        raise ValueError("Use a separate scoring artifact; raw generation output must remain intact")
    utils_path = args.official_utils or installed_utils()
    _init_worker(str(utils_path), args.math_verify)  # validate dependencies before creating workers
    samples_list = json.loads(args.dataset.read_text(encoding="utf-8"))
    samples = {str(row["id"]): row for row in samples_list}
    if len(samples) != len(samples_list):
        raise ValueError("Duplicate MATH dataset ids")
    records, files = read_records(args.results, samples)
    before = {str(path): sha256(path) for path in files}
    pairs = [(row, samples[str(row["id"])]) for row in records]
    started = time.monotonic()
    details = []
    executor = ProcessPoolExecutor(args.workers, initializer=_init_worker,
                                   initargs=(str(utils_path), args.math_verify)) if args.workers > 1 else None
    try:
        iterator = executor.map(_worker, pairs, chunksize=8) if executor else map(_worker, pairs)
        for detail in iterator:
            details.append(detail)
            if len(details) % 100 == 0 or len(details) == len(records):
                print(f"Scored {len(details)}/{len(records)} MATH examples in {time.monotonic()-started:.1f}s", flush=True)
    finally:
        if executor:
            executor.shutdown()
    if {str(path): sha256(path) for path in files} != before:
        raise RuntimeError("Generation shards changed while scoring; rerun after generation finishes")
    try:
        lm_eval_version = metadata.version("lm_eval")
    except metadata.PackageNotFoundError:
        lm_eval_version = "not installed; explicit source provided"
    output = {
        "metric": "lm_eval.minerva_math.exact_match",
        "scope": ("Official primary symbolic metric with strict official extraction; " +
                  ("official math_verify reported separately" if args.math_verify else "math_verify not computed")),
        "examples": len(details), "results": summarize(details),
        "scoring_seconds": time.monotonic() - started,
        "provenance": {
            "lm_eval_version": lm_eval_version,
            "sympy_version": metadata.version("sympy"),
            "antlr_version": metadata.version("antlr4-python3-runtime"),
            "math_verify_version": metadata.version("math_verify") if args.math_verify else None,
            "official_utils": str(utils_path), "official_utils_sha256": sha256(utils_path),
            "scorer_sha256": sha256(__file__), "dataset": str(args.dataset),
            "dataset_sha256": sha256(args.dataset), "generation_shards": before,
        },
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps(output["results"], indent=2), flush=True)


if __name__ == "__main__":
    main()
