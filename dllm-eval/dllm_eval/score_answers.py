"""Audit saved mathematical answers without changing the original benchmark.

Extraction never sees the reference answer. Only the selected final expression
is passed to Math-Verify; the chain of thought is never searched for a match.
Ambiguous/missing/unparseable answers are counted as failures and also reported
separately. This is a posthoc diagnostic policy, not a replacement paper score.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import random
import re
import time

from .score_math import installed_utils, load_metric, read_records, sha256


POLICY = {
    "version": "final-expression-v3",
    "math_extractor": "v2 unchanged",
    "gsm8k_extractor": "single numeric expression in explicit answer or final conclusion paragraph; multiple quantities unresolved",
    "selection": "latest explicit answer declaration; otherwise last boxed answer; otherwise standalone trailing math",
    "multiple_answers": "explicit enumeration compared as a complete set; conflicting scalar alternatives unresolved; never match any element",
    "delimiter_repair": "repair mismatched outer math delimiters only; preserve the entire payload and record the raw span",
    "comparison": "Math-Verify on isolated expressions, no string fallback; exact rational GSM8K comparison",
    "numeric_units": "ignore common textual units/currency; percent is /100 for MATH, numeric percent value for GSM8K",
    "weak_fallback": "standalone trailing math reported separately; no last-number search in prose",
    "unresolved": "count as incorrect and report reason; never search earlier predictions using gold",
    "scope": "retrospective diagnostic, uniformly applied to all methods; original scores untouched",
}
_MARKER = re.compile(r"####\s*|\b(?:final\s+answer\s*(?:is\s*)?[:=]?|(?:the\s+)?answer\s*(?:is|:|=))\s*", re.I)
_BOX = re.compile(r"\\(?:boxed|fbox)\s*\{")
_MATH = re.compile(r"\\\[(.*?)\\\]|\\\((.*?)\\\)|\$\$(.*?)\$\$|(?<!\$)\$([^$\n]+)\$(?!\$)", re.S)
_NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+|(?=\.\d))(?:\.\d+)?(?:[eE][+-]?\d+)?"
_NUMERIC = re.compile(rf"^({_NUMBER})(?:\s*/\s*({_NUMBER}))?$")
_NUMERIC_SPAN = re.compile(rf"(?<![\w.])({_NUMBER}(?:\s*/\s*{_NUMBER})?)(?![\w.])")
_CONCLUSION = re.compile(r"^\s*(?:therefore|thus|hence|so|in conclusion)\b\s*[,:]?\s*",re.I)
_UNITS = re.compile(r"\s+(?:dollars?|cents?|cm|mm|km|meters?|metres?|inches|feet|hours?|minutes?|seconds?|years?|degrees?)\s*$", re.I)
_METRIC = None


def policy_hash():
    return hashlib.sha256(json.dumps(POLICY, sort_keys=True).encode()).hexdigest()


def clean_expression(value):
    value = value.strip().replace("−", "-").replace("，", ",")
    value = re.sub(r"\s*I hope it is correct\.?\s*$", "", value, flags=re.I)
    value = value.strip(" \n\t:。")
    if value.endswith("."):
        value = value[:-1].rstrip()
    # Markdown decorations, not mathematical braces or parentheses.
    value = value.strip("*`")
    if value.startswith("\\[") and value.endswith("\\]"):
        value = value[2:-2].strip()
    elif value.startswith("\\(") and value.endswith("\\)"):
        value = value[2:-2].strip()
    elif value.startswith("$$") and value.endswith("$$"):
        value = value[2:-2].strip()
    elif value.startswith("$") and value.endswith("$") and value.count("$") == 2:
        value = value[1:-1].strip()
    else:
        # Formatting repair only: \(a,b$ and $a,b\) have a complete payload.
        # Missing math content/braces is never reconstructed from gold.
        opener = next((s for s in ("\\[","\\(","$$","$") if value.startswith(s)),None)
        closer = next((s for s in ("\\]","\\)","$$","$") if value.endswith(s)),None)
        if opener and closer and len(value)>len(opener)+len(closer):
            middle = value[len(opener):-len(closer)]
            if not re.search(r"\$|\\[\[\]()\]]",middle):
                value = middle.strip()
    return value


def boxes(text, offset=0):
    found = []
    for match in _BOX.finditer(text):
        depth, end = 1, match.end()
        while end < len(text) and depth:
            if text[end] in "{}" and (end == 0 or text[end-1] != "\\"):
                depth += 1 if text[end] == "{" else -1
            end += 1
        found.append(dict(answer=clean_expression(text[match.end():end-1]) if not depth else None,
                          span=[offset+match.start(), offset+end], malformed=bool(depth)))
    return found


def _selection(source, candidates, explicit, status="selected", enumeration=False):
    if any(c["answer"] is None or not c["answer"] for c in candidates):
        status = "malformed"
    return dict(status=status, source=source, explicit=explicit, candidates=candidates, enumeration=enumeration)


def _enumeration_separators(text, spans):
    return all(re.fullmatch(r"\s*(?:,?\s*(?:and|or)|,)\s*",text[a[1]:b[0]],re.I)
               for a,b in zip(spans,spans[1:]))


def extract_final(text):
    """Gold-independent extraction. No task label or reference is accepted."""
    if not isinstance(text, str) or not text.strip():
        return _selection("none", [], False, "missing")
    markers = list(_MARKER.finditer(text))
    all_boxes = boxes(text)
    if markers:
        marker = markers[-1]
        region = text[marker.end():]
        selected_boxes = boxes(region, marker.end())
        if selected_boxes:
            # A declaration containing several boxed alternatives needs audit;
            # equivalent duplicates are resolved later, independently of gold.
            spans = [c["span"] for c in selected_boxes]
            return _selection("explicit_box", selected_boxes, True,
                              enumeration=len(spans)>1 and _enumeration_separators(text,spans))
        region = region.lstrip()
        position = len(text) - len(region)
        math = _MATH.match(region)
        if math:
            maths = list(_MATH.finditer(region))
            tail = region[maths[-1].end():].strip(" \n\t.,。!*")
            if len(maths)>1 and _enumeration_separators(region,[m.span() for m in maths]) and (
                    not tail or re.fullmatch(r"I hope it is correct\.?",tail,re.I)):
                return _selection("explicit_enumeration",[dict(
                    answer=clean_expression(next(x for x in m.groups() if x is not None)),
                    span=[position+m.start(),position+m.end()],malformed=False) for m in maths],
                    True,enumeration=True)
            tail = region[math.end():].strip(" \n\t.,。!*")
            if tail and not re.fullmatch(r"I hope it is correct\.?", tail, flags=re.I):
                # Do not accept the first of several final alternatives.
                if _MATH.search(tail) or re.search(r"\b(?:or|and|instead|rather)\b|\d", tail, re.I):
                    return _selection("explicit_math", [], True, "ambiguous")
            expression = next(x for x in math.groups() if x is not None)
            return _selection("explicit_math", [dict(answer=clean_expression(expression),
                                span=[position, position+math.end()], malformed=False)], True)
        # Blank lines after "answer is:" are common. Take one answer line,
        # never a number somewhere in the rest of the reasoning.
        line = region.splitlines()[0] if region else ""
        line = re.sub(r"\s*I hope it is correct\.?\s*$", "", line, flags=re.I)
        return _selection("explicit_plain", [dict(answer=clean_expression(line),
                            span=[position, position+len(line)], malformed=False)], True)
    if all_boxes:
        return _selection("last_box", [all_boxes[-1]], True)
    maths = list(_MATH.finditer(text))
    if maths and not text[maths[-1].end():].strip(" \n\t.,。!*"):
        match = maths[-1]
        answer = next(x for x in match.groups() if x is not None)
        return _selection("trailing_math", [dict(answer=clean_expression(answer),
                            span=[match.start(), match.end()], malformed=False)], False)
    line = text.rstrip().splitlines()[-1]
    answer = clean_expression(line)
    if _NUMERIC.fullmatch(answer):
        return _selection("trailing_number", [dict(answer=answer,
                            span=[len(text.rstrip())-len(line), len(text.rstrip())], malformed=False)], False)
    return _selection("none", [], False, "missing")


def numeric_value(answer, gsm8k=False):
    value = clean_expression(answer)
    if gsm8k:
        value = _UNITS.sub("", value)
        value = value.lstrip("$").rstrip("%").strip()
    match = _NUMERIC.fullmatch(value)
    if not match:
        return None
    try:
        numerator = Fraction(match[1].replace(",", ""))
        return numerator / Fraction(match[2].replace(",", "")) if match[2] else numerator
    except (ValueError, ZeroDivisionError):
        return None


def gsm_numeric_claim(answer):
    """One scalar claim, never the last digit from arbitrary reasoning."""
    direct = numeric_value(answer,gsm8k=True)
    if direct is not None:
        return direct
    value = clean_expression(answer)
    latex_fraction = re.fullmatch(r"\\(?:d?frac)\{([+-]?\d+)\}\{([+-]?\d+)\}",value)
    if latex_fraction:
        try:
            return Fraction(int(latex_fraction[1]),int(latex_fraction[2]))
        except ZeroDivisionError:
            return None
    if re.search(r"\b(?:or|and|not|between|at least|at most|less than|more than)\b|[<>≤≥]",value,re.I):
        return None
    # An explicitly selected result equation uses its RHS, not an intermediate
    # number. No equation is evaluated or corrected using the reference.
    if "=" in value:
        value = value.rsplit("=",1)[-1]
    matches = list(_NUMERIC_SPAN.finditer(value))
    if len(matches) != 1:
        return None
    return numeric_value(matches[0][1],gsm8k=True)


def extract_gsm_final(text):
    selection = extract_final(text)
    if selection["status"] == "missing" and isinstance(text,str) and text.strip():
        paragraph = text.rstrip().rsplit("\n\n",1)[-1]
        conclusion = _CONCLUSION.match(paragraph)
        if conclusion:
            position = len(text.rstrip())-len(paragraph)+conclusion.end()
            payload = paragraph[conclusion.end():]
            selection = _selection("final_conclusion",[dict(answer=payload,
                                    span=[position,len(text.rstrip())],malformed=False)],True)
    return selection


class MathComparison:
    """Use the existing environment; never install or silently change a parser."""
    def __init__(self):
        from math_verify import LatexExtractionConfig, parse, verify
        self.parse_fn, self.verify_fn = parse, verify
        self.config = [LatexExtractionConfig(enforce_boxed_match=False)]

    def parse(self, value):
        # The isolated expression is the entire single math environment. Plain
        # expression extraction could turn "1/2" into "1" in old Math-Verify.
        value = clean_expression(value)
        number = numeric_value(value, gsm8k=True)
        if number is not None:
            from sympy import Rational
            if value.endswith("%"):
                number /= 100
            return Rational(number.numerator, number.denominator)
        value = re.sub(r"\\text\{\s*(?:and|or)\s*\}|\s+\b(?:and|or)\b\s+",",",value,flags=re.I)
        if not value or "$" in value or re.search(r"\b(?:or|and|instead|rather)\b", value, re.I):
            return None
        parsed = self.parse_fn("$"+value+"$", extraction_config=self.config, fallback_mode="no_fallback")
        return parsed[0] if len(parsed) == 1 else None

    def equivalent(self, left, right):
        return bool(self.verify_fn(left, right))


def assess(text, gold, task, comparison=None):
    selection = extract_gsm_final(text) if task == "gsm8k" else extract_final(text)
    result = dict(extraction=selection, status=selection["status"], correct=False,
                  explicit_correct=False, prediction=None, parsed_prediction=None)
    if selection["status"] != "selected":
        return result
    answers = [c["answer"] for c in selection["candidates"]]
    if task == "gsm8k":
        values = [gsm_numeric_claim(a) for a in answers]
        target = numeric_value(gold, gsm8k=True)
        if target is None:
            raise ValueError("Invalid GSM8K reference")
        same = lambda a, b: a == b
    else:
        comparison = comparison or MathComparison()
        values = [comparison.parse(a) for a in answers]
        target = comparison.parse(gold)
        if target is None:
            result["status"] = "reference_unparseable"
            return result
        same = comparison.equivalent
    if not values or any(value is None for value in values):
        result["status"] = "unparseable"
        return result
    prediction = values[0]
    if any(not same(values[0], value) for value in values[1:]):
        if task != "math" or not selection["enumeration"]:
            result["status"] = "ambiguous"
            return result
        from sympy import FiniteSet
        if not isinstance(target,FiniteSet):
            result["status"] = "ambiguous"
            return result
        prediction = FiniteSet(*values)
    result.update(status="correct" if same(target, prediction) else "wrong",
                  prediction=answers[0] if len(values)==1 else answers, parsed_prediction=str(prediction))
    result["correct"] = result["status"] == "correct"
    result["explicit_correct"] = result["correct"] and selection["explicit"]
    return result


def _init_worker():
    global _METRIC
    _METRIC = MathComparison()


def _worker(row):
    ident, gold, task, methods = row
    valid = _METRIC.parse(gold) is not None if task == "math" else numeric_value(gold,gsm8k=True) is not None
    if not valid:
        scored = {name:dict(extraction=extract_final(text),status="reference_unparseable",correct=False,
                            explicit_correct=False,prediction=None,parsed_prediction=None) for name,text in methods.items()}
    else:
        scored = {name:assess(text,gold,task,_METRIC) for name,text in methods.items()}
    return dict(id=ident,gold=gold,reference_parseable=valid,methods=scored)


def paired_interval(left, right, repeats=10000, seed=1234):
    import numpy as np
    if len(left) != len(right) or not left:
        raise ValueError("Paired sample sets must have equal nonzero length")
    wins = sum(bool(a) and not bool(b) for a, b in zip(left, right))
    losses = sum(bool(b) and not bool(a) for a, b in zip(left, right))
    n = len(left)
    samples = np.random.default_rng(seed).multinomial(n, [wins/n, losses/n, (n-wins-losses)/n], size=repeats)
    return dict(delta=(wins-losses)/n, wins=wins, losses=losses,
                paired_ci95=np.quantile((samples[:,0]-samples[:,1])/n,[.025,.975]).tolist(),
                repeats=repeats, seed=seed, scope="retrospective diagnostic, not noninferiority evidence")


def load_jobs(jobs, dataset, task):
    samples_list = json.loads(Path(dataset).read_text(encoding="utf-8"))
    samples = {str(row["id"]):row for row in samples_list}
    if len(samples) != len(samples_list):
        raise ValueError("Duplicate dataset IDs")
    metric = load_metric(installed_utils()) if task == "math" else None
    golds = {}
    for ident, sample in samples.items():
        if task == "math":
            boxed = metric["last_boxed_only_string"](sample["solution"])
            if boxed is None:
                raise ValueError("Missing original MATH reference: "+ident)
            golds[ident] = metric["remove_boxed"](boxed)
        else:
            answer = sample["answer"]
            if "####" not in answer:
                raise ValueError("Missing original GSM8K reference: "+ident)
            golds[ident] = answer.rsplit("####",1)[-1].strip()
    texts, old_scores, provenance, hashes, common_ids, identity = {}, {}, {}, {}, None, None
    for declaration in jobs:
        name, directory = declaration.split("=",1)
        if not name or name in texts:
            raise ValueError("Duplicate/empty job label")
        directory = Path(directory)
        summary_path = directory/"summary.json"
        summary = json.loads(summary_path.read_text())
        if summary["dataset_sha256"] != sha256(dataset):
            raise ValueError("Dataset hash mismatch")
        config = summary["configuration"]
        ident = (summary["model"], summary["revision"], config["gen_length"], config["block_length"], config["task"])
        if identity is not None and identity != ident:
            raise ValueError("Model revision/task/length mismatch")
        identity = ident
        if config["task"] != task:
            raise ValueError("Task mismatch")
        records, files = read_records(directory, samples)
        methods = config["methods"]
        if len(methods) != 1:
            raise ValueError("Each job must contain exactly one method")
        method = methods[0]
        table = {str(r["id"]):r[method]["text"] for r in records}
        ids = sorted(table)
        if set(ids) != {str(i) for i in summary["ids"]}:
            raise ValueError("Generation/summary IDs mismatch")
        if common_ids is not None and ids != common_ids:
            raise ValueError("All methods must have exactly the same prompt IDs")
        common_ids = ids
        files += [summary_path]
        prior = {}
        if task == "math" or summary.get("scoring_artifact"):
            path = Path(summary["scoring_artifact"])
            score = json.loads(path.read_text())
            prior = {str(r["id"]):r["methods"][method] for r in score["details"]}
            if task == "gsm8k":
                prior = {ident:dict(exact_match=value["flexible-extract"], strict_match=value["strict-match"])
                         for ident,value in prior.items()}
            if set(prior) != set(ids):
                raise ValueError("Official scoring IDs mismatch")
            files.append(path)
        texts[name], old_scores[name] = table, prior
        hashes.update({str(p):sha256(p) for p in files})
        provenance[name] = dict(directory=str(directory), method=method, identity=ident,
                                mean_seconds=summary["results"][method]["mean_seconds"])
    rows = [(ident,golds[ident],task,{name:table[ident] for name,table in texts.items()}) for ident in common_ids]
    return rows, texts, old_scores, provenance, hashes, samples


def summarize(details, old_scores):
    summaries = {}
    for name in details[0]["methods"]:
        rows = [d["methods"][name] for d in details]
        statuses = Counter(r["status"] for r in rows)
        sources = Counter(r["extraction"]["source"] for r in rows)
        disagreements = Counter()
        for detail in details:
            old = old_scores[name].get(detail["id"])
            if old:
                new = detail["methods"][name]["correct"]
                for key in ("exact_match", "math_verify"):
                    if key in old and bool(old[key]) != new:
                        disagreements[key+("_only" if old[key] else "_final_only")] += 1
        summaries[name] = dict(examples=len(rows), final_expression_accuracy=sum(r["correct"] for r in rows)/len(rows),
            explicit_final_accuracy=sum(r["explicit_correct"] for r in rows)/len(rows), statuses=dict(statuses),
            resolved_rate=(statuses["correct"]+statuses["wrong"])/len(rows),
            sources=dict(sources), disagreements=dict(disagreements))
        if old_scores[name]:
            summaries[name]["original_exact_match"] = sum(bool(old_scores[name][d["id"]]["exact_match"]) for d in details)/len(rows)
            if all("math_verify" in old_scores[name][d["id"]] for d in details):
                summaries[name]["original_math_verify"] = sum(bool(old_scores[name][d["id"]]["math_verify"]) for d in details)/len(rows)
    return summaries


def audit_sample(details, texts, old_scores, count=48, seed=1234):
    """Uniform ID sample from disagreements; method names hidden in review file."""
    eligible = []
    for d in details:
        if any((old_scores[name].get(d["id"], {}).get("math_verify",old_scores[name].get(d["id"], {}).get("exact_match")) is not None
                and bool(old_scores[name][d["id"]].get("math_verify",old_scores[name][d["id"]].get("exact_match"))) != r["correct"])
               or r["status"] in ("ambiguous","malformed","unparseable","reference_unparseable")
               for name,r in d["methods"].items()):
            eligible.append(d)
    rng = random.Random(seed)
    chosen = rng.sample(eligible, min(count, len(eligible)))
    reviews, key = [], {}
    for index, d in enumerate(chosen):
        audit_id = f"review_{index:03d}"
        names = list(d["methods"])
        rng.shuffle(names)
        candidates = {}
        key[audit_id] = dict(id=d["id"], mapping={})
        for j, name in enumerate(names):
            label = chr(65+j)
            key[audit_id]["mapping"][label] = name
            candidates[label] = dict(text=texts[name][d["id"]], extraction=d["methods"][name]["extraction"],
                                     human_final_answer=None, human_correct=None, notes=None)
        reviews.append(dict(audit_id=audit_id, gold=d["gold"], candidates=candidates))
    return dict(seed=seed, eligible_prompt_count=len(eligible), scope="disagreement-enriched audit, not accuracy estimate",
                reviews=reviews), key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("math","gsm8k"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--job", action="append", required=True, help="label=/completed/job/output")
    parser.add_argument("--output", type=Path, required=True, help="new directory; original scores are never overwritten")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or args.workers < 1:
        parser.error("Output must be a new directory; workers must be positive")
    rows, texts, old_scores, provenance, hashes, samples = load_jobs(args.job, args.dataset, args.task)
    args.output.mkdir(parents=True, exist_ok=False)
    import math_verify.parser as installed_parser
    import math_verify.grader as installed_grader
    manifest = dict(policy=POLICY, policy_sha256=policy_hash(), scorer_sha256=sha256(__file__),
                    dataset=str(args.dataset), dataset_sha256=sha256(args.dataset), inputs=hashes, jobs=provenance,
                    math_verify_version=metadata.version("math-verify"), sympy_version=metadata.version("sympy"),
                    parser_sha256=sha256(installed_parser.__file__), grader_sha256=sha256(installed_grader.__file__),
                    official_gold_utils_sha256=sha256(installed_utils()) if args.task == "math" else None,
                    raw_text_limit="Saved postprocessed text only; discarded tokens cannot be recovered by scoring")
    (args.output/"policy.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    started, details = time.monotonic(), []
    with ProcessPoolExecutor(args.workers, initializer=_init_worker) as executor:
        for detail in executor.map(_worker, rows, chunksize=8):
            details.append(detail)
            if len(details)%100 == 0 or len(details)==len(rows):
                print(f"Final-answer audit {len(details)}/{len(rows)}; {time.monotonic()-started:.1f}s",flush=True)
    if {p:sha256(p) for p in hashes} != hashes or sha256(args.dataset) != manifest["dataset_sha256"]:
        raise RuntimeError("Inputs changed during scoring")
    results = summarize(details, old_scores)
    names = list(results)
    pairs = []
    for i, left in enumerate(names):
        for right in names[i+1:]:
            pairs.append(dict(left=left,right=right,**paired_interval(
                [d["methods"][left]["correct"] for d in details],
                [d["methods"][right]["correct"] for d in details])))
    audit, key = audit_sample(details,texts,old_scores)
    for review in audit["reviews"]:
        sample = samples[key[review["audit_id"]]["id"]]
        review["question"] = sample.get("problem",sample.get("question"))
    for filename, value in (("scores.json",dict(results=results,pairs=pairs,details=details,
                                                scoring_seconds=time.monotonic()-started)),
                            ("blind_review.json",audit),("review_key.json",key)):
        (args.output/filename).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding="utf-8")
    (args.output/"complete").write_text("OK\n")
    print(json.dumps(dict(results=results,pairs=pairs),indent=2),flush=True)


if __name__ == "__main__":
    main()
