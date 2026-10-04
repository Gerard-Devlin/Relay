"""Portable data selection, prompt whitelist, hashing and atomic reporting."""
import hashlib,json,random
from pathlib import Path
def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def prompt_ids(tokenizer, question, task="gsm8k", *, preformatted=False):
    """Apply the LLaDA chat template to a task prompt.

    ``preformatted`` is used for prompts produced by lm-eval.  Those strings
    already contain the paper's few-shot demonstrations and task formatting;
    appending the local zero-shot instruction would change the benchmark.
    """
    if task == "gsm8k" and not preformatted:
        question = question + "\nExplain your reasoning and end with #### followed by the final number."
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=True,
        add_generation_prompt=True,
    )

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)

def select_samples(path, limit, offset, seed=51713):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    random.Random(seed).shuffle(rows)
    if limit <= 0 or offset < 0 or offset + limit > len(rows):
        raise ValueError("Invalid sample range")
    selected = rows[offset:offset + limit]
    ids = [str(s.get("id", s.get("task_id"))) for s in selected]
    if len(set(ids)) != len(ids) or "None" in ids:
        raise ValueError("Missing or duplicated sample IDs")
    return selected

def generation_prompt(sample):
    # Explicit whitelist: reference answers/tests/solutions never enter input.
    value = sample.get("paper_prompt", sample.get("prompt"))
    if not isinstance(value, str) or not value:
        raise ValueError("No legitimate prompt")
    return value
