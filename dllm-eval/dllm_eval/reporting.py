"""Evaluation presentation only: one transcript, progress bars and result tables."""
import os
import re
import sys
import threading
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_TASKS = {"gsm8k": "GSM8K", "math": "MATH", "humaneval": "HumanEval", "mbpp": "MBPP"}
_SHOTS = {"gsm8k": 5, "math": 4, "humaneval": 0, "mbpp": 3}


class _Tee:
    def __init__(self, console, logfile, lock):
        self.console, self.logfile, self.lock = console, logfile, lock

    def write(self, text):
        with self.lock:
            self.console.write(text)
            self.logfile.write(_ANSI.sub("", text).replace("\r", "\n"))
            self.logfile.flush()
        return len(text)

    def flush(self):
        with self.lock:
            self.console.flush()
            self.logfile.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


class _ProgressOutput(_Tee):
    """Redraw on a terminal; save readable, newline-delimited snapshots to disk."""
    def write(self, text):
        snapshots = [line.rstrip() for line in _ANSI.sub("", text).replace("\r", "\n").splitlines() if line.strip()]
        with self.lock:
            if self.console.isatty():
                self.console.write(text)
            else:
                for line in snapshots:
                    self.console.write(line + "\n")
            for line in snapshots:
                self.logfile.write(line + "\n")
            self.console.flush()
            self.logfile.flush()
        return len(text)


def result_table(cells):
    """Render the existing summary metrics without modifying them or their units."""
    headers = ["Task", "Length", "n-shot", "Correct / total", "Acc (%)", "Request (s)", "NFE"]
    rows = []
    for key, cell in cells.items():
        task, length = key.rsplit("_", 1)
        number = lambda value, digits: "N/A" if value is None else f"{value:.{digits}f}"
        rows.append([_TASKS.get(task, task), length, str(_SHOTS.get(task, "-")),
                     f"{cell['correct']}/{cell['samples']}", number(cell['accuracy_percent'], 2),
                     number(cell['mean_seconds'], 3), number(cell['mean_nfe'], 2)])
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    def line(row):
        return "| " + " | ".join(value.ljust(widths[i]) if i == 0 else value.rjust(widths[i])
                                    for i, value in enumerate(row)) + " |"
    separator = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    return "\n".join([line(headers), separator, *(line(row) for row in rows)])


class Reporter:
    def __init__(self, path, progress_stream):
        self.path, self.progress_stream = path, progress_stream
        self.bar = None

    def info(self, message):
        if self.bar is not None:
            self.bar.clear()
        print(f"{datetime.now().astimezone():%Y-%m-%d %H:%M:%S%z} | INFO | {message}", flush=True)
        if self.bar is not None:
            self.bar.refresh()

    def start_run(self, args, samples):
        total = sum(len(rows) for rows in samples.values()) * len(args.lengths)
        self.info(f"RelayCache | GPU {args.gpu} | shard {args.rank + 1}/{args.world_size} | "
                  f"{len(args.tasks) * len(args.lengths)} cells, {total} requests | resume={args.resume}")
        self.info("Request time: warmed prepared-prompt generation; setup, preparation, warm-up, "
                  "postprocessing, grading, disk I/O and reporting excluded.")

    def start_cell(self, task, length, total):
        self.close_progress()
        self.task, self.length = task, length
        self.count = self.correct = 0
        self.seconds = self.nfe = 0.0
        self.info(f"Starting {_TASKS.get(task, task)} | length={length} | n-shot={_SHOTS.get(task, '-')} | samples={total}")
        self.bar = tqdm(total=total, desc=f"{_TASKS.get(task, task)} / {length}", unit="req",
                        file=self.progress_stream, mininterval=1.0, miniters=1,
                        dynamic_ncols=self.progress_stream.isatty(),
                        ncols=None if self.progress_stream.isatty() else 120, ascii=True, leave=True,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]")

    def update(self, row):
        # Already graded scalar records only; no model tensors, prompts or references.
        self.count += 1
        self.correct += row['correct']
        self.seconds += row['seconds']
        self.nfe += row['nfe']
        self.bar.set_postfix_str(f"acc={100 * self.correct / self.count:.2f}%, "
                                f"request={self.seconds / self.count:.3f}s, nfe={self.nfe / self.count:.2f}", refresh=False)
        self.bar.update(1)

    def close_progress(self):
        if self.bar is not None:
            self.bar.close()
            self.bar = None

    def finish_cell(self):
        self.close_progress()
        n = self.count
        cell = dict(samples=n, correct=self.correct, accuracy_percent=100 * self.correct / n if n else None,
                    mean_seconds=self.seconds / n if n else None, mean_nfe=self.nfe / n if n else None)
        self.info(f"Completed {_TASKS.get(self.task, self.task)} / {self.length}")
        print(result_table({f"{self.task}_{self.length}": cell}), flush=True)

    def finish(self, cells, output):
        self.info("Evaluation complete. Final results:")
        print(result_table(cells), flush=True)
        self.info(f"Summary: {(output / 'summary.json').resolve()}")
        self.info(f"Log: {self.path}")


@contextmanager
def evaluation_log(directory, output, rank=0):
    """Capture one invocation, including startup failures and interrupts, then restore streams."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(output).name) or "relay"
    path = directory / f"{safe_name}_{stamp}_r{rank}_{os.getpid()}.log"
    original_out, original_err = sys.stdout, sys.stderr
    lock = threading.RLock()
    with path.open("x", encoding="utf-8", buffering=1) as logfile:
        reporter = Reporter(path, _ProgressOutput(original_err, logfile, lock))
        sys.stdout = _Tee(original_out, logfile, lock)
        sys.stderr = _Tee(original_err, logfile, lock)
        try:
            reporter.info(f"Log: {path}")
            yield reporter
        except BaseException:
            reporter.close_progress()
            reporter.info("Evaluation stopped; saved records remain available for explicit --resume.")
            traceback.print_exc(file=sys.stderr)
            raise
        finally:
            try:
                reporter.close_progress()
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                sys.stdout, sys.stderr = original_out, original_err
