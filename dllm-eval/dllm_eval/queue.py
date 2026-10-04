"""Durable CPU work queue for independent, resumable evaluation workers.

Claims have no time expiry: a slow model call must never become duplicate work.
Only a process proven dead on this host can have an abandoned claim recovered.
Completed records remain the source of truth when importing interrupted runs.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
import time
from typing import Iterable
import uuid


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_completed_records(paths: Iterable[Path], *, expected_count: int,
                           required_methods=(), allow_truncated_tail=False,
                           allow_identical_duplicates=False) -> dict[int, dict]:
    """Read durable prompt records, rejecting conflicts and out-of-range indices.

    A non-newline-terminated final JSON object is still read if it is valid JSON.
    Only an invalid final fragment can be ignored with allow_truncated_tail.
    Missing methods mean the prompt is incomplete, rather than reusable output.
    """
    records = {}
    origins = {}
    for raw_path in paths:
        path = Path(raw_path)
        raw = path.read_bytes()
        try:
            data = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            if not (allow_truncated_tail and error.end == len(raw)
                    and error.reason == "unexpected end of data"):
                raise
            data = raw[:error.start].decode("utf-8")
        lines = data.splitlines(keepends=True)
        for line_no, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if (allow_truncated_tail and line_no == len(lines)
                        and not line.endswith(("\n", "\r"))):
                    continue
                raise ValueError(f"Invalid JSON record: {path}:{line_no}") from None
            if not isinstance(row, dict):
                raise ValueError(f"Record is not an object: {path}:{line_no}")
            index = row.get("index")
            if type(index) is not int or not 0 <= index < expected_count:
                raise ValueError(f"Invalid prompt index {index!r}: {path}:{line_no}")
            missing = [name for name in required_methods
                       if not isinstance(row.get(name), dict)]
            if missing:
                raise ValueError(f"Incomplete prompt {index}, missing methods {missing}: {path}")
            if index in records:
                if not allow_identical_duplicates or records[index] != row:
                    raise ValueError(f"Duplicate/conflicting prompt {index}: "
                                     f"{origins[index]} and {path}:{line_no}")
                continue
            records[index] = row
            origins[index] = f"{path}:{line_no}"
    return records


@contextmanager
def _file_lock(path):
    # flock is process-safe on the Linux evaluation host. Windows support lets
    # the queue and recovery tests run in the local development checkout.
    with Path(path).open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            while True:
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _process_info(pid):
    """Return Linux process incarnation and state, if observable."""
    try:
        parts = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return parts[19], parts[0]
    except (OSError, IndexError):
        return None, None


def _owner_dead(owner):
    if owner["host"] != socket.gethostname():
        return False
    pid = owner["pid"]
    if os.name == "nt":
        # os.kill(pid, 0) does not have POSIX probe semantics on Windows.
        # Do not risk terminating a real worker while inspecting its claim.
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except (PermissionError, OSError):
        return False
    start, state = _process_info(pid)
    return (state == "Z" or
            (owner.get("process_start") is not None and start is not None
             and owner["process_start"] != start))


@dataclass(frozen=True)
class Lease:
    claim_id: str
    indices: tuple[int, ...]
    pid: int
    host: str
    process_start: str | None


class ElasticQueue:
    """Atomic index claims with strict identity and no GPU dependencies."""

    def __init__(self, root, *, total, identity, chunk_size=1,
                 completed_indices=()):
        if type(total) is not int or total < 0:
            raise ValueError("total must be a nonnegative integer")
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "queue.json"
        self.lock_path = self.root / "queue.lock"
        self.total = total
        self.identity = json.loads(_canonical(identity))
        self.identity_sha256 = hashlib.sha256(_canonical(identity).encode()).hexdigest()
        self.chunk_size = chunk_size
        completed_indices = self._indices(completed_indices)
        with _file_lock(self.lock_path):
            if self.path.exists():
                state = self._read()
            else:
                state = dict(version=1, total=total, identity=self.identity,
                             identity_sha256=self.identity_sha256,
                             completed=[], active={})
            self._reconcile(state, completed_indices)
            self._write(state)

    def _indices(self, values):
        values = list(values)
        if any(type(value) is not int or not 0 <= value < self.total for value in values):
            raise ValueError("Prompt indices must be integers in [0, total)")
        return set(values)

    def _read(self):
        state = json.loads(self.path.read_text(encoding="utf-8"))
        if (state.get("version") != 1 or state.get("total") != self.total
                or state.get("identity_sha256") != self.identity_sha256
                or state.get("identity") != self.identity):
            raise ValueError("Existing queue does not match dataset/configuration identity")
        done = self._indices(state["completed"])
        if len(done) != len(state["completed"]):
            raise ValueError("Duplicate completed indices in queue")
        occupied = set(done)
        for owner in state["active"].values():
            indices = self._indices(owner["indices"])
            if occupied & indices or len(indices) != len(owner["indices"]):
                raise ValueError("Queue has overlapping claims/completed indices")
            occupied.update(indices)
        return state

    def _write(self, state):
        state["updated_at"] = time.time()
        descriptor, name = tempfile.mkstemp(prefix="queue-", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(state, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def _reconcile(self, state, indices):
        state["completed"] = sorted(set(state["completed"]) | indices)
        for claim_id, owner in list(state["active"].items()):
            owner["indices"] = [index for index in owner["indices"] if index not in indices]
            if not owner["indices"]:
                del state["active"][claim_id]

    def reconcile_completed(self, indices):
        """Import indices backed by successfully validated durable records."""
        indices = self._indices(indices)
        with _file_lock(self.lock_path):
            state = self._read()
            self._reconcile(state, indices)
            self._write(state)

    def _recover(self, state):
        recovered = []
        for claim_id, owner in list(state["active"].items()):
            if _owner_dead(owner):
                recovered.extend(owner["indices"])
                del state["active"][claim_id]
        return sorted(recovered)

    def recover_dead(self):
        with _file_lock(self.lock_path):
            state = self._read()
            recovered = self._recover(state)
            if recovered:
                self._write(state)
            return recovered

    def remaining_indices(self):
        """Unfinished IDs (including in-flight IDs), for round-robin repartition."""
        with _file_lock(self.lock_path):
            done = set(self._read()["completed"])
            return [index for index in range(self.total) if index not in done]

    def claim(self, allowed_indices=None):
        # Match lmms-eval's per-rank document iterator, while retaining atomic
        # ownership across a worker restart or a change in available GPU count.
        allowed = (range(self.total) if allowed_indices is None else
                   sorted(self._indices(allowed_indices)))
        with _file_lock(self.lock_path):
            state = self._read()
            recovered = self._recover(state)
            occupied = set(state["completed"])
            for owner in state["active"].values():
                occupied.update(owner["indices"])
            indices = []
            for index in allowed:
                if index not in occupied:
                    indices.append(index)
                    if len(indices) == self.chunk_size:
                        break
            if not indices:
                if recovered:
                    self._write(state)
                return None
            claim_id = uuid.uuid4().hex
            start, _ = _process_info(os.getpid())
            owner = dict(indices=indices, pid=os.getpid(), host=socket.gethostname(),
                         process_start=start, claimed_at=time.time())
            state["active"][claim_id] = owner
            self._write(state)
            return Lease(claim_id, tuple(indices), owner["pid"], owner["host"], start)

    def _check_lease(self, state, lease):
        owner = state["active"].get(lease.claim_id)
        if owner is None:
            if set(lease.indices) <= set(state["completed"]):
                return None
            raise ValueError("Claim no longer exists; it may have been recovered")
        if any(owner[key] != getattr(lease, key) for key in ("pid", "host", "process_start")):
            raise ValueError("Claim owner does not match lease")
        if not set(owner["indices"]) <= set(lease.indices):
            raise ValueError("Claim indices do not match lease")
        return owner

    def complete(self, lease, completed_indices=None):
        """Complete durable indices; uncompleted chunk indices become available."""
        indices = self._indices(lease.indices if completed_indices is None else completed_indices)
        if not indices <= set(lease.indices):
            raise ValueError("Cannot complete indices outside this lease")
        with _file_lock(self.lock_path):
            state = self._read()
            owner = self._check_lease(state, lease)
            if owner is not None:
                del state["active"][lease.claim_id]
            state["completed"] = sorted(set(state["completed"]) | indices)
            self._write(state)

    def release(self, lease):
        self.complete(lease, completed_indices=())

    def status(self):
        with _file_lock(self.lock_path):
            state = self._read()
            active_count = sum(len(owner["indices"]) for owner in state["active"].values())
            done = len(state["completed"])
            return dict(total=self.total, completed=done, active=active_count,
                        pending=self.total - done - active_count,
                        complete=(done == self.total), claims=state["active"],
                        identity_sha256=self.identity_sha256)
