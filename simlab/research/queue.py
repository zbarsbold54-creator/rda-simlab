"""Resumable, single-instance run queue for the Research cockpit.

A queue is a directory (by default under ``runs/cockpit/<name>``) holding:

* ``queue.json`` — the persistent job list plus a serialized experiment,
* ``queue.lock`` — a lock file (created with O_EXCL) guaranteeing a single
  serving process, with staleness recovery,
* ``pause.flag`` / ``stop.flag`` — out-of-band control flags,
* ``results/<job>.json`` — per-run result metadata.

The serving loop processes one job at a time. It honors pause (between jobs),
stop, and resumes incomplete jobs after a crash/restart. Torch is never
imported here.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from simlab.research.experiment import (
    Experiment,
    arm_param_count,
    dump_experiment_toml,
    load_experiment_toml,
)

LOCK_FILE = "queue.lock"
PAUSE_FLAG = "pause.flag"
STOP_FLAG = "stop.flag"
QUEUE_JSON = "queue.json"


class QueueLockError(RuntimeError):
    pass


@dataclass
class Job:
    id: str
    arm_name: str
    train_seed: int
    status: str = "queued"  # queued | running | done | failed
    pid: int | None = None
    out_dir: str = ""
    exit_code: int | None = None


# ------------------------------------------------------------------ flags
def set_flag(queue_dir: str | Path, name: str) -> None:
    (Path(queue_dir) / name).write_text("", encoding="utf-8")


def clear_flag(queue_dir: str | Path, name: str) -> None:
    Path(queue_dir, name).unlink(missing_ok=True)


def has_flag(queue_dir: str | Path, name: str) -> bool:
    return Path(queue_dir, name).exists()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except ProcessLookupError:
        return False
    except OSError:
        return False
    return True


class QueueManager:
    """Owns one queue directory and can serve/launch/inspect it."""

    def __init__(self, queue_dir: str | Path) -> None:
        self.dir = Path(queue_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock_held = False

    # ------------------------------------------------------------ creation
    @classmethod
    def create(cls, queue_dir: str | Path, exp: Experiment) -> QueueManager:
        m = cls(queue_dir)
        jobs: list[Job] = []
        for i, arm in enumerate(exp.arms):
            for seed in exp.train_seeds:
                job_id = f"{i:02d}_{arm.name}_{seed}"
                jobs.append(
                    Job(
                        id=job_id,
                        arm_name=arm.name,
                        train_seed=seed,
                        out_dir=str(m.dir / "runs" / job_id),
                    )
                )
        m._write_experiment(exp)
        m._write_jobs(jobs)
        return m

    def _write_experiment(self, exp: Experiment) -> None:
        (self.dir / "experiment.toml").write_text(
            dump_experiment_toml(exp), encoding="utf-8"
        )

    def load_experiment(self) -> Experiment:
        return load_experiment_toml(self.dir / "experiment.toml")

    # ------------------------------------------------------------ persistence
    def _jobs_path(self) -> Path:
        return self.dir / QUEUE_JSON

    def _write_jobs(self, jobs: list[Job]) -> None:
        self._jobs_path().write_text(
            json.dumps([asdict(j) for j in jobs], indent=2),
            encoding="utf-8",
        )

    def load_jobs(self) -> list[Job]:
        if not self._jobs_path().exists():
            return []
        rows = json.loads(self._jobs_path().read_text(encoding="utf-8"))
        return [
            Job(
                id=r["id"],
                arm_name=r["arm_name"],
                train_seed=int(r["train_seed"]),
                status=r.get("status", "queued"),
                pid=r.get("pid"),
                out_dir=r.get("out_dir", ""),
                exit_code=r.get("exit_code"),
            )
            for r in rows
        ]

    def save_jobs(self, jobs: list[Job]) -> None:
        self._write_jobs(jobs)

    # ------------------------------------------------------------- lock
    def acquire_lock(self) -> None:
        lock = self.dir / LOCK_FILE
        for _ in range(2):
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                stale = False
                try:
                    pid = int(lock.read_text(encoding="utf-8").strip())
                    stale = not _pid_alive(pid)
                except (ValueError, OSError):
                    stale = True
                if not stale:
                    raise QueueLockError(
                        f"queue is already being served ({lock})"
                    ) from None
                lock.unlink(missing_ok=True)
                continue
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(str(os.getpid()))
            self._lock_held = True
            return
        raise QueueLockError(f"could not acquire lock {lock}")

    def release_lock(self) -> None:
        if self._lock_held:
            (self.dir / LOCK_FILE).unlink(missing_ok=True)
            self._lock_held = False

    # ------------------------------------------------------------- serving
    def _reconcile(self, jobs: list[Job]) -> list[Job]:
        """Reset jobs left 'running' by a dead server to 'queued'."""
        for j in jobs:
            if j.status == "running" and (j.pid is None or not _pid_alive(j.pid)):
                j.status = "queued"
                j.pid = None
        return jobs

    def next_queued(self, jobs: list[Job]) -> Job | None:
        for j in jobs:
            if j.status == "queued":
                return j
        return None

    def run_forever(
        self, runner: Callable[[Job, Experiment], int] | None = None
    ) -> int:
        """Serve the queue until all jobs finish or stop is requested."""
        self.acquire_lock()
        if runner is None:
            runner = lambda job, exp: _default_runner(self.dir, job, exp)
        try:
            exp = self.load_experiment()
            jobs = self._reconcile(self.load_jobs())
            self.save_jobs(jobs)
            while True:
                if has_flag(self.dir, STOP_FLAG):
                    return 0
                if has_flag(self.dir, PAUSE_FLAG):
                    time.sleep(0.25)
                    continue
                job = self.next_queued(jobs)
                if job is None:
                    return 0
                job.status = "running"
                job.pid = os.getpid()
                self.save_jobs(jobs)
                try:
                    code = runner(job, exp)
                except Exception as exc:  # noqa: BLE001
                    code = -1
                    self._record_error(job, f"{type(exc).__name__}: {exc}")
                job.exit_code = int(code)
                job.pid = None
                job.status = "done" if code == 0 else "failed"
                self.save_jobs(jobs)
                self._write_run_result(job, exp, failed=code != 0)
        finally:
            self.release_lock()
        return 0

    def _record_error(self, job: Job, message: str) -> None:
        err = self.dir / "results"
        err.mkdir(parents=True, exist_ok=True)
        (err / f"{job.id}.error").write_text(message, encoding="utf-8")

    def _write_run_result(
        self, job: Job, exp: Experiment, failed: bool = False
    ) -> None:
        from simlab.research.logs import run_summary

        arm = next(a for a in exp.arms if a.name == job.arm_name)
        _params, init_checksum, _cfg = arm_param_count(arm)
        summary = run_summary(Path(job.out_dir)) if not failed else {}
        result = {
            "job": job.id,
            "arm": job.arm_name,
            "seed": job.train_seed,
            "exit_code": job.exit_code,
            "workers": exp.workers,
            "init_checksum": init_checksum,
            "git_commit": _git_head(),
            "summary": summary,
            "failed": failed,
        }
        res_dir = self.dir / "results"
        res_dir.mkdir(parents=True, exist_ok=True)
        (res_dir / f"{job.id}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )

    # ------------------------------------------------------------- inspect
    def status(self) -> dict[str, Any]:
        jobs = self.load_jobs()
        counts = {"queued": 0, "running": 0, "done": 0, "failed": 0}
        for j in jobs:
            counts[j.status] = counts.get(j.status, 0) + 1
        return {"jobs": [asdict(j) for j in jobs], "counts": counts}

    def results(self) -> list[dict[str, Any]]:
        res_dir = self.dir / "results"
        if not res_dir.exists():
            return []
        out = []
        for p in sorted(res_dir.glob("*.json")):
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return out


def _git_head() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0:
            return proc.stdout.strip()
    except Exception:
        logging.getLogger(__name__).debug("git head failed", exc_info=True)
    return "unknown"


def _kill_tree(pid: int) -> None:
    """Kill a process tree without sending a console control event to the caller.

    On Windows, ``taskkill /T /F`` targets the child tree by PID and does not
    deliver a CTRL_C_EVENT to the queue/GUI process's console. Elsewhere, the
    process group is signaled directly.
    """
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _default_runner(queue_dir: str | Path, job: Job, exp: Experiment) -> int:
    from simlab.training.arch_spec import dump_arch_toml

    arm = next(a for a in exp.arms if a.name == job.arm_name)
    arch_path = Path(job.out_dir).parent / f"arm_{arm.name}.toml"
    arch_path.parent.mkdir(parents=True, exist_ok=True)
    arch_path.write_text(dump_arch_toml(arm.selection()), encoding="utf-8")
    out_dir = Path(job.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args = [
        sys.executable,
        "-m",
        "simlab.cli",
        "train",
        "--arch",
        str(arch_path),
        "--total-steps",
        str(exp.total_steps),
        "--workers",
        str(exp.workers),
        "--seed",
        str(job.train_seed),
        "--rollout",
        str(exp.rollout_length),
        "--out",
        str(out_dir),
    ]
    proc = subprocess.Popen(args)
    while proc.poll() is None:
        if has_flag(queue_dir, STOP_FLAG):
            _kill_tree(proc.pid)
            proc.wait()
            return -1
        time.sleep(0.25)
    return int(proc.returncode)


def main(argv: list[str] | None = None) -> int:
    """Small CLI for the queue server: ``python -m simlab.research.queue serve <dir>``."""
    import argparse

    parser = argparse.ArgumentParser(prog="simlab-research-queue")
    sub = parser.add_subparsers(dest="cmd")
    serve = sub.add_parser("serve")
    serve.add_argument("queue_dir")
    statusp = sub.add_parser("status")
    statusp.add_argument("queue_dir")
    args = parser.parse_args(argv)
    if args.cmd == "serve":
        return QueueManager(args.queue_dir).run_forever()
    if args.cmd == "status":
        import json as _json

        print(_json.dumps(QueueManager(args.queue_dir).status(), indent=2))
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
