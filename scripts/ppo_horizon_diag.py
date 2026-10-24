"""Horizon diagnostic driver (diag-v0).

Runs the three arms (R256, R1024, G999) at 300k env steps each, obs_only,
workers=4, for seeds 100000 and 100001. For every quick-eval marker the
trainer writes (each 20th iteration) it evals the saved checkpoint on dev
seeds 0-4, cap 10000. After each arm finishes it runs the final evals
(Balanced and Hard 5, dev seeds 0-19, cap 10000) and writes a result JSON.

Resumable: an arm/seed whose result JSON already exists is skipped. A lock
file prevents concurrent driver instances.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
LOG_ROOT = REPO / "logs" / "diag_horizon"
ARMS = [
    ("R256", {}),
    ("R1024", {"--rollout": "1024"}),
    ("G999", {"--gamma": "0.999"}),
]
SEEDS = [100000, 100001]
TOTAL_STEPS = 300000
WORKERS = 4
EVAL_EVERY = 20
CAP = 10000
DEV_SEEDS_MID = 5
DEV_SEEDS_FINAL = 20
LOCK = LOG_ROOT / ".lock"


def _env() -> dict[str, str]:
    e = dict(os.environ)
    e["OMP_NUM_THREADS"] = "1"
    e["MKL_NUM_THREADS"] = "1"
    return e


def _git_commit() -> str:
    r = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=False, cwd=str(REPO),
        capture_output=True, text=True,
    )
    return r.stdout.strip()


def _initial_checksum(seed: int, arm: dict[str, str]) -> str:
    from simlab.training.ppo import PPOConfig, _build_net

    kw: dict[str, object] = {}
    if "--rollout" in arm:
        kw["env_steps_per_rollout"] = int(arm["--rollout"])
    if "--gamma" in arm:
        kw["gamma"] = float(arm["--gamma"])
    policy = _build_net(PPOConfig(seed=seed, **kw))  # type: ignore[arg-type]
    h = hashlib.sha256()
    for k, v in sorted(policy.net_state_dict().items()):
        h.update(k.encode())
        h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()


def _eval_metrics(scenario: str, ckpt: Path, seeds: int, offset: int,
                  out_json: Path) -> dict[str, object]:
    cmd = [
        sys.executable, "-m", "simlab.cli", "eval",
        "--policy", "learned_torch",
        "--scenario", scenario,
        "--agents", "5",
        "--seeds", str(seeds),
        "--seed-offset", str(offset),
        "--cap", str(CAP),
        "--workers", str(WORKERS),
        "--checkpoint", str(ckpt),
        "--out", str(out_json),
    ]
    r = subprocess.run(cmd, check=False, cwd=str(REPO), capture_output=True,
                        text=True, env=_env())
    if r.returncode != 0:
        return {"error": r.stderr[-2000:], "returncode": r.returncode}
    if out_json.exists():
        return json.loads(out_json.read_text(encoding="utf-8"))
    return {"error": "no output json", "stdout": r.stdout[-2000:]}


def _exec_epochs(learning_csv: Path) -> int:
    if not learning_csv.exists():
        return 0
    n = 0
    for line in learning_csv.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        n += 1
    return n


def _run_arm(arm_name: str, arm: dict[str, str], seed: int) -> dict[str, object]:
    out_dir = LOG_ROOT / arm_name / str(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    result_json = out_dir / "result.json"
    if result_json.exists():
        return {"skipped": True, "result": result_json}

    checksum = _initial_checksum(seed, arm)

    log_file = out_dir / "train.log"
    cmd = [
        sys.executable, "-m", "simlab.cli", "train",
        "--obs", "obs_only",
        "--total-steps", str(TOTAL_STEPS),
        "--workers", str(WORKERS),
        "--seed", str(seed),
        "--eval-every", str(EVAL_EVERY),
        "--out", str(out_dir),
    ]
    for k, v in arm.items():
        cmd += [k, v]
    with log_file.open("w", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, cwd=str(REPO), env=_env(), stdout=f, stderr=subprocess.STDOUT)

    learning_csv = out_dir / "learning.csv"
    done_markers: set[int] = set()
    mid_evals: list[dict[str, object]] = []
    mid_out = out_dir / "mid"
    mid_out.mkdir(parents=True, exist_ok=True)
    poll = 0
    while proc.poll() is None:
        if learning_csv.exists():
            text = learning_csv.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                if line.startswith("# quick_eval it="):
                    it = int(line.split("it=", 1)[1].split(" ", 1)[0])
                    if it in done_markers:
                        continue
                    done_markers.add(it)
                    tmp = out_dir / "latest_mid.pt"
                    try:
                        shutil_copy(out_dir / "latest.pt", tmp)
                        mid_evals.append({
                            "iteration": it,
                            "eval": _eval_metrics("balanced", tmp, DEV_SEEDS_MID, 0,
                                                   mid_out / f"it{it}.json"),
                        })
                    except FileNotFoundError:
                        pass
        poll += 1
        if poll % 12 == 0:
            print(f"  [{arm_name} seed={seed}] training in progress (epochs done="
                  f"{_exec_epochs(learning_csv)})", flush=True)
        time.sleep(5)

    if proc.returncode != 0:
        return {"error": "training failed", "returncode": proc.returncode,
                "log_tail": (log_file.read_text(encoding="utf-8", errors="replace")[-2000:])}

    final = out_dir / "final.pt"
    balanced = _eval_metrics("balanced", final, DEV_SEEDS_FINAL, 0,
                             out_dir / "final_balanced.json")
    hard = _eval_metrics("hard", final, DEV_SEEDS_FINAL, 0,
                         out_dir / "final_hard.json")

    result = {
        "arm": arm_name,
        "seed": seed,
        "training_seed": seed,
        "initial_weights_checksum": checksum,
        "workers": WORKERS,
        "commit": _git_commit(),
        "total_env_steps": TOTAL_STEPS,
        "policy_updates": _exec_epochs(learning_csv),
        "mid_run_evals": mid_evals,
        "final_balanced": balanced,
        "final_hard": hard,
    }
    result_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return {"result": result_json}


def shutil_copy(src: Path, dst: Path) -> None:
    import shutil

    shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description="Horizon diagnostic driver")
    parser.add_argument("--arm", choices=[a for a, _ in ARMS], default=None)
    args = parser.parse_args()

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        print("another driver instance holds the lock; exiting", flush=True)
        return 1

    try:
        arms = ARMS if args.arm is None else [(args.arm, dict(ARMS)[args.arm])]
        for arm_name, arm in arms:
            for seed in SEEDS:
                print(f"== {arm_name} seed={seed} ==", flush=True)
                out = _run_arm(arm_name, arm, seed)
                print(f"   {json.dumps(out, default=str)}", flush=True)
    finally:
        try:
            os.remove(LOCK)
        except FileNotFoundError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
