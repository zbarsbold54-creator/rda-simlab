"""Regenerate BASELINES.md rows and docs/baselines.json.

Grid: scenarios (Balanced 1/5/20, Solo, Hard 1/5/20) x policies
(random / heuristic_noflee / heuristic / heuristic_v2), 20 seeds,
cap 10000. Multiprocessing, one process per episode. Also writes the
machine-readable baseline table to docs/baselines.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from multiprocessing import Pool
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from episode_lib import run_episode, summarize

from simlab.eval_lib import policy_source_hash, resolved_config_hash

REPO = Path(__file__).resolve().parents[1]

SEEDS = 20
CAP = 10000

SCENARIOS = [
    ("balanced_1", "Balanced 1", str(REPO / "configs/balanced.toml"), 1),
    ("balanced_5", "Balanced 5", str(REPO / "configs/balanced.toml"), 5),
    ("balanced_20", "Balanced 20", str(REPO / "configs/balanced.toml"), 20),
    ("solo", "Solo", str(REPO / "configs/solo.toml"), 1),
    ("hard_1", "Hard 1", str(REPO / "configs/hard.toml"), 1),
    ("hard_5", "Hard 5", str(REPO / "configs/hard.toml"), 5),
    ("hard_20", "Hard 20", str(REPO / "configs/hard.toml"), 20),
]
POLICIES = [
    "random",
    "heuristic_noflee",
    "heuristic",
    "heuristic_v2",
    "oracle_avoid",
]


def worker(task: tuple[str, str, int, int]) -> dict[str, Any]:
    config_path, policy, agents, seed = task
    track, food_left = run_episode(
        policy, seed, agents=agents, cap=CAP, config_path=config_path
    )
    classes = [t["class"] for t in track.values()]
    deaths = [t["death_step"] for t in track.values()]
    return {"seed": seed, "classes": classes, "deaths": deaths, "food_left": food_left}


def json_entry(s: dict[str, Any], policy: str) -> dict[str, Any]:
    return {
        "cap_pct": round(s["cap_pct"], 3),
        "ci": [round(s["ci"][0], 3), round(s["ci"][1], 3)],
        "median": int(s["median"]),
        "death_cause": {
            "starvation": int(s["starve"]),
            "hazard": int(s["hazard"]),
            "truncation": int(s["trunc"]),
        },
        "food_left_mean": round(s["food_left_mean"], 3),
        "policy_hash": policy_source_hash(policy),
    }


def run_grid(seed_offset: int) -> dict[str, Any]:
    tasks = []
    for _key, _label, config_path, agents in SCENARIOS:
        for policy in POLICIES:
            for seed in range(seed_offset, seed_offset + SEEDS):
                tasks.append((config_path, policy, agents, seed))

    with Pool() as pool:
        results = pool.map(worker, tasks)

    header = (
        f"{'scenario':<14}{'policy':<18}{'cap%':>7}{'95% CI':>14}"
        f"{'median':>8}{'starve':>8}{'hazard':>8}{'trunc':>8}{'food_left':>10}"
    )
    print(header)
    print("-" * len(header))

    scenarios_json: dict[str, Any] = {}
    idx = 0
    for key, label, config_path, agents in SCENARIOS:
        policies = {}
        for policy in POLICIES:
            rows = results[idx: idx + SEEDS]
            idx += SEEDS
            s = summarize(rows)
            policies[policy] = json_entry(s, policy)
            print(
                f"{label:<14}{policy:<18}"
                f"{s['cap_pct']:6.1f}%  "
                f"[{s['ci'][0]:5.1f},{s['ci'][1]:5.1f}]  "
                f"{s['median']:>8}  "
                f"{s['starve']:>8}{s['hazard']:>8}{s['trunc']:>8}  "
                f"{s['food_left_mean']:8.1f}"
            )
        scenarios_json[key] = {
            "label": label,
            "agents": agents,
            "config": str(Path(config_path).relative_to(REPO)).replace("\\", "/"),
            "config_hash": resolved_config_hash(config_path, agents),
            "policies": policies,
        }
    return scenarios_json


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-offset", type=int, default=None)
    args = parser.parse_args()

    if args.seed_offset is not None:
        sets = {"set": run_grid(args.seed_offset)}
        doc = {
            "generated": datetime.now(UTC).isoformat(),
            "git_describe": _git_describe(),
            "seed_list": list(range(args.seed_offset, args.seed_offset + SEEDS)),
            "step_cap": CAP,
            "scenarios": sets["set"],
        }
        out = REPO / "docs" / "baselines.json"
        out.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {out.relative_to(REPO)} (git_describe {doc['git_describe']})")
        return

    dev = run_grid(0)
    test = run_grid(1000)
    doc = {
        "generated": datetime.now(UTC).isoformat(),
        "git_describe": _git_describe(),
        "step_cap": CAP,
        "scenarios": dev,
        "seed_sets": {
            "dev": {"seed_list": list(range(SEEDS)), "scenarios": dev},
            "test": {
                "seed_list": list(range(1000, 1000 + SEEDS)),
                "scenarios": test,
            },
        },
    }
    out = REPO / "docs" / "baselines.json"
    out.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {out.relative_to(REPO)} (git_describe {doc['git_describe']})")


def _git_describe() -> str:
    return subprocess.run(
        ["git", "describe", "--tags", "--always"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


if __name__ == "__main__":
    main()
