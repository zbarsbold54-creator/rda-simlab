"""Step 2 (World v1): hazard-scaling diagnosis.

No config changes. Sweeps hazard count across agent counts for the sensor
heuristic and random policies (15 seeds, cap 10000), reporting cap percent
and the death-cause split. Also confirms the chase rule: each hazard
independently targets the nearest alive agent within its detection radius.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from episode_lib import split

from simlab.core.config import load_config
from simlab.core.runner import Runner

CAP = 10000
SEEDS = 15
CONFIG = "configs/balanced.toml"
NOMINAL_HAZARDS = 4


def run_seed(
    policy: str, seed: int, agents: int, hazard_count: int
) -> tuple[dict[str, dict[str, Any]], float]:
    base = load_config(CONFIG)
    cfg = replace(
        base,
        policy=policy,
        agent=replace(base.agent, count=agents),
        hazard=replace(base.hazard, count=hazard_count),
    )
    return _run(cfg, seed, agents)


def _run(cfg: Any, seed: int, agents: int) -> tuple[dict[str, dict[str, Any]], float]:
    runner = Runner(cfg, seed=seed, agents=agents)
    runner.setup()
    world = runner._world
    dt = world.dt
    track: dict[str, dict[str, Any]] = {}
    for aid in world._agents:
        track[aid] = {"hazard_loss": 0.0, "starve_loss": 0.0, "death_step": None, "class": None}
    alive = set(world._agents)
    for step in range(1, CAP + 1):
        runner.run_step(include_observations=False, include_agent_states=False)
        for aid in list(alive):
            state = world._agents[aid]
            t = track[aid]
            pos = state["position"]
            hazard_loss = 0.0
            for hazard in world._hazard_patches:
                dx = pos[0] - hazard["position"][0]
                dy = pos[1] - hazard["position"][1]
                if dx * dx + dy * dy <= hazard["radius"] ** 2:
                    hazard_loss += hazard["damage"] * dt
            t["hazard_loss"] += hazard_loss
            if state["energy"] <= 0.0:
                t["starve_loss"] += 0.5
            if not state["is_alive"]:
                t["death_step"] = step
                alive.discard(aid)
        if not alive:
            break
    food_left = sum(p["amount"] for p in world._food_patches)
    for t in track.values():
        if t["death_step"] is None:
            t["death_step"] = CAP
            t["class"] = "truncation"
        else:
            t["class"] = "hazard" if t["hazard_loss"] > t["starve_loss"] else "starvation"
    return track, food_left


def worker(task: tuple[str, int, int, int]) -> tuple[dict[str, dict[str, Any]], float]:
    policy, seed, agents, hazard_count = task
    return run_seed(policy, seed, agents, hazard_count)


def main() -> None:
    print("Chase rule: each hazard independently targets the nearest alive agent within its detection_radius (confirmed in world._move_hazards).\n")

    cells = [
        (1, [1, 2, NOMINAL_HAZARDS]),
        (5, [NOMINAL_HAZARDS, 2 * NOMINAL_HAZARDS]),
        (20, [NOMINAL_HAZARDS, 2 * NOMINAL_HAZARDS, 4 * NOMINAL_HAZARDS]),
    ]
    policies = ["heuristic", "random"]
    tasks = [
        (p, s, a, h)
        for a, hazards in cells
        for h in hazards
        for p in policies
        for s in range(SEEDS)
    ]
    print(f"running {len(tasks)} episodes across {SEEDS} seeds (cap {CAP})...")
    with Pool() as pool:
        results = pool.map(worker, tasks)
    idx = 0
    header = f"{'policy':<10}{'ag':>4}{'haz':>5}{'cap%':>7}{'starve':>7}{'hazard':>7}{'trunc':>7}"
    print(header)
    for a, hazards in cells:
        for h in hazards:
            for p in policies:
                chunk = results[idx: idx + SEEDS]
                idx += SEEDS
                causes = []
                cap_count = 0
                total = 0
                for track, _ in chunk:
                    for t in track.values():
                        total += 1
                        causes.append(t["class"])
                        if t["class"] == "truncation":
                            cap_count += 1
                st, hz, tr = split(causes)
                print(
                    f"{p:<10}{a:>4}{h:>5}{100.0 * cap_count / total:>7.1f}"
                    f"{st:>7}{hz:>7}{tr:>7}"
                )


if __name__ == "__main__":
    main()
