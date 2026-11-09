"""Step 1 (World v1.4): diagnose the identical 20 hazard deaths.

For Balanced 5-agent, 20-seed, sensor-heuristic runs, print per hazard-death
agent: seed, step of death, distance to the nearest hazard at reset, number of
hazard contacts, and flee-mode fraction in the final 200 steps. Also confirm
whether the same seeds die at every supply_ratio, and print a death-step
histogram (bins of 500).
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simlab.core.config import load_config
from simlab.core.runner import Runner

CAP = 10000
SEEDS = 20
AGENTS = 5
CONFIG = "configs/balanced.toml"
RATIOS = [0.8, 1.0, 1.2, 1.5]


def run_detail(seed: int) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    base = load_config(CONFIG)
    cfg = replace(
        base, policy="heuristic", agent=replace(base.agent, count=AGENTS)
    )
    runner = Runner(cfg, seed=seed, agents=AGENTS)
    runner.setup()
    world = runner._world
    dt = world.dt
    hazards = [tuple(h["position"]) for h in world._hazard_patches]
    reset_dist: dict[str, float] = {}
    for aid, st in world._agents.items():
        d2 = min(
            (st["position"][0] - hx) ** 2 + (st["position"][1] - hy) ** 2
            for hx, hy in hazards
        )
        reset_dist[aid] = d2 ** 0.5
    track: dict[str, dict[str, Any]] = {}
    for aid in world._agents:
        track[aid] = {
            "hazard_loss": 0.0,
            "starve_loss": 0.0,
            "contact_steps": [],
            "modes": [],
            "death_step": None,
            "energy_at_death": None,
        }
    alive = set(world._agents)
    for step in range(1, CAP + 1):
        result = runner.run_step(include_observations=False, include_agent_states=False)
        actions = runner.action_log[-1] if runner.action_log else {}
        for aid in list(alive):
            st = world._agents[aid]
            t = track[aid]
            pos = st["position"]
            damage = 0.0
            for h in world._hazard_patches:
                dx = pos[0] - h["position"][0]
                dy = pos[1] - h["position"][1]
                if dx * dx + dy * dy <= h["radius"] ** 2:
                    damage += h["damage"] * dt
            if damage > 0.0:
                t["hazard_loss"] += damage
                t["contact_steps"].append(step)
            if st["energy"] <= 0.0:
                t["starve_loss"] += 0.5
            act = actions.get(aid)
            t["modes"].append(act.action_id if act else "")
            if not st["is_alive"]:
                t["death_step"] = step
                t["energy_at_death"] = st["energy"]
                alive.discard(aid)
        if result.terminated:
            break
    return track, reset_dist


def run_ratio(ratio: float) -> tuple[float, set[int]]:
    from episode_lib import run_episode

    seeds: set[int] = set()
    for seed in range(SEEDS):
        track, _ = run_episode(
            "heuristic",
            seed,
            agents=AGENTS,
            cap=CAP,
            config_path=CONFIG,
            supply_ratio=ratio,
        )
        if any(t["class"] == "hazard" for t in track.values()):
            seeds.add(seed)
    return ratio, seeds


def main() -> None:
    print("=== Seed sets with hazard deaths by supply_ratio (heuristic, 5 agents) ===")
    with Pool() as pool:
        ratio_results = pool.map(run_ratio, RATIOS)
    per_ratio = {r: seeds for r, seeds in ratio_results}
    for r in RATIOS:
        print(f"ratio {r}: hazard-death seeds = {sorted(per_ratio[r])}")
    all_same = all(
        per_ratio[r] == per_ratio[RATIOS[0]] for r in RATIOS
    )
    print(f"same hazard-death seed set across ratios: {all_same}")

    print("\n=== Detail: hazard-death agents (heuristic, Balanced, 5 agents, 20 seeds) ===")
    header = f"{'seed':<5}{'agent':<9}{'death':<9}{'reset_dist':<12}{'contacts':<9}{'flee%_last200':<16}"
    print(header)
    print("-" * len(header))
    deathsteps: list[int] = []
    with Pool() as pool:
        details = pool.map(run_detail, range(SEEDS))
    nrow = 0
    for seed, (track, reset_dist) in enumerate(details):
        for aid, t in track.items():
            if t["death_step"] is not None and t["hazard_loss"] > t["starve_loss"]:
                deathsteps.append(t["death_step"])
                steps = t["modes"][max(0, t["death_step"] - 201): t["death_step"] - 1]
                flee = (
                    100.0 * sum(1 for m in steps if m == "heuristic_flee") / len(steps)
                    if steps
                    else 0.0
                )
                nrow += 1
                print(
                    f"{seed:<5}{aid:<9}{t['death_step']:<9}"
                    f"{reset_dist[aid]:<12.2f}{len(t['contact_steps']):<9}{flee:<16.1f}"
                )
    print(f"\ntotal hazard-death agents: {nrow}")
    print("\n=== Death-step histogram (bins of 500) ===")
    bins = Counter((d - 1) // 500 for d in deathsteps)
    for b in sorted(bins):
        print(f"step {b * 500 + 1}-{(b + 1) * 500}: {bins[b]}")


if __name__ == "__main__":
    main()
