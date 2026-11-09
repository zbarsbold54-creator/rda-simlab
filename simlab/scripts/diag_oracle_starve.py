"""Diagnose oracle_avoid starvation on Balanced 5 and Hard 5.

For each scenario, runs oracle_avoid over seeds 0..19 (dev protocol) and, for
the first 5 starved agents, prints mode fractions, eat ticks, the fraction of
steps with no valid food target, how a target is chosen when every patch lies
inside some hazard's detection radius, and a trace every 100 steps over the
last 1000 steps.
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simlab.core.config import load_config
from simlab.core.runner import Runner

DEV_SEEDS = list(range(20))


def run_scenario(script: str, agents: int = 5) -> list[dict[str, Any]]:
    config_path = str(Path(__file__).resolve().parents[1] / script)
    base = load_config(config_path)
    starved: list[dict[str, Any]] = []
    for seed in DEV_SEEDS:
        cfg = replace(base, policy="oracle_avoid", agent=replace(base.agent, count=agents))
        runner = Runner(cfg, seed=seed, agents=agents)
        runner.setup()
        policy = runner._policy
        world = runner._world
        dt = world.dt
        alive = set(world._agents)
        hazard_loss = {a: 0.0 for a in alive}
        starve_loss = {a: 0.0 for a in alive}
        hist: dict[str, list[dict[str, Any]]] = {a: [] for a in alive}
        for step in range(1, 10001):
            runner.run_step(include_observations=False, include_agent_states=False)
            actions = runner.action_log[-1]
            food = [
                (tuple(p["position"]), p["amount"]) for p in world._food_patches
            ]
            hazards = [tuple(p["position"]) for p in world._hazard_patches]
            for aid in list(alive):
                state = world._agents[aid]
                pos = state["position"]
                action = actions.get(aid)
                aid_id = action.action_id if action is not None else ""
                nhz = None
                dmin = None
                for h in hazards:
                    d2 = (pos[0] - h[0]) ** 2 + (pos[1] - h[1]) ** 2
                    if dmin is None or d2 < dmin:
                        dmin = d2
                if dmin is not None:
                    nhz = dmin ** 0.5
                target = None
                target_amount = None
                if aid_id not in ("oracle_avoid_flee", "oracle_avoid_wander"):
                    target = policy._select_food(pos, food, hazards)
                    if target is not None:
                        target_amount = next(
                            (a for p, a in food if p[0] == target[0] and p[1] == target[1]),
                            None,
                        )
                hist[aid].append(
                    {
                        "step": step,
                        "id": aid_id,
                        "energy": state["energy"],
                        "nhz": nhz,
                        "target": target,
                        "target_amount": target_amount,
                    }
                )
                for hz in world._hazard_patches:
                    dx = pos[0] - hz["position"][0]
                    dy = pos[1] - hz["position"][1]
                    if dx * dx + dy * dy <= hz["radius"] ** 2:
                        hazard_loss[aid] += hz["damage"] * dt
                if state["energy"] <= 0.0:
                    starve_loss[aid] += 0.5
                if not state["is_alive"]:
                    cls = (
                        "hazard"
                        if hazard_loss[aid] > starve_loss[aid]
                        else "starvation"
                    )
                    if cls == "starvation":
                        starved.append(
                            {
                                "seed": seed,
                                "aid": aid,
                                "death_step": step,
                                "food_eaten": state["food_eaten"],
                                "food_left": sum(p["amount"] for p in world._food_patches),
                                "log": hist[aid],
                            }
                        )
                    alive.discard(aid)
            if not alive:
                break
    return starved


def main() -> None:
    for name, script in (
        ("Balanced", "configs/balanced.toml"),
        ("Hard", "configs/hard.toml"),
    ):
        print(f"\n===== {name} 5 ===")
        starved = run_scenario(script)
        print(f"starved agents found: {len(starved)}")
        for s in starved[:5]:
            log = s["log"]
            n = len(log) or 1
            c = Counter(e["id"] for e in log)
            flee = c.get("oracle_avoid_flee", 0)
            wander = c.get("oracle_avoid_wander", 0)
            eat = c.get("oracle_avoid_eat", 0)
            seek = c.get("oracle_avoid_seek", 0) + c.get("oracle_avoid_close", 0)
            no_target = wander
            print(
                f"\nseed={s['seed']} agent={s['aid']} death={s['death_step']} "
                f"food_eaten={s['food_eaten']} food_left={s['food_left']:.0f}"
            )
            print(
                f"  mode%: seek={100.0*seek/n:.1f} eat={100.0*eat/n:.1f} "
                f"flee={100.0*flee/n:.1f} wander={100.0*wander/n:.1f}"
            )
            print(
                f"  eat_ticks={eat} no_valid_target_steps={no_target} "
                f"({100.0*no_target/n:.1f}%)"
            )
            seeklike = [
                e
                for e in log
                if e["id"] in ("oracle_avoid_seek", "oracle_avoid_close", "oracle_avoid_eat")
            ]
            if seeklike:
                last = seeklike[-1]
                print(
                    f"  last seek/eat/close target={last['target']} "
                    f"amount={last['target_amount']} nearest_hazard_dist={last['nhz']}"
                )
            if len(log) >= 1000:
                print("  trace (last 1000 steps, every 100):")
                for e in log[-1000::100]:
                    nhz = "   -  " if e["nhz"] is None else f"{e['nhz']:6.2f}"
                    mode = e["id"].replace("oracle_avoid_", "")
                    print(
                        f"    step {e['step']:6d} energy={e['energy']:6.2f} "
                        f"mode={mode:6s} nhz={nhz} target={e['target']} "
                        f"amount={e['target_amount']}"
                    )


if __name__ == "__main__":
    main()
