"""Step 1: foraging diagnosis for the sensor heuristic (no changes yet).

Runs the Balanced preset, 5 agents, 10 seeds with the historical sensor
heuristic (heuristic_noflee) and instruments every live step via the shared
compute_actions pipeline, so we observe the exact obs, action and per-agent
mode the policy actually produced.

Per seed prints:
  - group metrics (starved vs survivors): mode fractions, eat ticks, food
    units eaten, energy in vs out, distance travelled, mean thrust, and the
    fraction of steps with a food ray hit.
  - a trace every 100 steps over the last 1500 steps for up to 3 starved
    agents: step, position, energy, mode, nearest food ray distance, target
    patch id and amount, and whether eat was issued.
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simlab.core.config import load_config
from simlab.core.pipeline import compute_actions
from simlab.core.registry import get, load_plugins
from simlab.core.runner import Runner
from simlab.core.types import Action

MODE_EAT = "eat"
MODE_SEEK = "seek"
MODE_WANDER = "wander"
MODE_FLEE = "flee"
MODES = (MODE_EAT, MODE_SEEK, MODE_WANDER, MODE_FLEE)

THREAT_RANGE = 20.0
TRACE_STEP = 100
TRACE_MAX = 15  # 15 * 100 = last 1500 steps


def nearest_food(obs: Any) -> tuple[float | None, int | None]:
    best = None
    best_index = None
    for index, (dist, kind) in enumerate(zip(obs.rays, obs.ray_types)):
        if kind == 1 and (best is None or dist < best):
            best = dist
            best_index = index
    return best, best_index


def threat_of(obs: Any) -> float:
    hazards = [d for d, k in zip(obs.rays, obs.ray_types) if k == 2]
    nearest = min(hazards) if hazards else THREAT_RANGE
    return max(0.0, min(1.0, 1.0 - nearest / THREAT_RANGE))


def mode_of(obs: Any, action: Action, flee_enabled: bool) -> str:
    if flee_enabled and threat_of(obs) > 0.3:
        return MODE_FLEE
    if action.eat:
        return MODE_EAT
    best, _ = nearest_food(obs)
    if best is not None:
        return MODE_SEEK
    return MODE_WANDER


def find_target_patch(
    world: Any, aid: str, obs: Any, best_index: int | None
) -> tuple[str, float]:
    """Reconstruct which food patch the chosen food ray would hit first."""
    if best_index is None:
        return "", 0.0
    state = world._agents[aid]
    heading = state["heading"]
    count = len(obs.rays)
    offsets = [
        (-135.0 + 270.0 * i / (count - 1)) if count > 1 else 0.0
        for i in range(count)
    ]
    angle = heading + math.radians(offsets[best_index])
    dx, dy = math.cos(angle), math.sin(angle)
    px, py = state["position"]
    best_proj = None
    best_patch = None
    for idx, patch in enumerate(world._food_patches):
        if patch.get("amount", 0.0) <= 0.0:
            continue
        vx = patch["position"][0] - px
        vy = patch["position"][1] - py
        proj = vx * dx + vy * dy
        perp = abs(vx * dy - vy * dx)
        if (
            proj > 0.0
            and perp <= 0.75
            and proj <= 20.0
            and (best_proj is None or proj < best_proj)
        ):
            best_proj = proj
            best_patch = (idx, float(patch["amount"]))
    if best_patch is None:
        return "", 0.0
    return str(best_patch[0]), best_patch[1]


def empty_metrics(aid: str) -> dict[str, Any]:
    return {
        "id": aid,
        "mode": {m: 0 for m in MODES},
        "eat_ticks": 0,
        "wasted_eat": 0,
        "food_units": 0.0,
        "energy_in_food": 0.0,
        "energy_spent": 0.0,
        "distance": 0.0,
        "thrust_sum": 0.0,
        "live_steps": 0,
        "food_ray_hits": 0,
        "food_left_at_death": None,
        "hazard_loss": 0.0,
        "starve_loss": 0.0,
        "last_drain": 0.0,
        "energy_before": 0.0,
        "survival": 0,
        "class": "",
        "energy_at_death": 0.0,
        "trace": [],
    }


def run_seed(
    seed: int,
    agents: int = 5,
    cap: int = 10000,
    flee_enabled: bool = False,
    policy: str = "heuristic_noflee",
    hazard_count: int = 4,
) -> dict[str, dict[str, Any]]:
    base = load_config("configs/balanced.toml")
    cfg = replace(
        base,
        policy=policy,
        agent=replace(base.agent, count=agents),
        hazard=replace(base.hazard, count=hazard_count),
    )
    runner = Runner(cfg, seed=seed, agents=agents)
    runner.setup()
    world = runner._world
    dt = world.dt

    load_plugins()
    pol_cls = get("policy", policy)
    try:
        pol = pol_cls(seed=seed)
    except TypeError:
        pol = pol_cls()
    drive = get("drive", cfg.drive)()
    drive.reset(agents, world._rng)

    track = {aid: empty_metrics(aid) for aid in world._agents}
    alive = set(world._agents)

    for step in range(1, cap + 1):
        actions, _, observations = compute_actions(world, pol, drive, step - 1)
        for aid in list(alive):
            m = track[aid]
            obs = observations[aid]
            action = actions[aid]
            mode = mode_of(obs, action, flee_enabled)
            m["mode"][mode] += 1
            if action.eat:
                m["eat_ticks"] += 1
                if m["energy_before"] >= float(world._agents[aid]["max_energy"]) * 0.9:
                    m["wasted_eat"] += 1
            speed = action.thrust * 8.0
            drain = (0.08 + 0.04 * speed) * dt
            m["last_drain"] = drain
            m["energy_spent"] += drain
            m["distance"] += speed * dt
            m["thrust_sum"] += action.thrust
            m["live_steps"] += 1
            best, best_index = nearest_food(obs)
            if best is not None:
                m["food_ray_hits"] += 1
            m["energy_before"] = world._agents[aid]["energy"]
            if step % TRACE_STEP == 0:
                tid, tamount = find_target_patch(world, aid, obs, best_index)
                m["trace"].append(
                    (
                        step,
                        round(world._agents[aid]["position"][0], 1),
                        round(world._agents[aid]["position"][1], 1),
                        round(m["energy_before"], 1),
                        mode,
                        round(best, 2) if best is not None else None,
                        tid,
                        tamount,
                        bool(action.eat),
                    )
                )
                if len(m["trace"]) > TRACE_MAX:
                    m["trace"].pop(0)

        result = world.step(actions, include_observations=False, include_snapshot=False)
        for aid in list(alive):
            state = world._agents[aid]
            m = track[aid]
            gained = max(0.0, state["energy"] - m["energy_before"] + m["last_drain"])
            m["food_units"] += gained
            m["energy_in_food"] += gained
            pos = state["position"]
            for hazard in world._hazard_patches:
                dx = pos[0] - hazard["position"][0]
                dy = pos[1] - hazard["position"][1]
                if dx * dx + dy * dy <= hazard["radius"] ** 2:
                    m["hazard_loss"] += hazard["damage"] * dt
            if state["energy"] <= 0.0:
                m["starve_loss"] += 0.5
            if not state["is_alive"]:
                m["survival"] = step
                m["energy_at_death"] = state["energy"]
                m["class"] = (
                    "hazard" if m["hazard_loss"] > m["starve_loss"] else "starvation"
                )
                m["food_left_at_death"] = sum(
                    p["amount"] for p in world._food_patches
                )
                alive.discard(aid)
        if result.terminated:
            break

    for m in track.values():
        if m["survival"] == 0:
            m["survival"] = cap
            m["class"] = "truncation"
    return track


def fmt_group(metrics: list[dict[str, Any]], label: str) -> None:
    if not metrics:
        print(f"  {label}: none")
        return
    n = len(metrics)
    total_live = sum(x["live_steps"] for x in metrics) or 1
    mean = lambda key: sum(x[key] for x in metrics) / n
    per_agent_live = total_live / n
    mode = {m: sum(x["mode"][m] for x in metrics) / total_live for m in MODES}
    print(f"  {label} (n={n}):")
    print("    mode fractions: " + ", ".join(f"{m}={mode[m]:.3f}" for m in MODES))
    print(
        f"    eat ticks/agent={mean('eat_ticks'):.1f}  food units/agent={mean('food_units'):.1f}"
    )
    print(
        f"    energy in (food)/agent={mean('energy_in_food'):.1f}  "
        f"energy spent/agent={mean('energy_spent'):.1f}"
    )
    print(
        f"    distance/agent={mean('distance'):.1f}  "
        f"mean thrust={mean('thrust_sum')/per_agent_live:.3f}"
    )
    print(
        f"    food ray hit fraction={sum(x['food_ray_hits'] for x in metrics)/total_live:.3f}"
    )


def report_starved(m: dict[str, Any]) -> None:
    n = m["live_steps"] or 1
    mode = {k: m["mode"][k] / n for k in MODES}
    print(f"  starved agent {m['id']}:")
    print(f"    death step={m['survival']}  food eaten={m['food_units']:.1f} units")
    print(
        f"    food left at death={m['food_left_at_death']:.1f}  "
        f"max_food=400"
    )
    print(
        f"    energy gained (food)={m['energy_in_food']:.1f}  "
        f"energy spent={m['energy_spent']:.1f}"
    )
    print("    mode fractions: " + ", ".join(f"{k}={mode[k]:.3f}" for k in MODES))
    print(
        f"    food ray hit fraction={m['food_ray_hits']/n:.3f}  "
        f"eat ticks={m['eat_ticks']}  wasted_eat(energy>=0.9max)={m['wasted_eat']}"
    )
    print("      step   x     y    E  mode  foodray tgt amt eat")
    for row in m["trace"]:
        print(
            f"      {row[0]:5d} {row[1]:5.1f} {row[2]:5.1f} {row[3]:5.1f} "
            f"{row[4]:6s} {row[5]!s:5s} {row[6]:3s} {row[7]:4.1f} {int(row[8])}"
        )


def main() -> None:
    # Solo diagnosis: 1 agent, 1 hazard, Balanced (supply_ratio 2.0, radius
    # 16), 20 seeds, sensor heuristic with flee.
    seeds = list(range(20))
    agents = 1
    hazard_count = 1
    starved_total = 0
    other_total = 0
    for seed in seeds:
        track = run_seed(
            seed,
            agents=agents,
            hazard_count=hazard_count,
            flee_enabled=True,
            policy="heuristic",
        )
        m = next(iter(track.values()))
        cls = m["class"]
        print(f"=== seed {seed}: class={cls} survival={m['survival']} ===")
        if cls == "starvation":
            starved_total += 1
            report_starved(m)
        else:
            other_total += 1
    print(f"\nAGGREGATE over {len(seeds)} seeds: starved={starved_total} other={other_total}")
    print("(solo = 1 agent, 1 hazard, supply_ratio 2.0, radius 16, cap 10000)")


if __name__ == "__main__":
    main()
