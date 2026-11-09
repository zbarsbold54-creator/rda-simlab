"""Escape physics report and traces for Hard-scenario hazard-death agents.

Reports, from config + world code:
- hazard speed (units/s and per step), hazard turn behaviour
- agent max speed at thrust 1.0, sensor-heuristic flee thrust, agent max turn rate
- whether a chasing hazard can be outrun
- damage per contact step and contact steps to kill from full health

Then runs the Hard scenario (12 hazards, 5 agents), classifies per-agent deaths,
and for 5 hazard-death agents prints a trace every 25 steps over the final 300
steps: agent and nearest-hazard positions, distance, agent speed, hazard speed,
mode and thrust.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simlab.core.config import load_config
from simlab.core.runner import Runner

DT = 0.1
HAZARD_SPEED = 2.0  # units/s, constant in World._move_hazards


def escape_physics() -> None:
    cfg = load_config("configs/hard.toml")
    agent = cfg.agent
    hazard = cfg.hazard
    food = cfg.food
    flee_thrust = 0.8  # HeuristicPolicy flee branch
    eat_thrust = (0.7 * food.eat_dist * agent.max_turn_rate) / agent.speed_factor
    agent_max_speed = agent.speed_factor  # at thrust 1.0
    agent_flee_speed = flee_thrust * agent.speed_factor
    hazard_speed_step = HAZARD_SPEED * DT
    agent_max_speed_step = agent_max_speed * DT
    agent_flee_speed_step = agent_flee_speed * DT
    harm_step = hazard.damage_per_step * DT
    contact_steps_kill = 100.0 / harm_step

    print("=== HARD config (configs/hard.toml) ===")
    print(f"hazard.count={hazard.count} detection_radius={hazard.detection_radius} "
          f"damage_per_step={hazard.damage_per_step}")
    print(f"agent.count={agent.count} initial_energy={agent.initial_energy} "
          f"food_eat={agent.food_eat} eat_dist={food.eat_dist}")
    print(f"agent.speed_factor={agent.speed_factor} max_turn_rate={agent.max_turn_rate}")
    print(f"food.supply_ratio={food.supply_ratio} regrow_rate={food.regrow_rate}")

    print("\n=== Escape physics ===")
    print(f"dt = {DT} s/step")
    print(f"hazard speed = {HAZARD_SPEED} units/s = {hazard_speed_step:.3f} units/step "
          f"(constant; heading snaps to nearest live agent inside detection radius)")
    print("hazard wander turn (no target) = uniform(-0.4,0.4) rad/step")
    print(f"agent max speed at thrust 1.0 = {agent_max_speed} units/s = "
          f"{agent_max_speed_step:.3f} units/step")
    print(f"sensor heuristic flee thrust = {flee_thrust} -> speed {agent_flee_speed} "
          f"units/s = {agent_flee_speed_step:.3f} units/step")
    print(f"agent max turn rate = {agent.max_turn_rate} rad/s = "
          f"{agent.max_turn_rate * DT:.3f} rad/step")
    print(f"derived near-food thrust (0.7*eat_dist*omega/speed_factor) = {eat_thrust:.4f}")
    print(f"damage per contact step = {hazard.damage_per_step:.1f} * dt = "
          f"{harm_step:.3f} health/step")
    print(f"contact steps to kill from full health (100) = {contact_steps_kill:.0f}")
    ratio = agent_max_speed / HAZARD_SPEED
    print(f"agent/hazard straight-line speed ratio = {ratio:.1f}x -> a hazard chased "
          f"within detection radius CAN be outrun by an agent fleeing at thrust 1.0 "
          f"provided it picks a clear direction (no wall/corner/other-hazard trap).")


def hazard_death_traces(n_trace: int = 5) -> None:
    seeds = list(range(20))
    deaths: list[tuple[int, int]] = []  # (seed, agent_index)
    for seed in seeds:
        runner = Runner(load_config("configs/hard.toml"), seed=seed, agents=5)
        runner.setup()
        world = runner._world
        dt = world.dt
        hazard_loss = [0.0] * 5
        starve_loss = [0.0] * 5
        death_step = [None] * 5
        alive = set(range(5))
        for step in range(1, 10001):
            result = runner.run_step(
                include_observations=False, include_agent_states=False
            )
            for aid in list(alive):
                st = world._agents[f"agent_{aid}"]
                pos = st["position"]
                h = 0.0
                for hz in world._hazard_patches:
                    dx = pos[0] - hz["position"][0]
                    dy = pos[1] - hz["position"][1]
                    if dx * dx + dy * dy <= hz["radius"] ** 2:
                        h += hz["damage"] * dt
                hazard_loss[aid] += h
                if st["energy"] <= 0.0:
                    starve_loss[aid] += 0.5
                if not st["is_alive"]:
                    death_step[aid] = step
                    alive.discard(aid)
            if result.terminated:
                break
        for aid in range(5):
            if death_step[aid] is not None and hazard_loss[aid] > starve_loss[aid]:
                deaths.append((seed, aid))
        if len(deaths) >= n_trace:
            break

    print(f"\n=== Traces for first {n_trace} hazard-death agents ===")
    # Re-run each selected seed once to record per-step history for its agent.
    for seed, aid in deaths[:n_trace]:
        runner = Runner(load_config("configs/hard.toml"), seed=seed, agents=5)
        runner.setup()
        world = runner._world
        aid_key = f"agent_{aid}"
        history = []  # per step: (pos, energy, mode, thrust, [hazard positions])
        for step in range(1, 10001):
            result = runner.run_step(
                include_observations=False, include_agent_states=False
            )
            action = runner.action_log[step - 1].get(aid_key)
            st = world._agents[aid_key]
            mode = (action.action_id if action is not None else "")
            history.append(
                (
                    tuple(st["position"]),
                    st["energy"],
                    mode,
                    action.thrust if action is not None else 0.0,
                    tuple(hz["position"] for hz in world._hazard_patches),
                )
            )
            if not st["is_alive"]:
                dstep = step
                break
        else:
            dstep = 10000
        start = max(1, dstep - 300)
        print(f"\nseed={seed} agent={aid} death_step={dstep}")
        for step in range(start, dstep + 1, 25):
            pos, energy, mode, thrust, hazpos = history[step - 1]
            # nearest hazard
            best_d2 = None
            best_h = None
            for h in hazpos:
                d2 = (h[0] - pos[0]) ** 2 + (h[1] - pos[1]) ** 2
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best_h = h
            dist = math.sqrt(best_d2)
            a_speed = 0.0
            # speed is not stored in history; recompute from thrust
            a_speed = thrust * world.speed_factor
            print(
                f"step {step:5d} agent=({pos[0]:6.2f},{pos[1]:6.2f}) "
                f"hz=({best_h[0]:6.2f},{best_h[1]:6.2f}) dist={dist:6.2f} "
                f"energy={energy:6.2f} mode={mode or 'wander':14s} "
                f"thrust={thrust:4.2f} agent_speed={a_speed:5.2f} hz_speed={HAZARD_SPEED:.1f}"
            )


if __name__ == "__main__":
    escape_physics()
    hazard_death_traces()
