"""Focused oracle_avoid starvation debug: Balanced 5, seed 0.

Tracks the first starved agent and, in 500-step windows, reports the fraction
of eat-mode steps at which the nearest non-empty patch was out of eat range
(a proxy for failed eat attempts), plus a per-100-step trace. It also prints
the real successful-eat count (food_eaten) so a miss rate can be inferred.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simlab.core.config import load_config
from simlab.core.runner import Runner


def main() -> None:
    cfg = load_config("configs/balanced.toml")
    cfg = replace(
        cfg,
        policy="oracle_avoid",
        agent=replace(cfg.agent, count=5),
    )
    runner = Runner(cfg, seed=0, agents=5)
    runner.setup()
    world = runner._world
    eat_dist = world.eat_dist

    target_aid = None
    w_start = 0
    w_eat = 0
    w_miss = 0
    w_eaten0 = None
    for step in range(1, 10001):
        runner.run_step(include_observations=False, include_agent_states=False)
        actions = runner.action_log[-1]
        for aid, state in list(world._agents.items()):
            if not state["is_alive"]:
                continue
            action = actions.get(aid)
            aid_id = action.action_id if action is not None else ""
            if target_aid is None and aid_id.startswith("oracle_avoid"):
                target_aid = aid
            if aid != target_aid:
                continue
            pos = state["position"]
            food = world._food_patches
            best_d = None
            best_a = None
            for p in food:
                if p["amount"] <= 0.0:
                    continue
                dx = pos[0] - p["position"][0]
                dy = pos[1] - p["position"][1]
                d2 = dx * dx + dy * dy
                if best_d is None or d2 < best_d:
                    best_d = d2
                    best_a = p["amount"]
            dist = (best_d ** 0.5) if best_d is not None else None
            if (step - 1) // 500 != w_start:
                miss_pct = 100.0 * w_miss / w_eat if w_eat else 0.0
                print(
                    f"window {w_start*500:5d}-{(w_start+1)*500:5d} "
                    f"eat_attempts={w_eat} miss_outofrange_pct={miss_pct:.1f} "
                    f"eaten_total={state['food_eaten']}"
                )
                w_start = (step - 1) // 500
                w_eat = 0
                w_miss = 0
            if aid_id == "oracle_avoid_eat":
                w_eat += 1
                if dist is not None and dist > eat_dist:
                    w_miss += 1
            if step % 100 == 0:
                mode = aid_id.replace("oracle_avoid_", "")
                print(
                    f"  step {step:6d} en={state['energy']:6.2f} mode={mode:6s} "
                    f"nfood_dist={dist if dist is None else round(dist,2)} "
                    f"nfood_amt={best_a} eaten={state['food_eaten']}"
                )
            if state["energy"] <= 0.0:
                miss_pct = 100.0 * w_miss / w_eat if w_eat else 0.0
                print(
                    f"DEAD at step {step}, eaten={state['food_eaten']}, "
                    f"final window miss_pct={miss_pct:.1f}"
                )
                return
    if w_eat:
        miss_pct = 100.0 * w_miss / w_eat if w_eat else 0.0
        print(
            f"window {w_start*500:5d}-end eat_attempts={w_eat} "
            f"miss_outofrange_pct={miss_pct:.1f} eaten_total={w_eaten0}"
        )


if __name__ == "__main__":
    main()
