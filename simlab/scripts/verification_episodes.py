"""Print per-seed heuristic and random survival diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simlab.core.config import load_config
from simlab.core.runner import Runner
from simlab.interfaces.policy import HeuristicPolicy, RandomPolicy


def run_episode(seed: int, policy_name: str) -> tuple[int, str, int, int]:
    config = load_config()
    runner = Runner(config, seed=seed, agents=1)
    runner.setup()
    world = runner._world
    policy = HeuristicPolicy() if policy_name == "heuristic" else RandomPolicy(seed)
    energy_zero_step: int | None = None
    cause = "step-limit truncation"

    for step in range(1, runner._max_steps + 1):
        agent_id, state = next(iter(world._agents.items()))
        if isinstance(policy, HeuristicPolicy):
            action = policy.select_action(
                state,
                {"food_locations": [p["position"] for p in world._food_patches]},
                [],
                step - 1,
            )
        else:
            action = policy.act(None)
        previous_energy = state["energy"]
        runner.run_step(
            {agent_id: action},
            include_observations=False,
            include_snapshot=False,
            include_agent_states=False,
        )
        current = world._agents[agent_id]
        if previous_energy > 0.0 and current["energy"] <= 0.0:
            energy_zero_step = step
        if not current["is_alive"]:
            cause = "starvation" if energy_zero_step is not None else "hazard damage"
            break

    return (
        runner._step_count,
        cause,
        int(world._agents["agent_0"]["food_eaten"]),
        runner._max_steps,
    )


def main() -> None:
    print("seed policy survival_steps cause food_eaten configured_max_steps")
    for seed in range(1, 21):
        for policy in ("heuristic", "random"):
            survival, cause, food_eaten, max_steps = run_episode(seed, policy)
            print(
                f"{seed:02d} {policy:9s} {survival:14d} {cause:22s} "
                f"{food_eaten:10d} {max_steps:20d}"
            )


if __name__ == "__main__":
    main()
