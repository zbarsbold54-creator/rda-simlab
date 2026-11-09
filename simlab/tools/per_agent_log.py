"""Per-agent summaries for D/heuristic/do-nothing on Balanced seed 1000."""
import sys
sys.path.insert(0, ".")
import numpy as np
from dataclasses import replace
from simlab.core.config import load_config
from simlab.core.runner import Runner

def run(policy, checkpoint, label):
    base = load_config("configs/balanced.toml")
    cfg = replace(base, policy=policy, drive="simple" if policy != "learned_torch" else "rda8", agent=replace(base.agent, count=5), policy_checkpoint=checkpoint)
    # for learned_torch with checkpoint, drive should be rda8 (as in training); for heuristic/do_nothing, drive simple
    if policy == "learned_torch":
        cfg = replace(cfg, drive="rda8")
    runner = Runner(cfg, seed=1000, agents=5)
    runner.setup()
    world = runner._world
    # track per agent
    init_pos = {aid: tuple(st["position"]) for aid, st in world._agents.items()}
    dist = {aid: 0.0 for aid in world._agents}
    prev_pos = dict(init_pos)
    near_zero = {aid: 0 for aid in world._agents}
    total_ticks = {aid: 0 for aid in world._agents}
    energy_series = {aid: [] for aid in world._agents}
    for tick in range(10000):
        # get actions via runner (uses policy)
        actions = runner._component_actions()
        # log action magnitude
        for aid, act in actions.items():
            mag = (act.turn**2 + act.thrust**2) ** 0.5
            total_ticks[aid] += 1
            if mag < 0.01:
                near_zero[aid] += 1
        result = runner.run_step(agent_actions=actions, include_observations=False, include_snapshot=False, include_agent_states=False)
        # update distance and energy
        for aid, st in world._agents.items():
            pos = tuple(st["position"])
            if aid in prev_pos:
                dx = pos[0] - prev_pos[aid][0]
                dy = pos[1] - prev_pos[aid][1]
                dist[aid] += (dx*dx + dy*dy) ** 0.5
            prev_pos[aid] = pos
            if tick % 100 == 0:
                energy_series[aid].append(float(st["energy"]))
        if result.terminated:
            break
    print(f"\n=== {label} ===")
    for aid in sorted(world._agents.keys()):
        st = world._agents[aid]
        frac = near_zero[aid] / max(total_ticks[aid], 1)
        print(f"{aid} dist={dist[aid]:.1f} frac_near_zero={frac:.4f} food={st['food_eaten']} energy_series={energy_series[aid][:5]}...{energy_series[aid][-3:] if len(energy_series[aid])>5 else ''} final_energy={st['energy']:.1f} alive={st['is_alive']}")

if __name__ == "__main__":
    run("learned_torch", "logs/overnight/job1_D_100000/final.pt", "D")
    run("heuristic", None, "heuristic")
    run("do_nothing", None, "do_nothing")
