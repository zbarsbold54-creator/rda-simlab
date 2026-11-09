"""Step 2: survival report for sensor heuristic, oracle heuristic and random."""

from __future__ import annotations

import sys
from statistics import median

sys.path.insert(0, "C:\\AI\\SimulationCode")

from episode_lib import run_episode, split

SEEDS = range(1, 11)
AGENTS = 5
CAP = 10000


def main() -> None:
    print("survival_report agents=5 seeds=10 step_cap=10000 config=default")
    for policy in ("heuristic", "oracle_heuristic", "random"):
        outcomes: list[str] = []
        surv: list[int] = []
        for seed in SEEDS:
            track, _food_left = run_episode(policy, seed, AGENTS, CAP)
            for t in track.values():
                outcomes.append(t["class"])
                surv.append(t["death_step"])
        s = split(outcomes)
        med = median(surv)
        cap_pct = 100.0 * s[2] / len(surv)
        print(
            f"{policy:18s} median_survival={med:6.0f} cap_pct={cap_pct:5.1f} "
            f"starvation/hazard/truncation={s[0]}/{s[1]}/{s[2]}"
        )


if __name__ == "__main__":
    main()
