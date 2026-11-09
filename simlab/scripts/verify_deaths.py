"""Step 1: verify death-cause classification for the random policy."""

from __future__ import annotations

import sys
from statistics import median

sys.path.insert(0, "C:\\AI\\SimulationCode")

from episode_lib import run_episode, split

SEEDS = range(1, 11)
AGENTS = 5
CAP = 10000


def main() -> None:
    all_outcomes: list[str] = []
    all_surv: list[int] = []
    print(
        "seed agent survival_step energy_at_death health_at_death food_eaten "
        "contacts_final50 hazard_loss starve_loss class"
    )
    for seed in SEEDS:
        track, food_left = run_episode("random", seed, AGENTS, CAP)
        for aid, t in track.items():
            all_outcomes.append(t["class"])
            all_surv.append(t["death_step"])
            print(
                f"{seed:4d} {aid:8s} {t['death_step']:12d} "
                f"{t['energy_at_death']:15.2f} {t['health_at_death']:14.2f} "
                f"{t['food_eaten']:9d} {t.get('contacts_final_50', 0):14d} "
                f"{t['hazard_loss']:10.2f} {t['starve_loss']:9.2f} {t['class']}"
            )
        print(f"  seed {seed} food_left={food_left:.1f}")
    s = split(all_outcomes)
    print(
        f"random total agents={len(all_outcomes)} median_survival={median(all_surv):.0f} "
        f"starvation/hazard/truncation={s[0]}/{s[1]}/{s[2]}"
    )
    print(
        "old_classifier would_have_labeled: starvation only if energy hit zero "
        "at the death step; otherwise hazard -> most random deaths were labeled hazard"
    )


if __name__ == "__main__":
    main()
