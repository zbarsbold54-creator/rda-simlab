"""Aggregation of per-seed evaluation results for the Research cockpit.

Input rows match the per-seed output of ``simlab.eval_lib.run_episode_metrics``
(a dict with ``seed`` and ``per_agent`` list). The module computes per-seed
cap%/median/starve/hazard/trunc rows, per-arm mean +/- spread summaries, and
differences against a control arm. Torch is never imported here.
"""

from __future__ import annotations

import csv
import io
import json
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TRUNCATION = "truncation"

# Default control arm for RDA variants (rda_const is the control).
RDA_CONTROL = "rda_const"


def default_control(arm_name: str) -> str | None:
    if arm_name.startswith("rda") and arm_name != RDA_CONTROL:
        return RDA_CONTROL
    return None


@dataclass
class SeedRow:
    seed: int
    cap_pct: float
    median_survival: float
    starve: int
    hazard: int
    trunc: int
    food_eaten: float = 0.0
    hazard_contacts: float = 0.0
    mean_energy: float = 0.0


def rows_from_seed_results(results: Iterable[dict[str, Any]]) -> list[SeedRow]:
    """Summarize per-seed eval dicts into SeedRow list (sorted by seed)."""
    rows: list[SeedRow] = []
    for r in results:
        agents = r.get("per_agent") or []
        n = max(1, len(agents))
        classes = [a.get("class", TRUNCATION) for a in agents]
        trunc = sum(1 for c in classes if c == TRUNCATION)
        starve = sum(1 for c in classes if c == "starvation")
        hazard = sum(1 for c in classes if c == "hazard")
        deaths = [
            a.get("death_step") for a in agents if a.get("death_step") is not None
        ]
        median = statistics.median(deaths) if deaths else float("nan")
        rows.append(
            SeedRow(
                seed=int(r.get("seed", 0)),
                cap_pct=100.0 * trunc / n,
                median_survival=round(float(median), 3),
                starve=starve,
                hazard=hazard,
                trunc=trunc,
                food_eaten=round(
                    float(
                        statistics.mean(
                            [a.get("food_eaten", 0.0) or 0.0 for a in agents]
                        )
                    ),
                    3,
                )
                if agents
                else 0.0,
                hazard_contacts=round(
                    float(
                        statistics.mean(
                            [a.get("hazard_contacts", 0.0) or 0.0 for a in agents]
                        )
                    ),
                    3,
                )
                if agents
                else 0.0,
                mean_energy=round(
                    float(
                        statistics.mean(
                            [a.get("mean_energy", 0.0) or 0.0 for a in agents]
                        )
                    ),
                    3,
                )
                if agents
                else 0.0,
            )
        )
    rows.sort(key=lambda r: r.seed)
    return rows


METRIC_KEYS = (
    "cap_pct",
    "median_survival",
    "starve",
    "hazard",
    "trunc",
    "food_eaten",
    "hazard_contacts",
    "mean_energy",
)


def summarize_rows(rows: list[SeedRow]) -> dict[str, tuple[float, float]]:
    """Mean +/- spread (std) for every numeric metric across seeds."""
    result: dict[str, tuple[float, float]] = {}
    for key in METRIC_KEYS:
        values = [getattr(r, key) for r in rows if getattr(r, key) == getattr(r, key)]
        if not values:
            result[key] = (float("nan"), float("nan"))
            continue
        result[key] = (
            round(float(statistics.mean(values)), 3),
            round(float(statistics.pstdev(values)), 3) if len(values) > 1 else 0.0,
        )
    return result


def difference_rows(
    arm_rows: list[SeedRow], control_rows: list[SeedRow]
) -> list[dict[str, Any]]:
    """Per-seed differences (arm - control) for metric keys."""
    control_by_seed = {r.seed: r for r in control_rows}
    out: list[dict[str, Any]] = []
    for r in arm_rows:
        c = control_by_seed.get(r.seed)
        row: dict[str, Any] = {"seed": r.seed}
        if c is None:
            row["missing_control"] = True
            out.append(row)
            continue
        for key in METRIC_KEYS:
            row[key] = round(getattr(r, key) - getattr(c, key), 3)
        out.append(row)
    return out


def load_eval_file(path: str | Path) -> list[SeedRow]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "per_agent" in data:
        data = [data]
    if isinstance(data, dict) and "results" in data:
        data = data["results"]
    return rows_from_seed_results(data)


def to_csv(arm_rows: dict[str, list[SeedRow]], control: str | None = None) -> str:
    """CSV text: per arm per seed rows, with a diff block vs control."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["arm", "seed", *METRIC_KEYS])
    for arm, rows in arm_rows.items():
        for r in rows:
            writer.writerow([arm, r.seed, *[getattr(r, k) for k in METRIC_KEYS]])
    if control and control in arm_rows:
        writer.writerow([])
        writer.writerow([f"diff_vs_{control}", "seed", *METRIC_KEYS])
        control_rows = arm_rows[control]
        for arm, rows in arm_rows.items():
            if arm == control:
                continue
            for d in difference_rows(rows, control_rows):
                writer.writerow([arm, d["seed"], *[d.get(k, "") for k in METRIC_KEYS]])
    return buf.getvalue()
