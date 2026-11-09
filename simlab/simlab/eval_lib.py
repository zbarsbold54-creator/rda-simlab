"""Evaluation library for the `simlab.cli eval` subcommand.

Runs the full pipeline (sensing, registered DriveCore, registered Policy,
every step) with one multiprocessing worker per seed, aggregates per-agent
metrics, and computes a 95% bootstrap CI over seeds for the cap percent.
Results are hashed deterministically for reproducibility checks.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from simlab.core.config import load_config
from simlab.core.runner import Runner

REPO = Path(__file__).resolve().parents[1]
SCENARIO_CONFIGS = {
    "balanced": "configs/balanced.toml",
    "solo": "configs/solo.toml",
    "hard": "configs/hard.toml",
}


def scenario_baseline_key(scenario: str, agents: int) -> str:
    if scenario == "solo":
        return "solo"
    return f"{scenario}_{agents}"


def resolved_config_hash(config_path: str, agents: int) -> str:
    """Canonical hash of the resolved scenario config (incl. agent count)."""
    base = load_config(config_path)
    cfg = replace(base, agent=replace(base.agent, count=agents))
    return hashlib.sha256(
        json.dumps(cfg.to_dict(), sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def policy_source_hash(
    policy_name: str, checkpoint: str | Path | None = None
) -> str:
    """Hash the source files defining a registered policy and its bases.

    When a checkpoint path is supplied, the checkpoint file bytes are folded
    into the hash so ``--compare`` staleness checks detect a changed network
    (including its observation dimension and weights).
    """
    from simlab.core.registry import get

    cls = get("policy", policy_name)
    if cls is None:
        blob = policy_name.encode("utf-8")
        if checkpoint is not None:
            blob += Path(checkpoint).read_bytes()
        return hashlib.sha256(blob).hexdigest()[:16]
    files: list[str] = []
    seen: set[str] = set()

    def walk(klass: type[Any]) -> None:
        mod = inspect.getmodule(klass)
        file = getattr(mod, "__file__", None)
        if file is not None and file not in seen:
            seen.add(file)
            files.append(file)
        for base in klass.__bases__:
            walk(base)

    walk(cls)
    files.sort()
    blob = b"".join(Path(f).read_bytes() for f in files)
    if checkpoint is not None:
        blob += Path(checkpoint).read_bytes()
    return hashlib.sha256(blob).hexdigest()[:16]


def run_episode_metrics(
    policy: str,
    seed: int,
    agents: int,
    cap: int,
    config_path: str,
    drive: str = "simple",
    checkpoint: str | None = None,
) -> dict[str, Any]:
    """One full-pipeline episode; returns per-agent metrics for the seed."""
    base = load_config(config_path)
    cfg = replace(
        base,
        policy=policy,
        drive=drive,
        agent=replace(base.agent, count=agents),
        policy_checkpoint=checkpoint,
    )
    runner = Runner(cfg, seed=seed, agents=agents)
    runner.setup()
    world = runner._world
    dt = world.dt
    meta: dict[str, dict[str, Any]] = {}
    for aid in world._agents:
        meta[aid] = {
            "hazard_loss": 0.0,
            "starve_loss": 0.0,
            "n_hazard_contacts": 0,
            "energy_sum": 0.0,
            "steps_alive": 0,
            "death_step": None,
            "energy_at_death": None,
            "food_eaten": 0,
        }
    alive = set(world._agents)
    for step in range(1, cap + 1):
        result = runner.run_step(include_observations=False, include_agent_states=False)
        for aid in list(alive):
            st = world._agents[aid]
            m = meta[aid]
            pos = st["position"]
            contact = False
            for hz in world._hazard_patches:
                dx = pos[0] - hz["position"][0]
                dy = pos[1] - hz["position"][1]
                if dx * dx + dy * dy <= hz["radius"] ** 2:
                    m["hazard_loss"] += hz["damage"] * dt
                    contact = True
            if contact:
                m["n_hazard_contacts"] += 1
            if st["energy"] <= 0.0:
                m["starve_loss"] += 0.5
            m["energy_sum"] += st["energy"]
            m["steps_alive"] += 1
            if not st["is_alive"]:
                m["death_step"] = step
                m["energy_at_death"] = st["energy"]
                m["food_eaten"] = st["food_eaten"]
                alive.discard(aid)
        if result.terminated:
            break
    food_left = sum(p["amount"] for p in world._food_patches)
    per_agent = []
    for aid, m in meta.items():
        death = m["death_step"] if m["death_step"] is not None else cap
        if m["death_step"] is None:
            cls = "truncation"
        else:
            cls = "hazard" if m["hazard_loss"] > m["starve_loss"] else "starvation"
        per_agent.append(
            {
                "death_step": death,
                "class": cls,
                "energy_at_death": (
                    m["energy_at_death"] if m["death_step"] is not None else None
                ),
                "food_eaten": m["food_eaten"],
                "hazard_contacts": m["n_hazard_contacts"],
                "mean_energy": round(m["energy_sum"] / max(1, m["steps_alive"]), 4),
            }
        )
    per_agent.sort(key=lambda a: a["death_step"])
    return {"seed": seed, "per_agent": per_agent, "food_left": food_left}


def bootstrap_cap_ci(
    classes_by_seed: list[list[str]],
    n_boot: int = 1000,
    rng_seed: int = 1234,
) -> tuple[float, float]:
    rng = np.random.default_rng(rng_seed)
    caps = []
    n_seeds = len(classes_by_seed)
    for _ in range(n_boot):
        sample = rng.integers(0, n_seeds, size=n_seeds)
        cnt = sum(len(classes_by_seed[i]) for i in sample)
        cap_cnt = sum(
            sum(1 for cls in classes_by_seed[i] if cls == "truncation") for i in sample
        )
        caps.append(100.0 * cap_cnt / cnt)
    caps.sort()
    return caps[int(0.025 * len(caps))], caps[int(0.975 * len(caps))]


def aggregate(
    results: list[dict[str, Any]], wall_time: float, cap: int
) -> dict[str, Any]:
    """Collapse per-seed results into the summary dict + canonical hash."""
    results = sorted(results, key=lambda r: r["seed"])
    classes_by_seed = [[a["class"] for a in r["per_agent"]] for r in results]
    n = sum(len(c) for c in classes_by_seed)
    capn = sum(sum(1 for cls in c if cls == "truncation") for c in classes_by_seed)
    counts = {"starvation": 0, "hazard": 0, "truncation": 0}
    deaths = []
    food_eaten = []
    hazard_contacts = []
    mean_energy_list = []
    food_left = []
    for r in results:
        food_left.append(r["food_left"])
        for a in r["per_agent"]:
            deaths.append(a["death_step"])
            counts[a["class"]] += 1
            food_eaten.append(a["food_eaten"])
            hazard_contacts.append(a["hazard_contacts"])
            mean_energy_list.append(a["mean_energy"] or 0.0)
    deaths.sort()
    median = deaths[len(deaths) // 2]
    lo, hi = bootstrap_cap_ci(classes_by_seed)
    sum_ = {
        "cap_pct": 100.0 * capn / n,
        "ci": [round(lo, 3), round(hi, 3)],
        "median_survival": median,
        "death_cause": counts,
        "food_eaten_per_agent": round(float(np.mean(food_eaten)), 3),
        "hazard_contacts_per_1000": round(
            float(np.mean(hazard_contacts)) * 1000.0 / cap, 3
        ),
        "mean_energy": round(float(np.mean(mean_energy_list)), 3),
        "food_left_mean": round(float(np.mean(food_left)), 3),
        "wall_time_s": round(wall_time, 2),
    }
    hash_in = {
        "results": results,
    }
    sum_["hash"] = hashlib.sha256(
        json.dumps(hash_in, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return sum_


def compare_to_baselines(
    agg: dict[str, Any],
    baseline_key: str,
    baseline_path: Path,
    config_path: str,
    agents: int,
    policy: str,
    allow_stale: bool = False,
    seed_set: str | None = None,
    checkpoint: str | Path | None = None,
    cap: int | None = None,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Verdicts against the docs/baselines.json reference table.

    Verifies the current scenario config hash and policy source hash match what
    the baseline stored. If either is stale and ``allow_stale`` is False, the
    verdicts are refused (empty list) and the stale reasons are returned.

    When ``cap``/``seeds`` are supplied, the run is first checked to be
    comparable with the stored baseline (same step cap and seed list). A
    mismatch returns ``not_comparable`` and no verdicts.
    """
    if not baseline_path.exists():
        return {
            "stale": [],
            "verdicts": [
                {"against": "baselines.json", "verdict": "MISSING BASELINES FILE"}
            ],
        }
    doc = json.loads(baseline_path.read_text(encoding="utf-8"))

    comparability: list[str] = []
    ref_cap = doc.get("step_cap")
    if cap is not None and ref_cap is not None and cap != ref_cap:
        comparability.append(f"cap {cap} != baseline {ref_cap}")
    ref_seeds = None
    if seed_set is not None:
        ref_seeds = doc.get("seed_sets", {}).get(seed_set, {}).get("seed_list")
    if seeds is not None and ref_seeds is not None and list(seeds) != list(ref_seeds):
        comparability.append("seed list differs from baseline")
    if comparability:
        return {
            "stale": [],
            "verdicts": [],
            "not_comparable": "; ".join(comparability),
        }
    scenarios = doc.get("scenarios")
    if scenarios is None and seed_set is not None:
        scenarios = doc.get("seed_sets", {}).get(seed_set, {}).get("scenarios")
    if scenarios is None:
        scenarios = doc.get("scenarios", {})
    scenario = scenarios.get(baseline_key)
    if not scenario:
        return {
            "stale": [],
            "verdicts": [
                {"against": baseline_key, "verdict": "MISSING SCENARIO BASELINE"}
            ],
        }

    current_config_hash = resolved_config_hash(config_path, agents)
    current_policy_hash = policy_source_hash(policy, checkpoint)
    stale: list[str] = []
    if scenario.get("config_hash") != current_config_hash:
        stale.append(f"config hash differs for {baseline_key}")
    ref = scenario.get("policies", {}).get(policy)
    if ref is not None and ref.get("policy_hash") != current_policy_hash:
        stale.append(f"policy source hash differs for {policy}")

    if stale and not allow_stale:
        return {"stale": stale, "verdicts": []}

    verdicts = []
    lo, hi = agg["ci"]
    for ref_name in ("random", "heuristic_noflee", "heuristic"):
        r = scenario.get("policies", {}).get(ref_name)
        if not r:
            continue
        ref_lo, ref_hi = r["ci"]
        if lo > ref_hi:
            verdict = "BETTER"
        elif hi < ref_lo:
            verdict = "WORSE"
        else:
            verdict = "NOT DISTINGUISHABLE"
        verdicts.append({"against": ref_name, "verdict": verdict})
    return {"stale": stale, "verdicts": verdicts}
