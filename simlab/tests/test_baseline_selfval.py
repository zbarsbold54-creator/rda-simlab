"""Self-validating baseline tests (config/policy hashes + stale refusal)."""

from __future__ import annotations

import json
from pathlib import Path

from simlab.eval_lib import (
    compare_to_baselines,
    policy_source_hash,
    resolved_config_hash,
)

REPO = Path(__file__).resolve().parents[1]
BALANCED = str(REPO / "configs/balanced.toml")


def _write_fake_baseline(path: Path, config_hash: str, policy_hash: str) -> None:
    doc = {
        "scenarios": {
            "balanced_5": {
                "agents": 5,
                "config": "configs/balanced.toml",
                "config_hash": config_hash,
                "policies": {
                    "random": {"ci": [0.0, 5.0]},
                    "heuristic_noflee": {"ci": [20.0, 30.0]},
                    "heuristic": {"ci": [40.0, 60.0], "policy_hash": policy_hash},
                },
            }
        }
    }
    path.write_text(json.dumps(doc), encoding="utf-8")


def _agg() -> dict:
    return {"ci": [10.0, 20.0]}


def test_config_change_alters_resolved_hash(tmp_path) -> None:
    modified = tmp_path / "balanced_mod.toml"
    text = Path(BALANCED).read_text(encoding="utf-8").replace(
        "detection_radius = 16", "detection_radius = 17"
    )
    modified.write_text(text, encoding="utf-8")
    assert resolved_config_hash(BALANCED, 5) != resolved_config_hash(
        str(modified), 5
    )


def test_stale_config_hash_refuses_verdicts(tmp_path) -> None:
    modified = tmp_path / "balanced_mod.toml"
    text = Path(BALANCED).read_text(encoding="utf-8").replace(
        "detection_radius = 16", "detection_radius = 17"
    )
    modified.write_text(text, encoding="utf-8")
    baseline = tmp_path / "baselines.json"
    _write_fake_baseline(
        baseline, resolved_config_hash(BALANCED, 5), policy_source_hash("heuristic")
    )
    res = compare_to_baselines(
        _agg(),
        "balanced_5",
        baseline,
        str(modified),
        5,
        "heuristic",
        allow_stale=False,
    )
    assert any("config hash" in r for r in res["stale"])
    assert res["verdicts"] == []


def test_stale_policy_hash_refuses_verdicts(tmp_path) -> None:
    baseline = tmp_path / "baselines.json"
    _write_fake_baseline(
        baseline, resolved_config_hash(BALANCED, 5), "cafe"
    )
    res = compare_to_baselines(
        _agg(),
        "balanced_5",
        baseline,
        BALANCED,
        5,
        "heuristic",
        allow_stale=False,
    )
    assert any("policy source hash" in r for r in res["stale"])
    assert res["verdicts"] == []


def test_allow_stale_bypasses_refusal(tmp_path) -> None:
    baseline = tmp_path / "baselines.json"
    _write_fake_baseline(
        baseline, "deadbeef", "cafe"
    )
    res = compare_to_baselines(
        _agg(),
        "balanced_5",
        baseline,
        BALANCED,
        5,
        "heuristic",
        allow_stale=True,
    )
    assert res["stale"]
    assert len(res["verdicts"]) == 3


def test_mismatched_cap_is_not_comparable(tmp_path) -> None:
    baseline = tmp_path / "baselines.json"
    _write_fake_baseline(
        baseline, resolved_config_hash(BALANCED, 5), policy_source_hash("heuristic")
    )
    doc = json.loads(baseline.read_text(encoding="utf-8"))
    doc["step_cap"] = 10000
    baseline.write_text(json.dumps(doc), encoding="utf-8")
    res = compare_to_baselines(
        _agg(),
        "balanced_5",
        baseline,
        BALANCED,
        5,
        "heuristic",
        cap=2000,
    )
    assert "not_comparable" in res
    assert "cap 2000 != baseline 10000" in res["not_comparable"]
    assert res["verdicts"] == []


def test_mismatched_seeds_is_not_comparable(tmp_path) -> None:
    baseline = tmp_path / "baselines.json"
    _write_fake_baseline(
        baseline, resolved_config_hash(BALANCED, 5), policy_source_hash("heuristic")
    )
    doc = json.loads(baseline.read_text(encoding="utf-8"))
    doc["seed_sets"] = {"dev": {"seed_list": list(range(20))}}
    baseline.write_text(json.dumps(doc), encoding="utf-8")
    res = compare_to_baselines(
        _agg(),
        "balanced_5",
        baseline,
        BALANCED,
        5,
        "heuristic",
        seed_set="dev",
        seeds=[0],
    )
    assert "not_comparable" in res
    assert "seed list differs" in res["not_comparable"]
    assert res["verdicts"] == []


def test_matching_compare_is_comparable_and_emits_verdicts(tmp_path) -> None:
    baseline = tmp_path / "baselines.json"
    _write_fake_baseline(
        baseline, resolved_config_hash(BALANCED, 5), policy_source_hash("heuristic")
    )
    doc = json.loads(baseline.read_text(encoding="utf-8"))
    doc["step_cap"] = 10000
    doc["seed_sets"] = {"dev": {"seed_list": list(range(20))}}
    baseline.write_text(json.dumps(doc), encoding="utf-8")
    res = compare_to_baselines(
        _agg(),
        "balanced_5",
        baseline,
        BALANCED,
        5,
        "heuristic",
        seed_set="dev",
        cap=10000,
        seeds=list(range(20)),
    )
    assert "not_comparable" not in res
    assert len(res["verdicts"]) == 3
