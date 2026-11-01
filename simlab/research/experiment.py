"""Experiment builder and protocol guards for the Research cockpit.

An experiment is a grid over arms (ArchSpec backbones) crossed with a set of
training seeds. It is serialized to TOML in ``runs/`` and consumed by the run
queue. Validation enforces the research protocol:

* training seeds must be >= 100000,
* workers default to 4 (a visible warning is emitted when changed),
* dev seeds live in 0..19,
* test seeds live in 1000..1019 and require a confirm dialog ("final report
  only"),
* fewer than 3 seeds per arm produces a warning.

This module is torch-free at import; parameter counts import torch lazily.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from simlab.training.arch_spec import (
    build_policy_cfg,
    get_spec,
    list_specs,
    validate_config,
)

TRAIN_SEED_MIN = 100000
DEFAULT_WORKERS = 4
MIN_SEEDS_PER_ARM = 3
DEV_SEED_RANGE = (0, 19)
TEST_SEED_RANGE = (1000, 1019)
VALID_SCENARIOS = ("balanced", "hard", "solo")
FUSION_CHOICES = ("none", "concat", "film")
OBS_CHOICES = ("obs_only", "obs_drives")


@dataclass
class Arm:
    """One architecture arm in an experiment grid."""

    name: str
    backbone: str
    drive_core: str = "rda8"
    fusion: str = "film"
    obs_mode: str = "obs_drives"
    backbone_config: dict[str, Any] = field(default_factory=dict)
    drive_config: dict[str, Any] = field(default_factory=dict)

    def selection(self) -> dict[str, Any]:
        """ArchSpec selection dict as consumed by ``build_policy_cfg``."""
        return {
            "backbone": self.backbone,
            "drive_core": self.drive_core,
            "fusion": self.fusion,
            "obs_mode": self.obs_mode,
            "seed": 0,
            "backbone_config": dict(self.backbone_config),
            "drive_config": dict(self.drive_config),
        }

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.name.strip():
            errors.append("arm name is required")
        if self.backbone not in list_specs("backbone"):
            errors.append(f"unknown backbone {self.backbone!r}")
        if self.drive_core not in list_specs("drive_core"):
            errors.append(f"unknown drive core {self.drive_core!r}")
        if self.fusion not in FUSION_CHOICES:
            errors.append(f"unknown fusion {self.fusion!r}")
        if self.obs_mode not in OBS_CHOICES:
            errors.append(f"unknown obs mode {self.obs_mode!r}")
        bb = get_spec("backbone", self.backbone)
        if bb is not None:
            errors.extend(validate_config(bb, self.backbone_config))
        dc = get_spec("drive_core", self.drive_core)
        if dc is not None:
            errors.extend(validate_config(dc, self.drive_config))
        return errors


@dataclass
class Experiment:
    """A training grid: arms crossed with training seeds."""

    name: str
    arms: list[Arm]
    total_steps: int = 16384
    rollout_length: int = 1024
    workers: int = DEFAULT_WORKERS
    train_seeds: list[int] = field(default_factory=lambda: [TRAIN_SEED_MIN])
    dev_seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    test_seeds: list[int] = field(default_factory=list)
    eval_scenarios: list[str] = field(default_factory=lambda: ["balanced", "hard"])
    control_arm: str | None = None
    eval_every: int = 20

    def validate(self) -> tuple[list[str], list[str]]:
        """Return (errors, warnings) for the whole experiment."""
        errors: list[str] = []
        warnings: list[str] = []
        if not self.name.strip():
            errors.append("experiment name is required")
        if not self.arms:
            errors.append("at least one arm is required")
        arm_names = [a.name for a in self.arms]
        if len(arm_names) != len(set(arm_names)):
            errors.append("arm names must be unique")
        for s in self.train_seeds:
            if s < TRAIN_SEED_MIN:
                errors.append(f"training seed {s} must be >= {TRAIN_SEED_MIN}")
        if len(self.train_seeds) != len(set(self.train_seeds)):
            errors.append("training seeds must be unique")
        for s in self.dev_seeds:
            if not (DEV_SEED_RANGE[0] <= s <= DEV_SEED_RANGE[1]):
                errors.append(f"dev seed {s} must be within {DEV_SEED_RANGE}")
        for s in self.test_seeds:
            if not (TEST_SEED_RANGE[0] <= s <= TEST_SEED_RANGE[1]):
                errors.append(f"test seed {s} must be within {TEST_SEED_RANGE}")
        if self.workers != DEFAULT_WORKERS:
            warnings.append(
                f"workers {self.workers} differs from the pinned default "
                f"{DEFAULT_WORKERS}"
            )
        if self.workers < 1:
            errors.append("workers must be >= 1")
        if len(self.dev_seeds) < MIN_SEEDS_PER_ARM:
            warnings.append("fewer than 3 dev seeds per arm")
        if self.test_seeds and len(self.test_seeds) < MIN_SEEDS_PER_ARM:
            warnings.append("fewer than 3 test seeds per arm")
        for sc in self.eval_scenarios:
            if sc not in VALID_SCENARIOS:
                errors.append(f"unknown scenario {sc!r}")
        if self.control_arm is not None and self.control_arm not in arm_names:
            warnings.append(f"control arm {self.control_arm!r} is not among the arms")
        if self.total_steps < 1:
            errors.append("total_steps must be >= 1")
        if self.rollout_length < 1:
            errors.append("rollout_length must be >= 1")
        if self.eval_every < 0:
            errors.append("eval_every must be >= 0")
        for arm in self.arms:
            errors.extend(f"{arm.name}: {e}" for e in arm.validate())
        return errors, warnings

    def requires_test_confirm(self) -> bool:
        """True when test seeds are present and need the confirm dialog."""
        return bool(self.test_seeds)


# ------------------------------------------------------------------ toml
def _dump_value(key: str, value: Any, indent: int) -> list[str]:
    pad = " " * indent
    if isinstance(value, bool):
        return [f"{pad}{key} = {str(value).lower()}"]
    if isinstance(value, (int, float)):
        return [f"{pad}{key} = {value}"]
    if isinstance(value, str):
        return [f'{pad}{key} = "{value}"']
    if isinstance(value, dict):
        lines = [f"{pad}{key}"]
        for k, v in value.items():
            lines.extend(_dump_value(k, v, indent + 2))
        return lines
    if isinstance(value, (list, tuple)):
        items = ", ".join(json.dumps(v) for v in value)
        return [f"{pad}{key} = [{items}]"]
    return [f"{pad}{key} = {json.dumps(value)}"]


def dump_experiment_toml(exp: Experiment) -> str:
    lines = ["[experiment]"]
    lines.extend(_dump_value("name", exp.name, 0))
    lines.extend(_dump_value("total_steps", exp.total_steps, 0))
    lines.extend(_dump_value("rollout_length", exp.rollout_length, 0))
    lines.extend(_dump_value("workers", exp.workers, 0))
    lines.extend(_dump_value("eval_every", exp.eval_every, 0))
    lines.extend(_dump_value("train_seeds", list(exp.train_seeds), 0))
    lines.extend(_dump_value("dev_seeds", list(exp.dev_seeds), 0))
    lines.extend(_dump_value("test_seeds", list(exp.test_seeds), 0))
    lines.extend(_dump_value("eval_scenarios", list(exp.eval_scenarios), 0))
    if exp.control_arm is not None:
        lines.extend(_dump_value("control_arm", exp.control_arm, 0))
    for i, arm in enumerate(exp.arms):
        lines.append(f"[arms.{i}]")
        lines.extend(_dump_value("name", arm.name, 0))
        lines.extend(_dump_value("backbone", arm.backbone, 0))
        lines.extend(_dump_value("drive_core", arm.drive_core, 0))
        lines.extend(_dump_value("fusion", arm.fusion, 0))
        lines.extend(_dump_value("obs_mode", arm.obs_mode, 0))
        lines.append(f"[arms.{i}.backbone_config]")
        for k, v in arm.backbone_config.items():
            lines.extend(_dump_value(k, v, 2))
        lines.append(f"[arms.{i}.drive_config]")
        for k, v in arm.drive_config.items():
            lines.extend(_dump_value(k, v, 2))
    return "\n".join(lines) + "\n"


def _parse_arm(raw: dict[str, Any]) -> Arm:
    return Arm(
        name=str(raw.get("name", "")),
        backbone=str(raw.get("backbone", "")),
        drive_core=str(raw.get("drive_core", "rda8")),
        fusion=str(raw.get("fusion", "film")),
        obs_mode=str(raw.get("obs_mode", "obs_drives")),
        backbone_config=dict(raw.get("backbone_config") or {}),
        drive_config=dict(raw.get("drive_config") or {}),
    )


def load_experiment_toml(path: str | Path) -> Experiment:
    import tomllib

    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    exp_raw = raw.get("experiment", {})
    arms_raw = raw.get("arms", {})
    arms = [_parse_arm(arms_raw[k]) for k in sorted(arms_raw, key=int)]
    return Experiment(
        name=str(exp_raw.get("name", "")),
        arms=arms,
        total_steps=int(exp_raw.get("total_steps", 16384)),
        rollout_length=int(exp_raw.get("rollout_length", 1024)),
        workers=int(exp_raw.get("workers", DEFAULT_WORKERS)),
        train_seeds=[int(s) for s in exp_raw.get("train_seeds", [])],
        dev_seeds=[int(s) for s in exp_raw.get("dev_seeds", [])],
        test_seeds=[int(s) for s in exp_raw.get("test_seeds", [])],
        eval_scenarios=[str(s) for s in exp_raw.get("eval_scenarios", [])],
        control_arm=exp_raw.get("control_arm"),
        eval_every=int(exp_raw.get("eval_every", 20)),
    )


# ------------------------------------------------------------ param counts
def arm_param_count(arm: Arm) -> tuple[int, str, dict[str, Any]]:
    """Build the arm policy (torch, lazy) and return (params, hash, cfg)."""
    from simlab.training.arch_spec import arch_hash
    from simlab.training.torch_policy import make_policy

    selection = arm.selection()
    policy_cfg = build_policy_cfg(selection)
    policy = make_policy(policy_cfg)
    return policy.param_count(), arch_hash(selection), policy_cfg


def arm_param_counts(
    arms: list[Arm],
) -> list[dict[str, Any]]:
    """Table of {name, params, hash} for the arm list (torch, lazy)."""
    rows: list[dict[str, Any]] = []
    for arm in arms:
        params, h, _ = arm_param_count(arm)
        rows.append({"name": arm.name, "params": params, "hash": h})
    return rows
