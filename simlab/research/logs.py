"""Parsing of training-run artifacts for the Research cockpit.

The trainer writes ``learning.csv`` (one row per PPO iteration) with quick-eval
cap lines as comments, plus ``latest.pt``/``final.pt`` checkpoints. This module
reads those so the GUI can plot training curves and follow the latest
checkpoint. Torch is never imported here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

LEARNING_CSV = "learning.csv"
CHECKPOINT_NAMES = ("latest.pt", "final.pt")


def parse_learning_csv(path: str | Path) -> list[dict[str, Any]]:
    """Rows of {iteration, env_steps, mean_return, entropy, approx_kl,
    value_loss} from a learning.csv (empty list if missing/bad)."""
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 7:
                continue
            try:
                rows.append(
                    {
                        "iteration": int(parts[0]),
                        "env_steps": int(parts[1]),
                        "mean_episode_len": float(parts[2]),
                        "mean_return": float(parts[3]),
                        "entropy": float(parts[4]),
                        "approx_kl": float(parts[5]),
                        "value_loss": float(parts[6]),
                    }
                )
            except ValueError:
                continue
    return rows


def parse_quick_eval(path: str | Path) -> list[tuple[int, float]]:
    """List of (iteration, cap_pct) from '# quick_eval it=N cap_pct=X' lines."""
    path = Path(path)
    if not path.exists():
        return []
    out: list[tuple[int, float]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("# quick_eval"):
                continue
            try:
                it = int(line.split("it=")[1].split()[0])
                cap = float(line.split("cap_pct=")[1].strip())
            except (ValueError, IndexError):
                continue
            out.append((it, cap))
    return out


def latest_checkpoint(run_dir: str | Path) -> Path | None:
    """Newest checkpoint (by mtime) in a run directory, or None."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return None
    candidates = [run_dir / n for n in CHECKPOINT_NAMES if (run_dir / n).exists()]
    candidates += sorted(run_dir.glob("*.pt"))
    if not candidates:
        return None
    # prefer latest.pt/final.pt; otherwise newest by mtime
    for name in CHECKPOINT_NAMES:
        p = run_dir / name
        if p.exists():
            return p
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_summary(run_dir: str | Path) -> dict[str, Any]:
    """Compact summary of a finished run for the results table."""
    run_dir = Path(run_dir)
    rows = parse_learning_csv(run_dir / LEARNING_CSV)
    info: dict[str, Any] = {}
    if rows:
        last = rows[-1]
        info["iterations"] = last["iteration"] + 1
        info["env_steps"] = last["env_steps"]
        info["mean_return"] = last["mean_return"]
        info["entropy"] = last["entropy"]
        info["approx_kl"] = last["approx_kl"]
        info["value_loss"] = last["value_loss"]
    evals = parse_quick_eval(run_dir / LEARNING_CSV)
    if evals:
        info["last_cap_pct"] = evals[-1][1]
        info["eval_iterations"] = [it for it, _ in evals]
        info["cap_series"] = [cap for _, cap in evals]
    if latest_checkpoint(run_dir) is not None:
        info["has_checkpoint"] = True
    return info


def curve_series(run_dir: str | Path) -> dict[str, Any]:
    """Plot-ready series dict for one run directory."""
    run_dir = Path(run_dir)
    rows = parse_learning_csv(run_dir / LEARNING_CSV)
    series: dict[str, Any] = {
        "iteration": [r["iteration"] for r in rows],
        "mean_return": [r["mean_return"] for r in rows],
        "entropy": [r["entropy"] for r in rows],
        "approx_kl": [r["approx_kl"] for r in rows],
        "value_loss": [r["value_loss"] for r in rows],
        "env_steps": [r["env_steps"] for r in rows],
        "cap_iteration": [],
        "cap_pct": [],
    }
    evals = parse_quick_eval(run_dir / LEARNING_CSV)
    if evals:
        series["cap_iteration"] = [it for it, _ in evals]
        series["cap_pct"] = [cap for _, cap in evals]
    return series
