"""'Follow latest checkpoint' discovery for the Research cockpit.

The live mode polls a run directory for the newest checkpoint and decides when
a new checkpoint has appeared. Keeping this as pure functions makes it easily
testable offscreen.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simlab.research.logs import latest_checkpoint


def checkpoint_signature(path: str | Path | None) -> tuple[Any, ...] | None:
    """(mtime_ns, size) identity for a checkpoint file, or None."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    st = p.stat()
    return (st.st_mtime_ns, st.st_size)


@dataclass
class FollowState:
    last_sig: tuple[Any, ...] | None = None


def poll_latest(run_dir: str | Path, state: FollowState) -> Path | None:
    """Return the newest checkpoint if it changed since last poll, else None."""
    ckpt = latest_checkpoint(run_dir)
    sig = checkpoint_signature(ckpt)
    if sig is None or sig == state.last_sig:
        return None
    state.last_sig = sig
    return ckpt
