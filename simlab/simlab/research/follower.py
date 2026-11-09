"""'Follow latest checkpoint' discovery for the Research cockpit.

The live mode polls a run directory for the newest checkpoint and decides when
a new checkpoint has appeared. Keeping this as pure functions makes it easily
testable offscreen.
"""

from __future__ import annotations

import zipfile
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


def checkpoint_readable(path: str | Path | None) -> bool:
    """True if the path exists, is non-empty, and is an intact zip archive.

    Torch ``.pt`` files are ZIP archives; a truncated write leaves the central
    directory missing, so ``zipfile.is_zipfile`` rejects it without importing
    torch. Keeps the follower torch-free.
    """
    if path is None:
        return False
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return False
    return zipfile.is_zipfile(p)


@dataclass
class FollowState:
    last_sig: tuple[Any, ...] | None = None


def poll_latest(run_dir: str | Path, state: FollowState) -> Path | None:
    """Return the newest checkpoint if it changed and is readable since last poll.

    An unreadable (truncated/in-flight) checkpoint is ignored and the previous
    signature is kept, so a later poll retries it once the write completes.
    """
    ckpt = latest_checkpoint(run_dir)
    sig = checkpoint_signature(ckpt)
    if sig is None or sig == state.last_sig:
        return None
    if not checkpoint_readable(ckpt):
        return None
    state.last_sig = sig
    return ckpt
