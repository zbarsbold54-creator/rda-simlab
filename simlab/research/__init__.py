"""Research cockpit logic: experiment building, run queue, logs, results.

This package is torch-free at import time (it only touches stdlib, numpy,
and ``simlab.training.arch_spec``). Torch is imported lazily only when arm
parameter counts are computed.
"""

from __future__ import annotations
