"""Measure GUI rendering and simulation throughput for the Phase 6 gate."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from simlab.gui.main_window import MainWindow


def measure(window: MainWindow, speed: str, duration: float = 3.0) -> None:
    window.toolbar.speed.setCurrentText(speed)
    start_step = window.worker.latest_snapshot.step
    started = time.perf_counter()
    loop = QEventLoop()
    QTimer.singleShot(round(duration * 1000), loop.quit)
    loop.exec()
    elapsed = time.perf_counter() - started
    steps = window.worker.latest_snapshot.step - start_step
    print(
        f"agents=20 speed={speed} render_fps={window._render_fps:.1f} "
        f"sim_steps_per_second={steps / elapsed:.1f}"
    )


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(seed=1, agent_count=20)
    window.show()
    app.processEvents()
    for speed in ("10x", "Max"):
        measure(window, speed)
    window.close()
    app.processEvents()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
