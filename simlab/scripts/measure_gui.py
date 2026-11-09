"""Measure GUI rendering and simulation throughput at requested speeds."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication

from simlab.gui.app import MainWindow


def main() -> None:
    app = QApplication.instance() or QApplication([])
    for speed in ("10x", "Max"):
        window = MainWindow(seed=1, agent_count=20)
        window.show()
        window.toolbar.speed.setCurrentText(speed)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.005)
        print(
            f"agents=20 speed={speed} render_fps={window._render_fps:.1f} "
            f"sim_steps_per_second={window.worker.steps_per_second:.1f}"
        )
        window.close()
        app.processEvents()


if __name__ == "__main__":
    main()
