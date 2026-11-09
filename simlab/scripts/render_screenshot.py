"""Render the SimLab GUI to the Phase 3 review screenshot."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication

from simlab.gui import theme
from simlab.gui.app import MainWindow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--output", default="docs/screenshots/phase3.png")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--food-diagnostic-steps", type=int, default=2000)
    args = parser.parse_args()
    app = QApplication.instance() or QApplication([])
    window = MainWindow(seed=1, agent_count=args.agents)
    window.worker.set_playing(False)
    window.show()
    if args.steps > 0:
        window.worker.set_speed("Max")
        window.worker.set_playing(True)
        _pump_until_step(app, window, args.steps)
        window.worker.set_playing(False)
        app.processEvents()
    window._accept_frame(window.worker.latest_frame)
    window._select_agent("agent_0")
    window._accept_snapshot(window.worker.latest_snapshot)
    window._render_latest()
    app.processEvents()
    screenshot_snapshot = window.worker.latest_snapshot
    destination = Path(args.output)
    if not destination.is_absolute():
        destination = Path(__file__).resolve().parents[1] / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not window.grab().save(str(destination)):
        raise RuntimeError(f"could not save screenshot to {destination}")
    print(
        f"agents={len(screenshot_snapshot.agent_states)} steps={screenshot_snapshot.step} screenshot={destination}"
    )

    diagnostic_target = max(args.steps, args.food_diagnostic_steps)
    current_step = window.worker.latest_snapshot.step
    if diagnostic_target > current_step:
        window.worker.request_steps(diagnostic_target - current_step)
        _pump_until_step(app, window, diagnostic_target)
    diagnostic_snapshot = window.worker.latest_snapshot
    print(f"food_radius_diagnostics_after_step={diagnostic_snapshot.step}")
    for index, patch in enumerate(diagnostic_snapshot.food_patches[:5]):
        amount = float(patch["amount"])
        radius = window.world_view.food_draw_radius(amount)
        print(f"food_patch={index} amount={amount:.2f} drawn_radius={radius:.3f}")

    window.close()
    app.processEvents()


def _pump_until_step(app: QApplication, window: MainWindow, target_step: int) -> None:
    deadline = time.monotonic() + theme.SCREENSHOT_STEP_TIMEOUT_S
    while window.worker.latest_snapshot.step < target_step:
        app.processEvents()
        if time.monotonic() >= deadline:
            raise TimeoutError(f"simulation did not reach step {target_step}")
        time.sleep(theme.SCREENSHOT_PUMP_INTERVAL_S)


if __name__ == "__main__":
    main()
