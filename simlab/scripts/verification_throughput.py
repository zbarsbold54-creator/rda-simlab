"""Measure 20-agent GUI throughput and frame intervals for verification."""

from __future__ import annotations

import argparse
import functools
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from simlab.core.runner import Runner
from simlab.core.world import World
from simlab.gui.drive_plot import DrivePlotWidget
from simlab.gui.main_window import MainWindow
from simlab.gui.sim_worker import SimWorker
from simlab.gui.worker_components import WorkerComponents
from simlab.gui.world_view import WorldView


def enable_worker_profile() -> dict[str, float | int]:
    profile_data: dict[str, float | int] = {
        "snapshot_calls": 0,
        "snapshot_seconds": 0.0,
        "sleep_calls": 0,
        "sleep_seconds": 0.0,
    }
    original_snapshot = World.snapshot

    @functools.wraps(original_snapshot)
    def measured_snapshot(world: World):
        started = time.perf_counter()
        try:
            return original_snapshot(world)
        finally:
            profile_data["snapshot_calls"] += 1
            profile_data["snapshot_seconds"] += time.perf_counter() - started

    World.snapshot = measured_snapshot
    original_sleep = time.sleep

    def measured_sleep(seconds: float) -> None:
        started = time.perf_counter()
        try:
            original_sleep(seconds)
        finally:
            profile_data["sleep_calls"] += 1
            profile_data["sleep_seconds"] += time.perf_counter() - started

    time.sleep = measured_sleep
    original_run = SimWorker.run

    def time_method(owner, method_name: str, label: str) -> None:
        original = getattr(owner, method_name)

        @functools.wraps(original)
        def measured(instance, *args, **kwargs):
            started = time.perf_counter()
            try:
                return original(instance, *args, **kwargs)
            finally:
                profile_data[f"{label}_calls"] = (
                    int(profile_data.get(f"{label}_calls", 0)) + 1
                )
                profile_data[f"{label}_seconds"] = (
                    float(profile_data.get(f"{label}_seconds", 0.0))
                    + time.perf_counter()
                    - started
                )

        setattr(owner, method_name, measured)

    time_method(MainWindow, "_accept_frame", "accept_frame")
    time_method(MainWindow, "_accept_history", "accept_history")
    time_method(MainWindow, "_render_latest", "render_slot")
    time_method(WorldView, "paintEvent", "paint_event")
    time_method(DrivePlotWidget, "update_frame", "drive_plot")
    time_method(WorkerComponents, "actions_for", "actions_for")
    time_method(Runner, "run_step", "runner_step")

    @functools.wraps(original_run)
    def timed_run(worker: SimWorker) -> None:
        wall_start = time.perf_counter()
        cpu_start = time.thread_time()
        try:
            original_run(worker)
        finally:
            profile_data["worker_wall_seconds"] = (
                float(profile_data.get("worker_wall_seconds", 0.0))
                + time.perf_counter()
                - wall_start
            )
            profile_data["worker_cpu_seconds"] = (
                float(profile_data.get("worker_cpu_seconds", 0.0))
                + time.thread_time()
                - cpu_start
            )

    SimWorker.run = timed_run
    return profile_data


def wait_for(seconds: float) -> None:
    loop = QEventLoop()
    QTimer.singleShot(round(seconds * 1000), loop.quit)
    loop.exec()


def stop_window(window: MainWindow, app: QApplication) -> None:
    window._timer.stop()
    window.worker.stop()
    deadline = time.perf_counter() + 3.0
    while window._thread.isRunning() and time.perf_counter() < deadline:
        wait_for(0.01)
    window.close()
    app.processEvents()


def measure(
    window: MainWindow,
    speed: str,
    duration: float = 3.0,
    warmup: float = 1.0,
) -> None:
    window.toolbar.speed.setCurrentText(speed)
    wait_for(warmup)
    frame_times: list[float] = []
    last_frame = time.perf_counter()

    def on_frame() -> None:
        nonlocal last_frame
        current = time.perf_counter()
        frame_times.append(current - last_frame)
        last_frame = current

    window._timer.timeout.connect(on_frame)
    start_step = window.worker.latest_snapshot.step
    started = time.perf_counter()
    wait_for(duration)
    elapsed = time.perf_counter() - started
    steps = window.worker.latest_snapshot.step - start_step
    window._timer.timeout.disconnect(on_frame)
    intervals_ms = np.asarray(frame_times[1:], dtype=np.float64) * 1000.0
    p95 = float(np.percentile(intervals_ms, 95)) if intervals_ms.size else 0.0
    print(
        f"agents=20 overlays=all selected=agent_0 speed={speed} "
        f"render_fps={len(frame_times) / elapsed:.1f} "
        f"sim_steps_per_second={steps / elapsed:.1f} "
        f"p95_frame_time_ms={p95:.2f} frames={len(frame_times)} "
        f"latest_step={window.worker.latest_snapshot.step} "
        f"runner_step={window.worker._runner._step_count} "
        f"playing={window.worker._playing}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--speeds", nargs="+", default=("1x", "10x", "Max"))
    args = parser.parse_args()
    profile_data = enable_worker_profile() if args.profile else None
    app = QApplication.instance() or QApplication(sys.argv)
    signal_counts = {"frames": 0, "snapshots": 0, "history_batches": 0}
    for speed in args.speeds:
        window = MainWindow(seed=1, agent_count=20)
        window.show()
        for checkbox in window.session_toolbar.overlay_checks.values():
            checkbox.setChecked(True)
        window._select_agent("agent_0")
        app.processEvents()
        window.worker.frameReady.connect(
            lambda _frame: signal_counts.__setitem__(
                "frames", signal_counts["frames"] + 1
            )
        )
        window.worker.snapshotReady.connect(
            lambda _snapshot: signal_counts.__setitem__(
                "snapshots", signal_counts["snapshots"] + 1
            )
        )
        window.worker.historyReady.connect(
            lambda _batch: signal_counts.__setitem__(
                "history_batches", signal_counts["history_batches"] + 1
            )
        )
        measure(
            window,
            speed,
            duration=(0.45 if speed == "Max" else 1.25)
            if args.profile
            else (0.5 if speed == "Max" else 3.0),
            warmup=0.1 if speed == "Max" else (0.3 if args.profile else 1.0),
        )
        stop_window(window, app)
    if profile_data is not None:
        wall = float(profile_data["worker_wall_seconds"])
        cpu = float(profile_data["worker_cpu_seconds"])
        print(
            f"worker_wall_seconds={wall:.3f} worker_cpu_seconds={cpu:.3f} "
            f"worker_cpu_fraction={cpu / max(wall, 1e-9):.3f}"
        )
        print(
            f"snapshot_calls={profile_data['snapshot_calls']} "
            f"snapshot_total_ms={float(profile_data['snapshot_seconds']) * 1000:.1f} "
            f"sleep_calls={profile_data['sleep_calls']} "
            f"sleep_total_ms={float(profile_data['sleep_seconds']) * 1000:.1f}"
        )
        print(
            f"delivered_frame_signals={signal_counts['frames']} "
            f"delivered_snapshot_signals={signal_counts['snapshots']} "
            f"delivered_history_batches={signal_counts['history_batches']}"
        )
        for operation in (
            "accept_frame",
            "accept_history",
            "render_slot",
            "paint_event",
            "drive_plot",
            "actions_for",
            "runner_step",
        ):
            print(
                f"{operation}_calls={profile_data.get(f'{operation}_calls', 0)} "
                f"{operation}_total_ms={float(profile_data.get(f'{operation}_seconds', 0.0)) * 1000:.1f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
