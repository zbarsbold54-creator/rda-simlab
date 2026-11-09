"""Thread-safe request and state accessors for the simulation worker."""

from __future__ import annotations

import threading

from simlab.core.config import SimLabConfig
from simlab.core.registry import list_ as registry_list
from simlab.core.registry import load_plugins
from simlab.core.types import Snapshot
from simlab.gui import theme
from simlab.gui.frame import SimulationFrame


class WorkerControls:
    def _initialize_controls(self) -> None:
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._playing = True
        self._stopping = False
        self._step_budget = 0
        self._speed = theme.DEFAULT_SPEED
        self._requested_seed: int | None = None
        self._requested_snapshot: SimulationFrame | Snapshot | None = None
        self._requested_seek_step: int | None = None
        self._requested_scenario: tuple[int, SimLabConfig] | None = None
        self._requested_replay = False

    @property
    def latest_frame(self) -> SimulationFrame:
        with self._lock:
            return self._latest_frame

    @property
    def registry_options(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        load_plugins()
        return tuple(registry_list("policy")), tuple(registry_list("drive"))

    @property
    def latest_snapshot(self) -> Snapshot:
        with self._lock:
            return self._latest_snapshot

    @property
    def seed(self) -> int:
        with self._lock:
            return self._seed

    @property
    def steps_per_second(self) -> float:
        with self._lock:
            return self._steps_per_second

    @property
    def target_steps_per_second(self) -> float | None:
        with self._lock:
            speed = self._speed
        if speed == "Max":
            return None
        return theme.BASE_STEPS_PER_SECOND * float(speed.removesuffix("x"))

    def set_playing(self, playing: bool) -> None:
        with self._lock:
            self._playing = playing
        self._wake.set()

    def set_speed(self, speed: str) -> None:
        with self._lock:
            self._speed = speed
        self._wake.set()

    def request_step(self) -> None:
        self.request_steps(1)

    def request_steps(self, count: int) -> None:
        with self._lock:
            self._step_budget += max(0, int(count))
        self._wake.set()

    def request_reset(self, seed: int) -> None:
        with self._lock:
            self._requested_seed = int(seed)
        self._wake.set()

    def request_scenario(self, seed: int, config: SimLabConfig) -> None:
        with self._lock:
            self._requested_scenario = (int(seed), config)
        self._wake.set()

    def request_restore(self, frame: SimulationFrame | Snapshot) -> None:
        with self._lock:
            self._requested_snapshot = frame
            self._requested_seek_step = None
            self._playing = False
            snapshot = frame.snapshot if isinstance(frame, SimulationFrame) else frame
            self._history.resume_from(snapshot.step)
        self._wake.set()

    def request_seek(self, frame: SimulationFrame, target_step: int) -> None:
        with self._lock:
            self._requested_snapshot = frame
            self._requested_seek_step = max(frame.snapshot.step, int(target_step))
            self._playing = False
            self._history.resume_from(frame.snapshot.step)
        self._wake.set()

    def set_recording(self, enabled: bool) -> None:
        with self._lock:
            self._history.recording = bool(enabled)

    def request_replay(self) -> None:
        with self._lock:
            self._requested_replay = True
            self._playing = True
        self._wake.set()

    def stop(self) -> None:
        with self._lock:
            self._stopping = True
        self._wake.set()
