"""Simulation worker hosted by a dedicated QThread."""

from __future__ import annotations

import time
from types import MappingProxyType

from PySide6.QtCore import QObject, Signal, Slot

from simlab.core.config import SimLabConfig, load_config
from simlab.core.runner import Runner
from simlab.core.types import Action, StepResult
from simlab.gui import theme
from simlab.gui.frame import AgentFrame, SimulationFrame
from simlab.gui.worker_components import WorkerComponents
from simlab.gui.worker_controls import WorkerControls
from simlab.gui.worker_history import WorkerHistory

_TICK_BUDGET_MS = 150.0


class SimWorker(QObject, WorkerControls):
    """Runs simulation steps off the GUI thread and publishes immutable state."""

    snapshotReady = Signal(object)
    frameReady = Signal(object)
    historyReady = Signal(object)
    metricsReady = Signal(float, float, int, int)
    tickTimeReady = Signal(float)
    stopped = Signal()

    def __init__(
        self,
        config: SimLabConfig | None = None,
        seed: int = 1,
        agents: int | None = None,
    ) -> None:
        super().__init__()
        self._config = config or load_config()
        self._seed = int(seed)
        self._agent_count = max(
            1, int(self._config.agent.count if agents is None else agents)
        )
        self._runner = Runner(self._config, self._seed, agents=self._agent_count)
        self._runner.setup()
        self._latest_snapshot = self._runner._world.snapshot()
        self._initialize_controls()
        self._history = WorkerHistory(theme.SNAPSHOT_HISTORY, self._latest_snapshot)
        self._steps_per_second = 0.0
        self._tick_ms = 0.0
        self._components = WorkerComponents(
            self._config,
            self._seed,
            self._agent_count,
            self._runner._world._rng,
        )
        self._latest_frame = None
        self._latest_frame = SimulationFrame(
            self._latest_snapshot,
            MappingProxyType({}),
            self._components.drive_names,
        )

    def _advance_one_step(
        self,
    ) -> tuple[dict[str, Action], object, StepResult, object] | None:
        with self._lock:
            if self._stopping:
                return None
            world = self._runner._world
            action_step = self._runner._step_count
            actions = self._history.action_for(action_step)
        if actions is None:
            actions, details = self._components.actions_for(world, action_step)
        else:
            details = MappingProxyType(
                {
                    agent_id: AgentFrame(action, None, ())
                    for agent_id, action in actions.items()
                }
            )
        with self._lock:
            self._history.record(action_step, actions)
        result = self._runner.run_step(
            actions,
            include_observations=False,
            include_snapshot=False,
            include_agent_states=False,
        )
        checkpoint = None
        if self._runner._step_count % theme.HISTORY_SNAPSHOT_INTERVAL == 0:
            checkpoint = world.snapshot()
            with self._lock:
                self._history.add_snapshot(checkpoint)
        return actions, details, result, checkpoint

    @Slot()
    def run(self) -> None:
        previous = time.perf_counter()
        last_publish = previous
        last_metrics = previous
        window_steps = 0
        step_credit = 0.0
        seek_target: int | None = None
        latest_details = self._latest_frame.agents
        self.snapshotReady.emit(self.latest_snapshot)
        self.frameReady.emit(self.latest_frame)
        self.historyReady.emit(self._history.take_batch())

        while True:
            now = time.perf_counter()
            elapsed = now - previous
            previous = now
            with self._lock:
                if self._stopping:
                    break
                requested_seed = self._requested_seed
                self._requested_seed = None
                requested_snapshot = self._requested_snapshot
                self._requested_snapshot = None
                requested_seek_step = self._requested_seek_step
                self._requested_seek_step = None
                requested_scenario = self._requested_scenario
                self._requested_scenario = None
                requested_replay = self._requested_replay
                self._requested_replay = False
                playing = self._playing
                speed = self._speed
                step_budget = self._step_budget
                self._step_budget = 0

            if requested_scenario is not None:
                seek_target = None
                requested_seed, self._config = requested_scenario
                self._agent_count = max(1, int(self._config.agent.count))
                with self._lock:
                    self._seed = requested_seed
                    self._runner = Runner(
                        self._config, requested_seed, agents=self._agent_count
                    )
                    self._runner.setup()
                    self._latest_snapshot = self._runner._world.snapshot()
                    self._components.update_config(
                        self._config,
                        self._seed,
                        self._agent_count,
                        self._runner._world._rng,
                    )
                    self._latest_frame = SimulationFrame(
                        self._latest_snapshot,
                        MappingProxyType({}),
                        self._components.drive_names,
                    )
                    latest_details = self._latest_frame.agents
                    self._history.reset(self._latest_snapshot)
                    step_credit = 0.0
                    window_steps = 0
                self.snapshotReady.emit(self.latest_snapshot)
                self.frameReady.emit(self.latest_frame)
                self.historyReady.emit(self._history.take_batch())
            elif requested_seed is not None:
                seek_target = None
                with self._lock:
                    self._seed = requested_seed
                    self._runner = Runner(
                        self._config, requested_seed, agents=self._agent_count
                    )
                    self._runner.setup()
                    self._latest_snapshot = self._runner._world.snapshot()
                    self._components.update_config(
                        self._config,
                        self._seed,
                        self._agent_count,
                        self._runner._world._rng,
                    )
                    self._latest_frame = SimulationFrame(
                        self._latest_snapshot,
                        MappingProxyType({}),
                        self._components.drive_names,
                    )
                    latest_details = self._latest_frame.agents
                    self._history.reset(self._latest_snapshot)
                    step_credit = 0.0
                    window_steps = 0
                self.snapshotReady.emit(self.latest_snapshot)
                self.frameReady.emit(self.latest_frame)
                self.historyReady.emit(self._history.take_batch())
            if requested_replay:
                with self._lock:
                    recorded_steps = sorted(self._history.actions)
                    replay_snapshot = self._history.replay_origin()
                    if recorded_steps and replay_snapshot is not None:
                        self._runner.restore(replay_snapshot)
                        self._latest_snapshot = replay_snapshot
                        self._latest_frame = SimulationFrame(
                            replay_snapshot,
                            MappingProxyType({}),
                            self._components.drive_names,
                        )
                        self._history.replay_to(max(recorded_steps) + 1)
                        self.frameReady.emit(self._latest_frame)
                        self.snapshotReady.emit(replay_snapshot)
            if requested_snapshot is not None:
                with self._lock:
                    snapshot = (
                        requested_snapshot.snapshot
                        if isinstance(requested_snapshot, SimulationFrame)
                        else requested_snapshot
                    )
                    preserve_log = isinstance(requested_snapshot, SimulationFrame)
                    self._runner.restore(snapshot, preserve_action_log=preserve_log)
                    self._agent_count = len(snapshot.agent_states)
                    self._runner._world.width = snapshot.width
                    self._runner._world.height = snapshot.height
                    self._runner._world.dt = snapshot.dt
                    self._components.agent_count = self._agent_count
                    self._components.reset(self._runner._world._rng)
                    if not preserve_log:
                        self._history.reset(snapshot)
                        self._runner._snapshot_manager._snapshots.clear()
                    self._latest_snapshot = snapshot
                    self._latest_frame = (
                        requested_snapshot
                        if isinstance(requested_snapshot, SimulationFrame)
                        else SimulationFrame(
                            snapshot,
                            MappingProxyType({}),
                            self._components.drive_names,
                        )
                    )
                    latest_details = self._latest_frame.agents
                    if not preserve_log:
                        self._runner._snapshot_manager.add(snapshot)
                    if preserve_log:
                        self._history.add_snapshot(snapshot)
                    seek_target = requested_seek_step
                self.snapshotReady.emit(snapshot)
                self.frameReady.emit(self._latest_frame)
                if not preserve_log:
                    self.historyReady.emit(self._history.take_batch())

            if playing:
                if speed == "Max":
                    steps_to_run = theme.MAX_BATCH_STEPS
                else:
                    multiplier = float(speed.removesuffix("x"))
                    step_credit += elapsed * theme.BASE_STEPS_PER_SECOND * multiplier
                    steps_to_run = int(step_credit)
                    step_credit -= steps_to_run
                    if steps_to_run == 0 and step_budget == 0:
                        self._wake.wait(0.002)
                        self._wake.clear()
                        continue
            else:
                steps_to_run = 0
                if step_budget == 0 and seek_target is None:
                    self._wake.wait(0.05)
                    self._wake.clear()
                    continue

            if playing and self._tick_ms > _TICK_BUDGET_MS:
                # Auto-throttle: a tick overrunning the frame budget freezes
                # rendering, so lower the step count instead of stalling.
                steps_to_run = min(steps_to_run, 1)
            if seek_target is not None:
                steps_to_run = min(
                    theme.MAX_BATCH_STEPS,
                    max(0, seek_target - self._runner._step_count),
                )
            steps_to_run += step_budget
            checkpoint_snapshot = None
            for _ in range(steps_to_run):
                _t0 = time.perf_counter()
                advanced = self._advance_one_step()
                if advanced is None:
                    break
                _actions, details, result, checkpoint_snapshot = advanced
                latest_details = details
                window_steps += 1
                self._tick_ms = self._tick_ms * 0.9 + (
                    (time.perf_counter() - _t0) * 1000.0 * 0.1
                )
                if result.terminated:
                    seek_target = None
                    with self._lock:
                        self._playing = False
                    break
                if seek_target is not None and self._runner._step_count >= seek_target:
                    seek_target = None
                    break

            now = time.perf_counter()
            publish_now = now - last_publish >= theme.SNAPSHOT_EMIT_INTERVAL_S
            if publish_now or (not playing and window_steps > 0):
                snapshot = checkpoint_snapshot
                if snapshot is None or snapshot.step != self._runner._step_count:
                    snapshot = self._runner._world.snapshot()
                frame = SimulationFrame(
                    snapshot, latest_details, self._components.drive_names
                )
                with self._lock:
                    self._latest_snapshot = snapshot
                    self._latest_frame = frame
                if self._history.pending:
                    self.historyReady.emit(self._history.take_batch())
                last_publish = now
            elif self._history.pending and not playing:
                self.historyReady.emit(self._history.take_batch())
            if now - last_metrics >= theme.METRICS_INTERVAL_S:
                rate = window_steps / max(now - last_metrics, 1e-9)
                with self._lock:
                    self._steps_per_second = rate
                    current_step = self._runner._step_count
                    current_seed = self._seed
                self.metricsReady.emit(
                    rate, float(now - last_metrics), current_step, current_seed
                )
                self.tickTimeReady.emit(self._tick_ms)
                window_steps = 0
                last_metrics = now
            if playing and speed == "Max":
                time.sleep(theme.MAX_BATCH_YIELD_S)
        self.stopped.emit()
