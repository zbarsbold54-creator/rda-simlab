"""Main Simulation Studio window."""

from __future__ import annotations

import time
from collections import OrderedDict
from pathlib import Path
from types import MappingProxyType

from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QWidget,
)

from simlab.core.config import load_config
from simlab.core.types import Snapshot
from simlab.gui import theme
from simlab.gui.architecture import ArchitectureWidget
from simlab.gui.cockpit import CockpitWidget
from simlab.gui.drive_plot import DrivePlotWidget
from simlab.gui.frame import SimulationFrame
from simlab.gui.inspector import InspectorWidget
from simlab.gui.scenario import ScenarioWidget
from simlab.gui.sim_worker import SimWorker
from simlab.gui.toolbar import SessionToolbar, SimulationToolbar
from simlab.gui.world_view import WorldView


class MainWindow(QMainWindow):
    def __init__(self, seed: int = 1, agent_count: int | None = None) -> None:
        super().__init__()
        stylesheet_path = Path(__file__).with_name("style.qss")
        stylesheet = stylesheet_path.read_text(encoding="utf-8")
        chevron_path = (stylesheet_path.parent / "icons" / "chevron.svg").as_posix()
        stylesheet = stylesheet.replace(
            "url(icons/chevron.svg)", f'url("{chevron_path}")'
        )
        application = QApplication.instance()
        if application is not None:
            application.setStyleSheet(stylesheet)
            application.setFont(QFont(theme.UI_FONT, theme.UI_FONT_SIZE_PT))
        self.setWindowTitle("SimLab - Simulation Studio")
        self.resize(theme.WINDOW_WIDTH, theme.WINDOW_HEIGHT)
        self.setMinimumSize(theme.WINDOW_MIN_WIDTH, theme.WINDOW_MIN_HEIGHT)
        self.toolbar = SimulationToolbar(self)
        self.addToolBar(self.toolbar)
        self.session_toolbar = SessionToolbar(self)
        self.addToolBarBreak(Qt.ToolBarArea.TopToolBarArea)
        self.addToolBar(self.session_toolbar)
        self.world_view = WorldView(self)
        self.setCentralWidget(self.world_view)
        self._render_frames = 0
        self._render_fps = 0.0
        self._last_fps_time = time.perf_counter()
        self._step_rate = 0.0
        self._step = 0
        self._seed = seed
        self._thread = QThread(self)
        config = load_config()
        self.worker = SimWorker(config, seed, agents=agent_count)
        self._config = config
        self._frame_history = OrderedDict()
        self._snapshot_history = OrderedDict()
        self._selected_agent: str | None = None
        policies, drives = self.worker.registry_options
        self.scenario = ScenarioWidget(config, policies, drives, seed)
        self.scenario_dock = QDockWidget("Scenario", self)
        self.scenario_dock.setWidget(self.scenario)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.scenario_dock)
        self.inspector = InspectorWidget()
        self.inspector_dock = QDockWidget("Inspector", self)
        self.inspector_dock.setWidget(self.inspector)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.inspector_dock)
        self.arch = ArchitectureWidget(config)
        self.arch_dock = QDockWidget("Architecture", self)
        self.arch_dock.setWidget(self.arch)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.arch_dock)
        self.drive_plot = DrivePlotWidget()
        self.plot_dock = QDockWidget("Drives and Timeline", self)
        self.plot_dock.setWidget(self.drive_plot)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.plot_dock)
        self.cockpit = CockpitWidget(config)
        self.cockpit_dock = QDockWidget("Research Cockpit", self)
        self.cockpit_dock.setWidget(self.cockpit)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.cockpit_dock)
        self.cockpit_dock.hide()
        self.scenario_dock.setMinimumWidth(theme.INSPECTOR_WIDTH)
        self.inspector_dock.setMinimumWidth(theme.INSPECTOR_WIDTH)
        self.plot_dock.setMinimumHeight(theme.PLOT_HEIGHT)
        self.worker.moveToThread(self._thread)
        self._thread.started.connect(self.worker.run)
        self.worker.snapshotReady.connect(self._accept_snapshot)
        self.worker.frameReady.connect(self._accept_frame)
        self.worker.historyReady.connect(self._accept_history)
        self.worker.metricsReady.connect(self._accept_metrics)
        self.worker.stopped.connect(self._thread.quit)
        self.toolbar.playPause.connect(
            self.worker.set_playing, Qt.ConnectionType.DirectConnection
        )
        self.toolbar.singleStep.connect(
            self.worker.request_step, Qt.ConnectionType.DirectConnection
        )
        self.toolbar.speedChanged.connect(
            self.worker.set_speed, Qt.ConnectionType.DirectConnection
        )
        self.toolbar.resetRequested.connect(self._request_reset)
        self.session_toolbar.saveSnapshotRequested.connect(self._save_snapshot)
        self.session_toolbar.loadSnapshotRequested.connect(self._load_snapshot)
        self.session_toolbar.recordingChanged.connect(
            self.worker.set_recording, Qt.ConnectionType.DirectConnection
        )
        self.session_toolbar.replayRequested.connect(self._start_replay)
        self.world_view.agentSelected.connect(self._select_agent)
        self.scenario.applyRequested.connect(self._apply_scenario)
        self.arch.playbackRequested.connect(self._apply_arch)
        self.cockpit.playbackRequested.connect(self._apply_arch)
        self.drive_plot.timelineChanged.connect(self._seek_timeline)
        self.session_toolbar.follow.toggled.connect(self.world_view.set_follow_selected)
        for name, checkbox in self.session_toolbar.overlay_checks.items():
            checkbox.toggled.connect(
                lambda enabled, overlay=name: self.world_view.set_overlay(
                    overlay, enabled
                )
            )
        view_menu = self.menuBar().addMenu("&View")
        self._show_cockpit_action = view_menu.addAction(
            "Research Cockpit", self._toggle_cockpit
        )
        self._show_cockpit_action.setCheckable(True)
        self._timer = QTimer(self)
        self._timer.setInterval(theme.TIMER_INTERVAL_MS)
        self._timer.timeout.connect(self._render_latest)
        self._thread.start()
        self._timer.start()
        self.status_label = QLabel("Starting simulation…")
        status_bar = self.statusBar()
        status_bar.layout().setContentsMargins(
            theme.STATUS_TEXT_INSET,
            0,
            theme.STATUS_TEXT_INSET,
            0,
        )
        self._status_inset = QWidget()
        self._status_inset.setObjectName("statusInset")
        self._status_inset.setFixedWidth(theme.STATUS_TEXT_INSET)
        status_bar.addWidget(self._status_inset)
        status_bar.addWidget(self.status_label, 1)

    def _accept_snapshot(self, snapshot) -> None:
        self._step = snapshot.step
        self._seed = self.worker.seed
        self.world_view.set_snapshot(snapshot)

    def _accept_frame(self, frame) -> None:
        self._frame_history[frame.snapshot.step] = frame
        self._frame_history.move_to_end(frame.snapshot.step)
        while len(self._frame_history) > theme.SNAPSHOT_HISTORY:
            self._frame_history.popitem(last=False)
        self.world_view.set_frame(frame)
        self.inspector.set_agent(frame, self._selected_agent)
        self.drive_plot.update_frame(frame)

    def _accept_history(self, snapshots) -> None:
        for snapshot in snapshots:
            self._snapshot_history[snapshot.step] = snapshot
            self._snapshot_history.move_to_end(snapshot.step)
        while len(self._snapshot_history) > theme.SNAPSHOT_HISTORY:
            self._snapshot_history.popitem(last=False)
        if self._snapshot_history:
            self.drive_plot.set_timeline_range(
                next(iter(self._snapshot_history)),
                self.worker.latest_snapshot.step,
                self.worker.latest_snapshot.step,
            )

    def _request_reset(self, seed: int) -> None:
        self._frame_history.clear()
        self._snapshot_history.clear()
        self.worker.request_reset(seed)

    def _start_replay(self) -> None:
        self.toolbar.set_playing(True)
        self.worker.request_replay()

    def _select_agent(self, agent_id: str | None) -> None:
        self._selected_agent = agent_id
        self.drive_plot.select_agent(agent_id)
        for historical_frame in self._frame_history.values():
            self.drive_plot.update_frame(historical_frame)
        frame = self.worker.latest_frame
        self.inspector.set_agent(frame, agent_id)
        self.drive_plot.update_frame(frame)

    def _apply_scenario(self, config) -> None:
        self._config = config
        self.scenario._seed = self._seed
        self.scenario.summary.setText(
            f"Seed {self._seed} · {config.agent.count} agents · {config.hazard.count} hazards"
        )
        self._frame_history.clear()
        self._snapshot_history.clear()
        self.worker.request_scenario(self._seed, config)

    def _toggle_cockpit(self, visible: bool) -> None:
        self.cockpit_dock.setVisible(visible)

    def _apply_arch(self, config) -> None:
        self._config = config
        self._frame_history.clear()
        self._snapshot_history.clear()
        self.worker.request_scenario(self._seed, config)

    def _seek_timeline(self, step: int) -> None:
        if not self._snapshot_history:
            return
        eligible_steps = [
            snapshot_step
            for snapshot_step in self._snapshot_history
            if snapshot_step <= step
        ]
        snapshot_step = max(eligible_steps, default=min(self._snapshot_history))
        snapshot = self._snapshot_history[snapshot_step]
        frame = self._frame_history.get(snapshot_step)
        if frame is None:
            frame = SimulationFrame(snapshot, MappingProxyType({}), ())
        self.world_view.set_frame(frame)
        self.inspector.set_agent(frame, self._selected_agent)
        self.toolbar.set_playing(False)
        self.worker.request_seek(frame, step)

    def _save_snapshot(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save snapshot", "simlab_snapshot.json", "SimLab snapshot (*.json)"
        )
        if not path:
            return
        base_path = Path(path).with_suffix("")
        self.worker.latest_snapshot.save(str(base_path))
        self.status_label.setText(f"Snapshot saved: {base_path}.json + .npz")

    def _load_snapshot(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load snapshot", "", "SimLab snapshot (*.json *.npz)"
        )
        if not path:
            return
        try:
            snapshot = Snapshot.load(path)
        except (OSError, ValueError, KeyError) as error:
            QMessageBox.warning(self, "Could not load snapshot", str(error))
            return
        self._frame_history.clear()
        self._snapshot_history.clear()
        self.worker.request_restore(snapshot)

    def _accept_metrics(
        self, steps_per_second: float, elapsed: float, step: int, seed: int
    ) -> None:
        del elapsed
        self._step_rate = steps_per_second
        self._step = step
        self._seed = seed

    def _render_latest(self) -> None:
        frame = self.worker.latest_frame
        snapshot = frame.snapshot
        if (
            self.world_view._snapshot is None
            or self.world_view._snapshot.step != snapshot.step
        ):
            self._accept_frame(frame)
        self._step = snapshot.step
        self._seed = self.worker.seed
        self._render_frames += 1
        now = time.perf_counter()
        elapsed = now - self._last_fps_time
        if elapsed >= 1.0:
            self._render_fps = self._render_frames / elapsed
            self._render_frames = 0
            self._last_fps_time = now
        sim_time = self._step * snapshot.dt
        selected_count = sum(state.alive for state in snapshot.agent_states.values())
        self.status_label.setText(
            f"Step {self._step} | Sim time {sim_time:.1f}s | "
            f"Render {self._render_fps:.1f} FPS | Sim {self._step_rate:.1f} steps/s | Seed {self._seed} | Alive {selected_count}"
        )

    def closeEvent(self, event) -> None:
        self._timer.stop()
        self.worker.stop()
        self._thread.quit()
        self._thread.wait(5000)
        super().closeEvent(event)
