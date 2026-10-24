"""Research cockpit dock: build/run/watch/compare experiments in the GUI.

This widget is torch-free at import time (it only imports the torch-free
``simlab.research`` package and Qt). Torch is loaded lazily in a worker thread
only when arm parameter counts or gate traces are computed. The widget never
names a specific architecture class.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from simlab.core.config import load_config
from simlab.gui import theme
from simlab.research.experiment import (
    DEFAULT_WORKERS,
    Arm,
    Experiment,
    arm_param_counts,
    dump_experiment_toml,
    load_experiment_toml,
)
from simlab.research.follower import FollowState, checkpoint_readable, poll_latest
from simlab.research.gates import correlate_events, gate_stats, rail_warning
from simlab.research.logs import curve_series
from simlab.research.queue import (
    PAUSE_FLAG,
    STOP_FLAG,
    QueueManager,
    clear_flag,
    set_flag,
)
from simlab.research.results import (
    METRIC_KEYS,
    difference_rows,
    load_eval_file,
    to_csv,
)

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"
COCKPIT_DIR = RUNS_DIR / "cockpit"


class ParamCountThread(QThread):
    """Computes arm parameter counts (torch) off the GUI thread."""

    finishedTable = Signal(list)
    failed = Signal(str)

    def __init__(self, arms: list[Arm], parent=None) -> None:
        super().__init__(parent)
        self._arms = arms

    def run(self) -> None:
        try:
            self.finishedTable.emit(arm_param_counts(self._arms))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


def _parse_seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, _, hi = part.partition("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


class CockpitWidget(QWidget):
    """Research cockpit with six tabs."""

    playbackRequested = Signal(object)  # SimLabConfig for the world view

    def __init__(self, config, parent=None) -> None:
        super().__init__(parent)
        self._config = config or load_config()
        self._base = load_config()
        self._follow_state = FollowState()
        self._last_checkpoint: Path | None = None
        self._gate_data: dict[str, Any] = {}
        self._build_ui()

    # ------------------------------------------------------------ UI shell
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_experiment_tab(), "Experiment")
        self.tabs.addTab(self._build_queue_tab(), "Queue")
        self.tabs.addTab(self._build_live_tab(), "Live")
        self.tabs.addTab(self._build_curves_tab(), "Curves")
        self.tabs.addTab(self._build_results_tab(), "Results")
        self.tabs.addTab(self._build_gates_tab(), "Gates")
        root.addWidget(self.tabs)

    # ------------------------------------------------------- Experiment tab
    def _build_experiment_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        form = QFormLayout()
        self.exp_name = QLineEdit("rda_grid")
        self.total_steps = QSpinBox()
        self.total_steps.setRange(1024, 5_000_000)
        self.total_steps.setValue(16_384)
        self.rollout = QSpinBox()
        self.rollout.setRange(64, 4096)
        self.rollout.setValue(1024)
        self.workers = QSpinBox()
        self.workers.setRange(1, 16)
        self.workers.setValue(DEFAULT_WORKERS)
        self.train_seeds = QLineEdit("100000, 100001")
        self.dev_seeds = QLineEdit("0, 1, 2, 3, 4")
        self.test_seeds = QLineEdit("")
        self.control_arm = QComboBox()
        self.control_arm.addItem("(none)")
        form.addRow("Name", self.exp_name)
        form.addRow("Total steps", self.total_steps)
        form.addRow("Rollout length", self.rollout)
        form.addRow("Workers", self.workers)
        form.addRow("Train seeds", self.train_seeds)
        form.addRow("Dev seeds", self.dev_seeds)
        form.addRow("Test seeds", self.test_seeds)
        form.addRow("Control arm", self.control_arm)
        layout.addLayout(form)

        self.workers_warn = QLabel("")
        self.workers_warn.setStyleSheet(f"color: {theme.ENERGY_LOW};")
        layout.addWidget(self.workers_warn)
        self.workers.valueChanged.connect(self._update_workers_warn)
        self._update_workers_warn()

        self.scen_balanced = QCheckBox("balanced")
        self.scen_balanced.setChecked(True)
        self.scen_hard = QCheckBox("hard")
        self.scen_hard.setChecked(True)
        scen_row = QHBoxLayout()
        scen_row.addWidget(QLabel("Scenarios:"))
        scen_row.addWidget(self.scen_balanced)
        scen_row.addWidget(self.scen_hard)
        scen_row.addStretch(1)
        layout.addLayout(scen_row)

        layout.addWidget(QLabel("Arms (pick presets from the registry):"))
        self.arm_list = QListWidget()
        self.arm_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        layout.addWidget(self.arm_list, 1)

        self.param_table = QTableWidget(0, 3)
        self.param_table.setHorizontalHeaderLabels(["Arm", "Parameters", "Arch hash"])
        self.param_table.setMaximumHeight(160)
        layout.addWidget(self.param_table)

        self.exp_status = QLabel("")
        layout.addWidget(self.exp_status)

        row = QHBoxLayout()
        self.validate_btn = QPushButton("Validate")
        self.save_btn = QPushButton("Save TOML")
        row.addWidget(self.validate_btn)
        row.addWidget(self.save_btn)
        row.addStretch(1)
        layout.addLayout(row)

        self._populate_arms()
        self.arm_list.itemSelectionChanged.connect(self._schedule_param_counts)
        self.validate_btn.clicked.connect(self._validate_experiment)
        self.save_btn.clicked.connect(self._save_experiment)
        return page

    def _populate_arms(self) -> None:
        from simlab.training.arch_spec import list_specs

        for name in list_specs("backbone"):
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(
                Qt.CheckState.Checked
                if name.startswith("rda")
                else Qt.CheckState.Unchecked
            )
            self.arm_list.addItem(item)

    def _selected_arms(self) -> list[Arm]:
        arms: list[Arm] = []
        for i in range(self.arm_list.count()):
            item = self.arm_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                arms.append(
                    Arm(
                        name=item.text(),
                        backbone=item.text(),
                        drive_core="rda8",
                        fusion="film",
                        obs_mode="obs_drives",
                    )
                )
        return arms

    def _update_workers_warn(self) -> None:
        if self.workers.value() != DEFAULT_WORKERS:
            self.workers_warn.setText(
                f"Warning: workers = {self.workers.value()} differs from the "
                f"pinned default {DEFAULT_WORKERS}."
            )
        else:
            self.workers_warn.setText("")

    def _schedule_param_counts(self) -> None:
        arms = self._selected_arms()
        if not arms:
            self.param_table.setRowCount(0)
            return
        self.exp_status.setText("Computing parameter counts…")
        self._pc_thread = ParamCountThread(arms, self)
        self._pc_thread.finishedTable.connect(self._on_param_counts)
        self._pc_thread.failed.connect(self._on_param_count_failed)
        self._pc_thread.start()

    def _on_param_counts(self, rows: list[dict[str, Any]]) -> None:
        self.param_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            self.param_table.setItem(r, 0, QTableWidgetItem(row["name"]))
            self.param_table.setItem(r, 1, QTableWidgetItem(str(row["params"])))
            self.param_table.setItem(r, 2, QTableWidgetItem(row["hash"]))
        self.exp_status.setText(f"{len(rows)} arm(s); param counts computed.")
        self._refresh_control_arm()

    def _on_param_count_failed(self, msg: str) -> None:
        self.exp_status.setText(f"Param count failed: {msg}")

    def _refresh_control_arm(self) -> None:
        current = self.control_arm.currentText()
        self.control_arm.blockSignals(True)
        self.control_arm.clear()
        self.control_arm.addItem("(none)")
        for i in range(self.arm_list.count()):
            item = self.arm_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                self.control_arm.addItem(item.text())
        if current and current != "(none)":
            idx = self.control_arm.findText(current)
            if idx >= 0:
                self.control_arm.setCurrentIndex(idx)
        self.control_arm.blockSignals(False)

    def _build_experiment(self) -> Experiment:
        arms = self._selected_arms()
        scenarios = []
        if self.scen_balanced.isChecked():
            scenarios.append("balanced")
        if self.scen_hard.isChecked():
            scenarios.append("hard")
        control = self.control_arm.currentText()
        exp = Experiment(
            name=self.exp_name.text().strip() or "untitled",
            arms=arms,
            total_steps=self.total_steps.value(),
            rollout_length=self.rollout.value(),
            workers=self.workers.value(),
            train_seeds=_parse_seeds(self.train_seeds.text()),
            dev_seeds=_parse_seeds(self.dev_seeds.text()),
            test_seeds=_parse_seeds(self.test_seeds.text()),
            eval_scenarios=scenarios,
            control_arm=None if control == "(none)" else control,
        )
        return exp

    def _validate_experiment(self) -> None:
        exp = self._build_experiment()
        errors, warnings = exp.validate()
        parts = []
        if errors:
            parts.append("ERRORS:\n" + "\n".join(f"- {e}" for e in errors))
        if warnings:
            parts.append("WARNINGS:\n" + "\n".join(f"- {w}" for w in warnings))
        if not parts:
            parts.append("Experiment is valid.")
        self.exp_status.setText("\n".join(parts))

    def _save_experiment(self) -> None:
        exp = self._build_experiment()
        errors, _warnings = exp.validate()
        if errors:
            QMessageBox.warning(self, "Invalid experiment", "\n".join(errors))
            return
        if exp.requires_test_confirm():
            answer = QMessageBox.question(
                self,
                "Test seeds = final report only",
                "This experiment includes test seeds (1000-1019). These are "
                "for the final report only; continue?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        COCKPIT_DIR.mkdir(parents=True, exist_ok=True)
        default = str(COCKPIT_DIR / f"{exp.name}.toml")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save experiment", default, "TOML (*.toml)"
        )
        if not path:
            return
        Path(path).write_text(dump_experiment_toml(exp), encoding="utf-8")
        self.exp_status.setText(f"Saved {path}")

    # --------------------------------------------------------- Queue tab
    def _build_queue_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        row = QHBoxLayout()
        self.queue_dir_label = QLabel("Queue: ‐")
        row.addWidget(self.queue_dir_label)
        self.launch_btn = QPushButton("Launch grid")
        row.addWidget(self.launch_btn)
        row.addStretch(1)
        layout.addLayout(row)

        self.queue_status = QLabel("No queue running.")
        layout.addWidget(self.queue_status)

        self.queue_table = QTableWidget(0, 5)
        self.queue_table.setHorizontalHeaderLabels(
            ["Job", "Arm", "Seed", "Status", "Exit"]
        )
        layout.addWidget(self.queue_table, 1)

        controls = QHBoxLayout()
        self.pause_btn = QPushButton("Pause after current run")
        self.resume_btn = QPushButton("Resume")
        self.stop_btn = QPushButton("Stop")
        controls.addWidget(self.pause_btn)
        controls.addWidget(self.resume_btn)
        controls.addWidget(self.stop_btn)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.launch_btn.clicked.connect(self._launch_grid)
        self.pause_btn.clicked.connect(lambda: set_flag(self._queue_dir(), PAUSE_FLAG))
        self.resume_btn.clicked.connect(
            lambda: clear_flag(self._queue_dir(), PAUSE_FLAG)
        )
        self.stop_btn.clicked.connect(lambda: set_flag(self._queue_dir(), STOP_FLAG))

        self._queue_timer = QTimer(self)
        self._queue_timer.setInterval(1500)
        self._queue_timer.timeout.connect(self._refresh_queue)
        self._queue_timer.start()
        return page

    def _queue_dir(self) -> Path:
        return Path(self.queue_dir_label.text().removeprefix("Queue: "))

    def _launch_grid(self) -> None:
        exp = self._build_experiment()
        errors, _ = exp.validate()
        if errors:
            QMessageBox.warning(self, "Invalid experiment", "\n".join(errors))
            return
        qdir = COCKPIT_DIR / exp.name
        if (qdir / "queue.json").exists():
            answer = QMessageBox.question(
                self,
                "Queue exists",
                f"Queue {qdir} already exists. Resume it?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            QueueManager(qdir)
        else:
            QueueManager.create(qdir, exp)
        self.queue_dir_label.setText(f"Queue: {qdir}")
        # Spawn the detached server subprocess (one instance, lock file).
        proc = sys.executable
        args = [proc, "-m", "simlab.research.queue", "serve", str(qdir)]
        import subprocess

        self._server = subprocess.Popen(args)
        self.queue_status.setText("Queue server started.")
        self._refresh_queue()

    def _refresh_queue(self) -> None:
        try:
            qdir = self._queue_dir()
        except Exception:  # noqa: BLE001
            return
        if not qdir.exists() or not (qdir / "queue.json").exists():
            return
        manager = QueueManager(qdir)
        status = manager.status()
        jobs = status["jobs"]
        self.queue_table.setRowCount(len(jobs))
        for r, job in enumerate(jobs):
            self.queue_table.setItem(r, 0, QTableWidgetItem(job["id"]))
            self.queue_table.setItem(r, 1, QTableWidgetItem(job["arm_name"]))
            self.queue_table.setItem(r, 2, QTableWidgetItem(str(job["train_seed"])))
            self.queue_table.setItem(r, 3, QTableWidgetItem(job["status"]))
            self.queue_table.setItem(
                r, 4, QTableWidgetItem(str(job.get("exit_code") or ""))
            )
        counts = status["counts"]
        self.queue_status.setText(
            f"queued={counts['queued']} running={counts['running']} "
            f"done={counts['done']} failed={counts['failed']}"
        )

    # -------------------------------------------------------- Live tab
    def _build_live_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        form = QFormLayout()
        self.live_run = QComboBox()
        self.live_run.setEditable(True)
        form.addRow("Run directory", self.live_run)
        self.follow_cb = QCheckBox("Follow latest checkpoint")
        self.follow_interval = QSpinBox()
        self.follow_interval.setRange(100, 60_000)
        self.follow_interval.setValue(3000)
        self.follow_interval.setSuffix(" ms")
        self.follow_interval.setEnabled(False)
        form.addRow(self.follow_cb, self.follow_interval)
        layout.addLayout(form)

        self.live_status = QLabel("Select a run directory to follow.")
        layout.addWidget(self.live_status)

        self.live_plot = pg.PlotWidget()
        self.live_plot.setBackground(theme.PANEL)
        self.live_plot.showGrid(x=True, y=True, alpha=0.18)
        self.live_plot.addLegend()
        layout.addWidget(self.live_plot, 1)

        self.follow_cb.toggled.connect(self.follow_interval.setEnabled)
        self.follow_cb.toggled.connect(self._toggle_follow)
        self._follow_timer = QTimer(self)
        self._follow_timer.timeout.connect(self._poll_follow)
        return page

    def _toggle_follow(self, on: bool) -> None:
        if on:
            self._follow_timer.start(self.follow_interval.value())
            self.follow_interval.valueChanged.connect(
                lambda v: self._follow_timer.setInterval(v)
            )
        else:
            self._follow_timer.stop()

    def _live_run_dir(self) -> Path | None:
        text = self.live_run.currentText().strip()
        if not text:
            return None
        p = Path(text)
        return p if p.is_dir() else None

    def _poll_follow(self) -> None:
        run_dir = self._live_run_dir()
        if run_dir is None:
            self.live_status.setText("Run directory not found.")
            return
        ckpt = poll_latest(run_dir, self._follow_state)
        if ckpt is None:
            self.live_status.setText(f"Following {run_dir} (no new checkpoint).")
            return
        if not checkpoint_readable(ckpt):
            # Keep the previous good checkpoint and tell the user.
            self.live_status.setText(
                f"Following {run_dir} (new checkpoint unreadable, keeping previous)."
            )
            return
        self._last_checkpoint = ckpt
        # Rebuild a learned_torch playback config for the world view.
        arm = self._guess_arm(run_dir)
        cfg = replace(
            self._base,
            policy="learned_torch",
            drive=arm.drive_core,
            drive_config=dict(arm.drive_config) or None,
            policy_checkpoint=str(ckpt),
            policy_cfg=None,
        )
        self.live_status.setText(f"Loaded {ckpt.name}")
        self.playbackRequested.emit(cfg)

    def _guess_arm(self, run_dir: Path) -> Arm:
        # Find the arm from the parent queue's experiment.
        qdir = run_dir.parent.parent
        exp_path = qdir / "experiment.toml"
        if exp_path.exists():
            try:
                exp = load_experiment_toml(exp_path)
                for arm in exp.arms:
                    if run_dir.name.endswith(f"_{arm.name}_"):
                        return arm
            except Exception:
                logging.getLogger(__name__).debug("arm inference failed", exc_info=True)
        return Arm(name="rda_g1", backbone="rda_g1")

    # ------------------------------------------------------- Curves tab
    def _build_curves_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        row = QHBoxLayout()
        self.curves_run = QComboBox()
        self.curves_run.setEditable(True)
        row.addWidget(QLabel("Run directory"))
        row.addWidget(self.curves_run, 1)
        self.curves_metric = QComboBox()
        self.curves_metric.addItems(
            ["mean_return", "entropy", "approx_kl", "value_loss", "cap_pct"]
        )
        row.addWidget(QLabel("Metric"))
        row.addWidget(self.curves_metric)
        self.curves_refresh = QPushButton("Refresh")
        row.addWidget(self.curves_refresh)
        layout.addLayout(row)

        self.curves_plot = pg.PlotWidget()
        self.curves_plot.setBackground(theme.PANEL)
        self.curves_plot.showGrid(x=True, y=True, alpha=0.18)
        self.curves_plot.addLegend()
        layout.addWidget(self.curves_plot, 1)

        self.curves_refresh.clicked.connect(self._plot_curves)
        return page

    def _plot_curves(self) -> None:
        self.curves_plot.clear()
        metric = self.curves_metric.currentText()
        series = curve_series(self.curves_run.currentText())
        x = series["iteration"]
        y = series.get(metric, [])
        if not x or not y:
            self.curves_plot.setTitle("No data")
            return
        self.curves_plot.plot(x, y, pen=pg.mkPen(theme.ACCENT, width=2), name=metric)

    # ------------------------------------------------------- Results tab
    def _build_results_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        row = QHBoxLayout()
        self.results_load = QPushButton("Load eval files…")
        self.results_control = QComboBox()
        row.addWidget(self.results_load)
        row.addWidget(QLabel("Control arm"))
        row.addWidget(self.results_control, 1)
        self.results_csv = QPushButton("Export CSV")
        row.addWidget(self.results_csv)
        layout.addLayout(row)

        self.results_table = QTableWidget(0, 0)
        layout.addWidget(self.results_table, 1)
        self.results_status = QLabel("Load per-seed eval results (JSON) to aggregate.")
        layout.addWidget(self.results_status)

        self._results_rows: dict[str, list] = {}
        self.results_load.clicked.connect(self._load_results)
        self.results_csv.clicked.connect(self._export_results)
        return page

    def _load_results(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Load eval result files", str(RUNS_DIR), "JSON (*.json)"
        )
        if not paths:
            return
        arm_rows: dict[str, list] = {}
        for p in paths:
            arm = Path(p).stem.split("__")[0]
            try:
                rows = load_eval_file(p)
            except Exception as exc:  # noqa: BLE001
                self.results_status.setText(f"Load failed for {p}: {exc}")
                continue
            arm_rows.setdefault(arm, []).extend(rows)
        self._results_rows = arm_rows
        self._render_results_table()
        self._refresh_results_control()

    def _refresh_results_control(self) -> None:
        current = self.results_control.currentText()
        self.results_control.blockSignals(True)
        self.results_control.clear()
        self.results_control.addItem("(none)")
        for arm in self._results_rows:
            self.results_control.addItem(arm)
        if current in self._results_rows:
            self.results_control.setCurrentText(current)
        self.results_control.blockSignals(False)

    def _render_results_table(self) -> None:
        if not self._results_rows:
            self.results_table.setRowCount(0)
            return
        control = self.results_control.currentText()
        control = None if control == "(none)" else control
        # columns: arm, seed, metrics..., [diff metrics...]
        n_extra = 0
        if control in self._results_rows:
            for arm, rows in self._results_rows.items():
                if arm != control and rows:
                    n_extra = len(METRIC_KEYS)
                    break
        headers = ["arm", "seed", *METRIC_KEYS]
        if n_extra:
            headers += [f"{control}_diff_{k}" for k in METRIC_KEYS]
        self.results_table.setColumnCount(len(headers))
        self.results_table.setHorizontalHeaderLabels(headers)
        rows = []
        control_rows = self._results_rows.get(control, []) if control else []
        for arm, seed_rows in self._results_rows.items():
            for r in seed_rows:
                row = [arm, r.seed, *[getattr(r, k) for k in METRIC_KEYS]]
                if control and arm != control and control_rows:
                    diffs = difference_rows([r], control_rows)
                    if diffs and "missing_control" not in diffs[0]:
                        row += [diffs[0].get(k, "") for k in METRIC_KEYS]
                rows.append(row)
        self.results_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                self.results_table.setItem(r, c, QTableWidgetItem(str(val)))
        self.results_status.setText(
            f"{len(rows)} rows from {len(self._results_rows)} arm(s)."
        )

    def _export_results(self) -> None:
        if not self._results_rows:
            return
        control = self.results_control.currentText()
        control = None if control == "(none)" else control
        default = str(RUNS_DIR / "results_table.csv")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export results CSV", default, "CSV (*.csv)"
        )
        if path:
            Path(path).write_text(
                to_csv(self._results_rows, control=control), encoding="utf-8"
            )
            self.results_status.setText(f"Exported {path}")

    # ------------------------------------------------------- Gates tab
    def _build_gates_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        row = QHBoxLayout()
        self.gates_checkpoint = QPushButton("Load trace from checkpoint…")
        self.gates_follow = QPushButton("Use current follow data")
        row.addWidget(self.gates_checkpoint)
        row.addWidget(self.gates_follow)
        row.addStretch(1)
        layout.addLayout(row)

        self.gates_plot = pg.PlotWidget()
        self.gates_plot.setBackground(theme.PANEL)
        self.gates_plot.showGrid(x=True, y=True, alpha=0.18)
        self.gates_plot.addLegend()
        layout.addWidget(self.gates_plot, 1)

        self.gates_status = QLabel("No gate data loaded.")
        layout.addWidget(self.gates_status)

        self.gates_checkpoint.clicked.connect(self._load_gate_checkpoint)
        self.gates_follow.clicked.connect(self._plot_gate_data)
        return page

    def set_gate_data(self, gate: dict[str, Any]) -> None:
        self._gate_data = gate
        self._plot_gate_data()

    def _plot_gate_data(self) -> None:
        self.gates_plot.clear()
        gate = self._gate_data
        if not gate:
            self.gates_status.setText("No gate data.")
            return
        lam = np.asarray(gate.get("lam", []), dtype=float)
        mag = np.asarray(gate.get("mag", []), dtype=float)
        rail = float(gate.get("rail", 1.0))
        n = lam.size
        x = np.arange(n)
        if n:
            self.gates_plot.plot(
                x, lam, pen=pg.mkPen(theme.ACCENT, width=2), name="lambda_t"
            )
        if mag.size:
            self.gates_plot.plot(
                x, mag, pen=pg.mkPen(theme.ENERGY_HIGH, width=2), name="|E|"
            )
            rail_line = pg.InfiniteLine(
                pos=rail, angle=0, pen=pg.mkPen(theme.ENERGY_LOW, width=1)
            )
            self.gates_plot.addItem(rail_line)
        stats = self.analyze_gate_data(gate)
        self.gates_status.setText(
            f"fraction_at_max={stats['fraction_at_max']} "
            f"|E|-rail_warning_runs={len(stats['rail_runs'])} "
            f"corr_lam_events={stats.get('corr_lam_events', 'n/a')}"
        )

    def analyze_gate_data(self, gate: dict[str, Any]) -> dict[str, Any]:
        """Pure analysis reused by tests."""
        lam = np.asarray(gate.get("lam", []), dtype=float)
        mag = np.asarray(gate.get("mag", []), dtype=float)
        rail = float(gate.get("rail", 1.0))
        lam_max = float(gate.get("lam_max", lam.max() if lam.size else 1.0))
        stats = gate_stats(lam, lam_max)
        stats["rail_runs"] = rail_warning(mag, rail)
        events = gate.get("events")
        if isinstance(events, dict):
            corr = {}
            for name, ind in events.items():
                corr[name] = correlate_events(lam, np.asarray(ind, dtype=float))
            stats["corr_lam_events"] = corr
        return stats

    def _load_gate_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick checkpoint", str(RUNS_DIR), "PyTorch (*.pt)"
        )
        if not path:
            return
        try:
            from simlab.research.replay import collect_gate_trace

            gate = collect_gate_trace(path)
        except Exception as exc:  # noqa: BLE001
            self.gates_status.setText(f"Trace failed: {exc}")
            return
        self.set_gate_data(gate)
