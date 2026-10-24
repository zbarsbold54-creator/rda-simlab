"""Architecture panel: build, apply, save/load and launch policies.

This dock drives the architecture-agnostic learner slots. It only ever talks
to the torch-free ``simlab.training.arch_spec`` machinery at import time;
torch is loaded lazily inside the builder thread when the user chooses to
build a policy. The panel never names a specific architecture class.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from PySide6.QtCore import QProcess, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from simlab.training.arch_spec import (
    arch_dir,
    build_policy_cfg,
    dump_arch_toml,
    get_spec,
    list_specs,
    load_arch_toml,
    normalize_config,
    scan_plugin_dir,
    spec_schema,
    validate_config,
    write_template,
)

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"


class BuilderThread(QThread):
    """Runs the (torch) build off the GUI thread."""

    finishedOk = Signal(object, object)  # (param_count, hash)
    failed = Signal(str)

    def __init__(self, selection: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self._selection = selection

    def run(self) -> None:
        try:
            from simlab.training.arch_spec import arch_hash
            from simlab.training.torch_policy import make_policy

            policy_cfg = build_policy_cfg(self._selection)
            policy = make_policy(policy_cfg)
            self.finishedOk.emit(policy.param_count(), arch_hash(self._selection))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class _ProcessLauncher(QWidget):
    """Streams a train/eval subprocess into a log view."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._process: QProcess | None = None
        layout = QVBoxLayout(self)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(120)
        layout.addWidget(QLabel("Launch output"))
        layout.addWidget(self.log_view)
        row = QHBoxLayout()
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        row.addWidget(self.stop_button)
        row.addStretch(1)
        layout.addLayout(row)
        self.status = QLabel("Idle")
        layout.addWidget(self.status)
        self.stop_button.clicked.connect(self._stop)

    def launch(self, args: list[str]) -> None:
        if self._process is not None and self._process.state() != QProcess.ProcessState.NotRunning:
            return
        self.log_view.clear()
        self.status.setText("Starting…")
        self.stop_button.setEnabled(True)
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannel.MergedChannels)
        proc.readyReadStandardOutput.connect(lambda: self._read(proc))
        proc.finished.connect(lambda code, status: self._finished(proc, code, status))
        proc.start(sys.executable, args)
        self._process = proc

    def _read(self, proc: QProcess) -> None:
        data = bytes(proc.readAllStandardOutput())
        self.log_view.appendPlainText(data.decode("utf-8", "replace").rstrip())

    def _finished(self, proc: QProcess, code: int, _status) -> None:
        if proc is self._process:
            self.stop_button.setEnabled(False)
            self.status.setText(f"Exited with code {code}")

    def _stop(self) -> None:
        if self._process is not None:
            self._process.kill()
            self.stop_button.setEnabled(False)
            self.status.setText("Stopped by user")


class ArchitectureWidget(QWidget):
    """Architecture dock: backbone/drive-core/fusion selection + build/launch."""

    playbackRequested = Signal(object)  # SimLabConfig for the world view

    def __init__(self, config, parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._backbone_widgets: dict[str, QWidget] = {}
        self._drive_widgets: dict[str, QWidget] = {}
        self._checkpoint_path: str | None = None
        self._selection: dict[str, Any] = {
            "backbone": self._default_name("backbone"),
            "drive_core": "none",
            "fusion": "concat",
            "obs_mode": "obs_only",
            "seed": 0,
            "backbone_config": {},
            "drive_config": {},
        }
        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        form = QFormLayout()
        self.backbone = QComboBox()
        self.drive = QComboBox()
        self.fusion = QComboBox()
        self.fusion.addItems(["none", "concat", "film"])
        self.obs_mode = QComboBox()
        self.obs_mode.addItems(["obs_only", "obs_drives"])
        self.seed_box = QSpinBox()
        self.seed_box.setRange(0, 10**6)
        form.addRow("Backbone", self.backbone)
        form.addRow("DriveCore", self.drive)
        form.addRow("Fusion", self.fusion)
        form.addRow("Obs mode", self.obs_mode)
        form.addRow("Seed", self.seed_box)
        root.addLayout(form)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        self._config_form = QFormLayout(body)
        scroll.setWidget(body)
        root.addWidget(scroll)

        self.param_label = QLabel("Not built")
        root.addWidget(self.param_label)

        row = QHBoxLayout()
        self.rescan_button = QPushButton("Rescan")
        self.template_button = QPushButton("New architecture template")
        self.build_button = QPushButton("Build")
        self.apply_button = QPushButton("Apply to world")
        row.addWidget(self.rescan_button)
        row.addWidget(self.template_button)
        row.addWidget(self.build_button)
        row.addWidget(self.apply_button)
        root.addLayout(row)

        row2 = QHBoxLayout()
        self.save_button = QPushButton("Save experiment")
        self.load_button = QPushButton("Load experiment")
        self.checkpoint_button = QPushButton("Checkpoint…")
        row2.addWidget(self.save_button)
        row2.addWidget(self.load_button)
        row2.addWidget(self.checkpoint_button)
        root.addLayout(row2)

        self.error_list = QListWidget()
        self.error_list.setMaximumHeight(90)
        root.addWidget(QLabel("Plugin errors"))
        root.addWidget(self.error_list)

        self.launcher = _ProcessLauncher()
        root.addWidget(self.launcher)
        row3 = QHBoxLayout()
        self.train_button = QPushButton("Train (smoke)")
        self.eval_button = QPushButton("Eval")
        row3.addWidget(self.train_button)
        row3.addWidget(self.eval_button)
        root.addLayout(row3)

        self._rescan()
        self.backbone.currentTextChanged.connect(self._rebuild_forms)
        self.drive.currentTextChanged.connect(self._rebuild_forms)
        self.fusion.currentTextChanged.connect(self._rebuild_forms)
        self.obs_mode.currentTextChanged.connect(self._rebuild_forms)
        self.rescan_button.clicked.connect(self._rescan)
        self.template_button.clicked.connect(self._write_template)
        self.build_button.clicked.connect(self._build)
        self.apply_button.clicked.connect(self._apply)
        self.save_button.clicked.connect(self._save)
        self.load_button.clicked.connect(self._load)
        self.checkpoint_button.clicked.connect(self._pick_checkpoint)
        self.train_button.clicked.connect(self._launch_train)
        self.eval_button.clicked.connect(self._launch_eval)

    def _rescan(self) -> None:
        self._errors = scan_plugin_dir(arch_dir())
        self.error_list.clear()
        self.error_list.addItems(self._errors or ["No plugin errors"])
        self._populate_combo(
            self.backbone, list_specs("backbone"), self._default_name("backbone")
        )
        self._populate_combo(self.drive, list_specs("drive_core"), "none")
        self._rebuild_forms()

    @staticmethod
    def _default_name(kind: str) -> str:
        names = list_specs(kind)
        return names[0] if names else ""

    @staticmethod
    def _populate_combo(combo: QComboBox, names: list[str], default: str) -> None:
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(names)
        if current in names:
            combo.setCurrentText(current)
        else:
            combo.setCurrentText(default)
        combo.blockSignals(False)

    # ----------------------------------------------------------------- forms
    def _rebuild_forms(self) -> None:
        self._clear_form(self._config_form)
        self._backbone_widgets = self._build_field_row(
            self._config_form, "backbone", self.backbone.currentText()
        )
        self._drive_widgets = self._build_field_row(
            self._config_form, "drive_core", self.drive.currentText()
        )

    @staticmethod
    def _clear_form(form: QFormLayout) -> None:
        while form.rowCount():
            form.removeRow(0)

    def _build_field_row(
        self, form: QFormLayout, kind: str, name: str
    ) -> dict[str, QWidget]:
        spec = get_spec(kind, name)
        widgets: dict[str, QWidget] = {}
        if spec is None:
            return widgets
        for field in spec_schema(spec):
            w = self._widget_for(field)
            if w is None:
                continue
            label = field["name"]
            if field.get("help"):
                label = f'{field["name"]} (help: {field["help"]})'
            form.addRow(label, w)
            widgets[field["name"]] = w
        return widgets

    def _widget_for(self, field: dict[str, Any]) -> QWidget | None:
        default = field.get("default")
        ftype = field.get("type")
        if ftype == "int":
            box = QSpinBox()
            if field.get("min") is not None:
                box.setMinimum(int(field["min"]))
            if field.get("max") is not None:
                box.setMaximum(int(field["max"]))
            if default is not None:
                box.setValue(int(default))
            return box
        if ftype == "float":
            box = QDoubleSpinBox()
            box.setDecimals(4)
            box.setRange(-1e6, 1e6)
            if field.get("min") is not None:
                box.setMinimum(float(field["min"]))
            if field.get("max") is not None:
                box.setMaximum(float(field["max"]))
            if default is not None:
                box.setValue(float(default))
            return box
        if ftype == "bool":
            box = QCheckBox()
            if default:
                box.setChecked(True)
            return box
        if field.get("choices"):
            box = QComboBox()
            box.addItems([str(c) for c in field["choices"]])
            if default is not None:
                box.setCurrentText(str(default))
            return box
        box = QLineEdit()
        if default is not None:
            box.setText(str(default))
        return box

    def _collect(self) -> dict[str, Any]:
        selection = dict(self._selection)
        selection["backbone"] = self.backbone.currentText()
        selection["drive_core"] = self.drive.currentText()
        selection["fusion"] = self.fusion.currentText()
        selection["obs_mode"] = self.obs_mode.currentText()
        selection["seed"] = int(self.seed_box.value())
        selection["backbone_config"] = self._values_from(self._backbone_widgets)
        selection["drive_config"] = self._values_from(self._drive_widgets)
        return selection

    @staticmethod
    def _values_from(widgets: dict[str, QWidget]) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for name, w in widgets.items():
            if isinstance(w, (QSpinBox, QDoubleSpinBox)):
                values[name] = w.value()
            elif isinstance(w, QCheckBox):
                values[name] = w.isChecked()
            elif isinstance(w, QComboBox):
                values[name] = w.currentText()
            elif isinstance(w, QLineEdit):
                values[name] = w.text()
        return values

    def _restore_selection(self, selection: dict[str, Any]) -> None:
        self._selection = dict(selection)
        self.backbone.blockSignals(True)
        self.drive.blockSignals(True)
        self.fusion.blockSignals(True)
        self.obs_mode.blockSignals(True)
        self.backbone.setCurrentText(
            selection.get("backbone", self._default_name("backbone"))
        )
        self.drive.setCurrentText(selection.get("drive_core", "none"))
        self.fusion.setCurrentText(selection.get("fusion", "concat"))
        self.obs_mode.setCurrentText(selection.get("obs_mode", "obs_only"))
        self.seed_box.setValue(int(selection.get("seed", 0)))
        self.backbone.blockSignals(False)
        self.drive.blockSignals(False)
        self.fusion.blockSignals(False)
        self.obs_mode.blockSignals(False)
        self._rebuild_forms()
        self._set_widget_values(self._backbone_widgets, selection.get("backbone_config") or {})
        self._set_widget_values(self._drive_widgets, selection.get("drive_config") or {})

    @staticmethod
    def _set_widget_values(widgets: dict[str, QWidget], values: dict[str, Any]) -> None:
        for name, value in values.items():
            w = widgets.get(name)
            if w is None:
                continue
            if isinstance(w, QSpinBox):
                w.setValue(int(value))
            elif isinstance(w, QDoubleSpinBox):
                w.setValue(float(value))
            elif isinstance(w, QCheckBox):
                w.setChecked(bool(value))
            elif isinstance(w, QComboBox):
                w.setCurrentText(str(value))
            elif isinstance(w, QLineEdit):
                w.setText(str(value))

    # -------------------------------------------------------------- actions
    def _build(self) -> None:
        selection = self._collect()
        errors = self._validate(selection)
        if errors:
            self.param_label.setText("; ".join(errors))
            return
        self.param_label.setText("Building…")
        self._builder = BuilderThread(selection, self)
        self._builder.finishedOk.connect(self._on_built)
        self._builder.failed.connect(self._on_build_failed)
        self._builder.start()

    def _validate(self, selection: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        bb = get_spec("backbone", selection["backbone"])
        if bb is not None:
            cfg = dict(selection["backbone_config"])
            errors.extend(validate_config(bb, cfg))
            selection["backbone_config"] = normalize_config(bb, cfg)
        dc = get_spec("drive_core", selection["drive_core"])
        if dc is not None:
            cfg = dict(selection["drive_config"])
            errors.extend(validate_config(dc, cfg))
            selection["drive_config"] = normalize_config(dc, cfg)
        return errors

    def _on_built(self, param_count: int, h: str) -> None:
        self.param_label.setText(f"param_count={param_count}  arch_hash={h}")

    def _on_build_failed(self, msg: str) -> None:
        self.param_label.setText(f"Build failed: {msg}")

    def _apply(self) -> None:
        selection = self._collect()
        errors = self._validate(selection)
        if errors:
            self.param_label.setText("; ".join(errors))
            return
        base = self._config
        drive_core = selection["drive_core"]
        ckpt = self._checkpoint_path
        cfg = replace(
            base,
            policy="learned_torch",
            drive=drive_core,
            drive_config=dict(selection["drive_config"]) or None,
            policy_checkpoint=ckpt,
            policy_cfg=None if ckpt else build_policy_cfg(selection),
        )
        self.playbackRequested.emit(cfg)

    def _pick_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Pick checkpoint", "", "PyTorch (*.pt)")
        if path:
            self._checkpoint_path = path
            self.param_label.setText(f"Checkpoint: {path}")

    def _save(self) -> None:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        default = str(RUNS_DIR / "experiment.toml")
        path, _ = QFileDialog.getSaveFileName(self, "Save experiment", default, "TOML (*.toml)")
        if not path:
            return
        text = dump_arch_toml(self._collect())
        Path(path).write_text(text, encoding="utf-8")

    def _load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load experiment", "", "TOML (*.toml)")
        if not path:
            return
        try:
            selection = load_arch_toml(path)
        except Exception as exc:  # noqa: BLE001
            self.param_label.setText(f"Load failed: {exc}")
            return
        self._restore_selection(selection)

    def _write_template(self) -> None:
        try:
            path = write_template(arch_dir())
            self.param_label.setText(f"Wrote template: {path}")
            self._rescan()
        except Exception as exc:  # noqa: BLE001
            self.param_label.setText(f"Template failed: {exc}")

    def _launch_train(self) -> None:
        selection = self._collect()
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        arch_path = RUNS_DIR / "train_arch.toml"
        arch_path.write_text(dump_arch_toml(selection), encoding="utf-8")
        out_dir = RUNS_DIR / "train" / "smoke"
        out_dir.mkdir(parents=True, exist_ok=True)
        args = [
            "-m",
            "simlab.cli",
            "train",
            "--arch",
            str(arch_path),
            "--total-steps",
            "512",
            "--workers",
            "1",
            "--out",
            str(out_dir),
        ]
        self.launcher.launch(args)

    def _launch_eval(self) -> None:
        ckpt = self._checkpoint_path
        if not ckpt:
            self.param_label.setText("Select a checkpoint before eval")
            return
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        args = [
            "-m",
            "simlab.cli",
            "eval",
            "--policy",
            "learned_torch",
            "--checkpoint",
            ckpt,
            "--scenario",
            "balanced",
            "--agents",
            "5",
            "--seeds",
            "5",
            "--cap",
            "2000",
            "--out",
            str(RUNS_DIR / "eval_smoke.json"),
        ]
        self.launcher.launch(args)
