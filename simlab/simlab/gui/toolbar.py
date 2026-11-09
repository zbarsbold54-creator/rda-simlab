"""Simulation controls toolbar."""

from __future__ import annotations

import secrets

from PySide6.QtCore import Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QToolBar,
)

from simlab.gui import theme


class SimulationToolbar(QToolBar):
    playPause = Signal(bool)
    singleStep = Signal()
    speedChanged = Signal(str)
    resetRequested = Signal(int)

    SPEEDS = theme.SPEEDS

    def __init__(self, parent=None) -> None:
        super().__init__("Simulation", parent)
        self.setMovable(False)
        self.play_button = QPushButton("Pause")
        self.step_button = QPushButton("Single-step")
        self.speed = QComboBox()
        self.speed.addItems(self.SPEEDS)
        self.speed.setCurrentText(theme.DEFAULT_SPEED)
        self.speed.setMinimumContentsLength(max(map(len, self.SPEEDS)))
        self.seed = QLineEdit("1")
        self.seed.setFont(QFont(theme.NUMBER_FONT, theme.NUMBER_FONT_SIZE_PT))
        self.seed.setMaximumWidth(theme.SEED_FIELD_WIDTH)
        self.seed_label = QLabel("Seed")
        self.random_button = QPushButton("Random seed")
        self.reset_button = QPushButton("Reset")
        for widget in (self.play_button, self.step_button):
            self.addWidget(widget)
        self.addSeparator()
        self.addWidget(self.speed)
        self.addSeparator()
        self.addWidget(self.seed_label)
        self.addWidget(self.seed)
        self.addWidget(self.random_button)
        self.addWidget(self.reset_button)
        self.play_button.clicked.connect(self._toggle_play)
        self.step_button.clicked.connect(self.singleStep)
        self.speed.currentTextChanged.connect(self.speedChanged)
        self.random_button.clicked.connect(self._randomize_seed)
        self.reset_button.clicked.connect(self._reset)
        self._playing = True

    def _toggle_play(self) -> None:
        self.set_playing(not self._playing)

    def set_playing(self, playing: bool) -> None:
        self._playing = bool(playing)
        self.play_button.setText("Pause" if self._playing else "Play")
        self.playPause.emit(self._playing)

    def _randomize_seed(self) -> None:
        self.seed.setText(str(secrets.randbelow(2**31)))

    def _reset(self) -> None:
        try:
            seed = int(self.seed.text())
        except ValueError:
            seed = 1
            self.seed.setText(str(seed))
        self.resetRequested.emit(seed)


class SessionToolbar(QToolBar):
    """Snapshot, replay, camera, and rendering-overlay controls."""

    saveSnapshotRequested = Signal()
    loadSnapshotRequested = Signal()
    recordingChanged = Signal(bool)
    replayRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__("Snapshot and inspection", parent)
        self.setMovable(False)
        self.save_snapshot_button = QPushButton("Save")
        self.save_snapshot_button.setToolTip("Save snapshot as JSON and NPZ")
        self.load_snapshot_button = QPushButton("Load")
        self.load_snapshot_button.setToolTip("Load a JSON and NPZ snapshot")
        self.record_button = QPushButton("Record")
        self.record_button.setCheckable(True)
        self.record_button.setChecked(True)
        self.replay_button = QPushButton("Replay")
        self.follow = QCheckBox("Follow")
        self.overlay_checks = {}
        for name, title in (
            ("rays", "Rays"),
            ("vision", "Cone"),
            ("trails", "Trail"),
            ("grid", "Grid"),
            ("halo", "Halo"),
        ):
            checkbox = QCheckBox(title)
            checkbox.setChecked(name == "rays")
            self.overlay_checks[name] = checkbox
        for widget in (
            self.save_snapshot_button,
            self.load_snapshot_button,
            self.record_button,
            self.replay_button,
            self.follow,
            *self.overlay_checks.values(),
        ):
            self.addWidget(widget)
        self.save_snapshot_button.clicked.connect(self.saveSnapshotRequested)
        self.load_snapshot_button.clicked.connect(self.loadSnapshotRequested)
        self.record_button.toggled.connect(self._set_recording)
        self.replay_button.clicked.connect(self.replayRequested)

    def _set_recording(self, enabled: bool) -> None:
        self.record_button.setText("Recording" if enabled else "Record")
        self.recordingChanged.emit(enabled)
