"""Registry-backed scenario editor dock."""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from simlab.core.config import SimLabConfig
from simlab.gui import theme


class ScenarioWidget(QWidget):
    applyRequested = Signal(object)

    def __init__(
        self,
        config: SimLabConfig,
        policy_names: tuple[str, ...],
        drive_names: tuple[str, ...],
        seed: int,
    ) -> None:
        super().__init__()
        self._config = config
        self._seed = seed
        self._detection_radius = float(config.hazard.detection_radius)
        self._supply_ratio = config.food.supply_ratio
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.preset = QComboBox()
        self.preset.addItems(
            (
                "Custom",
                "Balanced",
                "Solo",
                "Hard",
                "Harsh (original)",
                "Sparse food",
                "Abundant food",
                "Predator heavy",
            )
        )
        self.agent_count = self._slider(1, 20, config.agent.count)
        self.food_density = self._slider(1, 50, round(config.food.spawn_rate * 100))
        self.food_regrow_rate = self._slider(
            0, 50, round(config.food.regrow_rate * 100)
        )
        self.hazard_count = self._slider(0, 20, config.hazard.count)
        self.policy = QComboBox()
        self.policy.addItems(sorted(policy_names))
        self.policy.setCurrentText(config.policy)
        self.drive = QComboBox()
        self.drive.addItems(sorted(drive_names))
        self.drive.setCurrentText(config.drive)
        form.addRow("Preset", self.preset)
        form.addRow("Agents", self.agent_count)
        form.addRow("Food density (initial patches)", self.food_density)
        form.addRow("Food regrowth (raw)", self.food_regrow_rate)
        form.addRow("Hazards", self.hazard_count)
        form.addRow("Policy", self.policy)
        form.addRow("DriveCore", self.drive)
        layout.addLayout(form)
        self.summary = QLabel()
        layout.addWidget(self.summary)
        self.apply_button = QPushButton("Apply and reset")
        layout.addWidget(self.apply_button)
        layout.addStretch(1)
        self.preset.currentTextChanged.connect(self._apply_preset)
        self.apply_button.clicked.connect(self.apply)
        self.apply()

    @staticmethod
    def _slider(minimum: int, maximum: int, value: int) -> QSlider:
        slider = QSlider()
        slider.setOrientation(
            __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.Orientation.Horizontal
        )
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        slider.setFixedWidth(theme.SCENARIO_SLIDER_WIDTH)
        return slider

    def _apply_preset(self, name: str) -> None:
        from simlab.core.config import load_config

        file_presets = {
            "Balanced": "configs/balanced.toml",
            "Solo": "configs/solo.toml",
            "Hard": "configs/hard.toml",
            "Harsh (original)": "configs/harsh_original.toml",
        }
        if name in file_presets:
            preset = load_config(file_presets[name])
            self.agent_count.setValue(preset.agent.count)
            self.food_density.setValue(round(preset.food.spawn_rate * 100))
            self.food_regrow_rate.setValue(round(preset.food.regrow_rate * 100))
            self.hazard_count.setValue(preset.hazard.count)
            self._detection_radius = float(preset.hazard.detection_radius)
            self._supply_ratio = preset.food.supply_ratio
            return
        self._supply_ratio = None
        presets = {
            "Sparse food": (0.025, self._config.hazard.count),
            "Abundant food": (0.25, self._config.hazard.count),
            "Predator heavy": (self._config.food.spawn_rate, 12),
        }
        if name in presets:
            rate, hazards = presets[name]
            self.food_density.setValue(round(rate * 100))
            self.hazard_count.setValue(hazards)

    def apply(self) -> None:
        config = replace(
            self._config,
            agent=replace(self._config.agent, count=self.agent_count.value()),
            food=replace(
                self._config.food,
                spawn_rate=self.food_density.value() / 100,
                regrow_rate=self.food_regrow_rate.value() / 100,
                supply_ratio=self._supply_ratio,
            ),
            hazard=replace(
                self._config.hazard,
                count=self.hazard_count.value(),
                detection_radius=self._detection_radius,
            ),
            policy=self.policy.currentText(),
            drive=self.drive.currentText(),
        )
        self.summary.setText(
            f"Seed {self._seed} · {config.agent.count} agents · {config.hazard.count} hazards"
        )
        self.applyRequested.emit(config)
