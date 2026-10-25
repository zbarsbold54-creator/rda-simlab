"""Live details and drive bars for the selected agent."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

from simlab.gui import theme
from simlab.gui.frame import SimulationFrame


class InspectorWidget(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(*([theme.DOCK_MARGIN] * 4))
        self._labels = {
            name: QLabel(f"{name}: —")
            for name in (
                "agent",
                "energy",
                "health",
                "age",
                "action",
                "sensors",
                "food",
                "death",
            )
        }
        for label in self._labels.values():
            self._layout.addWidget(label)
        self._retention_labels: dict[str, QLabel] = {}
        for name in ("E", "lam", "m", "s"):
            label = QLabel(f"{name}: —")
            label.hide()
            self._retention_labels[name] = label
            self._layout.addWidget(label)
        self._bars: dict[str, QProgressBar] = {}
        self._energy_plot = pg.PlotWidget()
        self._energy_plot.setBackground(theme.PANEL)
        self._energy_plot.showGrid(x=True, y=True, alpha=0.18)
        self._energy_plot.setLabel("left", "energy")
        self._energy_plot.setLabel("bottom", "step")
        self._energy_plot.setFixedHeight(120)
        self._energy_line = self._energy_plot.plot(
            pen=pg.mkPen(theme.ENERGY_HIGH, width=2)
        )
        self._layout.addWidget(self._energy_plot)
        self._layout.addStretch(1)

    def set_agent(self, frame: SimulationFrame, agent_id: str | None) -> None:
        state = frame.snapshot.agent_states.get(agent_id)
        detail = frame.agents.get(agent_id) if agent_id else None
        if state is None:
            for name, label in self._labels.items():
                label.setText(f"{name}: —")
            for bar in self._bars.values():
                bar.hide()
            for label in self._retention_labels.values():
                label.hide()
            self._energy_plot.hide()
            return
        self._energy_plot.show()
        self._labels["food"].setText(f"Food eaten: {state.food_eaten}")
        death = frame.deaths.get(state.agent_id)
        self._labels["death"].setText(
            f"Status: died at step {death[0]} ({death[1]})" if death else "Status: alive"
        )
        series = frame.energies.get(state.agent_id)
        if series and len(series) > 1:
            points = np.asarray(series, dtype=np.float64)
            self._energy_line.setData(points[:, 0], points[:, 1])
        else:
            self._energy_line.setData([], [])
        self._labels["agent"].setText(f"Agent: {state.agent_id}")
        self._labels["energy"].setText(
            f"Energy: {state.energy:.1f} / {state.max_energy:.1f}"
        )
        self._labels["health"].setText(f"Health: {state.health:.1f}")
        self._labels["age"].setText(f"Age: {state.age:.1f}s")
        action = detail.action if detail else None
        self._labels["action"].setText(
            f"Action: turn {action.turn:.2f}, thrust {action.thrust:.2f}, eat {action.eat}"
            if action
            else "Action: —"
        )
        observation = detail.observation if detail else None
        sensor_text = (
            "—"
            if observation is None
            else f"{len(observation.rays)} rays; nearest {min(observation.rays, default=0):.1f}"
        )
        self._labels["sensors"].setText(f"Sensors: {sensor_text}")
        names = frame.drive_names
        drives = detail.drives if detail else ()
        for index, name in enumerate(names):
            bar = self._bars.get(name)
            if bar is None:
                bar = QProgressBar()
                bar.setRange(0, 100)
                bar.setTextVisible(True)
                bar.setFormat(f"{name}  %p%")
                bar.setStyleSheet(
                    f"QProgressBar::chunk {{ background-color: {theme.DRIVE_COLORS[index % len(theme.DRIVE_COLORS)]}; }}"
                )
                self._bars[name] = bar
                self._layout.insertWidget(self._layout.count() - 1, bar)
            bar.show()
            bar.setValue(round(100 * drives[index])) if index < len(
                drives
            ) else bar.setValue(0)
        retention = detail.retention if detail else None
        for name, label in self._retention_labels.items():
            if retention and name in retention:
                label.setText(f"{name}: {retention[name]:.3f}")
                label.show()
            else:
                label.hide()
