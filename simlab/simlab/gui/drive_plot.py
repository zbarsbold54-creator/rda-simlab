"""Rolling pyqtgraph drive histories and snapshot timeline."""

from __future__ import annotations

import time
from collections import deque

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QSlider, QWidget

from simlab.gui import theme
from simlab.gui.frame import SimulationFrame


class DrivePlotWidget(QWidget):
    timelineChanged = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        layout = QHBoxLayout(self)
        self.plot = pg.PlotWidget()
        self.plot.setBackground(theme.PANEL)
        self.plot.showGrid(x=True, y=True, alpha=0.18)
        self.plot.setYRange(0, 1, padding=0)
        self.plot.addLegend()
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 0)
        self.timeline.setFixedWidth(theme.TIMELINE_WIDTH)
        layout.addWidget(self.plot, 1)
        layout.addWidget(self.timeline)
        self.timeline.valueChanged.connect(self.timelineChanged)
        self._scrubbing = False
        self.timeline.sliderPressed.connect(self._begin_scrub)
        self.timeline.sliderReleased.connect(self._end_scrub)
        self._selected: str | None = None
        self._series: dict[str, deque[tuple[int, float]]] = {}
        self._lines: dict[str, object] = {}
        self._last_plot_update = 0.0

    def select_agent(self, agent_id: str | None) -> None:
        self._selected = agent_id
        self._series.clear()
        for line in self._lines.values():
            self.plot.removeItem(line)
        self._lines.clear()
        self._last_plot_update = 0.0

    def set_timeline_range(self, first: int, last: int, current: int) -> None:
        self.timeline.blockSignals(True)
        self.timeline.setRange(first, max(first, last))
        self.timeline.setValue(min(max(current, first), max(first, last)))
        self.timeline.blockSignals(False)

    def _begin_scrub(self) -> None:
        self._scrubbing = True

    def _end_scrub(self) -> None:
        self._scrubbing = False

    def sync_live(self, step: int) -> None:
        """Keep the timeline pinned to the live simulation step while not scrubbing."""
        if self._scrubbing:
            return
        self.timeline.blockSignals(True)
        first = self.timeline.minimum()
        self.timeline.setRange(first, max(first, step))
        self.timeline.setValue(min(max(step, first), max(first, step)))
        self.timeline.blockSignals(False)

    def _drive_vector(self, frame: SimulationFrame) -> list[float] | None:
        drive_count = len(frame.drive_names)
        if drive_count == 0:
            return None
        selected = self._selected
        if selected is not None and selected in frame.agents:
            detail = frame.agents[selected]
            drives = getattr(detail, "drives", None) or ()
            return [
                float(drives[index]) if index < len(drives) else 0.0
                for index in range(drive_count)
            ]
        drives_by_agent = []
        for detail in frame.agents.values():
            drives = getattr(detail, "drives", None) or ()
            if not drives:
                continue
            drives_by_agent.append(
                [float(drives[i]) if i < len(drives) else 0.0 for i in range(drive_count)]
            )
        if not drives_by_agent:
            return None
        return [
            sum(samples[index] for samples in drives_by_agent) / len(drives_by_agent)
            for index in range(drive_count)
        ]

    def update_frame(self, frame: SimulationFrame) -> None:
        drives = self._drive_vector(frame)
        if drives is None:
            return
        for index, name in enumerate(frame.drive_names):
            values = self._series.setdefault(name, deque(maxlen=theme.PLOT_HISTORY))
            values.append((frame.snapshot.step, drives[index]))
        now = time.perf_counter()
        if now - self._last_plot_update < theme.DRIVE_PLOT_UPDATE_INTERVAL_S:
            return
        self._last_plot_update = now
        for index, name in enumerate(frame.drive_names):
            values = self._series[name]
            line = self._lines.get(name)
            if line is None:
                line = self.plot.plot(
                    name=name,
                    pen=pg.mkPen(
                        theme.DRIVE_COLORS[index % len(theme.DRIVE_COLORS)], width=2
                    ),
                )
                self._lines[name] = line
                if self.plot.plotItem.legend is not None:
                    for sample, label in self.plot.plotItem.legend.items:
                        if label.text == name:
                            label.mousePressEvent = lambda event, item=line: (
                                item.setVisible(not item.isVisible())
                            )
            samples = np.asarray(values, dtype=np.float64)
            line.setData(samples[:, 0], samples[:, 1])
