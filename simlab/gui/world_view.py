"""Painter-based world view with arena clipping, hover details, pan, and zoom."""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

from simlab.core.types import Snapshot
from simlab.gui import theme
from simlab.gui.frame import SimulationFrame


def agent_triangle_vertices(
    position: tuple[float, float], heading: float, radius: float
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Return the nose and symmetric rear vertices in world coordinates."""
    local_vertices = (
        (radius, 0.0),
        (theme.AGENT_TAIL_X * radius, theme.AGENT_TAIL_Y * radius),
        (theme.AGENT_TAIL_X * radius, -theme.AGENT_TAIL_Y * radius),
    )
    cosine, sine = math.cos(heading), math.sin(heading)
    px, py = position
    return tuple(
        (
            px + x * cosine - y * sine,
            py + x * sine + y * cosine,
        )
        for x, y in local_vertices
    )


class WorldView(QWidget):
    agentSelected = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._snapshot: Snapshot | None = None
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._drag_point: QPointF | None = None
        self._press_point: QPointF | None = None
        self._cursor_point: QPointF | None = None
        self._hovered_hazard: int | None = None
        self._selected_agent: str | None = None
        self._frame: SimulationFrame | None = None
        self._overlays = {
            "rays": True,
            "vision": False,
            "trails": False,
            "grid": False,
            "halo": False,
        }
        self._follow_selected = False
        self._trails: dict[str, list[tuple[float, float]]] = {}
        self._policy_info: dict | None = None
        self.setMouseTracking(True)
        self.setMinimumSize(theme.VIEW_MIN_WIDTH, theme.VIEW_MIN_HEIGHT)

    @staticmethod
    def food_draw_radius(amount: float) -> float:
        ratio = min(max(float(amount) / theme.DEFAULT_FOOD_AMOUNT, 0.0), 1.0)
        return theme.FOOD_SIZE_MIN + ratio * (theme.FOOD_SIZE_MAX - theme.FOOD_SIZE_MIN)

    def set_snapshot(self, snapshot: Snapshot) -> None:
        self._snapshot = snapshot
        self._refresh_hover()
        self.update()

    def set_frame(self, frame: SimulationFrame) -> None:
        self._frame = frame
        self.set_snapshot(frame.snapshot)
        for agent_id, state in frame.snapshot.agent_states.items():
            points = self._trails.setdefault(agent_id, [])
            points.append(state.position)
            del points[:-600]
        selected = frame.snapshot.agent_states.get(self._selected_agent)
        if self._follow_selected and selected is not None:
            scale, _ = self._transform(frame.snapshot)
            self._pan = QPointF(
                self.width() / 2 - selected.position[0] * scale,
                self.height() / 2 - selected.position[1] * scale,
            )
        self.update()

    def set_overlay(self, name: str, enabled: bool) -> None:
        self._overlays[name] = bool(enabled)
        self.update()

    def set_follow_selected(self, enabled: bool) -> None:
        self._follow_selected = bool(enabled)
        self.update()

    def _transform(self, snapshot: Snapshot) -> tuple[float, QPointF]:
        margin = float(theme.WORLD_MARGIN)
        scale = (
            min(
                (self.width() - 2 * margin) / max(snapshot.width, 1.0),
                (self.height() - 2 * margin) / max(snapshot.height, 1.0),
            )
            * self._zoom
        )
        origin = QPointF(
            (self.width() - snapshot.width * scale) / 2 + self._pan.x(),
            (self.height() - snapshot.height * scale) / 2 + self._pan.y(),
        )
        return scale, origin

    def _world_point(self, point: QPointF) -> QPointF | None:
        if self._snapshot is None:
            return None
        scale, origin = self._transform(self._snapshot)
        return QPointF(
            (point.x() - origin.x()) / scale, (point.y() - origin.y()) / scale
        )

    def _refresh_hover(self) -> None:
        self._hovered_hazard = None
        if self._snapshot is None or self._cursor_point is None:
            return
        world_point = self._world_point(self._cursor_point)
        if world_point is None:
            return
        closest = math.inf
        for index, patch in enumerate(self._snapshot.hazard_patches):
            x, y = patch["position"]
            distance = math.hypot(world_point.x() - x, world_point.y() - y)
            if (
                distance <= float(patch["radius"]) + theme.HAZARD_HOVER_TOLERANCE
                and distance < closest
            ):
                self._hovered_hazard = index
                closest = distance

    def paintEvent(self, event: object) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(theme.BACKGROUND))
        snapshot = self._snapshot
        if snapshot is None:
            painter.end()
            return

        scale, origin = self._transform(snapshot)
        painter.save()
        painter.translate(origin)
        painter.scale(scale, scale)
        arena = QRectF(0.0, 0.0, snapshot.width, snapshot.height)
        painter.setPen(
            QPen(QColor(theme.ARENA_BORDER), theme.ARENA_BORDER_WIDTH / scale)
        )
        painter.setBrush(QColor(theme.PANEL))
        painter.drawRect(arena)
        painter.setClipRect(arena, Qt.ClipOperation.IntersectClip)

        if self._overlays["grid"]:
            grid_pen = QPen(QColor(theme.GRID_COLOR), theme.GRID_WIDTH / scale)
            painter.setPen(grid_pen)
            for x in range(0, int(snapshot.width) + 1, theme.GRID_SPACING):
                painter.drawLine(QPointF(x, 0), QPointF(x, snapshot.height))
            for y in range(0, int(snapshot.height) + 1, theme.GRID_SPACING):
                painter.drawLine(QPointF(0, y), QPointF(snapshot.width, y))

        # Food is beneath hazards and agents so living entities remain legible.
        for patch in snapshot.food_patches:
            x, y = patch["position"]
            radius = self.food_draw_radius(float(patch["amount"]))
            outline = QColor(theme.FOOD_OUTLINE)
            outline.setAlpha(theme.FOOD_OUTLINE_ALPHA)
            painter.setPen(QPen(outline, theme.FOOD_OUTLINE_WIDTH / scale))
            painter.setBrush(QColor(theme.FOOD_FILL))
            painter.drawEllipse(QPointF(x, y), radius, radius)

        selected = snapshot.agent_states.get(self._selected_agent)
        for index, patch in enumerate(snapshot.hazard_patches):
            x, y = patch["position"]
            contact_radius = float(patch.get("radius", 0.0))
            detection_radius = float(patch.get("detection_radius", 0.0))
            ring = QColor(theme.HAZARD)
            ring.setAlpha(theme.HAZARD_RING_ALPHA)
            painter.setPen(QPen(ring, theme.WALL_WIDTH / scale))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(x, y), contact_radius, contact_radius)
            if self._detection_ring_visible(index, patch, selected):
                detection_ring = QColor(theme.HAZARD)
                detection_ring.setAlpha(theme.HAZARD_DETECTION_RING_ALPHA)
                painter.setPen(QPen(detection_ring, theme.WALL_WIDTH / scale))
                painter.drawEllipse(QPointF(x, y), detection_radius, detection_radius)

            outline = QColor(theme.HAZARD)
            outline.setAlpha(theme.HAZARD_OUTLINE_ALPHA)
            painter.setPen(QPen(outline, theme.WALL_WIDTH / scale))
            painter.setBrush(QColor(theme.HAZARD))
            size = theme.HAZARD_SIZE
            painter.drawPolygon(
                QPolygonF(
                    [
                        QPointF(x, y - size),
                        QPointF(x + size, y),
                        QPointF(x, y + size),
                        QPointF(x - size, y),
                    ]
                )
            )

        for state in snapshot.agent_states.values():
            if not state.alive:
                continue
            x, y = state.position
            angle = state.heading
            if self._overlays["trails"]:
                trail = self._trails.get(state.agent_id, [])
                if len(trail) > 1:
                    painter.setPen(
                        QPen(QColor(theme.TRAIL_COLOR), theme.TRAIL_WIDTH / scale)
                    )
                    rendered_trail = trail[:: theme.TRAIL_RENDER_STRIDE]
                    if rendered_trail[-1] != trail[-1]:
                        rendered_trail = [*rendered_trail, trail[-1]]
                    painter.drawPolyline(
                        QPolygonF([QPointF(*point) for point in rendered_trail])
                    )
            if state.agent_id == self._selected_agent and self._frame is not None:
                detail = self._frame.agents.get(state.agent_id)
                observation = detail.observation if detail else None
                if observation is not None and self._overlays["rays"]:
                    painter.setPen(
                        QPen(QColor(theme.RAY_COLOR), theme.RAY_WIDTH / scale)
                    )
                    offsets = [
                        -math.radians(135)
                        + index * math.radians(270) / max(len(observation.rays) - 1, 1)
                        for index in range(len(observation.rays))
                    ]
                    if len(offsets) > len(offsets) // 2:
                        offsets[len(offsets) // 2] = 0.0
                    for distance, offset in zip(observation.rays, offsets):
                        ray_angle = angle + offset
                        painter.drawLine(
                            QPointF(x, y),
                            QPointF(
                                x + math.cos(ray_angle) * distance,
                                y + math.sin(ray_angle) * distance,
                            ),
                        )
                    if self._overlays["vision"]:
                        cone = QPainterPath(QPointF(x, y))
                        cone.arcTo(
                            QRectF(x - 20, y - 20, 40, 40),
                            -math.degrees(angle) - 135,
                            270,
                        )
                        cone.closeSubpath()
                        tint = QColor(theme.RAY_COLOR)
                        tint.setAlpha(theme.VISION_ALPHA)
                        painter.fillPath(cone, tint)
                if self._overlays["halo"] and detail and detail.drives:
                    index = max(
                        range(len(detail.drives)), key=detail.drives.__getitem__
                    )
                    halo = QColor(theme.DRIVE_COLORS[index % len(theme.DRIVE_COLORS)])
                    halo.setAlpha(theme.HALO_ALPHA)
                    painter.setPen(QPen(halo, theme.HALO_WIDTH / scale))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawEllipse(
                        QPointF(x, y), theme.HALO_RADIUS, theme.HALO_RADIUS
                    )
            size = theme.AGENT_SIZE
            points = [
                QPointF(*point)
                for point in agent_triangle_vertices(state.position, angle, size)
            ]
            fraction = min(max(state.energy / max(state.max_energy, 1e-9), 0.0), 1.0)
            fill = self._energy_color(fraction)
            outline = QColor(theme.AGENT_OUTLINE)
            outline.setAlpha(theme.AGENT_OUTLINE_ALPHA)
            painter.setPen(QPen(outline, theme.AGENT_OUTLINE_WIDTH / scale))
            painter.setBrush(fill)
            painter.drawPath(self._rounded_triangle(points))

        painter.restore()
        self._draw_banner(painter)
        self._draw_agent_panel(painter)
        painter.end()

    def set_policy_info(self, info: dict | None) -> None:
        self._policy_info = info
        self.update()

    def _draw_banner(self, painter: QPainter) -> None:
        info = self._policy_info
        if not info:
            return
        margin = theme.DOCK_MARGIN
        x = margin
        y = margin
        width = min(540, self.width() - 2 * margin)
        lines: list[str] = []
        name = str(info.get("name", "policy"))
        lines.append(f"Policy: {name}")
        details = " | ".join(
            str(info[key])
            for key in ("preset", "backbone", "drive_core", "fusion", "obs_mode")
            if info.get(key)
        )
        if details:
            lines.append(details)
        params = info.get("param_count")
        arch_hash = info.get("arch_hash")
        arch_line = f"params: {params}" if params is not None else "params: ?"
        if arch_hash:
            arch_line += f"  arch:{str(arch_hash)[:8]}"
        lines.append(arch_line)
        trained = bool(info.get("trained"))
        checkpoint_path = info.get("checkpoint_path")
        train_step = info.get("train_step")
        if trained:
            status = "TRAINED"
            if checkpoint_path:
                status += f"  {checkpoint_path}"
            if train_step is not None:
                status += f"  (step {train_step})"
        else:
            status = "UNTRAINED"
            if checkpoint_path:
                status += "  checkpoint not loaded"
        lines.append(status)
        font = painter.font()
        font.setPointSize(theme.UI_FONT_SIZE_PT)
        painter.setFont(font)
        line_height = round(font.pointSizeF() * 1.6)
        total_height = line_height * len(lines) + 2 * margin
        rect = QRectF(x, y, width, total_height)
        painter.setPen(QPen(QColor(theme.BANNER_BORDER), 1.0))
        painter.setBrush(QColor(theme.BANNER_BG))
        painter.drawRoundedRect(rect, theme.PANEL_RADIUS, theme.PANEL_RADIUS)
        text_color = QColor(theme.TEXT)
        warning = not trained
        for index, text in enumerate(lines):
            painter.setPen(warning if index == len(lines) - 1 and warning else text_color)
            label_y = y + margin + index * line_height
            painter.drawText(
                QRectF(x + margin, label_y, width - 2 * margin, line_height),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                text,
            )
        if not trained:
            painter.setPen(QColor(theme.WARNING))
            painter.drawText(
                QRectF(
                    x + margin,
                    y + margin + len(lines) * line_height,
                    width - 2 * margin,
                    line_height,
                ),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                "Untrained (random weights): behaves roughly randomly. "
                "Train it from the Queue tab, then load the checkpoint.",
            )

    def _draw_agent_panel(self, painter: QPainter) -> None:
        frame = self._frame
        snapshot = self._snapshot
        selected = self._selected_agent
        if frame is None or snapshot is None or not selected:
            return
        state = snapshot.agent_states.get(selected)
        if state is None:
            return
        margin = theme.DOCK_MARGIN
        panel_width = 240
        x = self.width() - panel_width - margin
        y = margin
        lines: list[str] = []
        lines.append(f"Agent: {selected}")
        lines.append(f"energy: {state.energy:.1f} / {state.max_energy:.0f}")
        lines.append(f"food eaten: {state.food_eaten}")
        death = frame.deaths.get(selected)
        if death is not None:
            step, cause = death
            lines.append(f"died step {step} ({cause})")
        else:
            lines.append("alive")
        font = painter.font()
        font.setPointSize(theme.UI_FONT_SIZE_PT)
        painter.setFont(font)
        line_height = round(font.pointSizeF() * 1.6)
        text_height = line_height * len(lines) + 2 * margin
        series = frame.energies.get(selected)
        has_series = bool(series) and len(series) > 1
        plot_height = 70 if has_series else 0
        total_height = text_height + plot_height
        rect = QRectF(x, y, panel_width, total_height)
        painter.setPen(QPen(QColor(theme.BANNER_BORDER), 1.0))
        painter.setBrush(QColor(theme.BANNER_BG))
        painter.drawRoundedRect(rect, theme.PANEL_RADIUS, theme.PANEL_RADIUS)
        painter.setPen(QColor(theme.TEXT))
        for index, text in enumerate(lines):
            label_y = y + margin + index * line_height
            painter.drawText(
                QRectF(x + margin, label_y, panel_width - 2 * margin, line_height),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                text,
            )
        if has_series:
            plot_top = y + margin + len(lines) * line_height
            points = list(series)
            steps = [p[0] for p in points]
            values = [p[1] for p in points]
            min_s, max_s = min(steps), max(steps)
            min_v, max_v = min(values), max(values)
            span_s = (max_s - min_s) or 1.0
            span_v = (max_v - min_v) or 1.0
            plot_left = x + margin
            plot_right = x + panel_width - margin
            plot_bottom = plot_top + plot_height
            painter.setPen(QPen(QColor(theme.ACCENT), 1.5))
            path = QPainterPath()
            for idx, (step_value, value) in enumerate(points):
                px = plot_left + (step_value - min_s) / span_s * (plot_right - plot_left)
                py = plot_bottom - (value - min_v) / span_v * (plot_bottom - plot_top)
                if idx == 0:
                    path.moveTo(px, py)
                else:
                    path.lineTo(px, py)
            painter.drawPath(path)

    def _detection_ring_visible(self, index: int, hazard, selected_agent) -> bool:
        if self._hovered_hazard == index:
            return True
        return (
            selected_agent is not None
            and selected_agent.alive
            and math.dist(selected_agent.position, hazard["position"])
            <= float(hazard.get("detection_radius", 0.0))
        )

    @staticmethod
    def _energy_color(fraction: float) -> QColor:
        low, high = QColor(theme.ENERGY_LOW), QColor(theme.ENERGY_HIGH)
        return QColor(
            round(low.red() + fraction * (high.red() - low.red())),
            round(low.green() + fraction * (high.green() - low.green())),
            round(low.blue() + fraction * (high.blue() - low.blue())),
        )

    @staticmethod
    def _rounded_triangle(points: list[QPointF]) -> QPainterPath:
        path = QPainterPath()
        incoming = []
        outgoing = []
        for index, point in enumerate(points):
            previous_point = points[index - 1]
            next_point = points[(index + 1) % len(points)]
            incoming.append(
                point + (previous_point - point) * theme.AGENT_CORNER_ROUNDING
            )
            outgoing.append(point + (next_point - point) * theme.AGENT_CORNER_ROUNDING)
        path.moveTo(incoming[0])
        for index, point in enumerate(points):
            path.quadTo(point, outgoing[index])
            path.lineTo(incoming[(index + 1) % len(points)])
        path.closeSubpath()
        return path

    def wheelEvent(self, event: object) -> None:
        factor = (
            theme.ZOOM_FACTOR if event.angleDelta().y() > 0 else 1.0 / theme.ZOOM_FACTOR
        )
        self._zoom = min(theme.ZOOM_MAX, max(theme.ZOOM_MIN, self._zoom * factor))
        self._refresh_hover()
        self.update()

    def mousePressEvent(self, event: object) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_point = event.position()
            self._press_point = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: object) -> None:
        point = event.position()
        if self._drag_point is not None:
            self._pan += point - self._drag_point
            self._drag_point = point
        self._cursor_point = point
        self._refresh_hover()
        self.update()

    def mouseReleaseEvent(self, event: object) -> None:
        point = event.position()
        if (
            self._press_point is not None
            and self._snapshot is not None
            and (point - self._press_point).manhattanLength()
            <= theme.CLICK_DRAG_THRESHOLD
        ):
            world_point = self._world_point(point)
            if world_point is not None:
                nearest = min(
                    (
                        (
                            math.hypot(
                                world_point.x() - state.position[0],
                                world_point.y() - state.position[1],
                            ),
                            agent_id,
                        )
                        for agent_id, state in self._snapshot.agent_states.items()
                        if state.alive
                    ),
                    default=(math.inf, None),
                )
                self._selected_agent = (
                    nearest[1] if nearest[0] <= theme.AGENT_SELECT_RADIUS else None
                )
                self.agentSelected.emit(self._selected_agent)
        self._drag_point = None
        self._press_point = None
        self._cursor_point = point
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._refresh_hover()
        self.update()

    def leaveEvent(self, event: object) -> None:
        self._cursor_point = None
        self._refresh_hover()
        self.update()
        super().leaveEvent(event)
