"""Drawing engine: pygame.Surface wrapper with draw ops and undo stack."""

import math
import random
import pygame
from dataclasses import dataclass, field


@dataclass
class DrawState:
    color: tuple = (0, 0, 0)
    brush_size: int = 3
    background_color: tuple = (255, 255, 255)
    oil_paint: bool = False


class Canvas:
    MAX_UNDO = 50

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.state = DrawState()
        self.surface = pygame.Surface((width, height))
        self.surface.fill(self.state.background_color)
        self._undo_stack: list[pygame.Surface] = []

    def _save_undo(self):
        if len(self._undo_stack) >= self.MAX_UNDO:
            self._undo_stack.pop(0)
        self._undo_stack.append(self.surface.copy())

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        self.surface = self._undo_stack.pop()
        return True

    def execute(self, cmd: dict):
        action = cmd.get("action")
        method = getattr(self, f"_do_{action}", None)
        if method is None:
            raise ValueError(f"Unknown action: {action}")
        method(cmd)

    # --- State operations (no undo) ---

    def _do_set_color(self, cmd: dict):
        self.state.color = (cmd["r"], cmd["g"], cmd["b"])

    def _do_set_brush_size(self, cmd: dict):
        self.state.brush_size = cmd["size"]

    def _do_set_oil_paint(self, cmd: dict):
        self.state.oil_paint = cmd["enabled"]

    # --- Oil paint helpers ---

    def _sample_color(self, x: int, y: int) -> tuple:
        """Read the RGB color of a single pixel on the canvas."""
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))
        return tuple(self.surface.get_at((x, y))[:3])

    def _blend(self, base: tuple, top: tuple, ratio: float = 0.5) -> tuple:
        """Mix two colors. ratio=1.0 means all top, 0.0 means all base."""
        return tuple(int(b * (1 - ratio) + t * ratio) for b, t in zip(base, top))

    def _oil_dab(self, cx: int, cy: int, radius: int):
        """Paint a single oil-paint dab: sample the canvas center, blend with
        the brush color, and stamp a slightly irregular circle."""
        canvas_color = self._sample_color(cx, cy)
        # Heavy brush-color bias (0.6-0.8) so the stroke is visible but mixes
        mix_ratio = random.uniform(0.6, 0.8)
        blended = self._blend(canvas_color, self.state.color, mix_ratio)

        # Irregular dab: vary radius slightly per quadrant for a painterly feel
        for angle_step in range(0, 360, 6):
            a = math.radians(angle_step)
            r = radius * random.uniform(0.75, 1.0)
            ex = int(cx + r * math.cos(a))
            ey = int(cy + r * math.sin(a))
            pygame.draw.line(self.surface, blended, (cx, cy), (ex, ey), 1)

    def _oil_stroke(self, points: list[tuple[int, int]]):
        """Lay down oil-paint dabs along a series of points."""
        for px, py in points:
            size_jitter = random.uniform(0.8, 1.2)
            r = max(1, int(self.state.brush_size * size_jitter))
            self._oil_dab(px, py, r)

    @staticmethod
    def _interpolate(x1, y1, x2, y2, spacing=3) -> list[tuple[int, int]]:
        """Return evenly-spaced points along a line segment."""
        dx, dy = x2 - x1, y2 - y1
        dist = math.hypot(dx, dy)
        steps = max(1, int(dist / spacing))
        return [(int(x1 + dx * t / steps), int(y1 + dy * t / steps))
                for t in range(steps + 1)]

    # --- Drawing operations (save undo first) ---

    def _do_draw_point(self, cmd: dict):
        self._save_undo()
        if self.state.oil_paint:
            self._oil_dab(cmd["x"], cmd["y"], self.state.brush_size)
        else:
            pygame.draw.circle(
                self.surface, self.state.color,
                (cmd["x"], cmd["y"]), self.state.brush_size
            )

    def _do_draw_line(self, cmd: dict):
        self._save_undo()
        if self.state.oil_paint:
            pts = self._interpolate(cmd["x1"], cmd["y1"], cmd["x2"], cmd["y2"])
            self._oil_stroke(pts)
        else:
            pygame.draw.line(
                self.surface, self.state.color,
                (cmd["x1"], cmd["y1"]), (cmd["x2"], cmd["y2"]),
                self.state.brush_size
            )

    def _do_draw_rect(self, cmd: dict):
        self._save_undo()
        rect = pygame.Rect(cmd["x"], cmd["y"], cmd["width"], cmd["height"])
        width = 0 if cmd.get("filled", False) else self.state.brush_size
        pygame.draw.rect(self.surface, self.state.color, rect, width)

    def _do_draw_ellipse(self, cmd: dict):
        self._save_undo()
        rect = pygame.Rect(cmd["x"], cmd["y"], cmd["width"], cmd["height"])
        width = 0 if cmd.get("filled", False) else self.state.brush_size
        pygame.draw.ellipse(self.surface, self.state.color, rect, width)

    def _do_draw_path(self, cmd: dict):
        self._save_undo()
        points = cmd["points"]
        if len(points) < 2:
            return
        if self.state.oil_paint:
            # Interpolate between user-supplied points for smooth dab coverage
            dense = []
            for i in range(len(points) - 1):
                dense.extend(self._interpolate(
                    points[i][0], points[i][1],
                    points[i + 1][0], points[i + 1][1]))
            self._oil_stroke(dense)
        else:
            pygame.draw.lines(
                self.surface, self.state.color, False,
                points, self.state.brush_size
            )

    def _do_flood_fill(self, cmd: dict):
        self._save_undo()
        x, y = cmd["x"], cmd["y"]
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return

        pixel_array = pygame.PixelArray(self.surface)
        target_color = pixel_array[x, y]
        fill_color = self.surface.map_rgb(self.state.color)

        if target_color == fill_color:
            pixel_array.close()
            return

        stack = [(x, y)]
        visited = set()

        while stack:
            cx, cy = stack.pop()
            if cx < 0 or cx >= self.width or cy < 0 or cy >= self.height:
                continue
            if (cx, cy) in visited:
                continue
            if pixel_array[cx, cy] != target_color:
                continue

            visited.add((cx, cy))
            pixel_array[cx, cy] = fill_color

            stack.append((cx + 1, cy))
            stack.append((cx - 1, cy))
            stack.append((cx, cy + 1))
            stack.append((cx, cy - 1))

        pixel_array.close()

    def _do_clear(self, cmd: dict):
        self._save_undo()
        self.surface.fill(self.state.background_color)

    def _do_undo(self, cmd: dict):
        self.undo()

    # --- Read-only operations (no undo) ---

    def get_pixels_rgb(self, x: int = 0, y: int = 0,
                       w: int | None = None, h: int | None = None) -> list[list[list[int]]]:
        """Return a 2D list of [r, g, b] values (row-major) for the given region."""
        if w is None:
            w = self.width - x
        if h is None:
            h = self.height - y
        # Clamp to canvas bounds
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))
        w = min(w, self.width - x)
        h = min(h, self.height - y)

        pixel_array = pygame.PixelArray(self.surface)
        rows = []
        for row in range(y, y + h):
            r_list = []
            for col in range(x, x + w):
                mapped = pixel_array[col, row]
                r, g, b, _ = self.surface.unmap_rgb(mapped)
                r_list.append([r, g, b])
            rows.append(r_list)
        pixel_array.close()
        return rows
