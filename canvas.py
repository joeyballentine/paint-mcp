"""Drawing engine: pygame.Surface wrapper with draw ops and undo stack."""

import pygame
from dataclasses import dataclass, field


@dataclass
class DrawState:
    color: tuple = (0, 0, 0)
    brush_size: int = 3
    background_color: tuple = (255, 255, 255)


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

    # --- Drawing operations (save undo first) ---

    def _do_draw_point(self, cmd: dict):
        self._save_undo()
        pygame.draw.circle(
            self.surface, self.state.color,
            (cmd["x"], cmd["y"]), self.state.brush_size
        )

    def _do_draw_line(self, cmd: dict):
        self._save_undo()
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
