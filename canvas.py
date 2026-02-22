"""Drawing engine: pygame.Surface wrapper with draw ops and undo stack."""

import math
import random
import pygame
import numpy as np
from dataclasses import dataclass, field


VALID_BRUSH_SHAPES = ("round", "flat", "filbert", "fan", "palette_knife")


@dataclass
class DrawState:
    color: tuple = (0, 0, 0)
    brush_size: int = 3
    background_color: tuple = (255, 255, 255)
    oil_paint: bool = False
    brush_shape: str = "round"


class Canvas:
    MAX_UNDO = 50

    # --- Impasto lighting constants ---
    LIGHT_DIR = np.array([-0.3, -0.5, 1.0])
    LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)
    # AMBIENT is computed so that a flat surface (normal=(0,0,1)) gets
    # light_factor = 1.0 exactly, avoiding brightness shift on unpainted areas.
    DIFFUSE_STRENGTH = 0.25
    AMBIENT = 1.0 - DIFFUSE_STRENGTH * LIGHT_DIR[2]  # ~0.784
    SPECULAR_STRENGTH = 0.15
    SPECULAR_SHININESS = 24
    HEIGHT_SCALE = 0.6
    MAX_HEIGHT = 15.0

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.state = DrawState()
        self.surface = pygame.Surface((width, height))
        self.surface.fill(self.state.background_color)
        self.height_map = np.zeros((height, width), dtype=np.float32)
        self.display_surface = pygame.Surface((width, height))
        self._lighting_dirty = True
        self.impasto_visible = False
        self._undo_stack: list[tuple[pygame.Surface, np.ndarray]] = []

    def _save_undo(self):
        if len(self._undo_stack) >= self.MAX_UNDO:
            self._undo_stack.pop(0)
        self._undo_stack.append((self.surface.copy(), self.height_map.copy()))
        self._lighting_dirty = True

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        self.surface, self.height_map = self._undo_stack.pop()
        self._lighting_dirty = True
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

    def _do_set_brush_shape(self, cmd: dict):
        shape = cmd["shape"]
        if shape in VALID_BRUSH_SHAPES:
            self.state.brush_shape = shape

    # --- Oil paint helpers ---

    def _sample_color(self, x: int, y: int) -> tuple:
        """Read the RGB color of a single pixel on the canvas."""
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))
        return tuple(self.surface.get_at((x, y))[:3])

    def _avg_color_around(self, cx: int, cy: int, radius: int) -> tuple:
        """Average the canvas color in a disc around (cx, cy), sampling
        uniformly in all directions (N/S/E/W and diagonals)."""
        r_sum, g_sum, b_sum, count = 0, 0, 0, 0
        step = max(1, radius // 3)
        for dx in range(-radius, radius + 1, step):
            for dy in range(-radius, radius + 1, step):
                if dx * dx + dy * dy <= radius * radius:
                    c = self._sample_color(cx + dx, cy + dy)
                    r_sum += c[0]; g_sum += c[1]; b_sum += c[2]
                    count += 1
        if count == 0:
            return self._sample_color(cx, cy)
        return (r_sum // count, g_sum // count, b_sum // count)

    @staticmethod
    def _blend(base: tuple, top: tuple, ratio: float = 0.5) -> tuple:
        """Mix two colors. ratio=1.0 means all top, 0.0 means all base."""
        return tuple(int(b * (1 - ratio) + t * ratio) for b, t in zip(base, top))

    def _accumulate_height(self, cx: int, cy: int, radius: int, strength: float):
        """Add gaussian-ish height to the height_map in a disc around (cx, cy)."""
        r = max(1, radius)
        y_lo = max(0, cy - r)
        y_hi = min(self.height, cy + r + 1)
        x_lo = max(0, cx - r)
        x_hi = min(self.width, cx + r + 1)
        if y_lo >= y_hi or x_lo >= x_hi:
            return
        ys = np.arange(y_lo, y_hi, dtype=np.float32) - cy
        xs = np.arange(x_lo, x_hi, dtype=np.float32) - cx
        xx, yy = np.meshgrid(xs, ys)
        dist_sq = xx * xx + yy * yy
        r_sq = float(r * r)
        mask = dist_sq <= r_sq
        # Gaussian-ish falloff: peak at center, near zero at edge
        falloff = np.exp(-3.0 * dist_sq / r_sq)
        contribution = strength * falloff * mask
        self.height_map[y_lo:y_hi, x_lo:x_hi] += contribution
        np.clip(self.height_map[y_lo:y_hi, x_lo:x_hi], 0, self.MAX_HEIGHT,
                out=self.height_map[y_lo:y_hi, x_lo:x_hi])

    def _recompute_lighting(self):
        """Apply impasto lighting to raw surface → display_surface."""
        # Copy raw pixels and release the surface lock immediately
        raw = pygame.surfarray.pixels3d(self.surface)  # shape (W, H, 3)
        raw_hwc = np.transpose(raw, (1, 0, 2)).astype(np.float32)  # (H, W, 3) copy
        del raw  # release surface lock

        # Compute normals from height_map gradient
        dh_dy, dh_dx = np.gradient(self.height_map * self.HEIGHT_SCALE)
        # Normal vectors: (-dh/dx, -dh/dy, 1.0), then normalize
        norm = np.sqrt(dh_dx * dh_dx + dh_dy * dh_dy + 1.0)
        nx = -dh_dx / norm
        ny = -dh_dy / norm
        nz = 1.0 / norm

        # Diffuse: dot(N, L)
        lx, ly, lz = self.LIGHT_DIR
        diffuse = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)

        # Specular: Blinn-Phong with half-vector H = normalize(L + V)
        # View direction is straight up: V = (0, 0, 1)
        hv = np.array([lx, ly, lz + 1.0])
        hv = hv / np.linalg.norm(hv)
        n_dot_h = np.clip(nx * hv[0] + ny * hv[1] + nz * hv[2], 0.0, 1.0)
        specular = np.power(n_dot_h, self.SPECULAR_SHININESS)
        # Only show specular highlights on raised (painted) areas
        specular *= (self.height_map > 0.05)

        # Final color — AMBIENT is tuned so flat surfaces get factor=1.0
        light_factor = self.AMBIENT + self.DIFFUSE_STRENGTH * diffuse
        light_factor = light_factor[:, :, np.newaxis]  # broadcast to (H, W, 1)
        specular_term = (self.SPECULAR_STRENGTH * specular * 255.0)[:, :, np.newaxis]

        result = raw_hwc * light_factor + specular_term
        np.clip(result, 0, 255, out=result)

        # Write to display surface
        out = np.transpose(result.astype(np.uint8), (1, 0, 2))  # back to (W, H, 3)
        display_arr = pygame.surfarray.pixels3d(self.display_surface)
        display_arr[:] = out
        del display_arr  # release surface lock

        self._lighting_dirty = False

    def get_display_surface(self) -> pygame.Surface:
        """Return the lit display surface if impasto is visible, else raw."""
        if not self.impasto_visible:
            return self.surface
        if self._lighting_dirty:
            self._recompute_lighting()
        return self.display_surface

    def _oil_dab(self, cx: int, cy: int, radius: int, brush_color: tuple = None,
                 paint_strength: float = 0.75, direction: float = 0.0):
        """Paint a single oil-paint dab, dispatching to shape-specific method.

        brush_color: the paint on the brush (defaults to state color).
        paint_strength (0..1): how much brush color vs canvas color.
        direction: stroke direction in radians (0 = right).
        """
        if brush_color is None:
            brush_color = self.state.color

        shape = self.state.brush_shape
        if shape == "round":
            self._dab_round(cx, cy, radius, brush_color, paint_strength)
        elif shape == "flat":
            self._dab_flat(cx, cy, radius, brush_color, paint_strength, direction)
        elif shape == "filbert":
            self._dab_filbert(cx, cy, radius, brush_color, paint_strength, direction)
        elif shape == "fan":
            self._dab_fan(cx, cy, radius, brush_color, paint_strength, direction)
        elif shape == "palette_knife":
            self._dab_knife(cx, cy, radius, brush_color, paint_strength, direction)
        else:
            self._dab_round(cx, cy, radius, brush_color, paint_strength)

    def _dab_round(self, cx: int, cy: int, radius: int, brush_color: tuple,
                   paint_strength: float):
        """Original circular dab with soft edges and concentric rings."""
        # Paint concentric rings from outside in; each ring blends more brush
        rings = max(2, radius // 2)
        for ring in range(rings, -1, -1):
            t = ring / max(rings, 1)            # 1.0 at edge, 0.0 at center
            ring_r = max(1, int(radius * (t + 0.05)))

            # Cubic falloff for softer edges on large brushes
            edge_blend = t * t * t
            center_ratio = paint_strength * (1.0 - edge_blend * 0.85)

            canvas_color = self._avg_color_around(cx, cy, ring_r)
            blended = self._blend(canvas_color, brush_color, center_ratio)

            # Irregular ring with jittered radius for painterly texture
            for angle_step in range(0, 360, max(4, 8 - radius // 5)):
                a = math.radians(angle_step)
                r_jitter = ring_r * random.uniform(0.8, 1.0)
                ex = int(cx + r_jitter * math.cos(a))
                ey = int(cy + r_jitter * math.sin(a))
                pygame.draw.line(self.surface, blended, (cx, cy), (ex, ey), 1)

        # Accumulate height for impasto lighting.
        size_scale = 1.0 / (1.0 + max(0, radius - 12) * 0.06)
        self._accumulate_height(cx, cy, radius, paint_strength * 0.8 * size_scale)

    def _dab_shaped(self, cx: int, cy: int, radius: int, brush_color: tuple,
                    paint_strength: float, direction: float,
                    half_w: float, half_h: float, is_ellipse: bool = False,
                    hard_edge: bool = False, height_mult: float = 1.0):
        """Generic shaped dab renderer used by flat, filbert, fan, and knife.

        half_w, half_h: half-extents in local coords (before rotation).
        is_ellipse: True for elliptical mask (filbert), False for rectangular.
        hard_edge: True for minimal falloff (palette knife).
        height_mult: multiplier for impasto height accumulation.
        """
        cos_d = math.cos(-direction)
        sin_d = math.sin(-direction)

        # Bounding box in world coords (axis-aligned)
        extent = int(math.ceil(max(half_w, half_h))) + 1
        x_lo = max(0, cx - extent)
        x_hi = min(self.width, cx + extent + 1)
        y_lo = max(0, cy - extent)
        y_hi = min(self.height, cy + extent + 1)
        if x_lo >= x_hi or y_lo >= y_hi:
            return

        # Sample canvas color around the center for blending
        canvas_color = self._avg_color_around(cx, cy, max(1, int(min(half_w, half_h))))

        # We iterate in steps for performance (matching ring approach density)
        for wy in range(y_lo, y_hi):
            for wx in range(x_lo, x_hi):
                # Rotate to local coordinates
                dx = wx - cx
                dy = wy - cy
                lx = dx * cos_d - dy * sin_d
                ly = dx * sin_d + dy * cos_d

                # Test if inside shape
                if is_ellipse:
                    # Ellipse: (lx/half_w)^2 + (ly/half_h)^2 <= 1
                    nx = lx / half_w if half_w > 0 else 0
                    ny = ly / half_h if half_h > 0 else 0
                    dist_sq = nx * nx + ny * ny
                    if dist_sq > 1.0:
                        continue
                    t = math.sqrt(dist_sq)  # 0 at center, 1 at edge
                else:
                    # Rectangle: |lx| <= half_w and |ly| <= half_h
                    if abs(lx) > half_w or abs(ly) > half_h:
                        continue
                    # Normalized distance: max of the two axis ratios
                    tx = abs(lx) / half_w if half_w > 0 else 0
                    ty = abs(ly) / half_h if half_h > 0 else 0
                    t = max(tx, ty)

                # Compute paint strength with falloff
                if hard_edge:
                    local_strength = paint_strength * 0.95
                else:
                    # Cubic falloff from center to edge
                    edge_blend = t * t * t
                    local_strength = paint_strength * (1.0 - edge_blend * 0.85)

                # Add jitter for painterly texture (skip ~5% of edge pixels)
                if t > 0.7 and random.random() < 0.05:
                    continue

                pixel_color = self._sample_color(wx, wy)
                blended = self._blend(pixel_color, brush_color, local_strength)
                self.surface.set_at((wx, wy), blended)

        # Accumulate height for impasto
        effective_r = int(max(half_w, half_h))
        size_scale = 1.0 / (1.0 + max(0, effective_r - 12) * 0.06)
        self._accumulate_height(cx, cy, effective_r,
                                paint_strength * 0.8 * size_scale * height_mult)

    def _dab_flat(self, cx: int, cy: int, radius: int, brush_color: tuple,
                  paint_strength: float, direction: float):
        """Flat brush: wide rectangle, aspect ratio ~3:1."""
        half_w = radius * 1.5   # wide along stroke
        half_h = radius * 0.5   # narrow perpendicular
        self._dab_shaped(cx, cy, radius, brush_color, paint_strength, direction,
                         half_w, half_h, is_ellipse=False)

    def _dab_filbert(self, cx: int, cy: int, radius: int, brush_color: tuple,
                     paint_strength: float, direction: float):
        """Filbert brush: oval, aspect ratio ~2.5:1, soft edges."""
        half_w = radius * 1.25
        half_h = radius * 0.5
        self._dab_shaped(cx, cy, radius, brush_color, paint_strength, direction,
                         half_w, half_h, is_ellipse=True)

    def _dab_fan(self, cx: int, cy: int, radius: int, brush_color: tuple,
                 paint_strength: float, direction: float):
        """Fan brush: very wide, very thin rectangle, aspect ratio ~6:1."""
        half_w = radius * 3.0
        half_h = radius * 0.5
        # Fan brush has lighter paint application
        self._dab_shaped(cx, cy, radius, brush_color, paint_strength * 0.6,
                         direction, half_w, half_h, is_ellipse=False)

    def _dab_knife(self, cx: int, cy: int, radius: int, brush_color: tuple,
                   paint_strength: float, direction: float):
        """Palette knife: hard-edged rectangle, aspect ratio ~4:1, extra impasto."""
        half_w = radius * 2.0
        half_h = radius * 0.5
        self._dab_shaped(cx, cy, radius, brush_color, paint_strength, direction,
                         half_w, half_h, is_ellipse=False, hard_edge=True,
                         height_mult=1.8)

    def _oil_stroke(self, points: list[tuple[int, int]]):
        """Lay down oil-paint dabs along a series of points.

        Simulates a loaded brush: paint starts strong and depletes
        exponentially.  The brush also *picks up* canvas color as it
        moves, so the carried pigment shifts toward whatever it drags
        over — just like real oil paint."""
        n = len(points)
        if n == 0:
            return

        # --- paint-load parameters ---
        # Half-life in dab-steps: after this many dabs, paint is at 50%
        half_life = 140
        decay = 0.5 ** (1.0 / half_life)      # per-step multiplier

        paint_load = 0.95                       # current opacity of brush paint
        carried_color = self.state.color        # what's on the brush right now
        pickup_rate = 0.03                      # how fast canvas color mixes in

        for i, (px, py) in enumerate(points):
            # Compute stroke direction from surrounding points
            if i < n - 1:
                direction = math.atan2(points[i + 1][1] - py,
                                       points[i + 1][0] - px)
            elif i > 0:
                direction = math.atan2(py - points[i - 1][1],
                                       px - points[i - 1][0])
            else:
                direction = 0.0

            # Sample what's under the brush before we stamp
            canvas_color = self._avg_color_around(px, py, self.state.brush_size)

            # The brush picks up a bit of canvas color each dab
            carried_color = self._blend(canvas_color, carried_color,
                                        1.0 - pickup_rate)

            size_jitter = random.uniform(0.88, 1.12)
            # Brush also loses width as paint runs out
            size_factor = 0.8 + 0.2 * paint_load
            r = max(1, int(self.state.brush_size * size_jitter * size_factor))
            self._oil_dab(px, py, r, brush_color=carried_color,
                          paint_strength=paint_load, direction=direction)

            # Exponential depletion
            paint_load *= decay
            # Floor so the tail doesn't become invisible
            paint_load = max(paint_load, 0.18)

    def _blend_dab(self, cx: int, cy: int, radius: int, strength: float = 0.15,
                   dx_dir: float = 0.0, dy_dir: float = 0.0):
        """Smudge/blend dab: pulls colors in the stroke direction.

        Samples in all directions around the point, weighted so pixels
        *behind* the stroke direction contribute more (simulating a
        palette knife dragging paint forward).  strength is capped at
        0.25 to keep blending subtle."""
        strength = min(strength, 0.25)

        # If we have a stroke direction, weight samples behind the motion
        has_dir = abs(dx_dir) + abs(dy_dir) > 0.01
        if has_dir:
            mag = math.hypot(dx_dir, dy_dir)
            ndx, ndy = dx_dir / mag, dy_dir / mag
        else:
            ndx, ndy = 0.0, 0.0

        # Gather directional weighted average
        r_sum, g_sum, b_sum, w_sum = 0.0, 0.0, 0.0, 0.0
        step = max(1, radius // 3)
        for sx in range(-radius, radius + 1, step):
            for sy in range(-radius, radius + 1, step):
                dist_sq = sx * sx + sy * sy
                if dist_sq > radius * radius:
                    continue
                c = self._sample_color(cx + sx, cy + sy)
                # Distance weight: closer = more influence
                dist = math.sqrt(dist_sq) if dist_sq > 0 else 0.1
                w = 1.0 / (1.0 + dist * 0.3)
                # Directional weight: behind the stroke = more influence
                if has_dir and dist > 0.1:
                    dot = (-sx * ndx + -sy * ndy) / dist
                    w *= (1.0 + max(0, dot) * 1.5)
                r_sum += c[0] * w; g_sum += c[1] * w; b_sum += c[2] * w
                w_sum += w

        if w_sum < 0.01:
            return
        avg = (int(r_sum / w_sum), int(g_sum / w_sum), int(b_sum / w_sum))

        # Paint concentric rings, fading toward edge
        rings = max(2, radius // 2)
        for ring in range(rings, -1, -1):
            t = ring / max(rings, 1)
            ring_r = max(1, int(radius * (t + 0.05)))
            local_str = strength * (1.0 - t * t * 0.7)

            for angle_step in range(0, 360, max(4, 8 - radius // 5)):
                a = math.radians(angle_step)
                r_jitter = ring_r * random.uniform(0.88, 1.0)
                ex = int(cx + r_jitter * math.cos(a))
                ey = int(cy + r_jitter * math.sin(a))
                pixel_color = self._sample_color(ex, ey)
                blended = self._blend(pixel_color, avg, local_str)
                pygame.draw.line(self.surface, blended, (cx, cy), (ex, ey), 1)

    def _blend_stroke(self, points: list[tuple[int, int]], strength: float = 0.15):
        """Run the blend brush along a series of points, using stroke
        direction to weight the smudge like a palette knife."""
        for i, (px, py) in enumerate(points):
            # Compute local stroke direction from surrounding points
            dx_dir, dy_dir = 0.0, 0.0
            if i > 0:
                dx_dir = px - points[i - 1][0]
                dy_dir = py - points[i - 1][1]
            elif i < len(points) - 1:
                dx_dir = points[i + 1][0] - px
                dy_dir = points[i + 1][1] - py

            size_jitter = random.uniform(0.92, 1.08)
            r = max(1, int(self.state.brush_size * size_jitter))
            self._blend_dab(px, py, r, strength, dx_dir, dy_dir)

    @staticmethod
    def _interpolate(x1, y1, x2, y2, spacing=3) -> list[tuple[int, int]]:
        """Return evenly-spaced points along a line segment."""
        dx, dy = x2 - x1, y2 - y1
        dist = math.hypot(dx, dy)
        steps = max(1, int(dist / spacing))
        return [(int(x1 + dx * t / steps), int(y1 + dy * t / steps))
                for t in range(steps + 1)]

    # --- Normal-mode shape helpers ---

    def _shape_extents(self, radius: int) -> tuple[float, float]:
        """Return (half_w, half_h) for the current brush shape at given radius."""
        shape = self.state.brush_shape
        if shape == "flat":
            return radius * 1.5, radius * 0.5
        elif shape == "filbert":
            return radius * 1.25, radius * 0.5
        elif shape == "fan":
            return radius * 3.0, radius * 0.5
        elif shape == "palette_knife":
            return radius * 2.0, radius * 0.5
        return radius, radius  # round

    def _draw_shape_polygon(self, cx: int, cy: int, radius: int,
                            direction: float = 0.0):
        """Draw a filled rotated polygon for non-round shapes in normal mode."""
        shape = self.state.brush_shape
        half_w, half_h = self._shape_extents(radius)

        if shape == "filbert":
            # Approximate ellipse with polygon points
            cos_d = math.cos(direction)
            sin_d = math.sin(direction)
            pts = []
            for a_step in range(0, 360, 15):
                a = math.radians(a_step)
                lx = half_w * math.cos(a)
                ly = half_h * math.sin(a)
                wx = cx + lx * cos_d - ly * sin_d
                wy = cy + lx * sin_d + ly * cos_d
                pts.append((int(wx), int(wy)))
            if len(pts) >= 3:
                pygame.draw.polygon(self.surface, self.state.color, pts)
        else:
            # Rectangle shapes (flat, fan, palette_knife)
            cos_d = math.cos(direction)
            sin_d = math.sin(direction)
            corners = [(-half_w, -half_h), (half_w, -half_h),
                       (half_w, half_h), (-half_w, half_h)]
            pts = []
            for lx, ly in corners:
                wx = cx + lx * cos_d - ly * sin_d
                wy = cy + lx * sin_d + ly * cos_d
                pts.append((int(wx), int(wy)))
            pygame.draw.polygon(self.surface, self.state.color, pts)

    def _normal_draw_point(self, x: int, y: int):
        """Draw a single point in normal mode, respecting brush shape."""
        if self.state.brush_shape == "round":
            pygame.draw.circle(self.surface, self.state.color,
                               (x, y), self.state.brush_size)
        else:
            self._draw_shape_polygon(x, y, self.state.brush_size, 0.0)

    def _normal_draw_line(self, x1: int, y1: int, x2: int, y2: int):
        """Draw a line in normal mode, respecting brush shape."""
        if self.state.brush_shape == "round":
            pygame.draw.line(self.surface, self.state.color,
                             (x1, y1), (x2, y2), self.state.brush_size)
        else:
            direction = math.atan2(y2 - y1, x2 - x1)
            pts = self._interpolate(x1, y1, x2, y2)
            for px, py in pts:
                self._draw_shape_polygon(px, py, self.state.brush_size, direction)

    def _normal_draw_path(self, points: list):
        """Draw a path in normal mode, respecting brush shape."""
        if self.state.brush_shape == "round":
            pygame.draw.lines(self.surface, self.state.color, False,
                              points, self.state.brush_size)
        else:
            for i in range(len(points) - 1):
                p1, p2 = points[i], points[i + 1]
                direction = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                seg_pts = self._interpolate(p1[0], p1[1], p2[0], p2[1])
                for px, py in seg_pts:
                    self._draw_shape_polygon(px, py, self.state.brush_size,
                                             direction)

    # --- Drawing operations (save undo first) ---

    def _do_draw_point(self, cmd: dict):
        self._save_undo()
        if self.state.oil_paint:
            self._oil_dab(cmd["x"], cmd["y"], self.state.brush_size)
        else:
            self._normal_draw_point(cmd["x"], cmd["y"])

    def _do_draw_line(self, cmd: dict):
        self._save_undo()
        if self.state.oil_paint:
            pts = self._interpolate(cmd["x1"], cmd["y1"], cmd["x2"], cmd["y2"])
            self._oil_stroke(pts)
        else:
            self._normal_draw_line(cmd["x1"], cmd["y1"], cmd["x2"], cmd["y2"])

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
            self._normal_draw_path(points)

    def _do_batch_strokes(self, cmd: dict):
        """Execute many strokes in one command.  Each stroke gets a fresh
        brush load.  Accepts optional per-stroke color and brush_size."""
        self._save_undo()
        strokes = cmd["strokes"]
        for s in strokes:
            # Apply per-stroke overrides
            if "color" in s:
                self.state.color = tuple(s["color"])
            if "brush_size" in s:
                self.state.brush_size = s["brush_size"]
            if "brush_shape" in s:
                shape = s["brush_shape"]
                if shape in VALID_BRUSH_SHAPES:
                    self.state.brush_shape = shape

            kind = s.get("type", "path")
            if kind == "line":
                pts = self._interpolate(s["x1"], s["y1"], s["x2"], s["y2"])
                if self.state.oil_paint:
                    self._oil_stroke(pts)
                else:
                    self._normal_draw_line(s["x1"], s["y1"],
                                           s["x2"], s["y2"])
            elif kind == "path":
                points = s["points"]
                if len(points) < 2:
                    continue
                if self.state.oil_paint:
                    dense = []
                    for i in range(len(points) - 1):
                        dense.extend(self._interpolate(
                            points[i][0], points[i][1],
                            points[i + 1][0], points[i + 1][1]))
                    self._oil_stroke(dense)
                else:
                    self._normal_draw_path(points)
            elif kind == "point":
                if self.state.oil_paint:
                    self._oil_dab(s["x"], s["y"], self.state.brush_size)
                else:
                    self._normal_draw_point(s["x"], s["y"])

    def _do_blend_path(self, cmd: dict):
        self._save_undo()
        points = cmd["points"]
        strength = cmd.get("strength", 0.15)
        if len(points) < 2:
            return
        dense = []
        for i in range(len(points) - 1):
            dense.extend(self._interpolate(
                points[i][0], points[i][1],
                points[i + 1][0], points[i + 1][1]))
        self._blend_stroke(dense, strength)

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
        self.height_map[:] = 0

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
