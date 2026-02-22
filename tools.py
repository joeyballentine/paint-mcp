"""MCP tool definitions. Pushes drawing commands onto a thread-safe queue."""

import json
import queue
import threading
from typing import Optional
from mcp.server.fastmcp import FastMCP


def clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def create_mcp_server(command_queue: queue.Queue, width: int = 800, height: int = 600) -> FastMCP:
    mcp = FastMCP("paint-mcp")

    # Local state mirror so get_canvas_info can respond without touching pygame
    _color = [0, 0, 0]
    _brush_size = [3]
    _oil_paint = [False]

    @mcp.tool()
    def get_canvas_info() -> str:
        """Get canvas dimensions and current drawing settings."""
        return (
            f"Canvas: {width}x{height}, "
            f"color: rgb({_color[0]}, {_color[1]}, {_color[2]}), "
            f"brush_size: {_brush_size[0]}, "
            f"oil_paint_mode: {'on' if _oil_paint[0] else 'off'}"
        )

    @mcp.tool()
    def set_oil_paint_mode(enabled: bool) -> str:
        """Enable or disable oil-paint mode.

        When enabled, all strokes simulate oil paint: colors mix/blend with
        the existing canvas, brush strokes have natural variation, and only
        point, line, and path drawing are available (no rectangles, ellipses,
        or flood fill). Toggle off to return to normal drawing.

        IMPORTANT — call get_oil_painting_guide() after enabling this mode
        to learn proper oil-painting technique for realistic results."""
        _oil_paint[0] = enabled
        command_queue.put({"action": "set_oil_paint", "enabled": enabled})
        return f"Oil paint mode {'enabled' if enabled else 'disabled'}"

    @mcp.tool()
    def get_oil_painting_guide() -> str:
        """Return a comprehensive guide to oil-painting technique for this
        canvas engine.  Call this after enabling oil-paint mode so you
        understand how to use the tools realistically."""
        return """
=== OIL PAINTING TECHNIQUE GUIDE ===

This canvas simulates real oil paint physics.  The brush carries a finite
load of pigment that depletes and picks up canvas color as you stroke.
Follow these principles for realistic results.

--- USE batch_strokes — IT IS YOUR PRIMARY TOOL ---
The batch_strokes tool lets you execute MANY strokes in a single call.
Each stroke in the batch starts with a fresh brush load.  You can set
color and brush_size per-stroke inside the batch.  This is FAR more
efficient than calling draw_line/draw_path one at a time.

Example — block in a sky area with 8 overlapping strokes:
  batch_strokes([
    {"type":"line","color":[70,100,160],"brush_size":20,"x1":0,"y1":30,"x2":200,"y2":25},
    {"type":"line","color":[80,115,175],"brush_size":22,"x1":0,"y1":55,"x2":200,"y2":48},
    {"type":"line","color":[90,130,185],"brush_size":18,"x1":10,"y1":78,"x2":190,"y2":72},
    {"type":"line","color":[100,140,195],"brush_size":20,"x1":0,"y1":100,"x2":200,"y2":95},
    ...
  ])

ALWAYS prefer batch_strokes over individual draw_line/draw_path calls.
A typical painting requires 50-200+ strokes — batch them in groups of
10-30 per call.

--- STROKE MECHANICS ---
- USE MANY SHORT STROKES.  Each stroke has a generous paint load (~80 dabs
  before serious depletion), but oil painting is built from layered,
  overlapping strokes — not single sweeps.
- Each stroke in a batch_strokes call reloads the brush automatically.
- VARY BRUSH SIZE per stroke.  Use larger brushes (15-30) for broad base
  layers and washes, medium (8-15) for mid-detail, and small (3-7) for
  fine accents.  Set brush_size on each stroke in the batch.
- STROKE DIRECTION MATTERS.  Parallel strokes in one area create a visual
  "grain" like real brushwork.  Follow the form of what you're painting:
  horizontal for skies, vertical for tree trunks, curved for round objects.
- For paths, keep each path to 3-8 coordinate pairs.  For lines, each
  line segment is one stroke.

--- COLOR & LAYERING ---
- WORK DARK TO LIGHT.  Lay down darks and mid-tones first, then build up
  highlights.  Oil paint is opaque; lighter colors can cover darker ones
  with a fresh loaded stroke.
- LIMIT YOUR PALETTE.  Pick 4-6 base colors and mix by overlapping strokes
  rather than setting dozens of different RGB values.  Adjacent strokes
  naturally blend where they overlap — this is the key to rich color.
- VARY COLOR SLIGHTLY per stroke.  Don't use the exact same RGB for every
  stroke in an area.  Shift by 5-15 units per channel between strokes
  for richness: e.g. [70,100,160], [75,105,155], [65,95,165].
- OVERLAPPING STROKES MIX.  Because the brush picks up existing color,
  painting a yellow stroke over a blue area produces greenish tones
  naturally.  Use this intentionally for color transitions.
- TEMPERATURE SHIFTS.  Warm colors (reds, oranges, yellows) come forward;
  cool colors (blues, greens, purples) recede.  Use this for depth.

--- BLENDING ---
- USE blend_path SPARINGLY with LOW strength (0.05-0.15).  Multiple gentle
  passes are far better than one aggressive blend.  Real painters blend
  by lightly dragging a clean brush — not by smashing colors together.
- BLEND ALONG EDGES, not across whole areas.  Soften the boundary between
  two color zones with a short blend_path that follows the edge.
- LEAVE SOME EDGES HARD.  Not everything should be blended.  Crisp
  transitions between strokes give the painting energy and texture.

--- COMPOSITION WORKFLOW ---
1. BLOCK IN: batch_strokes with large brush (20-30), rough shapes and
   value masses.  Use just 3-4 colors.  20-40 short overlapping strokes
   per batch call.
2. DEVELOP: batch_strokes with medium brush (10-15), refine shapes and
   add mid-tone colors.  Let strokes follow the form of objects.
3. DETAIL: batch_strokes with small brush (3-8), add highlights, darks,
   and accents.  Very short strokes or single dabs.
4. SOFTEN (optional): Use blend_path at 0.05-0.10 strength to soften
   select edges.  Do NOT blend everything.

--- COMMON MISTAKES ---
- DO NOT call draw_line/draw_path one at a time in a loop.  Use
  batch_strokes to send many strokes in one call.
- DO NOT use long paths with 50+ points — the brush depletes and you
  just smear.  Keep paths to 3-8 points and use many of them.
- DO NOT blend at strength > 0.15 — it destroys texture and looks muddy.
- DO NOT paint with pure black (0,0,0) or pure white (255,255,255).
  Use dark blues/browns for darks and warm creams for highlights.
- DO NOT try to cover an area with one stroke.  Build up with many short
  overlapping strokes for rich, textured coverage.
"""

    @mcp.tool()
    def set_color(r: int, g: int, b: int) -> str:
        """Set the drawing color (RGB, each 0-255)."""
        r, g, b = clamp(r, 0, 255), clamp(g, 0, 255), clamp(b, 0, 255)
        _color[:] = [r, g, b]
        command_queue.put({"action": "set_color", "r": r, "g": g, "b": b})
        return f"Color set to rgb({r}, {g}, {b})"

    @mcp.tool()
    def set_brush_size(size: int) -> str:
        """Set the brush/stroke size (1-50 pixels)."""
        size = clamp(size, 1, 50)
        _brush_size[0] = size
        command_queue.put({"action": "set_brush_size", "size": size})
        return f"Brush size set to {size}"

    @mcp.tool()
    def draw_point(x: int, y: int) -> str:
        """Draw a single dot at (x, y)."""
        command_queue.put({"action": "draw_point", "x": x, "y": y})
        return f"Drew point at ({x}, {y})"

    @mcp.tool()
    def draw_line(x1: int, y1: int, x2: int, y2: int) -> str:
        """Draw a line from (x1, y1) to (x2, y2)."""
        command_queue.put({"action": "draw_line", "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        return f"Drew line from ({x1}, {y1}) to ({x2}, {y2})"

    @mcp.tool()
    def draw_rect(x: int, y: int, width: int, height: int, filled: bool = False) -> str:
        """Draw a rectangle. (x, y) is the top-left corner."""
        if _oil_paint[0]:
            return "Blocked: rectangles are not available in oil-paint mode. Use draw_line or draw_path instead."
        command_queue.put({
            "action": "draw_rect",
            "x": x, "y": y, "width": width, "height": height, "filled": filled,
        })
        mode = "filled" if filled else "outline"
        return f"Drew {mode} rectangle at ({x}, {y}) size {width}x{height}"

    @mcp.tool()
    def draw_ellipse(x: int, y: int, width: int, height: int, filled: bool = False) -> str:
        """Draw an ellipse bounded by the rectangle at (x, y) with given size."""
        if _oil_paint[0]:
            return "Blocked: ellipses are not available in oil-paint mode. Use draw_line or draw_path instead."
        command_queue.put({
            "action": "draw_ellipse",
            "x": x, "y": y, "width": width, "height": height, "filled": filled,
        })
        mode = "filled" if filled else "outline"
        return f"Drew {mode} ellipse at ({x}, {y}) size {width}x{height}"

    @mcp.tool()
    def draw_path(points: list[list[int]]) -> str:
        """Draw a freehand path through a list of [x, y] coordinate pairs."""
        command_queue.put({"action": "draw_path", "points": points})
        return f"Drew path through {len(points)} points"

    @mcp.tool()
    def batch_strokes(strokes: list[dict]) -> str:
        """Execute many strokes in a single command — MUCH more efficient than
        calling draw_line / draw_path one at a time.  Each stroke starts with
        a freshly loaded brush (paint reloads between strokes automatically).

        strokes: a list of stroke objects.  Each stroke object has:
          - "type": "line" | "path" | "point"
          - For "line":  {"type":"line", "x1":…, "y1":…, "x2":…, "y2":…}
          - For "path":  {"type":"path", "points":[[x,y], [x,y], …]}
          - For "point": {"type":"point", "x":…, "y":…}
          - Optional per-stroke overrides (applied before the stroke):
            - "color": [r, g, b]       — change color for this stroke
            - "brush_size": int        — change brush size for this stroke

        Example — paint three overlapping strokes with different colors:
        [
          {"type":"line", "color":[180,60,40], "brush_size":12, "x1":100, "y1":200, "x2":250, "y2":180},
          {"type":"line", "color":[60,120,180], "x1":120, "y1":210, "x2":260, "y2":190},
          {"type":"path", "color":[200,180,50], "brush_size":8, "points":[[300,100],[320,130],[310,160]]}
        ]

        This is the PREFERRED way to paint in oil-paint mode.  Use many
        short strokes (each 2-6 points for paths, or line segments) to
        build up coverage, rather than a few long strokes."""
        command_queue.put({"action": "batch_strokes", "strokes": strokes})
        return f"Executed batch of {len(strokes)} strokes"

    @mcp.tool()
    def blend_path(points: list[list[int]], strength: float = 0.10) -> str:
        """Blend/smudge brush: drag through [x, y] coordinate pairs to soften
        and merge existing colors — like dragging a clean palette knife.

        The blend is directional: it pulls color forward along the stroke
        direction, sampling vertically and horizontally in all directions.

        strength (0.0-0.25): how aggressively colors merge. Values above
        0.25 are clamped — subtle multi-pass blending looks far more
        realistic than a single heavy pass.  Recommended: 0.05-0.15.
        Works in both normal and oil-paint mode."""
        strength = max(0.0, min(0.25, strength))
        command_queue.put({"action": "blend_path", "points": points, "strength": strength})
        return f"Blended along {len(points)} points (strength={strength:.2f})"

    @mcp.tool()
    def flood_fill(x: int, y: int) -> str:
        """Bucket-fill the area at (x, y) with the current color."""
        if _oil_paint[0]:
            return "Blocked: flood fill is not available in oil-paint mode. Use draw_line or draw_path instead."
        command_queue.put({"action": "flood_fill", "x": x, "y": y})
        return f"Flood filled at ({x}, {y})"

    @mcp.tool()
    def clear_canvas() -> str:
        """Clear the entire canvas to white."""
        command_queue.put({"action": "clear"})
        return "Canvas cleared"

    @mcp.tool()
    def undo() -> str:
        """Undo the last drawing operation."""
        command_queue.put({"action": "undo"})
        return "Undo performed"

    def _request_response(cmd: dict, timeout: float = 5.0):
        """Send a command to the main thread and wait for a response."""
        event = threading.Event()
        result: dict = {}
        cmd["_event"] = event
        cmd["_result"] = result
        command_queue.put(cmd)
        if not event.wait(timeout):
            raise TimeoutError("Main thread did not respond in time")
        if "error" in result:
            raise RuntimeError(result["error"])
        return result["data"]

    @mcp.tool()
    def get_canvas_pixels(x: Optional[int] = None, y: Optional[int] = None,
                          width: Optional[int] = None, height: Optional[int] = None) -> str:
        """Return RGB pixel data from the canvas as a JSON 2D array of [r,g,b] values (row-major).

        All parameters are optional. Omit them to get the full canvas (800x600 = 480K pixels — very large!).
        For efficiency, request a small region instead, e.g. x=100, y=100, width=50, height=50."""
        cmd: dict = {"action": "get_pixels"}
        if x is not None:
            cmd["x"] = x
        if y is not None:
            cmd["y"] = y
        if width is not None:
            cmd["w"] = width
        if height is not None:
            cmd["h"] = height
        pixels = _request_response(cmd)
        return json.dumps(pixels)

    @mcp.tool()
    def save_canvas(file_path: str) -> str:
        """Save the current canvas to a PNG file at the given path."""
        result = _request_response({"action": "save_file", "path": file_path})
        return result

    return mcp
