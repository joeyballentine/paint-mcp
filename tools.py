"""MCP tool definitions. Pushes drawing commands onto a thread-safe queue."""

import json
import os
import queue
import tempfile
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
        to learn proper oil-painting technique for realistic results.
        When disabling, call get_drawing_guide() for standard mode tips."""
        _oil_paint[0] = enabled
        command_queue.put({"action": "set_oil_paint", "enabled": enabled})
        if enabled:
            return "Oil paint mode enabled. Call get_oil_painting_guide() to learn proper technique."
        return "Oil paint mode disabled. Call get_drawing_guide() for standard drawing tips."

    @mcp.tool()
    def get_oil_painting_guide() -> str:
        """Return a comprehensive guide to oil-painting technique for this
        canvas engine.  Call this after enabling oil-paint mode so you
        understand how to use the tools realistically."""
        return """
=== OIL PAINTING TECHNIQUE GUIDE ===

This canvas simulates real oil paint physics.  The brush carries pigment
that gradually depletes and picks up canvas color as you stroke.
Each stroke has a generous paint load that lasts well over 100 dabs,
so individual strokes are strong and opaque.  The key to realism is
MANY overlapping strokes building up texture and color variation.

=== CRITICAL: USE batch_strokes FOR EVERYTHING ===
batch_strokes is your PRIMARY and PREFERRED tool.  It executes many
strokes in one call.  Each stroke reloads the brush automatically.
You can set color and brush_size per-stroke inline.

You MUST use batch_strokes with LARGE batches.  A good painting needs
hundreds of strokes.  Send 20-50 strokes per batch_strokes call, and
make MULTIPLE batch_strokes calls to build up the painting.

DO NOT call draw_line or draw_path individually — always use
batch_strokes instead.

Example — block in a sky with many overlapping strokes:
  batch_strokes([
    {"type":"line","color":[70,100,160],"brush_size":20,"x1":0,"y1":20,"x2":250,"y2":15},
    {"type":"line","color":[75,108,168],"brush_size":22,"x1":0,"y1":38,"x2":260,"y2":32},
    {"type":"line","color":[80,115,175],"brush_size":18,"x1":5,"y1":55,"x2":245,"y2":50},
    {"type":"line","color":[72,105,162],"brush_size":20,"x1":0,"y1":70,"x2":250,"y2":66},
    {"type":"line","color":[85,118,178],"brush_size":22,"x1":0,"y1":85,"x2":255,"y2":80},
    {"type":"line","color":[90,125,182],"brush_size":19,"x1":5,"y1":100,"x2":248,"y2":96},
    {"type":"line","color":[78,110,170],"brush_size":21,"x1":0,"y1":115,"x2":250,"y2":110},
    {"type":"line","color":[95,130,188],"brush_size":20,"x1":0,"y1":130,"x2":252,"y2":125},
    ... continue for entire area ...
  ])

Notice: each stroke has SLIGHTLY different color (vary 5-15 per channel)
and slightly different brush_size.  This creates natural color richness.

--- STROKE MECHANICS ---
- USE MANY OVERLAPPING SHORT STROKES.  Each stroke is strong and opaque
  with plenty of paint.  The realism comes from layering many strokes,
  not from individual strokes running out of paint.
- OVERLAP STROKES DENSELY.  Adjacent strokes should overlap by ~30-50%
  of the brush width.  Don't leave gaps between strokes.
- VARY BRUSH SIZE per stroke (15-30 for base, 8-15 for mid, 3-7 for detail).
- STROKE DIRECTION MATTERS.  Follow the form: horizontal for skies,
  vertical for tree trunks, curved for round objects.
- For paths, keep each to 3-8 coordinate pairs.  For lines, use line type.
- COVER THE ENTIRE CANVAS.  Don't leave white gaps.  Every area should be
  painted with multiple overlapping strokes.

--- COLOR & LAYERING ---
- WORK DARK TO LIGHT.  Darks and mid-tones first, highlights last.
- LIMIT YOUR PALETTE to 4-6 base colors.  Mix by overlapping.
- ALWAYS VARY COLOR between strokes in the same area.  Shift 5-15 units
  per RGB channel.  Never use the exact same color for adjacent strokes.
- OVERLAPPING STROKES MIX.  Yellow over blue → green tones automatically.
- Use warm colors (reds/oranges/yellows) for foreground, cool colors
  (blues/greens/purples) for background and distance.

--- BLENDING ---
- USE blend_path SPARINGLY with LOW strength (0.05-0.15).
- BLEND ALONG EDGES only.  Don't blend entire areas.
- LEAVE SOME EDGES HARD for energy and texture.

--- WORKFLOW ---
1. BLOCK IN: 2-3 batch_strokes calls, 30-50 strokes each, large brush
   (20-30).  Cover the entire canvas with rough color masses.
   >>> PREVIEW after blocking in to check coverage and color balance. <<<
2. DEVELOP: 2-3 batch_strokes calls, 20-40 strokes each, medium brush
   (10-15).  Refine shapes, add color variation.
   >>> PREVIEW after each batch to catch problems early. <<<
3. DETAIL: 1-2 batch_strokes calls, 15-30 strokes each, small brush
   (3-8).  Highlights, darks, accents.
4. SOFTEN (optional): blend_path at 0.05-0.10 on select edges.
   >>> FINAL PREVIEW to verify the finished painting. <<<

Total for a complete painting: 5-10 batch_strokes calls, 150-400 strokes.

--- PREVIEWING YOUR WORK ---
CRITICAL: Use preview_canvas CONSTANTLY to check your work.  You cannot
see the canvas without it!  Call preview_canvas:
- AFTER EVERY LARGE CHANGE (blocking in, major color passes, big shapes).
  Large strokes can overpaint earlier work or leave unexpected gaps.
- After EACH development and detail pass to verify the result.
- BEFORE and AFTER blending to compare the effect.
- Whenever you are unsure whether strokes landed correctly.
If something looks wrong, fix it immediately before adding more layers.
Paint stacks — mistakes buried under new strokes are much harder to fix.

--- COMMON MISTAKES ---
- DO NOT use draw_line/draw_path individually.  Use batch_strokes.
- DO NOT send small batches of 3-5 strokes.  Send 20-50 per call.
- DO NOT use the same exact color for every stroke in an area.
- DO NOT leave white canvas showing through.  Cover everything.
- DO NOT use pure black or pure white.  Use dark blues/browns and
  warm creams instead.
"""

    @mcp.tool()
    def get_drawing_guide() -> str:
        """Return a guide to standard (non-oil-paint) drawing technique.
        Call this when working in normal mode for best results."""
        return """
=== STANDARD DRAWING GUIDE ===

Normal mode gives you clean, precise strokes — no paint mixing or
depletion.  Every stroke is exactly the color you set.  You have
access to ALL tools: lines, paths, rectangles, ellipses, flood fill,
and batch_strokes.

=== TOOL SELECTION ===
- batch_strokes: PREFERRED for any multi-stroke work.  Set per-stroke
  color and brush_size inline.  Much faster than individual calls.
- draw_rect / draw_ellipse: Use for geometric shapes.  Set filled=true
  for solid shapes, or leave false for outlines at current brush_size.
- flood_fill: Fill enclosed regions with the current color.  Great for
  backgrounds and large uniform areas.  Make sure the region is fully
  enclosed first, or paint will leak everywhere.
- draw_line / draw_path / draw_point: Fine for single strokes, but
  prefer batch_strokes when you need more than one or two.

=== BUILDING AN IMAGE ===
1. PLAN THE LAYOUT.  Decide where major shapes go before drawing.
   Normal mode strokes are opaque — later strokes cover earlier ones.
2. BACKGROUNDS FIRST.  Use flood_fill or large filled rectangles to
   lay down background colors before adding detail on top.
   >>> PREVIEW after setting up the background. <<<
3. LARGE SHAPES NEXT.  Block in major forms with filled rects,
   ellipses, or thick batch_strokes lines.
   >>> PREVIEW to verify shapes are positioned correctly. <<<
4. OUTLINES & STRUCTURE.  Add borders, edges, and structural lines
   with thinner strokes (brush_size 2-5).
5. DETAIL & TEXTURE.  Small strokes, dots, and thin paths for fine
   detail.  Use batch_strokes with many strokes per call.
   >>> PREVIEW after each detail pass. <<<
6. FINAL TOUCHES.  Highlights, shadows, small corrections.
   >>> FINAL PREVIEW to verify the finished image. <<<

=== PREVIEWING YOUR WORK ===
CRITICAL: Use preview_canvas CONSTANTLY.  You cannot see the canvas!
- AFTER EVERY MAJOR STEP (background, large shapes, detail passes).
- BEFORE adding detail on top of existing work — confirm the base
  looks right first, because later strokes will cover it.
- After flood_fill — verify it didn't leak into unintended areas.
- Whenever you are unsure if coordinates or sizes are correct.
Fix problems IMMEDIATELY.  Use undo() if a step went wrong, then redo
it correctly.  Don't keep building on top of mistakes.

=== COORDINATE TIPS ===
- Canvas is 800 wide x 600 tall.  Origin (0,0) is top-left.
- X increases rightward, Y increases downward.
- Rectangles and ellipses: (x, y) is the TOP-LEFT corner of the
  bounding box, not the center.
- Center of canvas: (400, 300).

=== COLOR STRATEGY ===
- Set color with set_color or per-stroke "color" in batch_strokes.
- For shading and depth, use darker variants of a color for shadows
  and lighter variants for highlights (shift all channels by 30-60).
- For outlines, use a color 40-80 units darker than the fill color.
- Avoid pure black (0,0,0) for outlines — use dark grey (30,30,30)
  or a dark saturated color instead.  It looks more natural.

=== COMMON MISTAKES ===
- DO NOT forget to preview.  Blind drawing leads to misaligned shapes
  and wasted undo steps.
- DO NOT flood_fill before closing a region.  The fill will leak across
  the entire canvas and you'll need to undo.
- DO NOT draw detail before confirming the background and layout.
- DO NOT use tiny brush_size (1-2) for large areas — it's slow and
  leaves visible gaps.  Use filled shapes or large brush_size instead.
- DO NOT guess coordinates for precise alignment.  Preview first, then
  adjust based on what you see.
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
    def preview_canvas() -> str:
        """Save the current canvas to a temporary PNG file and return its path.

        Use this to visually inspect your work.  The returned path can be
        opened with a vision/image-reading tool.  The file is placed in the
        system temp directory and will be cleaned up automatically."""
        fd, path = tempfile.mkstemp(suffix=".png", prefix="paint_mcp_preview_")
        os.close(fd)
        _request_response({"action": "save_file", "path": path})
        return path

    @mcp.tool()
    def save_canvas(file_path: str) -> str:
        """Save the current canvas to a PNG file at the given path."""
        result = _request_response({"action": "save_file", "path": file_path})
        return result

    return mcp
