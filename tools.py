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
        or flood fill). Toggle off to return to normal drawing."""
        _oil_paint[0] = enabled
        command_queue.put({"action": "set_oil_paint", "enabled": enabled})
        return f"Oil paint mode {'enabled' if enabled else 'disabled'}"

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
