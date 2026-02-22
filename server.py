"""Entry point: starts MCP server thread + pygame main loop."""

import os
# Suppress pygame welcome message before importing — it prints to stdout
# which would corrupt the MCP stdio JSON-RPC stream.
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import sys
import queue
import threading

import pygame
from canvas import Canvas
from tools import create_mcp_server

WIDTH, HEIGHT = 800, 600
TOOLBAR_H = 40
WINDOW_H = HEIGHT + TOOLBAR_H
FPS = 30

# Toolbar colours
TB_BG = (220, 220, 220)
TB_BTN = (180, 180, 180)
TB_BTN_HOVER = (160, 160, 160)
TB_TEXT = (30, 30, 30)


def run_mcp_server(mcp_server):
    """Target for the daemon thread — runs the MCP stdio server."""
    mcp_server.run(transport="stdio")


def _save_dialog_and_write(surface: pygame.Surface):
    """Open a Tk file-save dialog (runs on main thread) and write the PNG."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG image", "*.png"), ("All files", "*.*")],
        title="Save canvas as…",
    )
    root.destroy()
    if path:
        pygame.image.save(surface, path)


def _handle_request(cmd: dict, canvas: Canvas):
    """Process a request/response command from the MCP tool thread."""
    event: threading.Event = cmd["_event"]
    result: dict = cmd["_result"]
    action = cmd.get("action")
    try:
        if action == "get_pixels":
            data = canvas.get_pixels_rgb(
                cmd.get("x", 0), cmd.get("y", 0),
                cmd.get("w"), cmd.get("h"),
            )
            result["data"] = data
        elif action == "save_file":
            path = cmd["path"]
            pygame.image.save(canvas.get_display_surface(), path)
            result["data"] = f"Canvas saved to {path}"
        else:
            result["error"] = f"Unknown request action: {action}"
    except Exception as e:
        result["error"] = str(e)
    finally:
        event.set()


def main():
    # Shared command queue between MCP thread and pygame main thread
    command_queue = queue.Queue()

    # Create MCP server with tool definitions
    mcp_server = create_mcp_server(command_queue, WIDTH, HEIGHT)

    # Start MCP server in a background daemon thread
    mcp_thread = threading.Thread(target=run_mcp_server, args=(mcp_server,), daemon=True)
    mcp_thread.start()

    # Initialize pygame on the main thread
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, WINDOW_H))
    pygame.display.set_caption("Paint MCP")
    clock = pygame.time.Clock()

    canvas = Canvas(WIDTH, HEIGHT)

    font = pygame.font.SysFont(None, 24)
    save_btn_rect = pygame.Rect(10, 8, 70, 26)

    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()

        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if save_btn_rect.collidepoint(event.pos):
                    _save_dialog_and_write(canvas.get_display_surface())

        # Drain all pending commands from the queue
        while True:
            try:
                cmd = command_queue.get_nowait()
            except queue.Empty:
                break

            # Request/response bridge commands have an _event key
            if "_event" in cmd:
                _handle_request(cmd, canvas)
            else:
                try:
                    canvas.execute(cmd)
                except Exception as e:
                    print(f"Command error: {e}", file=sys.stderr)

        # --- Render ---
        # Toolbar
        pygame.draw.rect(screen, TB_BG, (0, 0, WIDTH, TOOLBAR_H))
        btn_color = TB_BTN_HOVER if save_btn_rect.collidepoint(mouse_pos) else TB_BTN
        pygame.draw.rect(screen, btn_color, save_btn_rect, border_radius=4)
        pygame.draw.rect(screen, TB_TEXT, save_btn_rect, width=1, border_radius=4)
        label = font.render("Save", True, TB_TEXT)
        label_rect = label.get_rect(center=save_btn_rect.center)
        screen.blit(label, label_rect)

        # Canvas (offset below toolbar)
        screen.blit(canvas.get_display_surface(), (0, TOOLBAR_H))
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
