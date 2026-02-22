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
FPS = 30


def run_mcp_server(mcp_server):
    """Target for the daemon thread — runs the MCP stdio server."""
    mcp_server.run(transport="stdio")


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
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Paint MCP")
    clock = pygame.time.Clock()

    canvas = Canvas(WIDTH, HEIGHT)

    running = True
    while running:
        # Handle pygame events (close button, etc.)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Drain all pending commands from the queue
        while True:
            try:
                cmd = command_queue.get_nowait()
            except queue.Empty:
                break
            try:
                canvas.execute(cmd)
            except Exception as e:
                print(f"Command error: {e}", file=sys.stderr)

        # Render
        screen.blit(canvas.surface, (0, 0))
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
