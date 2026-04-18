import argparse
import asyncio
from .app import create_app


def main():
    parser = argparse.ArgumentParser(description="Run the webrock server")
    parser.add_argument("--port", type=int, default=8000, help="Sanic server port (default: 8000)")
    parser.add_argument("--no-mcp", action="store_true", dest="no_mcp", help="Disable MCP server")
    parser.add_argument(
        "--mcp-transport",
        choices=["sse", "stdio"],
        default="sse",
        dest="mcp_transport",
        help="MCP transport: 'sse' starts an HTTP SSE server alongside Sanic (default); "
             "'stdio' starts only an MCP stdio server (no web UI)",
    )
    parser.add_argument("--mcp-port", type=int, default=8001, dest="mcp_port", help="MCP SSE server port (default: 8001)")
    parser.add_argument(
        "--project",
        default=None,
        help="Absolute path to the project directory (plugins + webrock.db). Defaults to cwd.",
    )
    args = parser.parse_args()

    if args.mcp_transport == "stdio" and not args.no_mcp:
        # Standalone stdio mode — no Sanic
        from .mcp_server import run_standalone_stdio
        run_standalone_stdio(project_dir=args.project)
        return

    app = asyncio.run(create_app())

    if not args.no_mcp:
        from .mcp_server import start_background_sse
        start_background_sse(app.ctx.plugins, app.ctx.metadata, args.mcp_port)

    app.run(host="0.0.0.0", port=args.port, single_process=True)


if __name__ == "__main__":
    main()
