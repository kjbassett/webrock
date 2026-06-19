import argparse
import asyncio
import threading
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
             "'stdio' runs MCP via stdio alongside Sanic",
    )
    parser.add_argument("--mcp-port", type=int, default=8001, dest="mcp_port", help="MCP SSE server port (default: 8001)")
    parser.add_argument(
        "--project",
        default=None,
        help="Absolute path to the project directory (plugins + webrock.db). Defaults to cwd.",
    )
    parser.add_argument(
        "--paused",
        action="store_true",
        dest="paused",
        help=(
            "Start all schedules paused — no schedules run until manually resumed via the UI. "
            "Use this after a long absence to reset stale schedule timers before they fire."
        ),
    )
    args = parser.parse_args()

    app = asyncio.run(create_app(project_dir=args.project, paused=args.paused))

    if not args.no_mcp:
        from .mcp_server import build_mcp_server, run_sse_blocking, start_background_stdio
        api_url = f"http://localhost:{args.port}"

        def _flat_metadata(app_ctx):
            """Build the same flat {plugin_id: meta} dict as GET /api/plugins."""
            result = {}
            for plugin_id in app_ctx.plugins:
                node = app_ctx.metadata
                try:
                    for part in plugin_id.split("."):
                        node = node[part]
                    result[plugin_id] = node
                except (KeyError, TypeError):
                    pass
            return result

        if args.mcp_transport == "sse":
            mcp_port = args.mcp_port

            @app.listener("after_server_start")
            async def start_mcp_sse(app, loop):
                server = build_mcp_server(api_url=api_url, plugin_metadata=_flat_metadata(app.ctx))
                t = threading.Thread(target=run_sse_blocking, args=(server, mcp_port), daemon=True)
                t.start()
                print(f"MCP SSE server started on http://0.0.0.0:{mcp_port}/sse")

        else:  # stdio
            @app.listener("after_server_start")
            async def start_mcp_stdio(app, loop):
                server = build_mcp_server(api_url=api_url, plugin_metadata=_flat_metadata(app.ctx))
                start_background_stdio(server)

    app.run(host="0.0.0.0", port=args.port, single_process=True)


if __name__ == "__main__":
    main()
