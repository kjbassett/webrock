"""
MCP server for webrock.

Exposes one scheduling tool per plugin (respecting mcp_disabled=True) plus
four management tools: list_schedules, edit_schedule, cancel_schedule,
get_run_history.

All tool calls are forwarded to the webrock REST API — no direct db access.

Transport modes
---------------
SSE (default when running `rock`):
    run_sse_blocking(server, port)  — called from after_server_start in run.py

stdio (for Claude Desktop / `rock-mcp`):
    run_stdio_blocking(server)      — called from after_server_start in run.py
    main_stdio_entry()              — standalone entry point, requires rock running
"""

import asyncio
import json
import sys
import threading
import urllib.request
import urllib.parse
import time
from typing import Any

from mcp.server import Server
from mcp.types import Tool, TextContent


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCHEDULE_FIELDS = frozenset({
    "schedule_type",
    "run_at",
    "interval_seconds",
    "after_schedule_id",
    "cron_minutes",
    "cron_hours",
    "cron_days_of_week",
    "cron_days_of_month",
    "cron_months",
})

_PYTHON_TO_JSON_TYPE: dict[str, dict] = {
    "str": {"type": "string"},
    "int": {"type": "integer"},
    "float": {"type": "number"},
    "bool": {"type": "boolean"},
}

_SCHEDULE_PROPERTIES: dict[str, dict] = {
    "schedule_type": {
        "type": "string",
        "enum": ["once", "interval", "cron", "after"],
        "default": "once",
        "description": (
            "How to schedule: 'once' (default) runs one time; "
            "'interval' repeats every interval_seconds; "
            "'cron' uses cron-style fields; "
            "'after' runs after another schedule completes."
        ),
    },
    "run_at": {
        "type": "string",
        "description": "ISO datetime to run once (e.g. '2025-06-01T09:00:00'). Omit to run immediately.",
    },
    "interval_seconds": {
        "type": "integer",
        "description": "For interval type: run every N seconds.",
        "minimum": 1,
    },
    "after_schedule_id": {
        "type": "integer",
        "description": "For after type: ID of the schedule whose completion triggers this job.",
    },
    "cron_minutes": {"type": "string", "default": "*", "description": "Cron minute field (0-59, *, ranges, lists)."},
    "cron_hours": {"type": "string", "default": "*", "description": "Cron hour field (0-23, *, ranges, lists)."},
    "cron_days_of_week": {"type": "string", "default": "*", "description": "Cron day-of-week (0-6 / mon-sun / *, ranges, lists)."},
    "cron_days_of_month": {"type": "string", "default": "*", "description": "Cron day-of-month (1-31, *, ranges, lists)."},
    "cron_months": {"type": "string", "default": "*", "description": "Cron month field (1-12 / jan-dec / *, ranges, lists)."},
}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _api_get(api_url: str, path: str) -> Any:
    with urllib.request.urlopen(f"{api_url}{path}", timeout=5) as r:
        return json.loads(r.read())


def _api_post(api_url: str, path: str, body: dict, method: str = "POST") -> Any:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{api_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def _api_delete(api_url: str, path: str) -> Any:
    req = urllib.request.Request(f"{api_url}{path}", method="DELETE")
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def _fetch_plugin_metadata(api_url: str, retries: int = 1, delay: float = 1.0) -> dict:
    for attempt in range(retries):
        try:
            return _api_get(api_url, "/api/plugins")
        except Exception as e:
            if attempt == retries - 1:
                raise RuntimeError(f"Cannot reach webrock API at {api_url}: {e}")
            time.sleep(delay)
    return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plugin_to_tool_name(plugin_id: str, all_plugin_ids: list[str]) -> str:
    func_name = plugin_id.rsplit(".", 1)[-1]
    if sum(1 for pid in all_plugin_ids if pid.rsplit(".", 1)[-1] == func_name) == 1:
        return f"schedule_{func_name}"
    return f"schedule_{plugin_id.replace('.', '__')}"


def _build_input_schema(plugin_meta: dict) -> dict:
    properties: dict[str, Any] = {}
    required: list[str] = []

    for arg in plugin_meta.get("args", []):
        name: str = arg["name"]
        python_type: str = arg.get("type", "any")
        json_type = _PYTHON_TO_JSON_TYPE.get(python_type, {})
        prop: dict[str, Any] = dict(json_type)

        doc = arg.get("doc") or arg.get("description") or ""
        if doc:
            prop["description"] = doc
        if arg.get("default") is not None:
            prop["default"] = arg["default"]
        if arg.get("min") is not None:
            prop["minimum"] = arg["min"]
        if arg.get("max") is not None:
            prop["maximum"] = arg["max"]

        properties[name] = prop
        if arg.get("default") is None:
            required.append(name)

    properties.update(_SCHEDULE_PROPERTIES)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _build_config_from_args(stype: str, arguments: dict) -> dict:
    if stype == "once":
        ts = arguments.get("run_at")
        return {"timestamp": ts if ts else "now"}
    if stype == "interval":
        return {"seconds": int(arguments.get("interval_seconds", 60))}
    if stype == "after":
        trigger_id = arguments.get("after_schedule_id")
        if trigger_id is None:
            raise ValueError("after_schedule_id is required for 'after' schedule type")
        return {"trigger_id": int(trigger_id)}
    if stype == "cron":
        return {
            "minutes": str(arguments.get("cron_minutes", "*")),
            "hours": str(arguments.get("cron_hours", "*")),
            "days_of_week": str(arguments.get("cron_days_of_week", "*")),
            "days_of_month": str(arguments.get("cron_days_of_month", "*")),
            "months": str(arguments.get("cron_months", "*")),
        }
    raise ValueError(f"Unknown schedule type: {stype!r}")


def _text(data: Any) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(data, default=str))]


# ---------------------------------------------------------------------------
# Server builder
# ---------------------------------------------------------------------------

def build_mcp_server(api_url: str, plugin_metadata: dict | None = None, retry: bool = False) -> Server:
    """
    Build the MCP Server.
    plugin_metadata: pre-loaded flat {plugin_id: meta} dict — skips the HTTP fetch.
                     Pass this from run.py where metadata is already in app.ctx.
    retry=True: retry up to 10 times when fetching from the API (rock-mcp standalone).
    retry=False: fetch once, fail fast.
    """
    if plugin_metadata is None:
        retries = 10 if retry else 1
        plugin_metadata = _fetch_plugin_metadata(api_url, retries=retries, delay=1.0)

    server = Server("webrock")

    all_plugin_ids = list(plugin_metadata.keys())
    tool_to_plugin: dict[str, str] = {}
    plugin_tools: list[Tool] = []

    for plugin_id, plugin_meta in plugin_metadata.items():
        if plugin_meta.get("mcp_disabled"):
            continue

        tool_name = _plugin_to_tool_name(plugin_id, all_plugin_ids)
        tool_to_plugin[tool_name] = plugin_id

        description = plugin_meta.get("doc") or f"Schedule the {plugin_id} plugin"
        plugin_tools.append(Tool(
            name=tool_name,
            description=description,
            inputSchema=_build_input_schema(plugin_meta),
        ))

    management_tools: list[Tool] = [
        Tool(
            name="list_schedules",
            description="List all active schedules, optionally filtered by plugin_id.",
            inputSchema={
                "type": "object",
                "properties": {
                    "plugin_id": {"type": "string", "description": "Filter to this plugin ID (omit for all)."},
                },
            },
        ),
        Tool(
            name="edit_schedule",
            description="Update the type, args, or config of an existing schedule in-place (preserves its ID).",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "Schedule ID to edit."},
                    "type": {"type": "string", "enum": ["once", "interval", "cron", "after"]},
                    "args": {"type": "object", "description": "New plugin argument values."},
                    "config": {"type": "object", "description": "New schedule config dict."},
                },
                "required": ["id", "type"],
            },
        ),
        Tool(
            name="cancel_schedule",
            description="Cancel (soft-delete) an active schedule by its ID.",
            inputSchema={
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "Schedule ID to cancel."},
                },
                "required": ["id"],
            },
        ),
        Tool(
            name="get_run_history",
            description="Retrieve recent run history for a plugin.",
            inputSchema={
                "type": "object",
                "properties": {
                    "plugin_id": {"type": "string", "description": "Plugin ID to fetch history for."},
                    "limit": {"type": "integer", "description": "Maximum rows to return (default 50).", "default": 50},
                },
                "required": ["plugin_id"],
            },
        ),
    ]

    all_tools = plugin_tools + management_tools

    @server.list_tools()
    async def handle_list_tools() -> list[Tool]:
        return all_tools

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict | None) -> list[TextContent]:
        args = arguments or {}

        if name in tool_to_plugin:
            plugin_id = tool_to_plugin[name]
            stype = args.get("schedule_type", "once")
            config_dict = _build_config_from_args(stype, args)
            args_dict = {k: v for k, v in args.items() if k not in _SCHEDULE_FIELDS}
            result = _api_post(api_url, "/api/schedules", {
                "plugin_id": plugin_id,
                "type": stype,
                "args": args_dict,
                "config": config_dict,
            })
            return _text(result)

        if name == "list_schedules":
            path = "/api/schedules"
            filter_pid = args.get("plugin_id")
            if filter_pid:
                path += f"?plugin_id={urllib.parse.quote(filter_pid)}"
            schedules_by_plugin = _api_get(api_url, path)
            flat = [s for jobs in schedules_by_plugin.values() for s in jobs]
            return _text(flat)

        if name == "edit_schedule":
            schedule_id = int(args["id"])
            result = _api_post(api_url, f"/api/schedules/{schedule_id}", {
                "type": args["type"],
                "args": args.get("args", {}),
                "config": args.get("config", {}),
            }, method="PATCH")
            return _text(result)

        if name == "cancel_schedule":
            schedule_id = int(args["id"])
            result = _api_delete(api_url, f"/api/schedules/{schedule_id}")
            return _text(result)

        if name == "get_run_history":
            plugin_id = args["plugin_id"]
            limit = int(args.get("limit", 50))
            result = _api_get(api_url, f"/api/runs/{urllib.parse.quote(plugin_id)}?limit={limit}")
            return _text(result)

        return _text({"error": f"Unknown tool: {name!r}"})

    return server


# ---------------------------------------------------------------------------
# SSE transport
# ---------------------------------------------------------------------------

def run_sse_blocking(server: Server, port: int) -> None:
    """Start an MCP SSE server via Starlette + uvicorn. Blocks until process exits."""
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Mount, Route
    import uvicorn

    sse_transport = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse_transport.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await server.run(
                streams[0], streams[1],
                server.create_initialization_options(),
            )

    starlette_app = Starlette(routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/messages", app=sse_transport.handle_post_message),
    ])

    uvicorn.run(starlette_app, host="0.0.0.0", port=port, log_level="warning")


def start_background_sse(server: Server, port: int) -> None:
    """Start the MCP SSE server in a background daemon thread."""
    t = threading.Thread(target=run_sse_blocking, args=(server, port), daemon=True)
    t.start()
    print(f"MCP SSE server started on http://0.0.0.0:{port}/sse")


# ---------------------------------------------------------------------------
# stdio transport
# ---------------------------------------------------------------------------

def run_stdio_blocking(server: Server) -> None:
    """Run the MCP stdio transport. Blocks until the client disconnects."""
    from mcp.server.stdio import stdio_server

    async def _main():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream, write_stream,
                server.create_initialization_options(),
            )

    asyncio.run(_main())


def start_background_stdio(server: Server) -> None:
    """Start the MCP stdio server in a background daemon thread."""
    t = threading.Thread(target=run_stdio_blocking, args=(server,), daemon=True)
    t.start()
    print("MCP stdio server started", file=sys.stderr)


# ---------------------------------------------------------------------------
# Standalone entry point (rock-mcp)
# ---------------------------------------------------------------------------

def main_stdio_entry() -> None:
    """
    Entry point registered as `rock-mcp` in pyproject.toml.
    Requires `rock` to already be running.  Connects to the webrock REST API
    and exposes its plugins as MCP tools over stdio.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run webrock MCP stdio bridge (requires rock running)")
    parser.add_argument("--api-url", default="http://localhost:8000", dest="api_url",
                        help="URL of the running webrock API server (default: http://localhost:8000)")
    args = parser.parse_args()

    server = build_mcp_server(api_url=args.api_url, retry=True)
    run_stdio_blocking(server)
