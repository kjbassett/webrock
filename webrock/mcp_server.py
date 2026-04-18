"""
MCP server for webrock.

Exposes one scheduling tool per plugin (respecting mcp_disabled=True) plus
four management tools: list_schedules, edit_schedule, cancel_schedule,
get_run_history.

Transport modes
---------------
SSE  (default when running `rock`):
    start_background_sse(plugins, metadata, port)  — daemon thread alongside Sanic

stdio (for Claude Desktop / `rock --mcp-transport stdio` / `rock-mcp`):
    run_standalone_stdio()  — loads project + DB independently, then blocks
"""

import asyncio
import json
import threading
from typing import Any

from mcp.server import Server
from mcp.types import Tool, TextContent

from . import db

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Argument names that carry schedule configuration, not plugin-specific values.
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
# Helpers
# ---------------------------------------------------------------------------

def _plugin_to_tool_name(plugin_id: str, all_plugin_ids: list[str]) -> str:
    """
    Map plugin_id (e.g. 'lights.turn_on') to a safe MCP tool name.
    Uses just the function name when unique, otherwise replaces dots with '__'.
    """
    func_name = plugin_id.rsplit(".", 1)[-1]
    if sum(1 for pid in all_plugin_ids if pid.rsplit(".", 1)[-1] == func_name) == 1:
        return f"schedule_{func_name}"
    return f"schedule_{plugin_id.replace('.', '__')}"


def _get_plugin_meta(plugin_id: str, metadata: dict) -> dict:
    """Walk the nested metadata dict using plugin_id as a dot-separated path."""
    node = metadata
    for part in plugin_id.split("."):
        node = node[part]
    return node


def _build_input_schema(plugin_meta: dict) -> dict:
    """Build a JSON Schema object for a per-plugin scheduling tool."""
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

    # Add schedule configuration fields
    properties.update(_SCHEDULE_PROPERTIES)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _build_config_from_args(stype: str, arguments: dict) -> dict:
    """Extract schedule-type config dict from MCP tool arguments."""
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
    """Wrap a value as a single MCP TextContent result."""
    return [TextContent(type="text", text=json.dumps(data, default=str))]


# ---------------------------------------------------------------------------
# Server builder
# ---------------------------------------------------------------------------

def build_mcp_server(plugins: dict, metadata: dict) -> Server:
    """
    Construct the MCP Server with all tools registered.
    `plugins`  — the dict loaded by load_project: {plugin_id: {"function": ..., "task": ...}}
    `metadata` — the nested metadata dict from load_project
    """
    server = Server("webrock")

    # Build tool-name → plugin_id mapping at construction time
    all_plugin_ids = list(plugins.keys())
    tool_to_plugin: dict[str, str] = {}
    plugin_tools: list[Tool] = []

    for plugin_id, plugin_info in plugins.items():
        try:
            plugin_meta = _get_plugin_meta(plugin_id, metadata)
        except (KeyError, TypeError):
            continue

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

        # --- Per-plugin scheduling tools ---
        if name in tool_to_plugin:
            plugin_id = tool_to_plugin[name]
            stype = args.get("schedule_type", "once")
            config_dict = _build_config_from_args(stype, args)
            args_dict = {k: v for k, v in args.items() if k not in _SCHEDULE_FIELDS}
            new_id = db.insert_schedule(plugin_id, stype, args_dict, config_dict, source="mcp")
            return _text({"status": "scheduled", "id": new_id, "plugin": plugin_id, "type": stype})

        # --- Management tools ---
        if name == "list_schedules":
            rows = db.get_active_schedules()
            filter_pid = args.get("plugin_id")
            if filter_pid:
                rows = [r for r in rows if r["plugin_id"] == filter_pid]
            result = []
            for row in rows:
                result.append({
                    "id": row["id"],
                    "plugin_id": row["plugin_id"],
                    "type": row["type"],
                    "args": row["args"],
                    "config": row["config"],
                    "next_run": db.format_ts(row["next_run"]),
                    "last_run": db.format_ts(row["last_run"]),
                    "source": row["source"],
                })
            return _text(result)

        if name == "edit_schedule":
            schedule_id = int(args["id"])
            stype = args["type"]
            args_dict = args.get("args", {})
            config_dict = args.get("config", {})
            db.update_schedule(schedule_id, stype, args_dict, config_dict)
            return _text({"status": "updated", "id": schedule_id})

        if name == "cancel_schedule":
            schedule_id = int(args["id"])
            db.soft_delete_schedule(schedule_id)
            return _text({"status": "cancelled", "id": schedule_id})

        if name == "get_run_history":
            plugin_id = args["plugin_id"]
            limit = int(args.get("limit", 50))
            runs = db.get_runs_for_plugin(plugin_id, limit=limit)
            return _text(runs)

        return _text({"error": f"Unknown tool: {name!r}"})

    return server


# ---------------------------------------------------------------------------
# SSE transport (runs alongside Sanic in a daemon thread)
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


def start_background_sse(plugins: dict, metadata: dict, port: int) -> None:
    """Build the MCP server and start its SSE endpoint in a background daemon thread."""
    server = build_mcp_server(plugins, metadata)
    t = threading.Thread(target=run_sse_blocking, args=(server, port), daemon=True)
    t.start()
    print(f"MCP SSE server started on http://0.0.0.0:{port}/sse")


# ---------------------------------------------------------------------------
# stdio transport (standalone process for Claude Desktop)
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


def run_standalone_stdio(project_dir: str | None = None) -> None:
    """
    Load the project and DB, build the MCP server, then run stdio transport.
    Used by `rock --mcp-transport stdio` and the `rock-mcp` entry point.

    project_dir: absolute path to the directory containing the plugins and
                 where webrock.db will be created/opened.  Defaults to cwd.
    """
    import os
    from .load_project import load_project

    folder = project_dir or os.getcwd()
    db_path = os.path.join(folder, "webrock.db")
    metadata, plugins, _ = asyncio.run(load_project(folder))
    db.init_db(db_path)
    server = build_mcp_server(plugins, metadata)
    run_stdio_blocking(server)


def main_stdio_entry() -> None:
    """Entry point registered as `rock-mcp` in pyproject.toml."""
    run_standalone_stdio()
