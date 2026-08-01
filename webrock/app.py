import asyncio
import os
from pathlib import Path
import sanic_jinja2

from .engine import Engine
from .load_project import load_project, load_builtins
from . import db
from sanic import Sanic, response
from sanic_jinja2 import SanicJinja2

_PACKAGE_DIR = Path(__file__).parent


async def create_app(project_dir: str | None = None, paused: bool = False):
    app = Sanic("YourApp")
    app.static("/static", str(_PACKAGE_DIR / "static"))

    print("CREATING APP")

    jinja = SanicJinja2(app, loader=sanic_jinja2.FileSystemLoader(str(_PACKAGE_DIR / "templates")))

    folder = os.path.abspath(project_dir or os.getcwd())
    metadata, plugins, shutdown_funcs = await load_project(folder)
    load_builtins(metadata, plugins)

    app.ctx.plugins = plugins
    app.ctx.metadata = metadata

    db_path = os.path.join(folder, "webrock.db")
    db.init_db(db_path)

    engine = Engine(plugins)
    if paused:
        engine.pause_system()
    app.ctx.engine = engine

    @app.listener("after_server_start")
    async def start_engine(app, loop):
        print("starting schedules")
        engine.start()

    @app.listener("before_server_stop")
    async def stop_engine(app):
        engine.stop()
        for func in shutdown_funcs:
            try:
                if asyncio.iscoroutinefunction(func):
                    await func()
                else:
                    func()
            except Exception as e:
                print(f"Error running shutdown procedure {func.__name__}: {e}")

    # --- Web UI ---
    @app.route("/")
    async def index(request):
        all_schedules = db.get_active_schedules()
        return jinja.render(
            "control_panel.html", request,
            metadata=metadata,
            plugins=plugins.keys(),
            all_schedules=all_schedules,
        )

    # --- API: System pause / resume ---

    @app.route("/api/system/status", methods=["GET"])
    async def api_system_status(request):
        return response.json({
            "paused": engine.system_paused,
            "pending_count": engine.pending_count,
        })

    @app.route("/api/system/pause", methods=["POST"])
    async def api_system_pause(request):
        engine.pause_system()
        return response.json({"paused": True})

    @app.route("/api/system/resume", methods=["POST"])
    async def api_system_resume(request):
        started = engine.resume_system()
        return response.json({"paused": False, "started": started})

    # --- API: Plugins ---

    @app.route("/api/plugins")
    async def api_plugins(request):
        result = {}
        for plugin_id in plugins:
            node = metadata
            try:
                for part in plugin_id.split("."):
                    node = node[part]
                result[plugin_id] = node
            except (KeyError, TypeError):
                pass
        return response.json(result)

    @app.route("/api/plugins/<plugin_id>/status")
    async def api_plugin_status(request, plugin_id):
        plugin_id = plugin_id.replace("__", ".")
        if plugin_id not in plugins:
            return response.json({"error": f"{plugin_id} not found"}, status=404)
        task_status = engine.get_task_status(plugin_id)
        plugin = plugins[plugin_id]
        result = None
        if plugin["task"] and plugin["task"].done():
            try:
                result = plugin["task"].result()
            except Exception:
                result = None
        return response.json({"task_status": task_status, "result": result})

    @app.route("/api/plugins/<plugin_id>/pause", methods=["POST"])
    async def api_plugin_pause(request, plugin_id):
        plugin_id = plugin_id.replace("__", ".")
        if plugin_id not in plugins:
            return response.json({"error": f"{plugin_id} not found"}, status=404)
        ok = engine.pause_task(plugin_id)
        if not ok:
            return response.json({"error": f"{plugin_id} is not running"}, status=409)
        return response.json({"task_status": "paused", "plugin_id": plugin_id})

    @app.route("/api/plugins/<plugin_id>/resume", methods=["POST"])
    async def api_plugin_resume(request, plugin_id):
        plugin_id = plugin_id.replace("__", ".")
        if plugin_id not in plugins:
            return response.json({"error": f"{plugin_id} not found"}, status=404)
        engine.resume_task(plugin_id)
        task_status = engine.get_task_status(plugin_id)
        return response.json({"task_status": task_status, "plugin_id": plugin_id})

    @app.route("/api/plugins/<plugin_id>/stop", methods=["POST"])
    async def api_plugin_stop(request, plugin_id):
        plugin_id = plugin_id.replace("__", ".")
        if plugin_id not in plugins:
            return response.json({"error": f"{plugin_id} not found"}, status=404)
        stopped, dependents = engine.stop_task(plugin_id)
        if not stopped:
            return response.json({"status": f"{plugin_id} is not running"})
        dep_info = [
            {"id": r["id"], "plugin_id": r["plugin_id"], "args": r["args"]}
            for r in dependents
        ]
        return response.json({
            "status": "stopped",
            "plugin_id": plugin_id,
            "dependent_schedules": dep_info,
        })

    # --- API: Schedules ---

    @app.route("/api/schedules", methods=["GET"])
    async def api_get_schedules(request):
        rows = db.get_active_schedules()
        filter_pid = request.args.get("plugin_id")
        if filter_pid:
            rows = [r for r in rows if r["plugin_id"] == filter_pid]
        result = {}
        for row in rows:
            pid = row["plugin_id"]
            if pid not in result:
                result[pid] = []
            result[pid].append({
                "id": row["id"],
                "type": row["type"],
                "args": row["args"],
                "config": row["config"],
                "next_run": db.format_ts(row["next_run"]),
                "last_run": db.format_ts(row["last_run"]),
                "source": row["source"],
                "disabled": bool(row["disabled"]),
                "task_status": engine.get_task_status(pid),
            })
        return response.json(result)

    def _coerce_args(plugin_id: str, args: dict) -> dict:
        """Coerce string values from web forms to their declared plugin arg types."""
        node = metadata
        try:
            for part in plugin_id.split("."):
                node = node[part]
        except (KeyError, TypeError):
            return args
        coerced = dict(args)
        for arg_meta in node.get("args", []):
            name = arg_meta["name"]
            if name not in coerced or not isinstance(coerced[name], str):
                continue
            atype = arg_meta.get("type")
            val = coerced[name]
            try:
                if atype == "int":
                    coerced[name] = int(val)
                elif atype == "float":
                    coerced[name] = float(val)
                elif atype == "bool":
                    coerced[name] = val.lower() not in ("false", "0", "")
            except (ValueError, TypeError):
                pass
        return coerced

    @app.route("/api/schedules", methods=["POST"])
    async def api_create_schedule(request):
        body = request.json or {}
        plugin_id = body.get("plugin_id")
        stype = body.get("type", "once")
        args_dict = _coerce_args(plugin_id, body.get("args", {})) if plugin_id else {}
        config_dict = body.get("config", {})
        if not plugin_id:
            return response.json({"error": "missing plugin_id"}, status=400)
        if plugin_id not in plugins:
            return response.json({"error": f"unknown plugin: {plugin_id}"}, status=404)
        new_id = db.insert_schedule(plugin_id, stype, args_dict, config_dict, source="web")
        return response.json({"status": "added", "plugin": plugin_id, "id": new_id})

    @app.route("/api/schedules/<schedule_id:int>", methods=["PATCH"])
    async def api_edit_schedule(request, schedule_id):
        body = request.json or {}
        stype = body.get("type")
        args_dict = body.get("args", {})
        config_dict = body.get("config", {})
        if not stype:
            return response.json({"error": "missing type"}, status=400)
        db.update_schedule(schedule_id, stype, args_dict, config_dict)
        return response.json({"status": "updated", "id": schedule_id})

    @app.route("/api/schedules/<schedule_id:int>", methods=["DELETE"])
    async def api_delete_schedule(request, schedule_id):
        db.soft_delete_schedule(schedule_id)
        return response.json({"status": "cancelled", "id": schedule_id})

    @app.route("/api/schedules/<schedule_id:int>/disabled", methods=["POST"])
    async def api_set_schedule_disabled(request, schedule_id):
        disabled = request.json.get("disabled", True)
        db.set_schedule_disabled(schedule_id, disabled)
        return response.json({"id": schedule_id, "disabled": disabled})

    @app.route("/api/schedules/bulk-set-disabled", methods=["POST"])
    async def api_bulk_set_disabled(request):
        ids = request.json.get("ids", [])
        disabled = request.json.get("disabled", True)
        if not ids:
            return response.json({"error": "ids required"}, status=400)
        db.bulk_set_disabled(ids, disabled)
        return response.json({"count": len(ids), "disabled": disabled})

    @app.route("/api/schedules/<schedule_id:int>/reset", methods=["POST"])
    async def api_reset_schedule(request, schedule_id):
        db.reset_schedule_next_run(schedule_id)
        return response.json({"status": "reset", "id": schedule_id})

    @app.route("/api/schedules/<schedule_id:int>/next-run", methods=["POST"])
    async def api_set_schedule_next_run(request, schedule_id):
        from datetime import datetime as _dt
        body = request.json or {}
        next_run_val = body.get("next_run")
        if next_run_val is None:
            return response.json({"error": "next_run required"}, status=400)
        if isinstance(next_run_val, str):
            try:
                next_run_ts = _dt.fromisoformat(next_run_val).timestamp()
            except ValueError:
                return response.json({"error": "invalid next_run format"}, status=400)
        else:
            next_run_ts = float(next_run_val)
        db.update_schedule_next_run(schedule_id, next_run_ts)
        return response.json({"status": "updated", "id": schedule_id})

    @app.route("/api/schedules/<schedule_id:int>/run-now", methods=["POST"])
    async def api_run_schedule_now(request, schedule_id):
        """Trigger a schedule immediately regardless of next_run."""
        rows = [r for r in db.get_active_schedules() if r["id"] == schedule_id]
        if not rows:
            return response.json({"error": "schedule not found"}, status=404)
        row = rows[0]
        plugin_id = row["plugin_id"]
        if plugin_id not in plugins:
            return response.json({"error": f"plugin {plugin_id} not loaded"}, status=404)
        plugin = plugins[plugin_id]
        run_id = db.insert_run(schedule_id, plugin_id, row["args"])
        await engine._run_job(plugin, plugin_id, row["args"], run_id, schedule_id)
        return response.json({"status": "started", "run_id": run_id, "plugin_id": plugin_id})

    @app.route("/api/runs/<plugin_id>")
    async def api_get_runs(request, plugin_id):
        limit = int(request.args.get("limit", 50))
        runs = db.get_runs_for_plugin(plugin_id, limit=limit)
        return response.json(runs)

    return app
