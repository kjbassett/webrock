import asyncio
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
import sanic_jinja2

from .schedule_utils import (
    form_to_schedule_parts,
    calculate_next_run_from_row,
)
from .load_project import load_project
from . import db
from sanic import Sanic, response
from sanic_jinja2 import SanicJinja2
from importlib.resources import files


async def create_app():
    app = Sanic("YourApp")
    webrock_path = files("webrock")
    app.static("/static", str(webrock_path / "static"))
    executor = ThreadPoolExecutor()

    print("CREATING APP")

    templates_path = webrock_path / "templates"
    jinja = SanicJinja2(app, loader=sanic_jinja2.FileSystemLoader(str(templates_path)))

    metadata, plugins, shutdown_funcs = await load_project()

    db.init_db("webrock.db")

    async def run_schedules():
        while True:
            triggered_this_tick = set()  # set of schedule IDs that fired this tick

            active = db.get_active_schedules()

            # Pass 1: time-based schedules
            for row in active:
                if row["plugin_id"] not in plugins:
                    continue
                if row["type"] == "after":
                    continue

                if row["next_run"] is None:
                    next_run = calculate_next_run_from_row(row)
                    if next_run is None:
                        continue
                    db.update_schedule_next_run(row["id"], next_run)
                    row["next_run"] = next_run

                now = time.time()
                if now < row["next_run"]:
                    continue

                args = row["args"]
                plugin = plugins[row["plugin_id"]]
                run_id = db.insert_run(row["id"], row["plugin_id"], args)
                await run_job(plugin, args, run_id)
                triggered_this_tick.add(row["id"])
                print(f"Scheduled start of {row['plugin_id']}")

                if row["type"] == "once":
                    db.soft_delete_schedule(row["id"])
                else:
                    new_next = calculate_next_run_from_row({**row, "last_run": now})
                    if new_next is not None:
                        db.update_schedule_next_run(row["id"], new_next, last_run=now)

            # Pass 2: "after" schedules
            if triggered_this_tick:
                for row in active:
                    if row["plugin_id"] not in plugins:
                        continue
                    if row["type"] != "after":
                        continue
                    trigger_id = row["config"].get("trigger_id")
                    if trigger_id in triggered_this_tick:
                        args = row["args"]
                        plugin = plugins[row["plugin_id"]]
                        run_id = db.insert_run(row["id"], row["plugin_id"], args)
                        await run_job(plugin, args, run_id)
                        print(f"After-triggered start of {row['plugin_id']}")

            await asyncio.sleep(1)

    async def run_job(plugin, args, run_id: int):
        if asyncio.iscoroutinefunction(plugin["function"]):
            plugin["task"] = asyncio.create_task(plugin["function"](**args))
        else:
            loop = asyncio.get_event_loop()
            plugin["task"] = loop.run_in_executor(
                executor,
                run_sync_function,
                plugin["function"],
                args,
            )
        plugin["task"].add_done_callback(complete_callback(plugin, run_id))

    @app.listener("before_server_stop")
    async def close_tasks(app):
        for func in shutdown_funcs:
            try:
                if asyncio.iscoroutinefunction(func):
                    await func()
                else:
                    func()
            except Exception as e:
                print(f"Error running shutdown procedure {func.__name__}: {e}")

    @app.listener("after_server_start")
    async def start_schedules(app, loop):
        print("starting schedules")
        asyncio.create_task(run_schedules())

    # --- Plugin Routes ---
    @app.route("/")
    async def index(request):
        all_schedules = db.get_active_schedules()
        return jinja.render(
            "control_panel.html", request,
            metadata=metadata,
            plugins=plugins.keys(),
            all_schedules=all_schedules,
        )

    @app.route("/schedule_job/<plugin_id>", methods=["POST"])
    async def schedule_job(request, plugin_id):
        meta = metadata
        for layer in plugin_id.split("."):
            meta = meta[layer]

        stype, args_dict, config_dict = form_to_schedule_parts(meta, request.form)
        new_id = db.insert_schedule(plugin_id, stype, args_dict, config_dict)
        return response.json({"status": "added", "plugin": plugin_id, "id": new_id})

    @app.route("/stop/<plugin_id>")
    async def stop(request, plugin_id):
        if plugin_id not in plugins:
            return response.json({"error": f"{plugin_id} not found"})
        plugin = plugins[plugin_id]
        if not plugin["task"]:
            return response.json({"status": f"{plugin_id} has not started"})
        if plugin["task"].done():
            return response.json({"status": f"{plugin_id} is already finished"})
        plugin["task"].cancel()
        return response.json({"status": f"{plugin_id} stopped"})

    @app.route("/status/<plugin_id>")
    async def status(request, plugin_id):
        if plugin_id not in plugins:
            return response.json({"error": f"{plugin_id} not found"})
        plugin = plugins[plugin_id]
        if not plugin["task"] or plugin["task"].done():
            result = None
            if plugin["task"]:
                try:
                    result = plugin["task"].result()
                except Exception:
                    result = None
            return response.json({"running": False, "result": result})
        else:
            return response.json({"running": True})

    @app.route("/get_schedules")
    async def get_schedules(request):
        rows = db.get_active_schedules()
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
            })
        return response.json(result)

    @app.route("/remove_schedule/<plugin_id>", methods=["POST"])
    async def remove_schedule(request, plugin_id):
        schedule_id = request.json.get("id")
        if schedule_id is None:
            return response.json({"error": "missing id"}, status=400)
        db.soft_delete_schedule(int(schedule_id))
        return response.json({"status": "removed", "id": schedule_id})

    @app.route("/get_runs/<plugin_id>")
    async def get_runs(request, plugin_id):
        runs = db.get_runs_for_plugin(plugin_id)
        return response.json(runs)

    return app


# --- Helper Functions ---
def complete_callback(plugin, run_id: int):
    def callback(task):
        try:
            result = task.result()
            print(f"Finished {plugin['function'].__name__}")
            print(f"Result: {result}")
            db.complete_run(run_id, "success", result=result)
        except Exception as e:
            print(f"Error in {plugin['function'].__name__}: {str(e)}")
            traceback.print_exc()
            db.complete_run(run_id, "error", error=str(e))
        plugin["task"] = None

    return callback


def run_sync_function(func, kwargs):
    return func(**kwargs)
