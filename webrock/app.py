import asyncio
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
import sanic_jinja2

from .schedule_utils import form_to_job_schedule, calculate_next_run, load_schedules, save_schedules
from .load_project import load_project
from sanic import Sanic, response
from sanic_jinja2 import SanicJinja2
from importlib.resources import files
from pathlib import Path


async def create_app():
    app = Sanic("YourApp")
    webrock_path = files("webrock")
    app.static("/static", str(webrock_path / "static"))
    executor = ThreadPoolExecutor()

    print("CREATING APP")

    templates_path = webrock_path / "templates"
    jinja = SanicJinja2(app, loader=sanic_jinja2.FileSystemLoader(str(templates_path)))

    metadata, plugins, shutdown_funcs = await load_project()

    schedule_file = Path("schedules.json")
    schedules = load_schedules(schedule_file)

    async def run_schedules():
        while True:
            for plugin_id, job_schedules in schedules.items():
                if plugin_id not in plugins:
                    continue
                plugin = plugins[plugin_id]
                i = 0
                while i < len(job_schedules):
                    js = job_schedules[i]
                    now = time.time()
                    next_run = js["schedule"].get("next_run", 0)
                    if now < next_run:
                        i += 1
                        continue
                    args = js.get("args", {})
                    await run_job(plugin, args)
                    print(f"Scheduled start of {plugin_id}")
                    if js["schedule"]["type"] == "once":
                        del job_schedules[i]
                    else:
                        js["schedule"]["last_run"] = now
                        js["schedule"]["next_run"] = calculate_next_run(js["schedule"])
                        i += 1
                    save_schedules(schedule_file, schedules)
            await asyncio.sleep(1)

    async def run_job(plugin, args):
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
        plugin["task"].add_done_callback(complete_callback(plugin))

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
        asyncio.create_task(run_schedules())

    # --- Plugin Routes ---
    @app.route("/")
    async def index(request):
        return jinja.render(
            "control_panel.html", request, metadata=metadata, plugins=plugins.keys()
        )

    @app.route("/schedule_job/<plugin_id>", methods=["POST"])
    async def schedule_job(request, plugin_id):
        if plugin_id not in schedules:
            schedules[plugin_id] = []

        # get the right metadata for the plugin
        meta = metadata
        for meta_layer in plugin_id.split("."):
            meta = meta[meta_layer]

        print(request.form)
        job_schedule = form_to_job_schedule(meta, request.form)
        print(job_schedule)
        schedules[plugin_id].append(job_schedule)
        save_schedules(schedule_file, schedules)
        return response.json({"status": "added", "plugin": plugin_id})

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
        return response.json(schedules)

    @app.route("/remove_schedule/<plugin_id>", methods=["POST"])
    async def remove_schedule(request, plugin_id):
        job_index = request.json.get("index")
        if plugin_id in schedules and 0 <= job_index < len(schedules[plugin_id]):
            schedules[plugin_id].pop(job_index)
            save_schedules(schedule_file, schedules)
            return response.json({"status": "removed", "plugin": plugin_id})
        return response.json({"error": "invalid index or plugin"})

    return app


# --- Helper Functions ---
def complete_callback(plugin):
    def callback(task):
        try:
            result = task.result()
            print(f"Finished {plugin['function'].__name__}")
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error in {plugin['function'].__name__}: {str(e)}")
            traceback.print_exc()
        plugin["task"] = None

    return callback


def run_sync_function(func, kwargs):
    return func(**kwargs)
