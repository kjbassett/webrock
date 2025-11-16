import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor

import sanic_jinja2
from webrock.scheduler import run_schedules, load_schedules, save_schedules

from .load_project import load_project
from sanic import Sanic, response
from sanic_jinja2 import SanicJinja2
from importlib.resources import files

async def create_app():
    app = Sanic("Your app")
    webrock_path = files('webrock')
    app.static("/static", str(webrock_path / "static"))
    executor = ThreadPoolExecutor()

    print("CREATING APP")

    templates_path = webrock_path / 'templates'
    jinja = SanicJinja2(app, loader=sanic_jinja2.FileSystemLoader(str(templates_path)))

    metadata, plugins, shutdown_funcs = await load_project()

    schedules = load_schedules()

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
    async def start_scheduler(app, loop):
        asyncio.create_task(run_schedules(app, schedules))

    # --- Plugin Routes ---
    @app.route("/")
    async def index(request):
        return jinja.render("control_panel.html", request, metadata=metadata, plugins=plugins.keys())

    @app.route("/start/<plugin_name>", methods=["POST"])
    async def start(request, plugin_name):
        if plugin_name not in plugins:
            return response.json({"error": f"{plugin_name} not found"})
        plugin = plugins[plugin_name]

        # get the right metadata for the plugin
        meta = metadata
        for meta_layer in plugin_name.split("."):
            meta = meta[meta_layer]

        if plugin["task"]:
            return response.json({"status": f"{plugin_name} is already running"})
        form_data = prepare_form_data(meta, request.form)
        if asyncio.iscoroutinefunction(plugin["function"]):
            plugin["task"] = asyncio.create_task(plugin["function"](**form_data))
        else:
            loop = asyncio.get_event_loop()
            plugin["task"] = loop.run_in_executor(executor, run_sync_function, plugin["function"], form_data)
        plugin["task"].add_done_callback(complete_callback(plugin))
        return response.json({"status": f"{plugin_name} started"})

    @app.route("/stop/<plugin_name>")
    async def stop(request, plugin_name):
        if plugin_name not in plugins:
            return response.json({"error": f"{plugin_name} not found"})
        plugin = plugins[plugin_name]
        if not plugin["task"]:
            return response.json({"status": f"{plugin_name} has not started"})
        if plugin["task"].done():
            return response.json({"status": f"{plugin_name} is already finished"})
        plugin["task"].cancel()
        return response.json({"status": f"{plugin_name} stopped"})

    @app.route("/status/<plugin_name>")
    async def status(request, plugin_name):
        if plugin_name not in plugins:
            return response.json({"error": f"{plugin_name} not found"})
        plugin = plugins[plugin_name]
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

    # --- Scheduler Endpoints ---
    @app.route("/update_schedule/<plugin_name>", methods=["POST"])
    async def update_schedule(request, plugin_name):
        job = request.json
        if plugin_name not in schedules:
            schedules[plugin_name] = []
        schedules[plugin_name].append(job)
        save_schedules(schedules)
        return response.json({"status": "added", "plugin": plugin_name})

    @app.route("/remove_schedule/<plugin_name>", methods=["POST"])
    async def remove_schedule(request, plugin_name):
        job_index = request.json.get("index")
        if plugin_name in schedules and 0 <= job_index < len(schedules[plugin_name]):
            schedules[plugin_name].pop(job_index)
            save_schedules(schedules)
            return response.json({"status": "removed", "plugin": plugin_name})
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


def prepare_form_data(meta, form):
    form_data = {}
    for arg in meta["args"]:
        if arg["name"] not in form:
            if "default" not in arg:
                raise ValueError(f"Missing required argument {arg['name']}")
            form_data[arg["name"]] = arg["default"]
            continue
        _type = arg["type"]
        value = form[arg["name"]][0]
        if _type == "bool":
            value = value.lower() == "true"
        elif _type != "any":
            try:
                _type = __builtins__[arg["type"]]
                value = _type(value)
            except KeyError:
                raise ValueError(f"Invalid type from  {arg['type']}. Type not in __builtins__.")
        form_data[arg["name"]] = value
    return form_data


def run_sync_function(func, kwargs):
    return func(**kwargs)
