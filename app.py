import asyncio

import sys

import sanic_jinja2
from webrock.load_plugins import load_plugins
from sanic import Sanic, response
from sanic_jinja2 import SanicJinja2
from importlib.resources import files


async def create_app():
    app = Sanic("Stonks")
    app.static("/static", "./static")

    # Leaving this here for now because we need to figure out how to handle situations like this
    # dao_manager = DAOManager()

    # Leaving this here for now because we need to figure out how to handle situations like this
    # await dao_manager.initialize()
    # is this useful???? app.ctx.dao_manager = dao_manager

    print("CREATING APP")

    # Leaving this here for now because we need to figure out how to handle situations like this
    # @app.listener("before_server_stop")
    # async def close_db(app, loop):
    #     print("Closing database connection")
    #     await dao_manager.db.close()

    templates_path = files('webrock').joinpath('templates')
    jinja = SanicJinja2(app, loader=sanic_jinja2.FileSystemLoader(str(templates_path)))

    # import main from all py files in data_sources
    metadata, plugins = load_plugins()

    @app.route("/")
    async def index(request):
        nonlocal plugins
        return jinja.render("control_panel.html", request, metadata=metadata)

    @app.route("/start/<plugin_name>", methods=["POST"])
    async def start(request, plugin_name):
        print(f"Received request to start {plugin_name}")
        nonlocal plugins
        if plugin_name not in plugins:
            return response.json({"error": f"{plugin_name} not found"})
        print("Found plugin")
        plugin = plugins[plugin_name]
        if plugin["task"]:
            return response.json({"status": f"{plugin_name} is already running"})
        form_data = prepare_form_data(plugin, request.form)
        plugin["task"] = asyncio.create_task(plugin["function"](**form_data))
        plugin["task"].add_done_callback(complete_callback(plugin))
        print(f"Started {plugin_name}")
        return response.json({"status": f"{plugin_name} started"})

    @app.route("/stop/<plugin_name>")
    async def stop(request, plugin_name):
        nonlocal plugins
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
        nonlocal plugins
        if plugin_name not in plugins:
            return response.json({"error": f"{plugin_name} not found"})
        plugin = plugins[plugin_name]
        if not plugin["task"]:
            return response.json({"running": False})
        if plugin["task"].done():
            return response.json({"running": False, "result": plugin["task"].result()})
        else:
            return response.json({"running": True})

    return app


def complete_callback(plugin):
    def callback(task):
        try:
            result = task.result()
            print(f"Finished {plugin['function'].__name__}")
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error in {plugin['function'].__name__}: {str(e)}")
        plugin["task"] = None

    return callback


def prepare_form_data(meta, form):
    # Since a web request can have duplicate keys, sanic puts everything in lists
    # Might have to change this later

    sys.stdout.write(meta)

    return {k: v[0] for k, v in form.items()}

