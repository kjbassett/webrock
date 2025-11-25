import asyncio
import calendar
from datetime import datetime, timedelta
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
import sanic_jinja2

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
                        i + 1
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


def form_to_job_schedule(meta, form):
    job_schedule = {"schedule": {}, "args": {}}

    def get_value(key, default=None):
        if key not in form or not form[key]:
            if not default:
                raise ValueError(f"Missing required item: {key}")
            return default
        v = form[key][0]  # form puts everything in lists
        if "," in v:
            v = v.split(",")
        return v

    # get schedule
    schedule_type = get_value("_schedule_type")
    if schedule_type not in ("once", "interval", "cron"):
        raise ValueError("Invalid schedule type")

    schedule = {"type": schedule_type}

    if schedule_type == "once":
        when = get_value("_timestamp", "now")
        if when == "now":
            schedule["timestamp"] = datetime.utcnow()
        else:
            # Let user pass an ISO timestamp or date/time input
            try:
                schedule["timestamp"] = datetime.fromisoformat(when)
            except ValueError:
                raise ValueError(f"Invalid timestamp: {when}")

    elif schedule_type == "interval":
        seconds_raw = get_value("_seconds")
        if seconds_raw is None:
            raise ValueError("interval in seconds is required for interval schedules")

        try:
            schedule["seconds"] = int(seconds_raw)
        except ValueError:
            raise ValueError("interval must be an integer")

    elif schedule_type == "cron":
        schedule["minutes"] = get_value("_minutes", "*")
        schedule["hours"] = get_value("_hours", "*")
        schedule["days_of_week"] = get_value("_days_of_week", "*")
        schedule["days_of_month"] = get_value("_days_of_month", "*")
        schedule["months"] = get_value("_months", "*")

    job_schedule["schedule"] = schedule

    # get function args
    for arg in meta["args"]:
        name = arg["name"]

        # Missing value
        if name not in form:
            if "default" not in arg:
                raise ValueError(f"Missing required argument {name}")
            form[name] = arg["default"]
            continue

        value = get_value(name, arg.get("default"))

        # Convert type
        _type = arg["type"]

        if _type == "bool":
            # Works for both scheduler & Sanic
            if isinstance(value, str):
                value = value.lower() == "true"
            else:
                value = bool(value)

        elif _type != "any":
            try:
                builtin_type = __builtins__[_type]
                value = builtin_type(value)
            except KeyError:
                raise ValueError(
                    f"Invalid type for {name}: '{_type}' not found in __builtins__."
                )

        job_schedule["args"][name] = value

    return job_schedule


def run_sync_function(func, kwargs):
    return func(**kwargs)


# Helpers

MONTH_LOOKUP = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
MONTH_LOOKUP.update(
    {name.lower(): i for i, name in enumerate(calendar.month_abbr) if name}
)


def parse_cron_field(raw):
    """
    Convert a string or list into a sorted set of allowed int values.
    Accepts '*', '1,2,5-7', etc.
    Returns None for wildcard.
    """
    if isinstance(raw, list):
        raw = ",".join(raw)

    raw = raw.strip()

    if raw == "*" or raw == "":
        return None  # wildcard meaning "all"

    result = set()
    parts = raw.split(",")

    for part in parts:
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            result.update(range(int(start), int(end) + 1))
        else:
            result.add(int(part))

    return sorted(result)


def parse_month_field(raw):
    """
    Same as parse_cron_field but supports month names.
    Returns None for wildcard.
    """
    if isinstance(raw, list):
        raw = ",".join(raw)

    raw = raw.strip()
    if raw == "*" or raw == "":
        return None

    result = set()
    parts = raw.split(",")

    for p in parts:
        p = p.strip().lower()
        if "-" in p:
            # Range, but may be names
            start, end = p.split("-", 1)
            start = MONTH_LOOKUP.get(start, int(start))
            end = MONTH_LOOKUP.get(end, int(end))
            result.update(range(start, end + 1))
            continue

        # Single item: name or int
        if p in MONTH_LOOKUP:
            result.add(MONTH_LOOKUP[p])
        else:
            result.add(int(p))

    return sorted(result)


def matches(value, allowed):
    """Return True if `value` is in allowed set or allowed is None (wildcard)."""
    return allowed is None or value in allowed


def calculate_next_run(schedule):
    """
    Given a schedule dict from your form_to_job_schedule output,
    return the next UTC datetime the job should run.
    """

    stype = schedule["type"]

    # --- ONCE ---
    if stype == "once":
        return schedule["timestamp"]

    last_run = schedule["last_run"]

    # --- INTERVAL ---
    if stype == "interval":
        seconds = schedule["seconds"]
        return last_run + seconds

    # --- CRON ---
    # Parse all cron fields
    minutes = parse_cron_field(schedule["minutes"])
    hours = parse_cron_field(schedule["hours"])
    dom = parse_cron_field(schedule["days_of_month"])
    dow = parse_cron_field(schedule["days_of_week"])
    months = parse_month_field(schedule["months"])

    # Start checking from now+1min
    t = datetime.fromtimestamp(last_run) + timedelta(minutes=1)
    t = t.replace(second=0, microsecond=0)

    # Hard stop: safety valve (5 years is generous)
    end = t + timedelta(days=365 * 5)

    while t < end:
        if (
            matches(t.minute, minutes)
            and matches(t.hour, hours)
            and matches(t.month, months)
            and (
                matches(t.day, dom)
                or matches(t.weekday(), dow)  # Python weekday: Mon=0..Sun=6
            )
        ):
            return int(t.timestamp())

        t += timedelta(minutes=1)

    raise RuntimeError(
        f"No next runtime found within 5 years. Cron may be impossible.\n{schedule}"
    )


def load_schedules(schedule_file):
    if schedule_file.exists():
        return json.loads(schedule_file.read_text())
    return {}


def save_schedules(schedule_file, schedules):
    schedule_file.write_text(json.dumps(schedules, indent=2))
