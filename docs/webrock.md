# Webrock Documentation

## Overview
Webrock is a lightweight web control panel for “plugin” functions inside any Python project. Mark functions with the `@plugin` decorator, start the server, and Webrock scans your project, renders a UI to configure arguments, and lets you schedule or stop those functions.

## How the app operates
1. **Startup**: The CLI entry point launches the Sanic server.
2. **Project scan**: The server walks the current project, imports modules, and identifies `@plugin`, `@init`, and `@shutdown` functions.
3. **Metadata build**: Plugin signatures and decorator metadata become UI fields.
4. **UI render**: Jinja templates render a control panel with forms for each plugin.
5. **Scheduling**: A background task checks schedules every second and triggers jobs when due.
6. **Persistence**: Schedules are saved to and loaded from `schedules.json`.

## Key concepts
- **Plugin**: A function decorated with `@plugin` that can be invoked and scheduled from the UI.
- **Init hook**: A function decorated with `@init` that runs at server start.
- **Shutdown hook**: A function decorated with `@shutdown` that runs when the server stops.
- **Schedules**: One-time, interval, or cron-like schedules stored as JSON.

## File structure
- [README.md](../README.md)
  Project summary.
- [pyproject.toml](../pyproject.toml)
  Package metadata and CLI entry point (`rock`).
- [requirements.txt](../requirements.txt)
  Runtime dependencies.

### webrock package
- [webrock/app.py](../webrock/app.py)
  Creates the Sanic app, defines routes, and runs scheduled jobs.
- [webrock/run.py](../webrock/run.py)
  CLI entry point that runs the server.
- [webrock/decorator.py](../webrock/decorator.py)
  Decorators for plugins, init, and shutdown hooks.
- [webrock/load_project.py](../webrock/load_project.py)
  Scans project modules, loads plugin metadata, and runs init hooks.
- [webrock/schedule_utils.py](../webrock/schedule_utils.py)
  Parses scheduling forms, computes next run times, loads/saves schedules.
- [webrock/__init__.py](../webrock/__init__.py)
  Package initializer (empty).

### UI templates
- [webrock/templates/control_panel.html](../webrock/templates/control_panel.html)
  Main UI page.
- [webrock/templates/_plugin_display.html](../webrock/templates/_plugin_display.html)
  Per-plugin form rendering.
- [webrock/templates/_form_macros.html](../webrock/templates/_form_macros.html)
  Form input macros.

### Static assets
- [webrock/static/control_panel.js](../webrock/static/control_panel.js)
  UI behavior for schedules and status.
- [webrock/static/control_panel.css](../webrock/static/control_panel.css)
  UI styling.

## Server routes
Defined in [webrock/app.py](../webrock/app.py):
- `GET /` — Render the control panel.
- `POST /schedule_job/<plugin_id>` — Add a schedule for a plugin.
- `GET /stop/<plugin_id>` — Cancel a running task.
- `GET /status/<plugin_id>` — Check task status and last result.
- `GET /get_schedules` — Return schedules JSON.
- `POST /remove_schedule/<plugin_id>` — Remove a scheduled job.

## Scheduling types
- **Once**: run immediately or at a specified timestamp.
- **Interval**: run every N seconds.
- **Cron-like**: run at specific minutes/hours/days/months.

## Example plugin
```python
from webrock.decorator import plugin, init, shutdown

@init
def setup():
    print("Init hook runs on server start.")

@shutdown
def cleanup():
    print("Shutdown hook runs on server stop.")

@plugin(
    description="Add two numbers",
    a={"ui_element": "number", "min": 0, "max": 100},
    b={"ui_element": "number", "min": 0, "max": 100}
)
def add(a: int, b: int) -> int:
    return a + b
```

## Notes
- The UI templates match the server’s current schedule API (once/interval/cron-like).
- Schedules are persisted in `schedules.json` at the project root.
