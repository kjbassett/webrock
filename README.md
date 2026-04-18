# Webrock

A lightweight scheduling server for Python projects. Add a single decorator to any function and webrock automatically exposes it through a web UI and an MCP server — giving you manual controls, cron/interval/chained scheduling, run history, and LLM tool access with zero boilerplate.

---

## Quick start

### Installation

Install webrock into your project's environment:

```bash
pip install -e /path/to/webrock
```

### Decorate your functions

```python
# my_project/tasks.py
from webrock.decorator import plugin

@plugin()
def send_report(email: str, subject: str = "Daily Report"):
    """Send the daily report to the given email address."""
    ...

@plugin(
    brightness={"ui_element": "slider", "min": 0, "max": 100}
)
def set_brightness(brightness: int = 50):
    """Set display brightness."""
    ...
```

### Run the server

```bash
rock                        # web UI on :8000, MCP SSE on :8001
rock --port 8080            # custom web UI port
rock --no-mcp               # web UI only, no MCP server
rock --mcp-port 8002        # custom MCP SSE port
```

Open `http://localhost:8000` to see all discovered plugins and schedule them.

---

## Decorators

### `@plugin(**kwargs)`

Marks a function as a webrock plugin. Webrock scans the project directory on startup and registers every decorated function automatically.

```python
from webrock.decorator import plugin

@plugin(
    description="Resize and compress an image",
    quality={"ui_element": "slider", "min": 1, "max": 100},
)
def process_image(path: str, quality: int = 80) -> str:
    ...
```

**Per-argument overrides** — pass a dict keyed by the argument name to customise its UI widget or add constraints:

| Key | Effect |
|---|---|
| `ui_element` | Override the default widget (`slider`, `textbox`, `checkbox`, `color`, `date`, `datetime_local`, `time`, `number`, …) |
| `min` / `max` | Numeric bounds |
| `default` | Default value shown in the form |

**Special decorator flags:**

| Flag | Effect |
|---|---|
| `mcp_disabled=True` | Exclude this plugin from the MCP server's tool list |

### `@init`

Runs once when the server starts, before scheduling begins. Supports both sync and async functions.

```python
from webrock.decorator import init

@init
def connect_to_database():
    ...
```

### `@shutdown`

Runs when the server stops. Supports both sync and async functions.

```python
from webrock.decorator import shutdown

@shutdown
def close_connections():
    ...
```

---

## Scheduling

Every plugin gets a scheduling form in the web UI with four modes:

| Type | Description |
|---|---|
| **Once** | Run at a specific datetime, or immediately |
| **Interval** | Repeat every N seconds |
| **Cron** | Run on a cron-style schedule (minute, hour, day-of-week, day-of-month, month) |
| **After** | Trigger automatically when another schedule's job completes |

Schedules are persisted in `webrock.db` (SQLite, created in the project directory). Each schedule row records its `source` (`web` or `mcp`). Completed runs are stored in a `runs` table with start time, finish time, status, and result/error.

---

## MCP server

Webrock ships a built-in [MCP](https://modelcontextprotocol.io) server so any MCP-compatible client (e.g. Claude Desktop) can schedule, inspect, edit, and cancel jobs directly.

### Tools exposed

One scheduling tool is generated per plugin (named `schedule_<function_name>`). Its input schema includes the plugin's own argument list plus schedule configuration fields:

| Field | Description |
|---|---|
| `schedule_type` | `"once"` (default), `"interval"`, `"cron"`, or `"after"` |
| `run_at` | ISO datetime string for a one-time run; omit to run immediately |
| `interval_seconds` | Repeat interval in seconds |
| `after_schedule_id` | ID of the schedule whose completion triggers this job |
| `cron_minutes/hours/days_of_week/days_of_month/months` | Cron fields (default `"*"`) |

Four management tools are always available:

| Tool | Description |
|---|---|
| `list_schedules` | List active schedules, optionally filtered by plugin |
| `edit_schedule` | Update an existing schedule's type/args/config in-place (same ID) |
| `cancel_schedule` | Soft-delete a schedule by ID |
| `get_run_history` | Fetch recent run history for a plugin |

### Transport modes

**SSE (default)** — MCP server starts automatically alongside Sanic:

```bash
rock                    # MCP SSE on http://localhost:8001/sse
rock --mcp-port 8002    # custom port
```

**stdio** — standalone process for Claude Desktop (no web UI):

```bash
rock --mcp-transport stdio --project /path/to/project
# or use the dedicated entry point:
rock-mcp --project /path/to/project
```

### Claude Desktop configuration

**SSE** (connect to a running `rock` instance):
```json
{
  "mcpServers": {
    "webrock": {
      "url": "http://localhost:8001/sse"
    }
  }
}
```

**stdio** (Claude Desktop spawns the process):
```json
{
  "mcpServers": {
    "webrock": {
      "command": "/path/to/venv/bin/python",
      "args": [
        "-m", "webrock.run",
        "--mcp-transport", "stdio",
        "--project", "/path/to/your/project"
      ]
    }
  }
}
```

The `--project` argument tells webrock where to scan for plugins and where to create `webrock.db`. It should point to the root directory of the project containing your decorated functions.

---

## CLI reference

```
rock [--port PORT] [--no-mcp] [--mcp-transport {sse,stdio}] [--mcp-port MCP_PORT] [--project PROJECT]

  --port PORT              Sanic web UI port (default: 8000)
  --no-mcp                 Disable the MCP server entirely
  --mcp-transport          'sse' starts an HTTP SSE server alongside Sanic (default)
                           'stdio' starts only a stdio MCP server (no web UI)
  --mcp-port MCP_PORT      MCP SSE server port (default: 8001)
  --project PROJECT        Absolute path to the project directory (plugins + webrock.db).
                           Defaults to the current working directory.
```

---

## Development

```bash
# Clone and install in editable mode
git clone https://github.com/kjbassett/webrock
cd webrock
pip install -e .

# Run against the bundled test plugins
rock
python -m webrock.run               # equivalent
python -m webrock.run --no-mcp
python -m webrock.run --mcp-transport stdio --project .
python -m webrock.run --mcp-transport sse --project .

# Run tests
python -m pytest tests/
```
