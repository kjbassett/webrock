import asyncio
import inspect
import time
import traceback
import types as _types
from concurrent.futures import ThreadPoolExecutor

from .schedule_utils import calculate_next_run_from_row
from . import db

TASK_IDLE = "idle"
TASK_RUNNING = "running"
TASK_PAUSED = "paused"


def _run_sync_function(func, kwargs):
    return func(**kwargs)


def _coerce_args(func, args: dict) -> dict:
    """Coerce string arg values to the types declared in func's annotations."""
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return args
    coerced = dict(args)
    for name, param in sig.parameters.items():
        if name not in coerced or not isinstance(coerced[name], str):
            continue
        ann = param.annotation
        if ann is inspect.Parameter.empty or isinstance(ann, _types.UnionType):
            continue
        val = coerced[name]
        try:
            if ann is int:
                coerced[name] = int(val)
            elif ann is float:
                coerced[name] = float(val)
            elif ann is bool:
                coerced[name] = val.lower() not in ("false", "0", "")
        except (ValueError, TypeError):
            pass
    return coerced


class Engine:
    """Scheduler and task executor.

    Task lifecycle: idle → running → idle (or paused → running → idle).
    System pause queues tasks that would start; resume drains the queue.
    Individual task pause uses asyncio.Event — cooperative plugins call
    ``await wait_if_paused(plugin_id)`` (from webrock.pause) at safe checkpoints.
    """

    def __init__(self, plugins: dict):
        self.plugins = plugins
        self._executor = ThreadPoolExecutor()
        self._scheduler_task: asyncio.Task | None = None
        self._system_paused: bool = False
        self._pending_runs: list[dict] = []
        self._stop_requested: set[str] = set()
        for plugin in plugins.values():
            plugin.setdefault("task", None)
            plugin.setdefault("task_status", TASK_IDLE)
            plugin.setdefault("pause_event", None)
            plugin.setdefault("current_schedule_id", None)
            plugin.setdefault("current_run_id", None)

    def start(self) -> None:
        """Start the scheduler loop. Must be called inside a running event loop."""
        from . import pause as _pause_module
        _pause_module._engine = self
        for plugin in self.plugins.values():
            plugin["pause_event"] = asyncio.Event()
            plugin["pause_event"].set()
        self._scheduler_task = asyncio.create_task(self._run_schedules())

    def stop(self) -> None:
        """Cancel the scheduler loop (does not stop running tasks)."""
        if self._scheduler_task:
            self._scheduler_task.cancel()

    # --- System pause / resume ---

    def pause_system(self) -> None:
        """Pause the system: prevent new tasks from starting."""
        self._system_paused = True

    def resume_system(self) -> list[str]:
        """Resume the system and start any queued pending runs.

        Returns:
            List of plugin_ids whose tasks were started.
        """
        self._system_paused = False
        started = []
        pending, self._pending_runs = list(self._pending_runs), []
        for entry in pending:
            run_id = db.insert_run(entry["schedule_id"], entry["plugin_id"], entry["args"])
            asyncio.create_task(
                self._run_job(entry["plugin"], entry["plugin_id"], entry["args"], run_id, entry["schedule_id"])
            )
            started.append(entry["plugin_id"])
        return started

    @property
    def system_paused(self) -> bool:
        return self._system_paused

    @property
    def pending_count(self) -> int:
        return len(self._pending_runs)

    # --- Per-task pause / resume ---

    def pause_task(self, plugin_id: str) -> bool:
        """Pause a running task. Cooperative tasks will stop at their next checkpoint.

        Returns:
            True if the task was running and was requested to pause.
        """
        plugin = self.plugins.get(plugin_id)
        if not plugin:
            return False
        task = plugin.get("task")
        if not task or task.done():
            return False
        if plugin.get("pause_event"):
            plugin["pause_event"].clear()
        plugin["task_status"] = TASK_PAUSED
        return True

    def resume_task(self, plugin_id: str) -> bool:
        """Resume a paused task.

        Returns:
            True if the task existed and the resume signal was sent.
        """
        plugin = self.plugins.get(plugin_id)
        if not plugin:
            return False
        if plugin.get("pause_event"):
            plugin["pause_event"].set()
        task = plugin.get("task")
        if task and not task.done():
            plugin["task_status"] = TASK_RUNNING
            return True
        plugin["task_status"] = TASK_IDLE
        return False

    async def wait_if_paused(self, plugin_id: str) -> None:
        """Cooperative pause point for plugin functions.

        Plugins import this indirectly via ``webrock.pause.wait_if_paused``.
        """
        plugin = self.plugins.get(plugin_id)
        if plugin and plugin.get("pause_event"):
            await plugin["pause_event"].wait()

    def get_task_status(self, plugin_id: str) -> str:
        """Return current task status string for a plugin."""
        plugin = self.plugins.get(plugin_id)
        if not plugin:
            return TASK_IDLE
        return plugin.get("task_status", TASK_IDLE)

    # --- Stop with dependent lookup ---

    def stop_task(self, plugin_id: str) -> tuple[bool, list[dict]]:
        """Cancel a running or pending task and return any dependent schedules.

        Handles two cases:
        - Pending task (queued while system is paused): removes from pending queue.
        - Active task (running or individually paused): cancels the asyncio task
          and marks plugin_id in _stop_requested so the done callback knows not
          to trigger after-jobs even if the task swallows CancelledError.

        Returns:
            (was_stopped, dependent_schedule_rows)
        """
        plugin = self.plugins.get(plugin_id)
        if not plugin:
            return False, []

        stopped = False
        schedule_id = None

        # Remove any queued pending runs for this plugin (system-paused queue).
        removed = [e for e in self._pending_runs if e["plugin_id"] == plugin_id]
        if removed:
            self._pending_runs = [e for e in self._pending_runs if e["plugin_id"] != plugin_id]
            plugin["task_status"] = TASK_IDLE
            if plugin.get("pause_event"):
                plugin["pause_event"].set()
            stopped = True
            schedule_id = removed[-1]["schedule_id"]

        # Cancel active asyncio task (running or individually paused).
        task = plugin.get("task")
        if task and not task.done():
            schedule_id = plugin.get("current_schedule_id")
            self._stop_requested.add(plugin_id)
            task.cancel()
            stopped = True

        if not stopped:
            return False, []

        dependents = self._get_dependent_schedules(schedule_id) if schedule_id else []
        return True, dependents

    def _get_dependent_schedules(self, schedule_id: int) -> list[dict]:
        return [
            row for row in db.get_active_schedules()
            if row["type"] == "after"
            and row["config"].get("trigger_id") == schedule_id
            and not row.get("disabled")
        ]

    # --- Internal scheduler loop ---

    async def _run_schedules(self) -> None:
        while True:
            active = db.get_active_schedules()

            for row in active:
                if row.get("disabled"):
                    continue
                if row["plugin_id"] not in self.plugins:
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

                plugin = self.plugins[row["plugin_id"]]

                if self._system_paused:
                    self._pending_runs.append({
                        "plugin": plugin,
                        "plugin_id": row["plugin_id"],
                        "args": row["args"],
                        "schedule_id": row["id"],
                    })
                    plugin["task_status"] = TASK_PAUSED
                    print(f"System paused — queued {row['plugin_id']}")
                else:
                    run_id = db.insert_run(row["id"], row["plugin_id"], row["args"])
                    print(f"Scheduled start of {row['plugin_id']}")
                    await self._run_job(plugin, row["plugin_id"], row["args"], run_id, row["id"])

                if row["type"] == "once":
                    db.soft_delete_schedule(row["id"])
                else:
                    new_next = calculate_next_run_from_row({**row, "last_run": now})
                    if new_next is not None:
                        db.update_schedule_next_run(row["id"], new_next, last_run=now)

            await asyncio.sleep(1)

    async def _trigger_after_jobs(self, completed_schedule_id: int) -> None:
        if self._system_paused:
            return
        for row in db.get_active_schedules():
            if row["type"] != "after" or row["plugin_id"] not in self.plugins:
                continue
            if row.get("disabled"):
                continue
            if row["config"].get("trigger_id") == completed_schedule_id:
                plugin = self.plugins[row["plugin_id"]]
                run_id = db.insert_run(row["id"], row["plugin_id"], row["args"])
                print(f"After-triggered start of {row['plugin_id']}")
                await self._run_job(plugin, row["plugin_id"], row["args"], run_id, row["id"])

    async def _run_job(self, plugin: dict, plugin_id: str, args: dict, run_id: int, schedule_id: int) -> None:
        args = _coerce_args(plugin["function"], args)
        plugin["current_schedule_id"] = schedule_id
        plugin["current_run_id"] = run_id
        plugin["task_status"] = TASK_RUNNING
        if asyncio.iscoroutinefunction(plugin["function"]):
            plugin["task"] = asyncio.create_task(plugin["function"](**args))
        else:
            loop = asyncio.get_event_loop()
            plugin["task"] = loop.run_in_executor(
                self._executor, _run_sync_function, plugin["function"], args
            )
        plugin["task"].add_done_callback(
            self._make_done_callback(plugin, plugin_id, run_id, schedule_id)
        )

    def _make_done_callback(self, plugin: dict, plugin_id: str, run_id: int, schedule_id: int):
        def callback(task):
            # Check stop_requested BEFORE discarding — handles the case where
            # the plugin swallows CancelledError and the task returns normally.
            was_explicitly_stopped = plugin_id in self._stop_requested
            self._stop_requested.discard(plugin_id)
            try:
                if task.cancelled() or was_explicitly_stopped:
                    db.complete_run(run_id, "stopped")
                else:
                    result = task.result()
                    print(f"Finished {plugin['function'].__name__}")
                    db.complete_run(run_id, "success", result=result)
                    asyncio.get_running_loop().create_task(
                        self._trigger_after_jobs(schedule_id)
                    )
            except Exception as e:
                print(f"Error in {plugin['function'].__name__}: {e}")
                db.complete_run(run_id, "error", error=traceback.format_exc())
            finally:
                plugin["task"] = None
                plugin["task_status"] = TASK_IDLE
                plugin["current_schedule_id"] = None
                plugin["current_run_id"] = None
                if plugin.get("pause_event"):
                    plugin["pause_event"].set()
        return callback
