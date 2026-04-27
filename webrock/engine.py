import asyncio
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from .schedule_utils import calculate_next_run_from_row
from . import db


def _run_sync_function(func, kwargs):
    return func(**kwargs)


class Engine:
    def __init__(self, plugins: dict):
        self.plugins = plugins
        self._executor = ThreadPoolExecutor()
        self._task: asyncio.Task | None = None

    def start(self):
        self._task = asyncio.create_task(self._run_schedules())

    def stop(self):
        if self._task:
            self._task.cancel()

    async def _run_schedules(self):
        while True:
            triggered_this_tick = set()
            active = db.get_active_schedules()

            # Pass 1: time-based schedules
            for row in active:
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

                args = row["args"]
                plugin = self.plugins[row["plugin_id"]]
                run_id = db.insert_run(row["id"], row["plugin_id"], args)
                await self._run_job(plugin, args, run_id)
                triggered_this_tick.add(row["id"])
                print(f"Scheduled start of {row['plugin_id']}")

                if row["type"] == "once":
                    db.soft_delete_schedule(row["id"])
                else:
                    new_next = calculate_next_run_from_row({**row, "last_run": now})
                    if new_next is not None:
                        db.update_schedule_next_run(row["id"], new_next, last_run=now)

            # Pass 2: "after" schedules — iterate until the chain is exhausted
            newly_triggered = set(triggered_this_tick)
            after_rows = [r for r in active if r["type"] == "after" and r["plugin_id"] in self.plugins]
            while newly_triggered:
                just_triggered = set()
                for row in after_rows:
                    if row["id"] in triggered_this_tick:
                        continue
                    trigger_id = row["config"].get("trigger_id")
                    if trigger_id in newly_triggered:
                        args = row["args"]
                        plugin = self.plugins[row["plugin_id"]]
                        run_id = db.insert_run(row["id"], row["plugin_id"], args)
                        await self._run_job(plugin, args, run_id)
                        print(f"After-triggered start of {row['plugin_id']}")
                        just_triggered.add(row["id"])
                triggered_this_tick |= just_triggered
                newly_triggered = just_triggered

            await asyncio.sleep(1)

    async def _run_job(self, plugin, args, run_id: int):
        if asyncio.iscoroutinefunction(plugin["function"]):
            plugin["task"] = asyncio.create_task(plugin["function"](**args))
        else:
            loop = asyncio.get_event_loop()
            plugin["task"] = loop.run_in_executor(
                self._executor,
                _run_sync_function,
                plugin["function"],
                args,
            )
        plugin["task"].add_done_callback(self._complete_callback(plugin, run_id))

    def _complete_callback(self, plugin, run_id: int):
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
