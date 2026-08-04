"""End-to-end tests for schedule-stop functionality.

These tests run a real Engine with an in-memory SQLite database to verify that
tasks are actually stopped when their stop conditions are met.
"""
import asyncio
import sys
import os
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from webrock import db
from webrock.engine import Engine
from webrock.decorator import plugin


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

@plugin()
async def _infinite_task() -> None:
    """Runs forever until cancelled."""
    await asyncio.sleep(9999)


def _make_engine() -> Engine:
    """Return a fresh Engine backed by an in-memory DB."""
    db.init_db(":memory:")
    pid = "tests.e2e.test_stop_scheduling._infinite_task"
    plugins = {
        pid: {
            "function": _infinite_task,
            "task": None,
        }
    }
    return Engine(plugins), pid


async def _run_engine_for(engine: Engine, seconds: float) -> None:
    """Start the engine, run for `seconds`, then stop it."""
    engine.start()
    await asyncio.sleep(seconds)
    engine.stop()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStopAfterDuration(unittest.IsolatedAsyncioTestCase):
    async def test_task_stopped_after_duration(self):
        engine, pid = _make_engine()
        schedule_id = db.insert_schedule(
            pid, "once", {}, {"timestamp": "now"},
            stop_config={"type": "duration", "seconds": 0.3},
        )

        await _run_engine_for(engine, 1.5)

        plugin = engine.plugins[pid]
        task = plugin.get("task")
        self.assertTrue(task is None or task.done(), "Task should have been stopped after 0.3s")
        self.assertNotIn(pid, engine._task_stop_at)


class TestStopAtTime(unittest.IsolatedAsyncioTestCase):
    async def test_task_stopped_at_specific_time(self):
        engine, pid = _make_engine()
        stop_at = (
            __import__("datetime").datetime.now()
            + __import__("datetime").timedelta(seconds=0.4)
        ).isoformat(timespec="seconds")
        schedule_id = db.insert_schedule(
            pid, "once", {}, {"timestamp": "now"},
            stop_config={"type": "at_time", "stop_at": stop_at},
        )

        await _run_engine_for(engine, 1.5)

        plugin = engine.plugins[pid]
        task = plugin.get("task")
        self.assertTrue(task is None or task.done(), "Task should have been stopped at the specified time")
        self.assertNotIn(pid, engine._task_stop_at)


class TestStopIfNewStart(unittest.IsolatedAsyncioTestCase):
    async def test_previous_run_cancelled_on_new_start(self):
        engine, pid = _make_engine()
        # Schedule to run every 0.4 seconds; stop_if_new_start cancels previous run
        db.insert_schedule(
            pid, "interval", {}, {"seconds": 1},
            stop_config={"stop_if_new_start": True},
        )

        # Manually trigger first run, record its task
        plugin = engine.plugins[pid]
        run_id_1 = db.insert_run(1, pid, {})
        await engine._run_job(plugin, pid, {}, run_id_1, 1, {"stop_if_new_start": True})
        first_task = plugin["task"]
        self.assertFalse(first_task.done(), "First task should still be running")

        # Trigger second run — should cancel first
        run_id_2 = db.insert_run(1, pid, {})
        await engine._run_job(plugin, pid, {}, run_id_2, 1, {"stop_if_new_start": True})

        # Give the event loop a tick to process the cancellation
        await asyncio.sleep(0.1)

        self.assertTrue(first_task.cancelled(), "First task should have been cancelled")
        second_task = plugin["task"]
        self.assertFalse(second_task.done(), "Second task should still be running")
        second_task.cancel()
        await asyncio.sleep(0)


class TestNoStop(unittest.IsolatedAsyncioTestCase):
    async def test_task_keeps_running_without_stop_config(self):
        engine, pid = _make_engine()
        db.insert_schedule(pid, "once", {}, {"timestamp": "now"})

        engine.start()
        await asyncio.sleep(0.8)

        plugin = engine.plugins[pid]
        task = plugin.get("task")
        self.assertIsNotNone(task)
        self.assertFalse(task.done(), "Task should still be running — no stop config")

        engine.stop()
        task.cancel()
        await asyncio.sleep(0)


if __name__ == "__main__":
    unittest.main()
