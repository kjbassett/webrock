import asyncio
import os
import tempfile
import time
import unittest
import unittest.mock

from webrock import db
from webrock.engine import Engine


class TestChainOrdering(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        fd, self._db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        db.init_db(self._db_path)

    def tearDown(self):
        if db._conn is not None:
            db._conn.close()
            db._conn = None
        try:
            os.unlink(self._db_path)
        except OSError:
            pass

    async def test_three_chained_jobs_each_start_after_previous_finishes(self):
        timeline = []

        async def job_a():
            timeline.append(("a", "start", time.time()))
            await asyncio.sleep(0.05)
            timeline.append(("a", "end", time.time()))

        async def job_b():
            timeline.append(("b", "start", time.time()))
            await asyncio.sleep(0.05)
            timeline.append(("b", "end", time.time()))

        async def job_c():
            timeline.append(("c", "start", time.time()))
            await asyncio.sleep(0.05)
            timeline.append(("c", "end", time.time()))

        plugins = {
            "job_a": {"function": job_a, "task": None},
            "job_b": {"function": job_b, "task": None},
            "job_c": {"function": job_c, "task": None},
        }

        id_a = db.insert_schedule("job_a", "once", {}, {"timestamp": "now"})
        id_b = db.insert_schedule("job_b", "after", {}, {"trigger_id": id_a})
        id_c = db.insert_schedule("job_c", "after", {}, {"trigger_id": id_b})

        engine = Engine(plugins)
        engine.start()

        # 3 jobs × 0.05s + engine tick overhead + generous margin
        await asyncio.sleep(1.0)
        engine.stop()

        self.assertEqual(len(timeline), 6, f"Expected 6 timeline events, got: {timeline}")

        times = {(job, event): ts for job, event, ts in timeline}

        self.assertGreater(
            times[("b", "start")], times[("a", "end")],
            "job_b started before job_a finished"
        )
        self.assertGreater(
            times[("c", "start")], times[("b", "end")],
            "job_c started before job_b finished"
        )


class TestAfterScheduleTriggerConditionsUnit(unittest.IsolatedAsyncioTestCase):
    """Unit tests for _trigger_after_jobs filtering logic.

    Uses a real in-memory DB but mocks _run_job to isolate the flag logic.
    """

    def setUp(self):
        fd, self._db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        db.init_db(self._db_path)

    def tearDown(self):
        if db._conn is not None:
            db._conn.close()
            db._conn = None
        try:
            os.unlink(self._db_path)
        except OSError:
            pass

    def _make_engine_with_mock_run(self):
        async def noop(): pass
        plugins = {
            "parent": {"function": noop},
            "child":  {"function": noop},
        }
        engine = Engine(plugins)
        for p in engine.plugins.values():
            p["pause_event"] = asyncio.Event()
            p["pause_event"].set()
        engine._run_job = unittest.mock.AsyncMock()
        return engine

    async def _run_trigger(self, engine, parent_id, status):
        await engine._trigger_after_jobs(parent_id, status)

    async def test_fires_on_success_when_trigger_on_success_true(self):
        engine = self._make_engine_with_mock_run()
        parent_id = db.insert_schedule("parent", "interval", {}, {"seconds": 3600})
        db.insert_schedule("child", "after", {}, {
            "trigger_id": parent_id, "trigger_on_success": True, "trigger_on_error": False,
        })
        await self._run_trigger(engine, parent_id, "success")
        engine._run_job.assert_awaited_once()

    async def test_does_not_fire_on_error_when_trigger_on_error_false(self):
        engine = self._make_engine_with_mock_run()
        parent_id = db.insert_schedule("parent", "interval", {}, {"seconds": 3600})
        db.insert_schedule("child", "after", {}, {
            "trigger_id": parent_id, "trigger_on_success": True, "trigger_on_error": False,
        })
        await self._run_trigger(engine, parent_id, "error")
        engine._run_job.assert_not_awaited()

    async def test_fires_on_error_when_trigger_on_error_true(self):
        engine = self._make_engine_with_mock_run()
        parent_id = db.insert_schedule("parent", "interval", {}, {"seconds": 3600})
        db.insert_schedule("child", "after", {}, {
            "trigger_id": parent_id, "trigger_on_success": False, "trigger_on_error": True,
        })
        await self._run_trigger(engine, parent_id, "error")
        engine._run_job.assert_awaited_once()

    async def test_does_not_fire_on_success_when_trigger_on_success_false(self):
        engine = self._make_engine_with_mock_run()
        parent_id = db.insert_schedule("parent", "interval", {}, {"seconds": 3600})
        db.insert_schedule("child", "after", {}, {
            "trigger_id": parent_id, "trigger_on_success": False, "trigger_on_error": True,
        })
        await self._run_trigger(engine, parent_id, "success")
        engine._run_job.assert_not_awaited()

    async def test_legacy_config_fires_on_success(self):
        engine = self._make_engine_with_mock_run()
        parent_id = db.insert_schedule("parent", "interval", {}, {"seconds": 3600})
        db.insert_schedule("child", "after", {}, {"trigger_id": parent_id})
        await self._run_trigger(engine, parent_id, "success")
        engine._run_job.assert_awaited_once()

    async def test_legacy_config_does_not_fire_on_error(self):
        engine = self._make_engine_with_mock_run()
        parent_id = db.insert_schedule("parent", "interval", {}, {"seconds": 3600})
        db.insert_schedule("child", "after", {}, {"trigger_id": parent_id})
        await self._run_trigger(engine, parent_id, "error")
        engine._run_job.assert_not_awaited()


class TestAfterScheduleTriggerConditionsIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests: real engine, real DB, real async functions.

    Verifies that trigger_on_success / trigger_on_error flags control whether
    the child job actually executes after a parent success or failure.
    """

    def setUp(self):
        fd, self._db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        db.init_db(self._db_path)

    def tearDown(self):
        if db._conn is not None:
            db._conn.close()
            db._conn = None
        try:
            os.unlink(self._db_path)
        except OSError:
            pass

    async def _run_scenario(
        self,
        parent_succeeds: bool,
        trigger_on_success: bool,
        trigger_on_error: bool,
        legacy: bool = False,
    ) -> bool:
        """Return True if the child job ran within the timeout."""
        child_event = asyncio.Event()

        async def parent_fn():
            if not parent_succeeds:
                raise RuntimeError("deliberate parent failure")

        async def child_fn():
            child_event.set()

        plugins = {
            "parent": {"function": parent_fn},
            "child":  {"function": child_fn},
        }
        engine = Engine(plugins)
        engine.start()

        parent_id = db.insert_schedule("parent", "once", {}, {"timestamp": "now"})
        config = {"trigger_id": parent_id}
        if not legacy:
            config["trigger_on_success"] = trigger_on_success
            config["trigger_on_error"] = trigger_on_error
        db.insert_schedule("child", "after", {}, config)

        fired = False
        try:
            await asyncio.wait_for(child_event.wait(), timeout=2.0)
            fired = True
        except asyncio.TimeoutError:
            fired = False
        finally:
            engine.stop()

        return fired

    async def test_child_fires_when_parent_succeeds_and_on_success_true(self):
        fired = await self._run_scenario(parent_succeeds=True, trigger_on_success=True, trigger_on_error=False)
        self.assertTrue(fired, "child should have fired after parent success")

    async def test_child_does_not_fire_when_parent_errors_and_on_error_false(self):
        child_event = asyncio.Event()

        async def parent_fn():
            raise RuntimeError("deliberate parent failure")

        async def child_fn():
            child_event.set()

        plugins = {
            "parent": {"function": parent_fn},
            "child":  {"function": child_fn},
        }
        engine = Engine(plugins)
        engine.start()

        parent_id = db.insert_schedule("parent", "once", {}, {"timestamp": "now"})
        db.insert_schedule("child", "after", {}, {
            "trigger_id": parent_id, "trigger_on_success": True, "trigger_on_error": False,
        })

        await asyncio.sleep(0.5)
        engine.stop()

        self.assertFalse(child_event.is_set(), "child should not have fired after parent error with trigger_on_error=False")

    async def test_child_fires_when_parent_errors_and_on_error_true(self):
        fired = await self._run_scenario(parent_succeeds=False, trigger_on_success=False, trigger_on_error=True)
        self.assertTrue(fired, "child should have fired after parent error with trigger_on_error=True")

    async def test_child_does_not_fire_when_parent_succeeds_and_on_success_false(self):
        child_event = asyncio.Event()

        async def parent_fn():
            pass

        async def child_fn():
            child_event.set()

        plugins = {
            "parent": {"function": parent_fn},
            "child":  {"function": child_fn},
        }
        engine = Engine(plugins)
        engine.start()

        parent_id = db.insert_schedule("parent", "once", {}, {"timestamp": "now"})
        db.insert_schedule("child", "after", {}, {
            "trigger_id": parent_id, "trigger_on_success": False, "trigger_on_error": True,
        })

        await asyncio.sleep(0.5)
        engine.stop()

        self.assertFalse(child_event.is_set(), "child should not have fired after parent success with trigger_on_success=False")

    async def test_legacy_config_fires_on_parent_success(self):
        fired = await self._run_scenario(parent_succeeds=True, trigger_on_success=True, trigger_on_error=False, legacy=True)
        self.assertTrue(fired, "legacy after-schedule should fire when parent succeeds")

    async def test_legacy_config_does_not_fire_on_parent_error(self):
        child_event = asyncio.Event()

        async def parent_fn():
            raise RuntimeError("deliberate parent failure")

        async def child_fn():
            child_event.set()

        plugins = {
            "parent": {"function": parent_fn},
            "child":  {"function": child_fn},
        }
        engine = Engine(plugins)
        engine.start()

        parent_id = db.insert_schedule("parent", "once", {}, {"timestamp": "now"})
        db.insert_schedule("child", "after", {}, {"trigger_id": parent_id})

        await asyncio.sleep(0.5)
        engine.stop()

        self.assertFalse(child_event.is_set(), "legacy after-schedule should not fire when parent errors")

    async def test_two_siblings_only_matching_one_fires(self):
        """When parent errors, only the on_error sibling fires, not the on_success sibling."""
        success_event = asyncio.Event()
        error_event = asyncio.Event()

        async def parent_fn():
            raise RuntimeError("deliberate parent failure")

        async def on_success_child():
            success_event.set()

        async def on_error_child():
            error_event.set()

        plugins = {
            "parent":          {"function": parent_fn},
            "on_success_child": {"function": on_success_child},
            "on_error_child":   {"function": on_error_child},
        }
        engine = Engine(plugins)
        engine.start()

        parent_id = db.insert_schedule("parent", "once", {}, {"timestamp": "now"})
        db.insert_schedule("on_success_child", "after", {}, {
            "trigger_id": parent_id, "trigger_on_success": True, "trigger_on_error": False,
        })
        db.insert_schedule("on_error_child", "after", {}, {
            "trigger_id": parent_id, "trigger_on_success": False, "trigger_on_error": True,
        })

        try:
            await asyncio.wait_for(error_event.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass
        finally:
            engine.stop()

        self.assertTrue(error_event.is_set(), "on_error_child should have fired")
        self.assertFalse(success_event.is_set(), "on_success_child should not have fired")


if __name__ == "__main__":
    unittest.main()
