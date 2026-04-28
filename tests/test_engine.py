import asyncio
import os
import tempfile
import time
import unittest

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


if __name__ == "__main__":
    unittest.main()
