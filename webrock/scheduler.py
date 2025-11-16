import asyncio
import json
import time
from pathlib import Path

SCHEDULE_FILE = Path("schedules.json")

async def run_schedules(app, schedules):
    while True:
        now = time.time()
        for plugin_id, jobs in schedules.items():
            for job in jobs:
                if should_run(job, now):
                    # Internal call to /start/<plugin> route
                    form_args = job.get("args", {})
                    data = {k: [str(v)] for k, v in form_args.items()}  # mimic form data
                    _, resp = await app.test_client.post(f"/start/{plugin_id}", form=data)
                    print(f"Scheduled start of {plugin_id}: {resp.status}")
                    job["last_run"] = now
        save_schedules(schedules)
        await asyncio.sleep(1)


def should_run(job, now):
    job_type = job.get("type")
    last_run = job.get("last_run")
    if job_type == "interval":
        interval = job.get("interval_seconds", 60)
        return last_run is None or now - last_run >= interval
    elif job_type == "cron":
        cron_time = job.get("time")
        if not cron_time:
            return False
        hh, mm = map(int, cron_time.split(":"))
        local = time.localtime(now)
        return local.tm_hour == hh and local.tm_min == mm and (last_run is None or now - last_run >= 60)
    elif job_type == "once":
        return last_run is None
    return False


def load_schedules():
    if SCHEDULE_FILE.exists():
        return json.loads(SCHEDULE_FILE.read_text())
    return {}


def save_schedules(schedules):
    SCHEDULE_FILE.write_text(json.dumps(schedules, indent=2))
