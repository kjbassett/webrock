"""Webrock built-in plugins that are always available regardless of the user project."""

from .decorator import plugin
from . import db


@plugin()
async def resume_schedules(schedule_ids: str) -> str:
    """Resume a list of paused schedules at the time this plugin fires.

    Intended to be scheduled as a one-time job via the "Resume at..." button
    in the schedule manager. Select the schedules you want to resume, click
    "Resume at...", pick a future date and time, and confirm. A new Specific
    Time schedule will be created that calls this plugin at that time.

    Args:
        schedule_ids: Comma-separated integer schedule IDs to resume.

    Returns:
        Summary string listing the count and IDs that were unpaused.
    """
    ids = [int(x.strip()) for x in schedule_ids.split(",") if x.strip().isdigit()]
    if not ids:
        return "No valid schedule IDs provided."
    db.bulk_set_paused(ids, False)
    return f"Resumed {len(ids)} schedule(s): {ids}"
