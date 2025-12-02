import calendar
import json
from datetime import datetime, timedelta


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
        ts = get_value("_timestamp", "now")
        if ts != "now":
            try:
                datetime.fromisoformat(ts)
            except ValueError:
                raise ValueError(f"Invalid timestamp: {ts}")
        schedule['timestamp'] = ts

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
                value = value.lower() in ("true", "on")
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


MONTH_LOOKUP = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
MONTH_LOOKUP.update(
    {name.lower(): i for i, name in enumerate(calendar.month_abbr) if name}
)
MONTH_LOOKUP.update({str(n): n for n in range(1, 13)})


def parse_cron_field(raw):
    """
    Convert a string into a sorted set of allowed int values.
    Accepts '*', '1,2,5-7', etc.
    Returns None for wildcard.
    """
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
            start = MONTH_LOOKUP.get(start)
            end = MONTH_LOOKUP.get(end)
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
        return int(datetime.fromisoformat(schedule["timestamp"]).timestamp())

    last_run = schedule["last_run"]

    # --- INTERVAL ---
    if stype == "interval":
        if "last_run" not in schedule:  # recently created or resumed
            return int(datetime.now().timestamp())
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
        schedule = json.loads(schedule_file.read_text())

        # clear next_run times if already missed
        for plugin in schedule.keys():
            for sched in schedule[plugin]:
                if "next_run" in sched['schedule'] and sched['schedule']['next_run'] < datetime.now().timestamp():
                    del sched['schedule']["next_run"]
    return {}


def save_schedules(schedule_file, schedules):
    schedule_file.write_text(json.dumps(schedules, indent=2))
