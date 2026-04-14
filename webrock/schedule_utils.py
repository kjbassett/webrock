import calendar
import json
from datetime import datetime, timedelta


def form_to_schedule_parts(meta, form):
    """Parse a schedule form POST into (stype, args_dict, config_dict)."""
    job_args = {}
    config = {}

    def get_value(key, default=None):
        if key not in form or not form[key]:
            if not default:
                raise ValueError(f"Missing required item: {key}")
            return default
        v = form[key][0]  # form puts everything in lists
        if "," in v:
            v = v.split(",")
        return v

    # get schedule type
    schedule_type = get_value("_schedule_type")
    if schedule_type not in ("once", "interval", "cron", "after"):
        raise ValueError("Invalid schedule type")

    if schedule_type == "once":
        ts = get_value("_timestamp", "now")
        if ts != "now":
            try:
                datetime.fromisoformat(ts)
            except ValueError:
                raise ValueError(f"Invalid timestamp: {ts}")
        config["timestamp"] = ts

    elif schedule_type == "after":
        trigger_id_raw = get_value("_trigger_schedule_id")
        try:
            config["trigger_id"] = int(trigger_id_raw)
        except (ValueError, TypeError):
            raise ValueError("_trigger_schedule_id must be an integer")

    elif schedule_type == "interval":
        seconds_raw = get_value("_seconds")
        try:
            config["seconds"] = int(seconds_raw)
        except ValueError:
            raise ValueError("interval must be an integer")

    elif schedule_type == "cron":
        config["minutes"] = get_value("_minutes", "*")
        config["hours"] = get_value("_hours", "*")
        config["days_of_week"] = get_value("_days_of_week", "*")
        config["days_of_month"] = get_value("_days_of_month", "*")
        config["months"] = get_value("_months", "*")

    # get function args
    for arg in meta["args"]:
        name = arg["name"]

        if name not in form:
            if "default" not in arg:
                raise ValueError(f"Missing required argument {name}")
            form[name] = arg["default"]
            continue

        value = get_value(name, arg.get("default"))

        _type = arg["type"]

        if _type == "bool":
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

        job_args[name] = value

    return schedule_type, job_args, config


def calculate_next_run_from_row(row: dict) -> float | None:
    """Convert a DB schedule row to the flat dict format and call calculate_next_run."""
    config = row["config"] if isinstance(row["config"], dict) else json.loads(row["config"])
    flat = {"type": row["type"], **config}
    if row.get("last_run") is not None:
        flat["last_run"] = row["last_run"]
    return calculate_next_run(flat)


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
    Given a schedule dict, return the next UTC timestamp the job should run.
    Accepts the flat dict format: {"type": ..., "seconds": ..., "last_run": ..., ...}
    """
    stype = schedule["type"]

    # --- AFTER (event-driven, no time-based next_run) ---
    if stype == "after":
        return None

    # --- ONCE ---
    if stype == "once":
        if schedule["timestamp"] == "now":
            return datetime.now().timestamp()
        return int(datetime.fromisoformat(schedule["timestamp"]).timestamp())

    last_run = schedule.get("last_run", int(datetime.now().timestamp()))

    # --- INTERVAL ---
    if stype == "interval":
        if "last_run" not in schedule:  # recently created or resumed
            return datetime.now().timestamp()
        seconds = schedule["seconds"]
        return last_run + seconds

    # --- CRON ---
    minutes = parse_cron_field(schedule["minutes"])
    hours = parse_cron_field(schedule["hours"])
    dom = parse_cron_field(schedule["days_of_month"])
    dow = parse_cron_field(schedule["days_of_week"])
    months = parse_month_field(schedule["months"])

    t = datetime.fromtimestamp(last_run) + timedelta(minutes=1)
    t = t.replace(second=0, microsecond=0)

    end = t + timedelta(days=365 * 5)

    while t < end:
        if (
            matches(t.minute, minutes)
            and matches(t.hour, hours)
            and matches(t.month, months)
            and matches(t.day, dom)
            and matches(t.weekday(), dow)
        ):
            return int(t.timestamp())

        t += timedelta(minutes=1)

    raise RuntimeError(
        f"No next runtime found within 5 years. Cron may be impossible.\n{schedule}"
    )
