import json
import sqlite3
import time
from datetime import datetime

_conn: sqlite3.Connection | None = None


def init_db(db_path: str) -> sqlite3.Connection:
    global _conn
    _conn = sqlite3.connect(db_path, check_same_thread=False)
    _conn.row_factory = sqlite3.Row
    _conn.execute("PRAGMA journal_mode=WAL")
    _conn.executescript("""
        CREATE TABLE IF NOT EXISTS schedules (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            plugin_id   TEXT NOT NULL,
            type        TEXT NOT NULL,
            args        TEXT NOT NULL,
            config      TEXT NOT NULL,
            next_run    REAL,
            last_run    REAL,
            created_at  REAL NOT NULL,
            deleted_at  REAL
        );

        CREATE TABLE IF NOT EXISTS runs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            schedule_id  INTEGER NOT NULL REFERENCES schedules(id),
            plugin_id    TEXT NOT NULL,
            args         TEXT NOT NULL,
            started_at   REAL NOT NULL,
            finished_at  REAL,
            status       TEXT NOT NULL DEFAULT 'running',
            result       TEXT,
            error        TEXT
        );
    """)
    _conn.commit()
    return _conn


def get_conn() -> sqlite3.Connection:
    if _conn is None:
        raise RuntimeError("Database not initialised — call init_db() first")
    return _conn


# --- Schedules ---

def insert_schedule(plugin_id: str, stype: str, args_dict: dict, config_dict: dict) -> int:
    cur = get_conn().execute(
        "INSERT INTO schedules (plugin_id, type, args, config, created_at) VALUES (?, ?, ?, ?, ?)",
        (plugin_id, stype, json.dumps(args_dict), json.dumps(config_dict), time.time()),
    )
    get_conn().commit()
    return cur.lastrowid


def get_active_schedules() -> list[dict]:
    rows = get_conn().execute(
        "SELECT * FROM schedules WHERE deleted_at IS NULL ORDER BY id"
    ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["args"] = json.loads(d["args"])
        d["config"] = json.loads(d["config"])
        result.append(d)
    return result


def update_schedule_next_run(schedule_id: int, next_run: float, last_run: float | None = None):
    if last_run is not None:
        get_conn().execute(
            "UPDATE schedules SET next_run = ?, last_run = ? WHERE id = ?",
            (next_run, last_run, schedule_id),
        )
    else:
        get_conn().execute(
            "UPDATE schedules SET next_run = ? WHERE id = ?",
            (next_run, schedule_id),
        )
    get_conn().commit()


def soft_delete_schedule(schedule_id: int):
    get_conn().execute(
        "UPDATE schedules SET deleted_at = ? WHERE id = ?",
        (time.time(), schedule_id),
    )
    get_conn().commit()


# --- Runs ---

def insert_run(schedule_id: int, plugin_id: str, args_dict: dict) -> int:
    cur = get_conn().execute(
        "INSERT INTO runs (schedule_id, plugin_id, args, started_at, status) VALUES (?, ?, ?, ?, 'running')",
        (schedule_id, plugin_id, json.dumps(args_dict), time.time()),
    )
    get_conn().commit()
    return cur.lastrowid


def complete_run(run_id: int, status: str, result=None, error: str | None = None):
    get_conn().execute(
        "UPDATE runs SET finished_at = ?, status = ?, result = ?, error = ? WHERE id = ?",
        (time.time(), status, json.dumps(result) if result is not None else None, error, run_id),
    )
    get_conn().commit()


def get_runs_for_plugin(plugin_id: str, limit: int = 50) -> list[dict]:
    rows = get_conn().execute(
        "SELECT * FROM runs WHERE plugin_id = ? ORDER BY started_at DESC LIMIT ?",
        (plugin_id, limit),
    ).fetchall()
    return [dict(row) for row in rows]


# --- Formatting ---

def format_ts(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")
