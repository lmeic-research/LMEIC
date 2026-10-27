"""
SQLite persistence for the human annotation web tool.
Version: v1.1 | Date: 2026-10-23 (v1.1: per-step and server-side timing columns, read-only mode)

Every submission is written immediately. Re-submissions of an item move the
previous version to annotation_history, so no judgement is ever overwritten
without a trace. Each annotation round uses its own database file; a frozen
round is opened read-only.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS annotations (
    annotator TEXT NOT NULL,
    item_id TEXT NOT NULL,
    phase TEXT NOT NULL,
    label TEXT NOT NULL,
    suggested_label TEXT NOT NULL,
    override_reason TEXT,
    confidence INTEGER NOT NULL,
    note TEXT,
    high_checks TEXT NOT NULL,
    medium_checks TEXT NOT NULL,
    active_seconds REAL,
    opened_at TEXT,
    submitted_at TEXT NOT NULL,
    revision INTEGER NOT NULL,
    server_elapsed_s REAL,
    step_seconds TEXT,
    PRIMARY KEY (annotator, item_id)
);
CREATE TABLE IF NOT EXISTS annotation_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    annotator TEXT NOT NULL,
    item_id TEXT NOT NULL,
    phase TEXT NOT NULL,
    label TEXT NOT NULL,
    suggested_label TEXT NOT NULL,
    override_reason TEXT,
    confidence INTEGER NOT NULL,
    note TEXT,
    high_checks TEXT NOT NULL,
    medium_checks TEXT NOT NULL,
    active_seconds REAL,
    opened_at TEXT,
    submitted_at TEXT NOT NULL,
    revision INTEGER NOT NULL,
    server_elapsed_s REAL,
    step_seconds TEXT,
    archived_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS gates (
    name TEXT PRIMARY KEY,
    value INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS onboarding (
    annotator TEXT PRIMARY KEY,
    acknowledged_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS review_opinions (
    opinion_id INTEGER PRIMARY KEY AUTOINCREMENT,
    annotator TEXT NOT NULL,
    topic TEXT NOT NULL,
    text TEXT NOT NULL,
    saved_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS review_notes (
    note_id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    saved_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    annotator TEXT,
    event TEXT NOT NULL,
    detail TEXT,
    ts TEXT NOT NULL
);
"""

ANNOTATION_COLUMNS = [
    "annotator", "item_id", "phase", "label", "suggested_label", "override_reason",
    "confidence", "note", "high_checks", "medium_checks", "active_seconds",
    "opened_at", "submitted_at", "revision", "server_elapsed_s", "step_seconds",
]
_ADDED_COLUMNS = {"server_elapsed_s": "REAL", "step_seconds": "TEXT"}

GATE_REVIEW = "calibration_review_open"
GATE_MAIN = "main_open"
TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime(TS_FORMAT)


def parse_ts(ts: str) -> datetime:
    return datetime.strptime(ts, TS_FORMAT).replace(tzinfo=timezone.utc)


class Store:
    def __init__(self, db_path: Path, read_only: bool = False):
        self.db_path = Path(db_path)
        self.read_only = read_only
        self._local = threading.local()
        self._write_lock = threading.Lock()
        if read_only:
            if not self.db_path.exists():
                raise FileNotFoundError(self.db_path)
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)
            c.execute("PRAGMA journal_mode=WAL")
            for table in ("annotations", "annotation_history"):
                have = {r["name"] for r in c.execute(f"PRAGMA table_info({table})")}
                for col, typ in _ADDED_COLUMNS.items():
                    if col not in have:
                        c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            if self.read_only:
                conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, check_same_thread=False)
            else:
                conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def _writable(self) -> None:
        if self.read_only:
            raise PermissionError(f"{self.db_path} is opened read-only")

    # ---- annotations -------------------------------------------------------

    def get(self, annotator: str, item_id: str) -> dict[str, Any] | None:
        row = self._conn().execute(
            "SELECT * FROM annotations WHERE annotator=? AND item_id=?", (annotator, item_id)
        ).fetchone()
        return _decode(row) if row else None

    def done_ids(self, annotator: str, phase: str) -> set[str]:
        rows = self._conn().execute(
            "SELECT item_id FROM annotations WHERE annotator=? AND phase=?", (annotator, phase)
        ).fetchall()
        return {r["item_id"] for r in rows}

    def all_for_phase(self, phase: str) -> list[dict[str, Any]]:
        rows = self._conn().execute("SELECT * FROM annotations WHERE phase=?", (phase,)).fetchall()
        return [_decode(r) for r in rows]

    def all_rows(self) -> list[dict[str, Any]]:
        return [_decode(r) for r in self._conn().execute("SELECT * FROM annotations").fetchall()]

    def history_rows(self) -> list[dict[str, Any]]:
        return [_decode(r) for r in self._conn().execute("SELECT * FROM annotation_history").fetchall()]

    def last_open(self, annotator: str, item_id: str) -> datetime | None:
        row = self._conn().execute(
            "SELECT ts FROM events WHERE annotator=? AND event='open_item' AND detail=? "
            "ORDER BY event_id DESC LIMIT 1", (annotator, item_id)).fetchone()
        return parse_ts(row["ts"]) if row else None

    def save(self, rec: dict[str, Any]) -> int:
        self._writable()
        with self._write_lock, self._conn() as c:
            prev = c.execute(
                "SELECT * FROM annotations WHERE annotator=? AND item_id=?",
                (rec["annotator"], rec["item_id"]),
            ).fetchone()
            revision = 1
            if prev is not None:
                revision = int(prev["revision"]) + 1
                cols = ANNOTATION_COLUMNS + ["archived_at"]
                c.execute(
                    f"INSERT INTO annotation_history ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                    [prev[k] for k in ANNOTATION_COLUMNS] + [utc_now()],
                )
            row = dict(rec)
            row["high_checks"] = json.dumps(rec.get("high_checks", []))
            row["medium_checks"] = json.dumps(rec.get("medium_checks", []))
            row["step_seconds"] = json.dumps(rec["step_seconds"]) if rec.get("step_seconds") else None
            row["submitted_at"] = utc_now()
            row["revision"] = revision
            c.execute(
                f"INSERT OR REPLACE INTO annotations ({','.join(ANNOTATION_COLUMNS)}) "
                f"VALUES ({','.join('?' * len(ANNOTATION_COLUMNS))})",
                [row.get(k) for k in ANNOTATION_COLUMNS],
            )
            return revision

    # ---- gates, onboarding, notes, events ----------------------------------

    def gate(self, name: str) -> bool:
        row = self._conn().execute("SELECT value FROM gates WHERE name=?", (name,)).fetchone()
        return bool(row and row["value"])

    def set_gate(self, name: str, value: bool) -> None:
        self._writable()
        with self._write_lock, self._conn() as c:
            c.execute("INSERT OR REPLACE INTO gates (name, value, updated_at) VALUES (?,?,?)",
                      (name, int(value), utc_now()))

    def acknowledged(self, annotator: str) -> bool:
        row = self._conn().execute("SELECT 1 FROM onboarding WHERE annotator=?", (annotator,)).fetchone()
        return row is not None

    def acknowledge(self, annotator: str) -> None:
        self._writable()
        with self._write_lock, self._conn() as c:
            c.execute("INSERT OR IGNORE INTO onboarding (annotator, acknowledged_at) VALUES (?,?)",
                      (annotator, utc_now()))

    def latest_review_note(self) -> str:
        row = self._conn().execute(
            "SELECT text FROM review_notes ORDER BY note_id DESC LIMIT 1").fetchone()
        return row["text"] if row else ""

    def review_notes(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._conn().execute("SELECT * FROM review_notes").fetchall()]

    def add_review_opinion(self, annotator: str, topic: str, text: str) -> None:
        """Append an annotator's pre-meeting opinion on one topic; unchanged text is not stored again."""
        self._writable()
        with self._write_lock, self._conn() as c:
            row = c.execute("SELECT text FROM review_opinions WHERE annotator=? AND topic=? "
                            "ORDER BY opinion_id DESC LIMIT 1", (annotator, topic)).fetchone()
            if (row["text"] if row else "") == text:
                return
            c.execute("INSERT INTO review_opinions (annotator, topic, text, saved_at) VALUES (?,?,?,?)",
                      (annotator, topic, text, utc_now()))

    def review_opinions(self, annotator: str | None = None) -> dict[str, dict[str, dict[str, str]]]:
        """Latest opinion per annotator and topic: {annotator: {topic: {text, saved_at}}}."""
        sql, args = "SELECT * FROM review_opinions", ()
        if annotator is not None:
            sql, args = sql + " WHERE annotator=?", (annotator,)
        out: dict[str, dict[str, dict[str, str]]] = {}
        for r in self._conn().execute(sql + " ORDER BY opinion_id", args).fetchall():
            out.setdefault(r["annotator"], {})[r["topic"]] = {"text": r["text"], "saved_at": r["saved_at"]}
        return out

    def review_opinion_history(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._conn().execute("SELECT * FROM review_opinions ORDER BY opinion_id").fetchall()]

    def add_review_note(self, text: str) -> None:
        self._writable()
        with self._write_lock, self._conn() as c:
            c.execute("INSERT INTO review_notes (text, saved_at) VALUES (?,?)", (text, utc_now()))

    def log(self, annotator: str | None, event: str, detail: str = "") -> None:
        self._writable()
        with self._write_lock, self._conn() as c:
            c.execute("INSERT INTO events (annotator, event, detail, ts) VALUES (?,?,?,?)",
                      (annotator, event, detail, utc_now()))

    def events(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._conn().execute("SELECT * FROM events").fetchall()]


def _decode(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    for k in ("high_checks", "medium_checks", "step_seconds"):
        if k in d and isinstance(d[k], str):
            d[k] = json.loads(d[k])
    return d
