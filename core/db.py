"""core/db.py – SQLite schema, connection, and CRUD helpers."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Generator

import yaml

_CFG = yaml.safe_load(Path("config.yaml").read_text())
_DB_PATH = Path(_CFG["database"]["sqlite_path"])


@contextmanager
def _conn() -> Generator[sqlite3.Connection, None, None]:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(_DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


# ─── Schema ───────────────────────────────────────────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS facilities (
    id   TEXT PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS items (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    unit        TEXT NOT NULL DEFAULT 'units'
);

CREATE TABLE IF NOT EXISTS stock (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    facility_id     TEXT    NOT NULL REFERENCES facilities(id),
    item_id         TEXT    NOT NULL REFERENCES items(id),
    qty             REAL    NOT NULL,
    batch_no        TEXT,
    expiry_date     TEXT,   -- ISO-8601 date or NULL
    as_of_date      TEXT    NOT NULL  -- ISO-8601 date
);

CREATE TABLE IF NOT EXISTS consumption (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    facility_id TEXT NOT NULL REFERENCES facilities(id),
    item_id     TEXT NOT NULL REFERENCES items(id),
    qty         REAL NOT NULL,
    log_date    TEXT NOT NULL  -- ISO-8601 date
);

CREATE TABLE IF NOT EXISTS departments (
    id   TEXT PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS item_dept_map (
    item_id TEXT NOT NULL REFERENCES items(id),
    dept_id TEXT NOT NULL REFERENCES departments(id),
    PRIMARY KEY (item_id, dept_id)
);

CREATE TABLE IF NOT EXISTS disruption_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    facility_id TEXT NOT NULL,
    item_id     TEXT NOT NULL,
    raw_text    TEXT NOT NULL,
    extracted   TEXT,          -- JSON blob
    eta_days    INTEGER,
    risk_level  TEXT,
    created_at  TEXT NOT NULL
);
"""


def init_db() -> None:
    """Create all tables (idempotent)."""
    with _conn() as con:
        con.executescript(DDL)


# ─── Reads ────────────────────────────────────────────────────────────────────

def get_stock(facility_id: str, item_id: str) -> list[dict]:
    sql = """
        SELECT s.*, i.name AS item_name, i.unit
        FROM stock s JOIN items i ON s.item_id = i.id
        WHERE s.facility_id = ? AND s.item_id = ?
        ORDER BY s.expiry_date ASC
    """
    with _conn() as con:
        rows = con.execute(sql, (facility_id, item_id)).fetchall()
    return [dict(r) for r in rows]


def get_all_stock(facility_id: str) -> list[dict]:
    sql = """
        SELECT s.*, i.name AS item_name, i.unit
        FROM stock s JOIN items i ON s.item_id = i.id
        WHERE s.facility_id = ?
        ORDER BY i.name, s.expiry_date ASC
    """
    with _conn() as con:
        rows = con.execute(sql, (facility_id,)).fetchall()
    return [dict(r) for r in rows]


def get_consumption(facility_id: str, item_id: str, days: int = 30) -> list[dict]:
    sql = """
        SELECT * FROM consumption
        WHERE facility_id = ? AND item_id = ?
          AND log_date >= date('now', ?)
        ORDER BY log_date ASC
    """
    with _conn() as con:
        rows = con.execute(sql, (facility_id, item_id, f"-{days} days")).fetchall()
    return [dict(r) for r in rows]


def get_affected_departments(item_id: str) -> list[dict]:
    sql = """
        SELECT d.id, d.name FROM departments d
        JOIN item_dept_map m ON d.id = m.dept_id
        WHERE m.item_id = ?
    """
    with _conn() as con:
        rows = con.execute(sql, (item_id,)).fetchall()
    return [dict(r) for r in rows]


def find_surplus_facilities(item_id: str, exclude_facility: str) -> list[dict]:
    """Return facilities (other than exclude_facility) that have stock for item_id."""
    sql = """
        SELECT f.id, f.name, SUM(s.qty) AS total_qty
        FROM stock s
        JOIN facilities f ON s.facility_id = f.id
        WHERE s.item_id = ? AND s.facility_id != ?
        GROUP BY f.id, f.name
        HAVING total_qty > 0
        ORDER BY total_qty DESC
    """
    with _conn() as con:
        rows = con.execute(sql, (item_id, exclude_facility)).fetchall()
    return [dict(r) for r in rows]


def get_all_facilities() -> list[dict]:
    with _conn() as con:
        rows = con.execute("SELECT * FROM facilities").fetchall()
    return [dict(r) for r in rows]


def get_all_items() -> list[dict]:
    with _conn() as con:
        rows = con.execute("SELECT * FROM items").fetchall()
    return [dict(r) for r in rows]


def get_item_by_name(name: str) -> dict | None:
    sql = "SELECT * FROM items WHERE LOWER(name) = LOWER(?)"
    with _conn() as con:
        row = con.execute(sql, (name,)).fetchone()
    return dict(row) if row else None


# ─── Writes ───────────────────────────────────────────────────────────────────

def log_disruption(
    facility_id: str,
    item_id: str,
    raw_text: str,
    extracted: str,
    eta_days: int | None,
    risk_level: str,
) -> int:
    sql = """
        INSERT INTO disruption_events
          (facility_id, item_id, raw_text, extracted, eta_days, risk_level, created_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
    """
    with _conn() as con:
        cur = con.execute(sql, (facility_id, item_id, raw_text, extracted, eta_days, risk_level))
        return cur.lastrowid  # type: ignore[return-value]


def update_stock(facility_id: str, item_id: str, new_qty: float, as_of: str) -> None:
    """Upsert: update the latest stock row's qty."""
    sql = """
        UPDATE stock SET qty = ?
        WHERE facility_id = ? AND item_id = ?
          AND as_of_date = (
              SELECT MAX(as_of_date) FROM stock
              WHERE facility_id = ? AND item_id = ?
          )
    """
    with _conn() as con:
        con.execute(sql, (new_qty, facility_id, item_id, facility_id, item_id))
