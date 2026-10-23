"""tests/test_db.py – Database CRUD and schema tests (uses a temp in-memory DB)."""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# Patch the DB path before importing db module
@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Redirect all DB operations to a temp file for each test."""
    db_file = tmp_path / "test.db"
    monkeypatch.setattr("core.db._DB_PATH", db_file)
    # Re-init schema
    from core.db import init_db
    init_db()
    yield db_file


def _seed_minimal():
    """Insert minimal data needed for tests."""
    from core.db import _conn
    with _conn() as con:
        con.execute("INSERT OR IGNORE INTO facilities VALUES ('F-A', 'Facility A')")
        con.execute("INSERT OR IGNORE INTO items VALUES ('insulin', 'Insulin', 'vial')")
        con.execute("INSERT OR IGNORE INTO departments VALUES ('icu', 'ICU')")
        con.execute(
            "INSERT INTO stock (facility_id, item_id, qty, batch_no, expiry_date, as_of_date) "
            "VALUES ('F-A', 'insulin', 200.0, 'B001', '2027-06-01', '2026-10-23')"
        )
        con.execute(
            "INSERT INTO consumption (facility_id, item_id, qty, log_date) "
            "VALUES ('F-A', 'insulin', 30.0, '2026-10-22')"
        )
        con.execute(
            "INSERT OR IGNORE INTO item_dept_map VALUES ('insulin', 'icu')"
        )


class TestDBSchema:
    def test_tables_created(self, temp_db):
        con = sqlite3.connect(str(temp_db))
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        con.close()
        expected = {"facilities", "items", "stock", "consumption",
                    "departments", "item_dept_map", "disruption_events"}
        assert expected.issubset(tables)

    def test_init_idempotent(self):
        from core.db import init_db
        init_db()
        init_db()  # second call should not raise


class TestStockCRUD:
    def test_get_stock(self):
        _seed_minimal()
        from core.db import get_stock
        rows = get_stock("F-A", "insulin")
        assert len(rows) == 1
        assert rows[0]["qty"] == pytest.approx(200.0)

    def test_get_all_stock(self):
        _seed_minimal()
        from core.db import get_all_stock
        rows = get_all_stock("F-A")
        assert len(rows) >= 1
        assert all(r["facility_id"] == "F-A" for r in rows)

    def test_get_stock_empty(self):
        from core.db import get_stock
        rows = get_stock("F-A", "nonexistent")
        assert rows == []


class TestConsumption:
    def test_get_consumption(self):
        _seed_minimal()
        from core.db import get_consumption
        rows = get_consumption("F-A", "insulin", days=30)
        assert len(rows) == 1
        assert rows[0]["qty"] == pytest.approx(30.0)

    def test_get_consumption_empty(self):
        from core.db import get_consumption
        rows = get_consumption("F-A", "nonexistent", days=30)
        assert rows == []


class TestDepartments:
    def test_get_affected_departments(self):
        _seed_minimal()
        from core.db import get_affected_departments
        depts = get_affected_departments("insulin")
        assert len(depts) == 1
        assert depts[0]["id"] == "icu"

    def test_get_affected_departments_unknown_item(self):
        from core.db import get_affected_departments
        depts = get_affected_departments("unknown_item")
        assert depts == []


class TestSurplus:
    def test_find_surplus_facilities(self):
        _seed_minimal()
        from core.db import _conn, find_surplus_facilities
        # Add stock at another facility
        with _conn() as con:
            con.execute("INSERT OR IGNORE INTO facilities VALUES ('F-B', 'Facility B')")
            con.execute(
                "INSERT INTO stock (facility_id, item_id, qty, as_of_date) "
                "VALUES ('F-B', 'insulin', 900.0, '2026-10-23')"
            )
        surplus = find_surplus_facilities("insulin", "F-A")
        assert len(surplus) == 1
        assert surplus[0]["id"] == "F-B"

    def test_find_surplus_excludes_current_facility(self):
        _seed_minimal()
        from core.db import find_surplus_facilities
        surplus = find_surplus_facilities("insulin", "F-A")
        assert all(f["id"] != "F-A" for f in surplus)


class TestItemLookup:
    def test_get_item_by_name_case_insensitive(self):
        _seed_minimal()
        from core.db import get_item_by_name
        item = get_item_by_name("INSULIN")
        assert item is not None
        assert item["id"] == "insulin"

    def test_get_item_by_name_not_found(self):
        from core.db import get_item_by_name
        assert get_item_by_name("nonexistent_drug") is None


class TestDisruptionLog:
    def test_log_disruption(self):
        _seed_minimal()
        from core.db import log_disruption, _conn
        row_id = log_disruption(
            facility_id="F-A",
            item_id="insulin",
            raw_text="Test email",
            extracted='{"item":"Insulin"}',
            eta_days=9,
            risk_level="HIGH",
        )
        assert row_id > 0
        with _conn() as con:
            row = con.execute(
                "SELECT * FROM disruption_events WHERE id = ?", (row_id,)
            ).fetchone()
        assert row["risk_level"] == "HIGH"
        assert row["eta_days"] == 9
