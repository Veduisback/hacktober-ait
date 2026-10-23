"""seed.py – Populate the database and vector store with synthetic data.

⚠️  SYNTHETIC DATA NOTICE: All data in this project is entirely fictional.
    No real patient data, procurement records, or supplier information is used.
    This is a demonstration system only.

Usage:
    python seed.py            # seed DB + build RAG index
    python seed.py --reset    # drop and re-create everything
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _check_deps() -> None:
    missing = []
    for pkg in ["yaml", "chromadb", "pandas"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[seed] Missing packages: {missing}. Run: pip install -r requirements.txt")
        sys.exit(1)


def _check_ollama() -> None:
    try:
        import httpx
        import yaml
        cfg = yaml.safe_load(Path("config.yaml").read_text())["ollama"]
        r = httpx.get(f"{cfg['base_url']}/api/version", timeout=5)
        r.raise_for_status()
        print(f"[seed] Ollama ✓  (base_url={cfg['base_url']})")
    except Exception as exc:
        print(f"\n[seed] ⚠️  Ollama not reachable: {exc}")
        print("       Start Ollama first: run `ollama serve` in a terminal.")
        print("       Skipping RAG index build — re-run seed.py after starting Ollama.\n")
        return


def seed_database(reset: bool = False) -> None:
    import sqlite3
    from core.db import init_db, _DB_PATH, _conn

    if reset and _DB_PATH.exists():
        _DB_PATH.unlink()
        print(f"[seed] Removed existing DB: {_DB_PATH}")

    print("[seed] Initialising schema …")
    init_db()

    with _conn() as con:
        # ── Facilities ────────────────────────────────────────────────────────
        facilities = [
            ("Facility-A", "District Hospital – City Centre"),
            ("Facility-B", "Community Health Centre – North"),
            ("Facility-C", "Community Health Centre – East"),
        ]
        con.executemany(
            "INSERT OR IGNORE INTO facilities (id, name) VALUES (?, ?)", facilities
        )

        # ── Items ─────────────────────────────────────────────────────────────
        items = [
            ("insulin",        "Insulin (Regular 100 IU/mL)",  "vial"),
            ("paracetamol",    "Paracetamol 500mg",            "tablet"),
            ("amoxicillin",    "Amoxicillin 500mg",            "capsule"),
            ("iv_saline",      "IV Normal Saline 0.9% 500mL",  "bag"),
            ("surgical_gloves","Surgical Gloves (sterile)",     "pair"),
            ("metformin",      "Metformin 500mg",              "tablet"),
            ("oxygen_cylinder","Oxygen Cylinder (B-type)",      "cylinder"),
        ]
        con.executemany(
            "INSERT OR IGNORE INTO items (id, name, unit) VALUES (?, ?, ?)", items
        )

        # ── Departments ───────────────────────────────────────────────────────
        departments = [
            ("icu",           "Intensive Care Unit (ICU)"),
            ("medicine",      "General Medicine Ward"),
            ("surgery",       "Surgical Ward"),
            ("emergency",     "Emergency / Casualty"),
            ("endocrinology", "Endocrinology / Diabetic Clinic"),
            ("pediatrics",    "Pediatric Ward"),
            ("opd",           "Outpatient Department (OPD)"),
            ("maternity",     "Maternity / Labour Ward"),
        ]
        con.executemany(
            "INSERT OR IGNORE INTO departments (id, name) VALUES (?, ?)", departments
        )

        # ── Item → Department map ──────────────────────────────────────────
        item_dept = [
            # insulin → ICU, Medicine, Surgery, Emergency, Endocrinology
            ("insulin", "icu"), ("insulin", "medicine"), ("insulin", "surgery"),
            ("insulin", "emergency"), ("insulin", "endocrinology"),
            # paracetamol → everywhere
            ("paracetamol", "medicine"), ("paracetamol", "pediatrics"),
            ("paracetamol", "opd"), ("paracetamol", "surgery"),
            ("paracetamol", "emergency"), ("paracetamol", "maternity"),
            # amoxicillin → medicine, pediatrics, OPD, surgery
            ("amoxicillin", "medicine"), ("amoxicillin", "pediatrics"),
            ("amoxicillin", "opd"), ("amoxicillin", "surgery"),
            # iv_saline → all clinical
            ("iv_saline", "icu"), ("iv_saline", "surgery"), ("iv_saline", "medicine"),
            ("iv_saline", "emergency"), ("iv_saline", "maternity"),
            # surgical_gloves → all clinical
            ("surgical_gloves", "icu"), ("surgical_gloves", "surgery"),
            ("surgical_gloves", "emergency"), ("surgical_gloves", "medicine"),
            ("surgical_gloves", "maternity"),
            # metformin → endocrinology, medicine, OPD
            ("metformin", "endocrinology"), ("metformin", "medicine"), ("metformin", "opd"),
            # oxygen → ICU, emergency, surgery
            ("oxygen_cylinder", "icu"), ("oxygen_cylinder", "emergency"),
            ("oxygen_cylinder", "surgery"), ("oxygen_cylinder", "maternity"),
        ]
        con.executemany(
            "INSERT OR IGNORE INTO item_dept_map (item_id, dept_id) VALUES (?, ?)", item_dept
        )

        # ── Stock ─────────────────────────────────────────────────────────────
        stock_file = Path("data/stock.csv")
        with stock_file.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                con.execute(
                    """INSERT INTO stock
                       (facility_id, item_id, qty, batch_no, expiry_date, as_of_date)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        row["facility_id"],
                        row["item_id"],
                        float(row["qty"]),
                        row["batch_no"] or None,
                        row["expiry_date"] or None,
                        row["as_of_date"],
                    ),
                )

        # ── Consumption ───────────────────────────────────────────────────────
        consumption_file = Path("data/consumption.csv")
        with consumption_file.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                con.execute(
                    """INSERT INTO consumption
                       (facility_id, item_id, qty, log_date)
                       VALUES (?, ?, ?, ?)""",
                    (row["facility_id"], row["item_id"], float(row["qty"]), row["log_date"]),
                )

    print("[seed] Database seeded ✓")
    print(f"       → {len(facilities)} facilities, {len(items)} items, "
          f"{len(departments)} departments")


def build_rag_index(force: bool = False) -> None:
    print("[seed] Building RAG index …")
    try:
        from core.rag import build_index
        n = build_index(force=force)
        print(f"[seed] RAG index built ✓  ({n} chunks indexed)")
    except Exception as exc:
        print(f"[seed] ⚠️  RAG index failed: {exc}")
        print("       Ensure Ollama is running and nomic-embed-text is pulled.")
        print("       Run: ollama pull nomic-embed-text")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the hospital supply-chain demo database.")
    parser.add_argument("--reset", action="store_true", help="Drop and re-create the database.")
    parser.add_argument("--db-only", action="store_true", help="Seed DB only, skip RAG index.")
    args = parser.parse_args()

    print("\n⚠️  NOTICE: All data is SYNTHETIC and fictional. No real data is used.\n")

    _check_deps()
    seed_database(reset=args.reset)

    if not args.db_only:
        _check_ollama()
        build_rag_index(force=args.reset)

    print("\n[seed] Done! Run: streamlit run app.py\n")


if __name__ == "__main__":
    main()
