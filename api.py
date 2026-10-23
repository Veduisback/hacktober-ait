"""api.py – FastAPI REST layer for the Hospital Supply-Chain Analyst.

Endpoints:
  POST /analyze          – Extract + forecast + risk + agent recommendation
  GET  /risks/{facility} – Risk table for all items at a facility
  GET  /stock/{facility}/{item} – Raw stock for one item
  GET  /health           – System health check

Run:  uvicorn api:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

_CFG = yaml.safe_load(Path("config.yaml").read_text())

app = FastAPI(
    title=_CFG["app"]["title"],
    description=(
        "AI-powered supply-chain disruption analysis for government hospitals. "
        "100% open-source: Ollama LLM + ChromaDB + SQLite. No paid APIs."
    ),
    version=_CFG["app"]["version"],
    license_info={"name": "MIT"},
)


# ─── Request / Response models ────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    raw_text: str
    facility_id: str = _CFG["app"]["default_facility"]
    run_agent: bool = True


class ExtractionOut(BaseModel):
    item: str
    qty: float | None
    new_eta_days: int
    reason: str
    confidence: float
    language_detected: str


class ForecastOut(BaseModel):
    days_of_stock: float
    daily_consumption: float
    method: str
    unit: str


class RiskOut(BaseModel):
    stock_risk: str
    expiry_risk: str
    nearest_expiry_date: str | None
    days_until_expiry: int | None
    surplus_facilities: list[dict]
    recommended_transfer_from: str | None
    recommended_transfer_qty: float | None


class AnalyzeResponse(BaseModel):
    facility_id: str
    extraction: ExtractionOut
    forecast: ForecastOut | None
    risk: RiskOut | None
    recommendation: str | None
    trace_steps: int
    citations: list[dict]


class RiskItem(BaseModel):
    item_id: str
    item_name: str
    on_hand: float
    unit: str
    days_of_stock: float
    stock_risk: str
    expiry_risk: str
    nearest_expiry_date: str | None


class HealthResponse(BaseModel):
    status: str
    ollama: str
    db: str
    rag_chunks: int


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["system"])
def health_check() -> HealthResponse:
    """Check that Ollama, SQLite, and ChromaDB are all operational."""
    # Ollama
    try:
        import httpx
        r = httpx.get(f"{_CFG['ollama']['base_url']}/api/version", timeout=5)
        r.raise_for_status()
        ollama_status = "ok"
    except Exception as exc:
        ollama_status = f"error: {exc}"

    # DB
    try:
        from core.db import get_all_facilities
        get_all_facilities()
        db_status = "ok"
    except Exception as exc:
        db_status = f"error: {exc}"

    # RAG
    try:
        import chromadb
        client = chromadb.PersistentClient(path=_CFG["chroma"]["persist_dir"])
        col = client.get_or_create_collection(_CFG["chroma"]["collection_name"])
        rag_chunks = col.count()
    except Exception:
        rag_chunks = -1

    overall = "healthy" if ollama_status == "ok" and db_status == "ok" else "degraded"
    return HealthResponse(
        status=overall,
        ollama=ollama_status,
        db=db_status,
        rag_chunks=rag_chunks,
    )


@app.post("/analyze", response_model=AnalyzeResponse, tags=["analysis"])
def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    """
    Full pipeline: extract disruption from raw text → forecast → risk → agent.

    - Accepts English or Hindi supplier emails / POs.
    - Returns structured extraction, stock forecast, risk assessment, and
      an AI-generated recommendation with guideline citations.
    """
    from core.extract import extract_disruption
    from core.db import get_item_by_name, get_stock, get_consumption, find_surplus_facilities
    from core.forecast import forecast_days_of_stock
    from core.risk import assess_stock_risk, assess_expiry_risk, build_risk_assessment

    # Step 1 – Extract
    try:
        ext = extract_disruption(req.raw_text)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Extraction failed: {exc}")

    extraction_out = ExtractionOut(
        item=ext.item,
        qty=ext.qty,
        new_eta_days=ext.new_eta_days,
        reason=ext.reason,
        confidence=ext.confidence,
        language_detected=ext.language_detected,
    )

    # Step 2 – Forecast + Risk
    forecast_out: ForecastOut | None = None
    risk_out: RiskOut | None = None

    item_rec = get_item_by_name(ext.item)
    if item_rec:
        stock_rows = get_stock(req.facility_id, item_rec["id"])
        total_stock = sum(r["qty"] for r in stock_rows)
        consumption = get_consumption(req.facility_id, item_rec["id"], days=30)
        fc = forecast_days_of_stock(
            consumption_records=consumption,
            current_stock=total_stock,
            item_id=item_rec["id"],
            facility_id=req.facility_id,
            unit=item_rec.get("unit", "units"),
        )
        forecast_out = ForecastOut(
            days_of_stock=fc.days_of_stock if fc.days_of_stock != float("inf") else 9999,
            daily_consumption=fc.daily_consumption,
            method=fc.method,
            unit=fc.unit,
        )

        stock_risk = assess_stock_risk(fc.days_of_stock, ext.new_eta_days)
        expiry_risk, nearest_expiry, days_until_expiry = assess_expiry_risk(stock_rows)
        surplus = find_surplus_facilities(item_rec["id"], req.facility_id)

        rec_from = surplus[0]["name"] if surplus and stock_risk in ("HIGH", "MEDIUM") else None
        rec_qty = round(surplus[0]["total_qty"] * 0.5, 1) if rec_from else None

        risk_out = RiskOut(
            stock_risk=stock_risk,
            expiry_risk=expiry_risk,
            nearest_expiry_date=nearest_expiry,
            days_until_expiry=days_until_expiry,
            surplus_facilities=surplus,
            recommended_transfer_from=rec_from,
            recommended_transfer_qty=rec_qty,
        )

    # Step 3 – Agent (optional)
    recommendation: str | None = None
    trace_steps = 0
    citations: list[dict] = []

    if req.run_agent:
        try:
            from core.agent import run_agent
            agent_result = run_agent(ext, req.facility_id)
            recommendation = agent_result.recommendation
            trace_steps = len(agent_result.trace)
            citations = agent_result.citations
        except Exception as exc:
            recommendation = f"Agent unavailable: {exc}"

    return AnalyzeResponse(
        facility_id=req.facility_id,
        extraction=extraction_out,
        forecast=forecast_out,
        risk=risk_out,
        recommendation=recommendation,
        trace_steps=trace_steps,
        citations=citations,
    )


@app.get("/risks/{facility_id}", response_model=list[RiskItem], tags=["analysis"])
def get_risks(facility_id: str) -> list[RiskItem]:
    """Get the current risk table for all items at a facility."""
    from core.db import get_all_stock, get_consumption
    from core.forecast import forecast_days_of_stock
    from core.risk import assess_stock_risk, assess_expiry_risk

    stock_rows = get_all_stock(facility_id)
    if not stock_rows:
        raise HTTPException(status_code=404, detail=f"No stock found for facility '{facility_id}'")

    items_seen: dict[str, Any] = {}
    for row in stock_rows:
        iid = row["item_id"]
        if iid not in items_seen:
            items_seen[iid] = {"name": row["item_name"], "unit": row["unit"],
                               "total_qty": 0.0, "rows": []}
        items_seen[iid]["total_qty"] += row["qty"]
        items_seen[iid]["rows"].append(row)

    result = []
    for iid, info in items_seen.items():
        consumption = get_consumption(facility_id, iid, days=30)
        fc = forecast_days_of_stock(
            consumption_records=consumption,
            current_stock=info["total_qty"],
            item_id=iid,
            facility_id=facility_id,
            unit=info["unit"],
        )
        stock_risk = assess_stock_risk(fc.days_of_stock, 0)
        expiry_risk, nearest_expiry, _ = assess_expiry_risk(info["rows"])
        result.append(RiskItem(
            item_id=iid,
            item_name=info["name"],
            on_hand=info["total_qty"],
            unit=info["unit"],
            days_of_stock=fc.days_of_stock if fc.days_of_stock != float("inf") else 9999,
            stock_risk=stock_risk,
            expiry_risk=expiry_risk,
            nearest_expiry_date=nearest_expiry,
        ))

    return result


@app.get("/stock/{facility_id}/{item_id}", response_model=list[dict], tags=["stock"])
def get_stock_endpoint(facility_id: str, item_id: str) -> list[dict]:
    """Get raw stock records for a specific item at a facility."""
    from core.db import get_stock
    rows = get_stock(facility_id, item_id)
    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No stock for item '{item_id}' at facility '{facility_id}'",
        )
    return rows
