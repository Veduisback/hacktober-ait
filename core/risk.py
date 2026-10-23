"""core/risk.py – Risk engine: stock-out risk, expiry risk (FEFO), transfer suggestions.

Rules (all thresholds in config.yaml):
  HIGH  – days_of_stock < new_eta_days + high_risk_gap_days
  MEDIUM – days_of_stock < new_eta_days * 1.5
  LOW   – days_of_stock >= new_eta_days * 1.5
  CRITICAL_EXPIRY – nearest expiry < critical_expiry_days
  EXPIRY_WARNING  – nearest expiry < expiry_warn_days
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import yaml

from core.forecast import ForecastResult

_CFG = yaml.safe_load(Path("config.yaml").read_text())["risk"]
HIGH_GAP: int = int(_CFG["high_risk_gap_days"])
EXPIRY_WARN: int = int(_CFG["expiry_warn_days"])
CRITICAL_EXPIRY: int = int(_CFG["critical_expiry_days"])

RiskLevel = Literal["HIGH", "MEDIUM", "LOW", "UNKNOWN"]
ExpiryRisk = Literal["CRITICAL", "WARNING", "OK", "UNKNOWN"]


@dataclass
class RiskAssessment:
    item_id: str
    item_name: str
    facility_id: str
    days_of_stock: float
    new_eta_days: int
    stock_risk: RiskLevel
    expiry_risk: ExpiryRisk
    nearest_expiry_date: str | None       # ISO-8601 or None
    days_until_expiry: int | None
    surplus_facilities: list[dict]        # [{id, name, total_qty}]
    recommended_transfer_from: str | None
    recommended_transfer_qty: float | None


def assess_stock_risk(days_of_stock: float, new_eta_days: int) -> RiskLevel:
    """Classify stock-out risk given current stock forecast and ETA delay."""
    if days_of_stock == float("inf"):
        return "LOW"
    gap = days_of_stock - new_eta_days
    if gap <= HIGH_GAP:
        return "HIGH"
    if gap <= new_eta_days * 0.5:
        return "MEDIUM"
    return "LOW"


def assess_expiry_risk(stock_rows: list[dict]) -> tuple[ExpiryRisk, str | None, int | None]:
    """
    FEFO: find the nearest expiry date across all stock batches.
    Returns (risk_level, nearest_expiry_iso, days_until_expiry).
    """
    today = date.today()
    soonest: date | None = None

    for row in stock_rows:
        exp_raw = row.get("expiry_date")
        if not exp_raw:
            continue
        try:
            exp_date = date.fromisoformat(str(exp_raw))
        except ValueError:
            continue
        if soonest is None or exp_date < soonest:
            soonest = exp_date

    if soonest is None:
        return "UNKNOWN", None, None

    days_left = (soonest - today).days
    if days_left <= CRITICAL_EXPIRY:
        risk: ExpiryRisk = "CRITICAL"
    elif days_left <= EXPIRY_WARN:
        risk = "WARNING"
    else:
        risk = "OK"

    return risk, soonest.isoformat(), days_left


def build_risk_assessment(
    forecast: ForecastResult,
    item_name: str,
    new_eta_days: int,
    stock_rows: list[dict],
    surplus_facilities: list[dict],
) -> RiskAssessment:
    """Combine forecasting, FEFO expiry, and transfer suggestion into one assessment."""
    stock_risk = assess_stock_risk(forecast.days_of_stock, new_eta_days)
    expiry_risk, nearest_expiry, days_until_expiry = assess_expiry_risk(stock_rows)

    # Suggest transfer from facility with most surplus (if stock risk is HIGH/MEDIUM)
    recommended_from: str | None = None
    recommended_qty: float | None = None
    if stock_risk in ("HIGH", "MEDIUM") and surplus_facilities:
        best = surplus_facilities[0]
        recommended_from = best["name"]
        # Suggest transferring half their surplus (conservative)
        recommended_qty = round(best["total_qty"] * 0.5, 1)

    return RiskAssessment(
        item_id=forecast.item_id,
        item_name=item_name,
        facility_id=forecast.facility_id,
        days_of_stock=forecast.days_of_stock,
        new_eta_days=new_eta_days,
        stock_risk=stock_risk,
        expiry_risk=expiry_risk,
        nearest_expiry_date=nearest_expiry,
        days_until_expiry=days_until_expiry,
        surplus_facilities=surplus_facilities,
        recommended_transfer_from=recommended_from,
        recommended_transfer_qty=recommended_qty,
    )
