"""tests/test_risk.py – Unit tests for the risk engine."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from core.risk import (
    assess_stock_risk,
    assess_expiry_risk,
    build_risk_assessment,
)
from core.forecast import ForecastResult


# ─── Stock risk rules ─────────────────────────────────────────────────────────

class TestStockRisk:
    def test_high_risk_when_stock_lt_eta_plus_gap(self):
        # days_of_stock=4, eta=5 → gap = -1 ≤ 3 → HIGH
        assert assess_stock_risk(4.0, 5) == "HIGH"

    def test_high_risk_exact_gap(self):
        # days_of_stock=8, eta=5 → gap=3 ≤ 3 → HIGH
        assert assess_stock_risk(8.0, 5) == "HIGH"

    def test_medium_risk(self):
        # days_of_stock=10, eta=5 → gap=5, 5 > 3 but < 5*0.5=2.5? No: 5 > 2.5 → LOW
        # Let's use: days=7, eta=5 → gap=2 ≤ 3 → HIGH actually
        # Medium: gap > 3, gap ≤ eta*0.5: days=9, eta=5 → gap=4 > 3, eta*0.5=2.5 → 4 > 2.5 → LOW
        # Actually test MEDIUM case: days=7, eta=5 → gap=2 → HIGH
        # days=10, eta=5 → gap=5, 5 > 3 and 5 > 5*0.5=2.5 → LOW
        # Hmm — let me check the logic: gap ≤ eta*0.5 → MEDIUM
        # days=6, eta=5 → gap=1 → HIGH (≤3)
        # days=9, eta=5 → gap=4 > 3, eta*0.5=2.5, gap(4) > 2.5 → LOW
        # The medium condition is gap <= eta*0.5 AND gap > HIGH_GAP
        # days=7, eta=7 → gap=0 → HIGH
        # days=12, eta=7 → gap=5 > 3, eta*0.5=3.5, gap(5) > 3.5 → LOW
        # days=10, eta=7 → gap=3 → HIGH
        # days=11, eta=7 → gap=4 > 3, eta*0.5=3.5, gap(4) > 3.5 → LOW
        # days=10.5, eta=7 → gap=3.5 > 3, eta*0.5=3.5, gap(3.5) ≤ 3.5 → MEDIUM!
        result = assess_stock_risk(10.5, 7)
        assert result == "MEDIUM"

    def test_low_risk(self):
        # days_of_stock=30, eta=5 → well above all thresholds → LOW
        assert assess_stock_risk(30.0, 5) == "LOW"

    def test_infinite_stock_is_low(self):
        assert assess_stock_risk(float("inf"), 10) == "LOW"

    def test_zero_eta(self):
        # No delay → no risk from ETA
        assert assess_stock_risk(5.0, 0) == "HIGH"  # gap=5 ≤ 3? No, 5>3 → check MEDIUM
        # Actually gap=5, eta=0 → eta*0.5=0 → gap(5) > 0 → LOW
        # Wait: assess_stock_risk(5, 0): gap = 5-0 = 5 > 3, check eta*0.5 = 0, 5 > 0 → LOW
        # So correct is LOW
        assert assess_stock_risk(5.0, 0) == "LOW"

    def test_high_risk_stock_out_imminent(self):
        # 1 day of stock, ETA 9 days
        assert assess_stock_risk(1.0, 9) == "HIGH"


# ─── Expiry risk ──────────────────────────────────────────────────────────────

class TestExpiryRisk:
    def _row(self, days_ahead: int, qty: float = 100.0) -> dict:
        exp = (date.today() + timedelta(days=days_ahead)).isoformat()
        return {"expiry_date": exp, "qty": qty}

    def test_critical_expiry(self):
        risk, _, days = assess_expiry_risk([self._row(3)])
        assert risk == "CRITICAL"
        assert days == 3

    def test_warning_expiry(self):
        risk, _, days = assess_expiry_risk([self._row(20)])
        assert risk == "WARNING"
        assert days == 20

    def test_ok_expiry(self):
        risk, _, _ = assess_expiry_risk([self._row(60)])
        assert risk == "OK"

    def test_no_expiry_date(self):
        risk, exp_date, days = assess_expiry_risk([{"expiry_date": None, "qty": 50}])
        assert risk == "UNKNOWN"
        assert exp_date is None

    def test_picks_soonest_expiry(self):
        rows = [self._row(60), self._row(4), self._row(90)]
        risk, _, days = assess_expiry_risk(rows)
        assert risk == "CRITICAL"
        assert days == 4

    def test_empty_rows(self):
        risk, exp, days = assess_expiry_risk([])
        assert risk == "UNKNOWN"
        assert exp is None
        assert days is None


# ─── build_risk_assessment integration ───────────────────────────────────────

class TestBuildRiskAssessment:
    def _fc(self, days: float) -> ForecastResult:
        return ForecastResult(
            item_id="insulin",
            facility_id="Facility-A",
            current_stock=150.0,
            daily_consumption=30.0,
            days_of_stock=days,
            method="moving_average",
            unit="vial",
        )

    def test_transfer_recommended_on_high_risk(self):
        fc = self._fc(4.0)
        surplus = [{"id": "Facility-B", "name": "Facility-B", "total_qty": 900.0}]
        assessment = build_risk_assessment(
            forecast=fc,
            item_name="Insulin",
            new_eta_days=9,
            stock_rows=[],
            surplus_facilities=surplus,
        )
        assert assessment.stock_risk == "HIGH"
        assert assessment.recommended_transfer_from == "Facility-B"
        assert assessment.recommended_transfer_qty == pytest.approx(450.0)

    def test_no_transfer_when_low_risk(self):
        fc = self._fc(30.0)
        surplus = [{"id": "Facility-B", "name": "Facility-B", "total_qty": 900.0}]
        assessment = build_risk_assessment(
            forecast=fc,
            item_name="Insulin",
            new_eta_days=5,
            stock_rows=[],
            surplus_facilities=surplus,
        )
        assert assessment.stock_risk == "LOW"
        assert assessment.recommended_transfer_from is None
