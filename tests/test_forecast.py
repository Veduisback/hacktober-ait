"""tests/test_forecast.py – Unit tests for the forecasting module."""

from __future__ import annotations

import math

import pytest

from core.forecast import forecast_days_of_stock, ForecastResult, _moving_average, _exp_smoothing
import pandas as pd


# ─── Moving average ───────────────────────────────────────────────────────────

class TestMovingAverage:
    def test_basic_mean(self):
        series = pd.Series([10.0, 20.0, 30.0])
        result = _moving_average(series, window=3)
        assert result == pytest.approx(20.0)

    def test_window_clips_to_tail(self):
        series = pd.Series([100.0, 10.0, 20.0, 30.0])
        # Window=3 → mean of last 3: (10+20+30)/3 = 20
        result = _moving_average(series, window=3)
        assert result == pytest.approx(20.0)

    def test_empty_series_returns_zero(self):
        result = _moving_average(pd.Series([], dtype=float), window=7)
        assert result == 0.0

    def test_single_value(self):
        result = _moving_average(pd.Series([42.0]), window=7)
        assert result == pytest.approx(42.0)


# ─── Forecast function ────────────────────────────────────────────────────────

class TestForecastDaysOfStock:
    def _make_records(self, daily_qty: float, n_days: int) -> list[dict]:
        from datetime import date, timedelta
        base = date(2026, 10, 1)
        return [
            {"log_date": (base + timedelta(days=i)).isoformat(), "qty": daily_qty}
            for i in range(n_days)
        ]

    def test_known_daily_rate(self):
        """With 300 units stock and 30 units/day consumption, expect ~10 days."""
        records = self._make_records(30.0, 14)
        result = forecast_days_of_stock(
            consumption_records=records,
            current_stock=300.0,
            item_id="insulin",
            facility_id="Facility-A",
            method="moving_average",
        )
        assert isinstance(result, ForecastResult)
        assert result.days_of_stock == pytest.approx(10.0, abs=1.0)

    def test_no_consumption_returns_inf(self):
        result = forecast_days_of_stock(
            consumption_records=[],
            current_stock=100.0,
            item_id="paracetamol",
            facility_id="Facility-A",
        )
        assert math.isinf(result.days_of_stock)
        assert result.method == "no_data"

    def test_zero_stock_returns_zero_days(self):
        records = self._make_records(10.0, 7)
        result = forecast_days_of_stock(
            consumption_records=records,
            current_stock=0.0,
            item_id="amoxicillin",
            facility_id="Facility-A",
            method="moving_average",
        )
        assert result.days_of_stock == pytest.approx(0.0, abs=0.1)

    def test_unit_preserved(self):
        records = self._make_records(5.0, 7)
        result = forecast_days_of_stock(
            consumption_records=records,
            current_stock=50.0,
            item_id="oxygen_cylinder",
            facility_id="Facility-A",
            unit="cylinder",
        )
        assert result.unit == "cylinder"

    def test_exp_smoothing_method_label(self):
        records = self._make_records(10.0, 10)
        result = forecast_days_of_stock(
            consumption_records=records,
            current_stock=100.0,
            item_id="insulin",
            facility_id="Facility-A",
            method="exp_smoothing",
        )
        assert result.method == "exp_smoothing"

    def test_forecast_result_fields(self):
        records = self._make_records(20.0, 7)
        result = forecast_days_of_stock(
            consumption_records=records,
            current_stock=100.0,
            item_id="test_item",
            facility_id="Facility-A",
        )
        assert result.item_id == "test_item"
        assert result.facility_id == "Facility-A"
        assert result.current_stock == pytest.approx(100.0)
        assert result.daily_consumption > 0
