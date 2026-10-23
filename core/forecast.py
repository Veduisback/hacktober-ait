"""core/forecast.py – Days-of-stock-left forecasting (conventional, no LLM).

Two methods:
  1. moving_average  – simple N-day rolling mean of daily consumption.
  2. exp_smoothing   – Holt exponential smoothing via statsmodels (falls back
                       to moving average if statsmodels not available or too
                       few data points).

Returns a ForecastResult with days_of_stock, daily_consumption, and method used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

_CFG = yaml.safe_load(Path("config.yaml").read_text())["forecast"]
WINDOW_DAYS: int = int(_CFG["window_days"])
ALPHA: float = float(_CFG["alpha"])


@dataclass
class ForecastResult:
    item_id: str
    facility_id: str
    current_stock: float
    daily_consumption: float   # units/day
    days_of_stock: float       # current_stock / daily_consumption
    method: str                # "moving_average" | "exp_smoothing"
    unit: str = "units"


def _moving_average(series: pd.Series, window: int) -> float:
    """Return the rolling-window mean of the last `window` days."""
    if len(series) == 0:
        return 0.0
    tail = series.tail(window)
    return float(tail.mean())


def _exp_smoothing(series: pd.Series, alpha: float) -> float:
    """Holt simple exponential smoothing; falls back to moving average on error."""
    if len(series) < 3:
        return _moving_average(series, WINDOW_DAYS)
    try:
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing  # type: ignore

        model = SimpleExpSmoothing(series.values, initialization_method="estimated")
        fit = model.fit(smoothing_level=alpha, optimized=False)
        forecast = float(fit.forecast(1)[0])
        return max(forecast, 0.0)
    except Exception:
        return _moving_average(series, WINDOW_DAYS)


def forecast_days_of_stock(
    consumption_records: list[dict],
    current_stock: float,
    item_id: str,
    facility_id: str,
    unit: str = "units",
    method: str = "exp_smoothing",
) -> ForecastResult:
    """
    Forecast how many days of stock remain.

    Parameters
    ----------
    consumption_records : list of dicts with keys {log_date, qty}
    current_stock       : total on-hand quantity
    item_id, facility_id: identifiers
    unit                : display unit
    method              : "moving_average" | "exp_smoothing"
    """
    if not consumption_records:
        # No consumption history: assume very low usage → many days of stock
        return ForecastResult(
            item_id=item_id,
            facility_id=facility_id,
            current_stock=current_stock,
            daily_consumption=0.0,
            days_of_stock=float("inf"),
            method="no_data",
            unit=unit,
        )

    df = pd.DataFrame(consumption_records)
    df["log_date"] = pd.to_datetime(df["log_date"])
    df = df.sort_values("log_date")

    # Aggregate by day (in case of multiple entries per day)
    daily = df.groupby(df["log_date"].dt.date)["qty"].sum().reset_index()
    daily.columns = ["date", "qty"]
    series = daily["qty"].astype(float)

    if method == "exp_smoothing":
        daily_rate = _exp_smoothing(series, ALPHA)
        used_method = "exp_smoothing"
    else:
        daily_rate = _moving_average(series, WINDOW_DAYS)
        used_method = "moving_average"

    if daily_rate <= 0:
        days_left = float("inf")
    else:
        days_left = round(current_stock / daily_rate, 1)

    return ForecastResult(
        item_id=item_id,
        facility_id=facility_id,
        current_stock=current_stock,
        daily_consumption=round(daily_rate, 2),
        days_of_stock=days_left,
        method=used_method,
        unit=unit,
    )
