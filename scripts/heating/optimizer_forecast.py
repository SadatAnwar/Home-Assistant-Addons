"""Forecast extraction helpers for HeatingOptimizer."""

from __future__ import annotations

import logging
from datetime import datetime, time

from .config import DEFAULTS

logger = logging.getLogger(__name__)


def extract_overnight_temps(
    forecast: list[dict] | None,
    default: float,
) -> list[float]:
    """Extract overnight temperature predictions from forecast."""
    if not forecast:
        logger.debug(f"No forecast available, using default temp: {default}°C")
        return [default] * 8

    temps = []
    for entry in forecast[:8]:  # Next 8 hours
        temp = entry.get("temperature")
        if temp is not None:
            temps.append(temp)

    if not temps:
        logger.debug(f"No temps in forecast, using default: {default}°C")
        return [default] * 8

    if DEFAULTS.forecast_temp_bias != 0:
        temps = [t + DEFAULTS.forecast_temp_bias for t in temps]
    logger.debug(f"Extracted overnight temps from forecast: {temps}")
    return temps


def extract_forecast_temp_at_time(
    forecast: list[dict] | None,
    target_time: time,
    default: float,
) -> float:
    """Extract forecast temperature at a specific time of day."""
    if not forecast:
        logger.debug(f"No forecast for {target_time}, using default: {default}°C")
        return default

    target_hour = target_time.hour

    for entry in forecast:
        dt_str = entry.get("datetime", "")
        if not dt_str:
            continue

        try:
            # Parse forecast datetime
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            # Convert to local time if needed (naive comparison)
            forecast_hour = dt.hour

            # Find entry matching target hour (today or tomorrow)
            if forecast_hour == target_hour:
                temp = entry.get("temperature")
                if temp is not None:
                    temp += DEFAULTS.forecast_temp_bias
                    logger.debug(f"Forecast at {target_time}: {temp}°C (from {dt_str})")
                    return temp
        except ValueError:
            continue

    logger.debug(f"No forecast match for {target_time}, using default: {default}°C")
    return default


def extract_daytime_temps(
    forecast: list[dict] | None,
    morning_time: time,
    evening_time: time,
    default: float,
) -> list[float]:
    """Extract forecast temperatures for daytime hours."""
    if not forecast:
        return [default]

    temps = []
    for entry in forecast:
        dt_str = entry.get("datetime", "")
        if not dt_str:
            continue

        try:
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            hour = dt.hour

            # Check if within daytime hours
            if morning_time.hour <= hour <= evening_time.hour:
                temp = entry.get("temperature")
                if temp is not None:
                    temps.append(temp)
        except ValueError:
            continue

    if temps:
        if DEFAULTS.forecast_temp_bias != 0:
            temps = [t + DEFAULTS.forecast_temp_bias for t in temps]
        logger.debug(
            f"Daytime forecast temps ({morning_time}-{evening_time}): min={min(temps)}, max={max(temps)}, avg={sum(temps) / len(temps):.1f}"
        )
        return temps

    return [default]


def estimate_solar_contribution(
    forecast: list[dict] | None,
    morning_time: time,
    evening_time: time,
) -> float:
    """Estimate solar heat contribution based on forecast."""
    if not forecast:
        return 0.0

    total_contribution = 0.0

    for entry in forecast:
        condition = entry.get("condition", "").lower()

        # Check if during daytime
        dt_str = entry.get("datetime", "")
        if dt_str:
            try:
                dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                if dt.hour < morning_time.hour or dt.hour > evening_time.hour:
                    continue
            except ValueError:
                continue

        # Estimate solar contribution based on condition
        if condition in ("sunny", "clear"):
            total_contribution += 0.5
        elif condition in ("partlycloudy", "partly_cloudy"):
            total_contribution += 0.2
        elif condition in ("cloudy", "overcast"):
            total_contribution += 0.0

    return min(total_contribution, 3.0)  # Cap at 3°C
