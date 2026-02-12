"""Hourly plan construction helpers for HeatingOptimizer."""

from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Any, Callable


def build_hourly_plan(
    *,
    thermal_model,
    switch_on_time: time,
    switch_off_time: time | None,
    optimal_setpoint: float,
    bedroom_temp: float,
    outside_temp: float,
    weather_forecast: list[dict] | None,
    current_time: datetime | None,
    plan_factory: Callable[..., Any],
) -> list[Any]:
    """Build a 24-hour heating plan."""
    plans = []
    current_temp = bedroom_temp
    if current_time is None:
        current_time = datetime.now()
    start_dt = current_time.replace(minute=0, second=0, microsecond=0)

    # Build hourly outside temp lookup from forecast
    hourly_outside: dict[datetime, float] = {}
    if weather_forecast:
        for entry in weather_forecast:
            dt_str = entry.get("datetime", "")
            if dt_str:
                try:
                    dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                    if dt.tzinfo:
                        dt = dt.replace(tzinfo=None)
                    temp = entry.get("temperature")
                    if temp is not None:
                        hourly_outside[
                            dt.replace(minute=0, second=0, microsecond=0)
                        ] = temp
                except ValueError:
                    pass

    for step in range(24):
        sim_dt = start_dt + timedelta(hours=step)
        hour = sim_dt.hour
        # Determine if heating should be on
        hour_time = time(hour, 0)

        if switch_off_time is None:
            # Continuous heating - always on
            system_on = True
        elif switch_on_time <= switch_off_time:
            # Normal case (e.g., on at 05:00, off at 22:00)
            system_on = switch_on_time <= hour_time < switch_off_time
        else:
            # Wrap-around case (e.g., on at 22:00, off at 06:00)
            system_on = hour_time >= switch_on_time or hour_time < switch_off_time

        # Use per-hour outside temp from forecast, fall back to flat value
        hour_outside = hourly_outside.get(sim_dt, outside_temp)

        if system_on:
            # Predict modulation
            modulation = thermal_model.predict_modulation(
                outside_temp=hour_outside,
                setpoint=optimal_setpoint,
                room_temp=current_temp,
            )
            # Heating: use effective rate consistent with predict_heating_duration
            base_rate = max(thermal_model.mean_heating_rate, 1.0)
            gap = max(0, optimal_setpoint + 0.5 - current_temp)
            current_temp += base_rate * min(1.0, gap / 2.0)
            # Room won't exceed setpoint + small overshoot
            current_temp = min(current_temp, optimal_setpoint + 0.5)
        else:
            modulation = 0
            # Physics-based cooling: Newton's law dT/dt = -k * (T_in - T_out)
            current_temp -= thermal_model.k * (current_temp - hour_outside)

        current_temp = max(15, min(25, current_temp))

        plans.append(
            plan_factory(
                hour=hour,
                system_state="on" if system_on else "off",
                setpoint=optimal_setpoint if system_on else None,
                expected_modulation=round(modulation, 1),
                expected_room_temp=round(current_temp, 1),
            )
        )

    return plans
