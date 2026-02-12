"""Metrics and schedule assembly helpers for HeatingOptimizer."""

from __future__ import annotations

from datetime import datetime, time
from typing import Any, Callable


def estimate_temp_range(hours: list[Any]) -> tuple[float, float]:
    """Estimate min and max temperatures from hourly plan."""
    temps = [h.expected_room_temp for h in hours]
    return min(temps), max(temps)


def estimate_gas_usage(hours: list[Any], gas_base_rate: float) -> float:
    """Estimate gas usage based on modulation and hours."""
    total_kwh = 0.0
    for hour in hours:
        if hour.system_state == "on":
            # Scale by modulation (higher modulation = more gas)
            hourly_kwh = gas_base_rate * (hour.expected_modulation / 50)
            total_kwh += hourly_kwh

    return round(total_kwh, 1)


def calculate_burner_hours(
    switch_on_time: time,
    switch_off_time: time | None,
) -> float:
    """Calculate expected burner operation hours."""
    if switch_off_time is None:
        # Continuous heating - 24 hours
        return 24.0

    on_mins = switch_on_time.hour * 60 + switch_on_time.minute
    off_mins = switch_off_time.hour * 60 + switch_off_time.minute

    if off_mins > on_mins:
        # Same day: simple subtraction
        duration_mins = off_mins - on_mins
    else:
        # Crosses midnight: on_time to midnight + midnight to off_time
        duration_mins = (24 * 60 - on_mins) + off_mins

    return round(duration_mins / 60, 1)


def calculate_avg_modulation(hours: list[Any]) -> float:
    """Calculate average modulation when system is ON."""
    on_hours = [h for h in hours if h.system_state == "on"]
    if not on_hours:
        return 0.0

    total_mod = sum(h.expected_modulation for h in on_hours)
    return round(total_mod / len(on_hours), 1)


def find_temp_at_hour(hours: list[Any], target_hour: int) -> float | None:
    """Find expected room temp at a specific hour in the hourly plan."""
    for hour in hours:
        if hour.hour == target_hour:
            return hour.expected_room_temp
    return None


def build_schedule(
    *,
    current_time: datetime,
    hours: list[Any],
    switch_on_time: time,
    switch_off_time: time | None,
    optimal_setpoint: float,
    solar_contribution: float,
    reasoning: list[str],
    expected_switch_on_temp: float | None,
    target_warm_time: time,
    gas_base_rate: float,
    schedule_factory: Callable[..., Any],
) -> Any:
    """Build final schedule object with computed metrics."""
    expected_min, expected_max = estimate_temp_range(hours)

    return schedule_factory(
        date=current_time,
        hours=hours,
        switch_on_time=switch_on_time,
        switch_off_time=switch_off_time,
        optimal_setpoint=optimal_setpoint,
        cycles_per_day=1,
        expected_gas_usage=estimate_gas_usage(hours, gas_base_rate),
        expected_min_temp=expected_min,
        expected_max_temp=expected_max,
        solar_contribution=solar_contribution,
        reasoning=reasoning,
        expected_switch_on_temp=expected_switch_on_temp,
        expected_target_time_temp=find_temp_at_hour(hours, target_warm_time.hour),
        expected_switch_off_temp=(
            find_temp_at_hour(hours, switch_off_time.hour)
            if switch_off_time is not None
            else None
        ),
        expected_burner_hours=calculate_burner_hours(switch_on_time, switch_off_time),
        expected_avg_modulation=calculate_avg_modulation(hours),
    )
