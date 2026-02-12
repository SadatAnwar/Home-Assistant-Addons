"""Case resolution logic for heating scheduler."""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Any, Callable

from .optimizer import DailyHeatingSchedule, HeatingOptimizer

logger = logging.getLogger(__name__)


class UpdateMode(Enum):
    """What HA helpers to update after schedule calculation."""

    ALL = "all"
    SWITCH_OFF_AND_SETPOINT = "switch_off_and_setpoint"


def time_ge(a: time, b: time) -> bool:
    """Check if time a >= time b (simple comparison, no midnight wrapping)."""
    return (a.hour, a.minute) >= (b.hour, b.minute)


def determine_and_calculate(
    *,
    now: datetime,
    existing_record: Any,
    heating_is_on: bool,
    target_warm_time: time,
    target_night_time: time,
    settings: dict[str, Any],
    current_state: dict[str, Any],
    optimizer: HeatingOptimizer,
    parse_time: Callable[[str], time],
) -> tuple[Any, UpdateMode, str]:
    """Determine the current case and calculate the appropriate schedule."""
    room_temps = current_state.get("room_temps", {})
    outside_temp = current_state.get("outside_temp", 5)
    forecast = current_state.get("forecast")

    common_kwargs = {
        "target_warm_time": target_warm_time,
        "target_night_time": target_night_time,
        "target_temp": settings["target_temp"],
        "min_overnight_temp": settings["min_bedroom_temp"],
        "min_daytime_temp": settings["min_daytime_temp"],
        "current_temps": room_temps,
        "outside_temp": outside_temp,
        "weather_forecast": forecast,
    }

    # CASE A: No prediction exists for today
    if existing_record is None:
        logger.info("Case A: First run of the day — full calculation")
        schedule = optimizer.calculate_optimal_schedule(**common_kwargs)
        return schedule, UpdateMode.ALL, "A"

    # We have an existing prediction — determine sub-case
    pred = existing_record.prediction
    switch_on_time = parse_time(pred.switch_on_time)
    switch_off_time = (
        parse_time(pred.switch_off_time) if pred.switch_off_time != "CONTINUOUS" else None
    )

    current_time = now.time()

    if heating_is_on:
        # CASE D: Heating is ON — mid-day recalculation
        logger.info("Case D: Heating ON — recalculating switch-off and setpoint")
        schedule = optimizer.recalculate_mid_day(
            **common_kwargs,
            current_time=now,
            original_switch_on_time=switch_on_time,
        )
        return schedule, UpdateMode.SWITCH_OFF_AND_SETPOINT, "D"

    # Heating is OFF
    if switch_off_time is not None and time_ge(current_time, switch_off_time):
        # Past switch-off time — could be after today's cycle or before tomorrow's
        # Check if we're past the preferred off time too
        if time_ge(current_time, target_night_time):
            # CASE E: Heating done for today — this is effectively tomorrow's calc
            logger.info(
                "Case E: Past switch-off and night time — " "calculating tomorrow's schedule"
            )
            tomorrow = now + timedelta(days=1)
            schedule = optimizer.calculate_optimal_schedule(
                **common_kwargs,
                current_time=tomorrow,
            )
            return schedule, UpdateMode.ALL, "E"

    # Heating OFF, before switch-on time
    if not time_ge(current_time, switch_on_time):
        # CASE B: Before heating starts — recalculate with fresh conditions
        logger.info("Case B: Before switch-on — recalculating with fresh data")
        schedule = optimizer.calculate_optimal_schedule(**common_kwargs)
        return schedule, UpdateMode.ALL, "B"

    # Heating OFF but past switch-on time and before switch-off
    if time_ge(current_time, switch_on_time) and (
        switch_off_time is None or not time_ge(current_time, switch_off_time)
    ):
        if not time_ge(current_time, target_warm_time):
            # CASE C: Should have started but hasn't — trigger immediately
            logger.info("Case C: Past switch-on but heating OFF — triggering now+2min")
            immediate_on = (now + timedelta(minutes=2)).time()
            schedule = optimizer.recalculate_mid_day(
                **common_kwargs,
                current_time=now,
                original_switch_on_time=immediate_on,
            )
            # Override switch-on to now+2min
            schedule = DailyHeatingSchedule(
                date=schedule.date,
                hours=schedule.hours,
                switch_on_time=immediate_on,
                switch_off_time=schedule.switch_off_time,
                optimal_setpoint=schedule.optimal_setpoint,
                cycles_per_day=schedule.cycles_per_day,
                expected_gas_usage=schedule.expected_gas_usage,
                expected_min_temp=schedule.expected_min_temp,
                expected_max_temp=schedule.expected_max_temp,
                solar_contribution=schedule.solar_contribution,
                reasoning=schedule.reasoning
                + [f"Late start: switch-on set to {immediate_on.strftime('%H:%M')}"],
                expected_switch_on_temp=schedule.expected_switch_on_temp,
                expected_target_time_temp=schedule.expected_target_time_temp,
                expected_switch_off_temp=schedule.expected_switch_off_temp,
                expected_burner_hours=schedule.expected_burner_hours,
                expected_avg_modulation=schedule.expected_avg_modulation,
            )
            return schedule, UpdateMode.ALL, "C"

    # Fallback: heating OFF, past warm time but before switch-off
    # This could happen if heating was turned off manually
    # CASE E: treat as done for today
    logger.info("Case E: Heating OFF post warm-time — calculating tomorrow's schedule")
    tomorrow = now + timedelta(days=1)
    schedule = optimizer.calculate_optimal_schedule(
        **common_kwargs,
        current_time=tomorrow,
    )
    return schedule, UpdateMode.ALL, "E"
