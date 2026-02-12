"""Switch-off and day-window helpers for HeatingOptimizer."""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta

logger = logging.getLogger(__name__)


def is_daytime(hour: int, target_warm_time: time, preferred_off_time: time) -> bool:
    """Check if a clock hour falls within daytime (warm_time to off_time)."""
    warm_h = target_warm_time.hour
    off_h = preferred_off_time.hour
    if warm_h <= off_h:
        return warm_h <= hour < off_h
    return hour >= warm_h or hour < off_h


def hours_until(current: time, target: time) -> float:
    """Calculate hours from current time until target time."""
    current_mins = current.hour * 60 + current.minute
    target_mins = target.hour * 60 + target.minute

    if target_mins <= current_mins:
        # Target is tomorrow
        target_mins += 24 * 60

    return (target_mins - current_mins) / 60


def simulate_cooling_stays_warm(
    *,
    thermal_model,
    candidate_time: time,
    target_temp: float,
    outside_temp: float,
    hours_off: float,
    min_daytime_temp: float,
    min_overnight_temp: float,
    target_warm_time: time,
    preferred_off_time: time,
) -> bool:
    """Check if switching off at candidate_time keeps the house warm enough."""
    cooling = thermal_model.predict_cooling_curve(
        start_temp=target_temp,
        outside_temp=outside_temp,
        hours=int(hours_off),
    )

    for i, temp in enumerate(cooling.temperatures):
        sim_hour = (candidate_time.hour + i) % 24
        threshold = (
            min_daytime_temp
            if is_daytime(sim_hour, target_warm_time, preferred_off_time)
            else min_overnight_temp
        )
        if temp < threshold:
            return False
    return True


def calculate_switch_off_time(
    *,
    thermal_model,
    preferred_off_time: time,
    target_temp: float,
    min_overnight_temp: float,
    min_daytime_temp: float,
    outside_temp: float,
    target_warm_time: time,
    switch_on_time: time,
    min_off_duration_hours: float,
) -> time | None:
    """Find earliest switch-off time keeping bedroom above active threshold."""
    logger.debug(
        f"Switch-off calc: preferred={preferred_off_time}, "
        f"target={target_temp}°C, min_day={min_daytime_temp}°C, "
        f"min_night={min_overnight_temp}°C, "
        f"outside={outside_temp}°C, switch_on={switch_on_time}"
    )

    sim_kwargs = {
        "thermal_model": thermal_model,
        "target_temp": target_temp,
        "outside_temp": outside_temp,
        "min_daytime_temp": min_daytime_temp,
        "min_overnight_temp": min_overnight_temp,
        "target_warm_time": target_warm_time,
        "preferred_off_time": preferred_off_time,
    }

    # Build candidate times at 30-minute intervals from target_warm_time to preferred_off_time
    candidates = []
    cursor = datetime.combine(datetime.today(), target_warm_time)
    end_dt = datetime.combine(datetime.today(), preferred_off_time)
    if end_dt <= cursor:
        end_dt += timedelta(days=1)
    while cursor <= end_dt:
        candidates.append(cursor.time())
        cursor += timedelta(minutes=30)

    # Search for earliest safe switch-off time
    for candidate_time in candidates:
        off_hours = hours_until(candidate_time, switch_on_time)
        if off_hours < min_off_duration_hours:
            continue

        if simulate_cooling_stays_warm(
            candidate_time=candidate_time,
            hours_off=off_hours,
            **sim_kwargs,
        ):
            logger.info(
                f"Earliest safe switch-off: {candidate_time.strftime('%H:%M')} "
                f"(off for {off_hours:.0f}h, stays above "
                f"daytime={min_daytime_temp}°C / overnight={min_overnight_temp}°C)"
            )
            return candidate_time

    # No safe time found before preferred_off_time — search beyond it
    # (extend up to 2 hours past preferred in 30-min steps, then give up)
    logger.debug("No safe off time before preferred time, searching later...")
    for extra_step in range(1, 5):
        extra_dt = datetime.combine(datetime.today(), preferred_off_time) + timedelta(
            minutes=extra_step * 30
        )
        candidate_time = extra_dt.time()

        off_hours = hours_until(candidate_time, switch_on_time)
        if off_hours < min_off_duration_hours:
            continue

        if simulate_cooling_stays_warm(
            candidate_time=candidate_time,
            hours_off=off_hours,
            **sim_kwargs,
        ):
            logger.info(
                f"Extended switch-off: {candidate_time.strftime('%H:%M')} "
                f"({extra_step * 30}min past preferred)"
            )
            return candidate_time

    # Nothing works — continuous heating
    logger.info("No safe switch-off time found, using continuous heating")
    return None
