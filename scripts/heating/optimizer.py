"""Heating schedule optimizer.

Calculates optimal daily heating schedules based on:
- User settings (target warm time, temperatures)
- Thermal model predictions
- Weather forecasts
- Safety constraints (min temp, extreme cold)
"""

import logging
from dataclasses import dataclass
from datetime import datetime, time, timedelta

from .config import DEFAULTS
from .optimizer_forecast import (
    estimate_solar_contribution as estimate_solar_contribution_helper,
    extract_daytime_temps as extract_daytime_temps_helper,
    extract_forecast_temp_at_time as extract_forecast_temp_at_time_helper,
    extract_overnight_temps as extract_overnight_temps_helper,
)
from .optimizer_hourly_plan import build_hourly_plan as build_hourly_plan_helper
from .optimizer_metrics import (
    build_schedule as build_schedule_helper,
    calculate_avg_modulation as calculate_avg_modulation_helper,
    calculate_burner_hours as calculate_burner_hours_helper,
    estimate_gas_usage as estimate_gas_usage_helper,
    estimate_temp_range as estimate_temp_range_helper,
    find_temp_at_hour as find_temp_at_hour_helper,
)
from .optimizer_switch_off import (
    calculate_switch_off_time as calculate_switch_off_time_helper,
    hours_until as hours_until_helper,
    is_daytime as is_daytime_helper,
    simulate_cooling_stays_warm as simulate_cooling_stays_warm_helper,
)
from .thermal_model import ThermalModel

logger = logging.getLogger(__name__)


@dataclass
class HourlyHeatingPlan:
    """Heating plan for a single hour."""

    hour: int
    system_state: str  # "on" or "off"
    setpoint: float | None
    expected_modulation: float
    expected_room_temp: float


@dataclass
class DailyHeatingSchedule:
    """Complete daily heating schedule."""

    date: datetime
    hours: list[HourlyHeatingPlan]
    switch_on_time: time
    switch_off_time: time | None  # None means no switch-off (continuous heating)
    optimal_setpoint: float
    cycles_per_day: int
    expected_gas_usage: float
    expected_min_temp: float
    expected_max_temp: float
    solar_contribution: float
    reasoning: list[str]
    expected_switch_on_temp: float | None = (
        None  # Predicted room temp at switch-on time
    )
    expected_target_time_temp: float | None = None  # Predicted temp at target warm time
    expected_switch_off_temp: float | None = None  # Predicted temp at switch-off time
    expected_burner_hours: float | None = None  # Expected hours of burner operation
    expected_avg_modulation: float | None = None  # Expected average modulation %


class HeatingOptimizer:
    """Calculates optimal heating schedules."""

    def __init__(self, thermal_model: ThermalModel):
        self.thermal_model = thermal_model

    def _is_daytime(
        self, hour: int, target_warm_time: time, preferred_off_time: time
    ) -> bool:
        """Check if a clock hour falls within daytime (warm_time to off_time)."""
        return is_daytime_helper(hour, target_warm_time, preferred_off_time)

    def calculate_optimal_schedule(
        self,
        target_warm_time: time,
        target_night_time: time,
        target_temp: float,
        min_overnight_temp: float,
        min_daytime_temp: float,
        current_temps: dict[str, float],
        outside_temp: float,
        weather_forecast: list[dict] | None = None,
        current_time: datetime | None = None,
    ) -> DailyHeatingSchedule:
        """Calculate optimal heating schedule for today/tomorrow.

        Args:
            target_warm_time: When house should be warm (e.g., 07:00)
            target_night_time: When to switch off heating (e.g., 22:00)
            target_temp: Desired room temperature
            min_overnight_temp: Hard minimum for bedroom overnight
            min_daytime_temp: Comfort minimum for bedroom during the day
            current_temps: Current room temperatures
            outside_temp: Current outside temperature
            weather_forecast: Weather forecast data from HA
            current_time: Current time (defaults to now)

        Returns:
            DailyHeatingSchedule with optimal times and setpoints
        """
        if current_time is None:
            current_time = datetime.now()

        reasoning = []
        bedroom_temp = current_temps.get("bedroom", 20.0)

        # Extract forecast temperatures for different periods
        # 1. Overnight temps (for cooling prediction)
        overnight_temps = self._extract_overnight_temps(weather_forecast, outside_temp)
        min_overnight = min(overnight_temps) if overnight_temps else outside_temp

        # 2. Morning temp (for heating duration calculation)
        forecast_morning_temp = self._extract_forecast_temp_at_time(
            weather_forecast, target_warm_time, min_overnight
        )

        # 3. Daytime temps (for setpoint calculation)
        daytime_temps = self._extract_daytime_temps(
            weather_forecast, target_warm_time, target_night_time, outside_temp
        )
        min_daytime_outside = min(daytime_temps)
        avg_daytime_temp = sum(daytime_temps) / len(daytime_temps)

        reasoning.append(
            f"Current bedroom: {bedroom_temp:.1f}°C, current outside: {outside_temp:.1f}°C"
        )
        reasoning.append(f"Forecast overnight min: {min_overnight:.1f}°C")
        reasoning.append(
            f"Forecast morning ({target_warm_time}): {forecast_morning_temp:.1f}°C"
        )
        reasoning.append(
            f"Forecast daytime: min={min_daytime_outside:.1f}°C, avg={avg_daytime_temp:.1f}°C"
        )
        if DEFAULTS.forecast_temp_bias != 0:
            reasoning.append(
                f"Forecast bias correction: +{DEFAULTS.forecast_temp_bias}°C applied"
            )

        # 1. Predict overnight cooling curve (system OFF)
        hours_until_morning = self._hours_until(current_time.time(), target_warm_time)
        cooling_hours = max(1, int(hours_until_morning) + 1)

        cooling_prediction = self.thermal_model.predict_cooling_curve(
            start_temp=bedroom_temp,
            outside_temp=min_overnight,
            hours=cooling_hours,
        )

        # Find expected morning temperature (at target_warm_time)
        morning_idx = min(
            int(hours_until_morning), len(cooling_prediction.temperatures) - 1
        )
        morning_start_temp = cooling_prediction.temperatures[morning_idx]

        reasoning.append(
            f"Expected morning temp (no heating): {morning_start_temp:.1f}°C"
        )

        # 2. Calculate required pre-heat time using FORECAST morning outside temp
        heating_duration = self.thermal_model.predict_heating_duration(
            start_temp=morning_start_temp,
            target_temp=target_temp,
            outside_temp=forecast_morning_temp,  # Use forecast, not current!
            setpoint=DEFAULTS.default_setpoint,
        )

        # Add safety buffer
        total_preheat = heating_duration + DEFAULTS.safety_buffer_minutes
        reasoning.append(
            f"Heating duration: {heating_duration}min + {DEFAULTS.safety_buffer_minutes}min buffer"
        )

        # 3. Calculate switch-on time
        target_warm_dt = datetime.combine(current_time.date(), target_warm_time)
        if current_time.time() > target_warm_time:
            # Target is tomorrow
            target_warm_dt += timedelta(days=1)

        switch_on_dt = target_warm_dt - timedelta(minutes=total_preheat)
        switch_on_time = switch_on_dt.time()

        reasoning.append(f"Switch-on time: {switch_on_time.strftime('%H:%M')}")

        # 4. Predict solar contribution (for daytime)
        solar_contribution = self._estimate_solar_contribution(
            weather_forecast,
            target_warm_time,
            target_night_time,
        )
        reasoning.append(f"Expected solar contribution: +{solar_contribution:.1f}°C")

        # 5. Calculate optimal setpoint using FORECAST daytime temps
        optimal_setpoint = target_temp

        # Adjust for solar (reduce setpoint if significant solar gain expected)
        if solar_contribution > 0.5:
            optimal_setpoint -= 1
            reasoning.append("Reduced setpoint by 1°C for solar gain")

        # Clamp to configured bounds
        optimal_setpoint = round(
            max(DEFAULTS.min_setpoint, min(DEFAULTS.max_setpoint, optimal_setpoint))
        )
        reasoning.append(f"Optimal setpoint: {optimal_setpoint}°C")

        # 6. Calculate optimal switch-off time
        # Find earliest time heating can stop while maintaining min_temp until morning
        switch_off_time = self._calculate_switch_off_time(
            preferred_off_time=target_night_time,
            target_temp=target_temp,
            min_overnight_temp=min_overnight_temp,
            min_daytime_temp=min_daytime_temp,
            outside_temp=min_overnight,  # Overnight forecast min for cooling calc
            target_warm_time=target_warm_time,
            switch_on_time=switch_on_time,
        )

        if switch_off_time is None:
            reasoning.append(
                "Switch-off: CONTINUOUS (can't maintain min temp overnight)"
            )
        elif switch_off_time.hour < target_night_time.hour:
            # We found an earlier time than the user's preferred
            saved_hours = self._hours_until(switch_off_time, target_night_time)
            reasoning.append(
                f"Switch-off time: {switch_off_time.strftime('%H:%M')} "
                f"({saved_hours:.0f}h earlier than preferred {target_night_time.strftime('%H:%M')})"
            )
        else:
            reasoning.append(f"Switch-off time: {switch_off_time.strftime('%H:%M')}")

        # 7. Build hourly plan
        hours = self._build_hourly_plan(
            switch_on_time=switch_on_time,
            switch_off_time=switch_off_time,
            optimal_setpoint=optimal_setpoint,
            bedroom_temp=bedroom_temp,
            outside_temp=outside_temp,
            weather_forecast=weather_forecast,
            current_time=current_time,
        )

        # 8. Build final schedule with metrics
        return self._build_schedule(
            current_time=current_time,
            hours=hours,
            switch_on_time=switch_on_time,
            switch_off_time=switch_off_time,
            optimal_setpoint=optimal_setpoint,
            solar_contribution=solar_contribution,
            reasoning=reasoning,
            expected_switch_on_temp=morning_start_temp,
            target_warm_time=target_warm_time,
        )

    def recalculate_mid_day(
        self,
        target_warm_time: time,
        target_night_time: time,
        target_temp: float,
        min_overnight_temp: float,
        min_daytime_temp: float,
        current_temps: dict[str, float],
        outside_temp: float,
        weather_forecast: list[dict] | None = None,
        current_time: datetime | None = None,
        original_switch_on_time: time | None = None,
    ) -> DailyHeatingSchedule:
        """Recalculate schedule during active heating period.

        Uses actual measured temps instead of cooling predictions.
        Preserves switch-on time, recalculates switch-off and setpoint.
        """
        if current_time is None:
            current_time = datetime.now()

        reasoning = []
        bedroom_temp = current_temps.get("bedroom", 20.0)
        reasoning.append(f"Mid-day recalc: actual bedroom {bedroom_temp:.1f}°C")

        # Preserve original switch-on time
        switch_on_time = original_switch_on_time or target_warm_time

        # Overnight forecast for switch-off cooling sim
        overnight_temps = self._extract_overnight_temps(weather_forecast, outside_temp)
        min_overnight = min(overnight_temps) if overnight_temps else outside_temp

        # Daytime temps for setpoint
        # Switch-on time guard: if original switch-on is less than 2 hours before
        # target_warm_time, use target_warm_time - 2hrs for switch-off calculation
        # to prevent unrealistically short off-period estimates
        effective_switch_on = switch_on_time
        hours_before_warm = self._hours_until(switch_on_time, target_warm_time)
        if hours_before_warm < 2.0 and hours_before_warm > 0:
            guard_hour = (target_warm_time.hour - 2) % 24
            effective_switch_on = time(guard_hour, target_warm_time.minute)
            reasoning.append(
                f"Switch-on guard: using {effective_switch_on.strftime('%H:%M')} "
                f"for off-time calc (original {switch_on_time.strftime('%H:%M')} "
                f"too close to warm time)"
            )

        # Recalculate switch-off time using actual bedroom temp
        switch_off_time = self._calculate_switch_off_time(
            preferred_off_time=target_night_time,
            target_temp=target_temp,
            min_overnight_temp=min_overnight_temp,
            min_daytime_temp=min_daytime_temp,
            outside_temp=min_overnight,
            target_warm_time=target_warm_time,
            switch_on_time=effective_switch_on,
        )

        off_str = switch_off_time.strftime("%H:%M") if switch_off_time else "CONTINUOUS"
        reasoning.append(f"Recalculated switch-off: {off_str}")

        # Solar contribution
        solar_contribution = self._estimate_solar_contribution(
            weather_forecast, target_warm_time, target_night_time
        )

        # Calculate setpoint fresh from current target_temp
        setpoint = target_temp

        # Apply solar reduction (consistent with initial calculation)
        if solar_contribution > 0.5:
            setpoint -= 1
            reasoning.append("Reduced setpoint by 1°C for solar gain")

        # Clamp setpoint to safe bounds
        setpoint = round(
            max(DEFAULTS.min_setpoint, min(DEFAULTS.max_setpoint, setpoint))
        )
        reasoning.append(f"Setpoint: {setpoint}°C")

        # Build hourly plan from current hour onward
        hours = self._build_hourly_plan(
            switch_on_time=switch_on_time,
            switch_off_time=switch_off_time,
            optimal_setpoint=setpoint,
            bedroom_temp=bedroom_temp,
            outside_temp=outside_temp,
            weather_forecast=weather_forecast,
            current_time=current_time,
        )

        return self._build_schedule(
            current_time=current_time,
            hours=hours,
            switch_on_time=switch_on_time,
            switch_off_time=switch_off_time,
            optimal_setpoint=setpoint,
            solar_contribution=solar_contribution,
            reasoning=reasoning,
            expected_switch_on_temp=bedroom_temp,
            target_warm_time=target_warm_time,
        )

    def apply_safety_overrides(
        self,
        schedule: DailyHeatingSchedule,
        current_temps: dict[str, float],
        min_overnight_temp: float,
        min_daytime_temp: float,
        target_warm_time: time,
        preferred_off_time: time,
    ) -> tuple[DailyHeatingSchedule, list[str]]:
        """Apply safety overrides to a schedule.

        Returns modified schedule and list of overrides applied.
        """
        overrides = []
        bedroom_temp = current_temps.get("bedroom", 20.0)

        # Determine active threshold based on current time of day
        now_hour = datetime.now().time().hour
        if self._is_daytime(now_hour, target_warm_time, preferred_off_time):
            active_min = min_daytime_temp
            label = "daytime"
        else:
            active_min = min_overnight_temp
            label = "overnight"

        # Bedroom too cold override
        if bedroom_temp < active_min:
            overrides.append(
                f"Bedroom below {label} minimum {active_min}°C: "
                f"recommend immediate heating ({bedroom_temp:.1f}°C)"
            )

        return schedule, overrides

    def _hours_until(self, current: time, target: time) -> float:
        """Calculate hours from current time until target time."""
        return hours_until_helper(current, target)

    def _extract_overnight_temps(
        self,
        forecast: list[dict] | None,
        default: float,
    ) -> list[float]:
        """Extract overnight temperature predictions from forecast."""
        return extract_overnight_temps_helper(forecast, default)

    def _extract_forecast_temp_at_time(
        self,
        forecast: list[dict] | None,
        target_time: time,
        default: float,
    ) -> float:
        """Extract forecast temperature at a specific time of day.

        Args:
            forecast: Hourly forecast data
            target_time: Time to find forecast for
            default: Default temp if forecast unavailable

        Returns:
            Forecast temperature for the target time
        """
        return extract_forecast_temp_at_time_helper(forecast, target_time, default)

    def _extract_daytime_temps(
        self,
        forecast: list[dict] | None,
        morning_time: time,
        evening_time: time,
        default: float,
    ) -> list[float]:
        """Extract forecast temperatures for daytime hours.

        Returns list of temps between morning and evening times.
        """
        return extract_daytime_temps_helper(forecast, morning_time, evening_time, default)

    def _estimate_solar_contribution(
        self,
        forecast: list[dict] | None,
        morning_time: time,
        evening_time: time,
    ) -> float:
        """Estimate solar heat contribution based on forecast."""
        return estimate_solar_contribution_helper(forecast, morning_time, evening_time)

    def _simulate_cooling_stays_warm(
        self,
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
        return simulate_cooling_stays_warm_helper(
            thermal_model=self.thermal_model,
            candidate_time=candidate_time,
            target_temp=target_temp,
            outside_temp=outside_temp,
            hours_off=hours_off,
            min_daytime_temp=min_daytime_temp,
            min_overnight_temp=min_overnight_temp,
            target_warm_time=target_warm_time,
            preferred_off_time=preferred_off_time,
        )

    def _calculate_switch_off_time(
        self,
        preferred_off_time: time,
        target_temp: float,
        min_overnight_temp: float,
        min_daytime_temp: float,
        outside_temp: float,
        target_warm_time: time,
        switch_on_time: time,
    ) -> time | None:
        """Find earliest switch-off time keeping bedroom above the active threshold.

        Each simulated cooling hour is checked against the correct threshold
        for that clock hour: min_daytime_temp during waking hours
        (target_warm_time → preferred_off_time) and min_overnight_temp otherwise.
        """
        return calculate_switch_off_time_helper(
            thermal_model=self.thermal_model,
            preferred_off_time=preferred_off_time,
            target_temp=target_temp,
            min_overnight_temp=min_overnight_temp,
            min_daytime_temp=min_daytime_temp,
            outside_temp=outside_temp,
            target_warm_time=target_warm_time,
            switch_on_time=switch_on_time,
            min_off_duration_hours=DEFAULTS.min_off_duration_hours,
        )

    def _build_hourly_plan(
        self,
        switch_on_time: time,
        switch_off_time: time | None,
        optimal_setpoint: float,
        bedroom_temp: float,
        outside_temp: float,
        weather_forecast: list[dict] | None = None,
        current_time: datetime | None = None,
    ) -> list[HourlyHeatingPlan]:
        """Build 24-hour heating plan."""
        return build_hourly_plan_helper(
            thermal_model=self.thermal_model,
            switch_on_time=switch_on_time,
            switch_off_time=switch_off_time,
            optimal_setpoint=optimal_setpoint,
            bedroom_temp=bedroom_temp,
            outside_temp=outside_temp,
            weather_forecast=weather_forecast,
            current_time=current_time,
            plan_factory=HourlyHeatingPlan,
        )

    def _estimate_temp_range(
        self,
        hours: list[HourlyHeatingPlan],
    ) -> tuple[float, float]:
        """Estimate min and max temperatures from hourly plan."""
        return estimate_temp_range_helper(hours)

    def _estimate_gas_usage(
        self,
        hours: list[HourlyHeatingPlan],
    ) -> float:
        """Estimate gas usage based on modulation and hours."""
        return estimate_gas_usage_helper(hours, self.thermal_model.gas_base_rate_kwh)

    def _calculate_burner_hours(
        self,
        switch_on_time: time,
        switch_off_time: time | None,
    ) -> float:
        """Calculate expected burner operation hours."""
        return calculate_burner_hours_helper(switch_on_time, switch_off_time)

    def _calculate_avg_modulation(self, hours: list[HourlyHeatingPlan]) -> float:
        """Calculate average modulation when system is ON."""
        return calculate_avg_modulation_helper(hours)

    def _find_temp_at_hour(
        self, hours: list[HourlyHeatingPlan], target_hour: int
    ) -> float | None:
        """Find expected room temp at a specific hour in the hourly plan."""
        return find_temp_at_hour_helper(hours, target_hour)

    def _build_schedule(
        self,
        current_time: datetime,
        hours: list[HourlyHeatingPlan],
        switch_on_time: time,
        switch_off_time: time | None,
        optimal_setpoint: float,
        solar_contribution: float,
        reasoning: list[str],
        expected_switch_on_temp: float | None,
        target_warm_time: time,
    ) -> DailyHeatingSchedule:
        """Build a DailyHeatingSchedule with computed metrics."""
        return build_schedule_helper(
            current_time=current_time,
            hours=hours,
            switch_on_time=switch_on_time,
            switch_off_time=switch_off_time,
            optimal_setpoint=optimal_setpoint,
            solar_contribution=solar_contribution,
            reasoning=reasoning,
            expected_switch_on_temp=expected_switch_on_temp,
            target_warm_time=target_warm_time,
            gas_base_rate=self.thermal_model.gas_base_rate_kwh,
            schedule_factory=DailyHeatingSchedule,
        )

    def generate_schedule_summary(self, schedule: DailyHeatingSchedule) -> str:
        """Generate human-readable schedule summary."""
        off_time_str = (
            schedule.switch_off_time.strftime("%H:%M")
            if schedule.switch_off_time
            else "CONTINUOUS"
        )
        lines = [
            f"Heating Schedule for {schedule.date.strftime('%Y-%m-%d')}",
            "",
            f"Switch ON:  {schedule.switch_on_time.strftime('%H:%M')}",
            f"Switch OFF: {off_time_str}",
            f"Setpoint:   {schedule.optimal_setpoint}°C",
            "",
            "Expected temperatures:",
            f"  Min: {schedule.expected_min_temp}°C",
            f"  Max: {schedule.expected_max_temp}°C",
            "",
            f"Solar contribution: +{schedule.solar_contribution}°C",
            f"Expected gas usage: ~{schedule.expected_gas_usage} kWh",
            "",
            "Reasoning:",
        ]

        for reason in schedule.reasoning:
            lines.append(f"  - {reason}")

        return "\n".join(lines)
