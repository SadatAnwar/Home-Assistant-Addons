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
from typing import Any

from .config import DEFAULTS, SOLAR_GAIN_ROOMS
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


class HeatingOptimizer:
    """Calculates optimal heating schedules."""

    def __init__(self, thermal_model: ThermalModel):
        self.thermal_model = thermal_model

    def calculate_optimal_schedule(
        self,
        target_warm_time: time,
        target_night_time: time,
        target_temp: float,
        min_bedroom_temp: float,
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
            min_bedroom_temp: Hard minimum for bedroom
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
        min_daytime_temp = min(daytime_temps)
        avg_daytime_temp = sum(daytime_temps) / len(daytime_temps)

        reasoning.append(f"Current bedroom: {bedroom_temp:.1f}°C, current outside: {outside_temp:.1f}°C")
        reasoning.append(f"Forecast overnight min: {min_overnight:.1f}°C")
        reasoning.append(f"Forecast morning ({target_warm_time}): {forecast_morning_temp:.1f}°C")
        reasoning.append(f"Forecast daytime: min={min_daytime_temp:.1f}°C, avg={avg_daytime_temp:.1f}°C")

        # 1. Predict overnight cooling curve (system OFF)
        cooling_prediction = self.thermal_model.predict_cooling_curve(
            start_temp=bedroom_temp,
            outside_temp=min_overnight,
            hours=8,
        )

        # Find expected morning temperature (at target_warm_time)
        hours_until_morning = self._hours_until(current_time.time(), target_warm_time)
        morning_idx = min(int(hours_until_morning), len(cooling_prediction.temperatures) - 1)
        morning_start_temp = cooling_prediction.temperatures[morning_idx]

        reasoning.append(f"Expected morning temp (no heating): {morning_start_temp:.1f}°C")

        # 2. Calculate required pre-heat time using FORECAST morning outside temp
        heating_duration = self.thermal_model.predict_heating_duration(
            start_temp=morning_start_temp,
            target_temp=target_temp,
            outside_temp=forecast_morning_temp,  # Use forecast, not current!
            setpoint=DEFAULTS.default_setpoint,
        )

        # Add safety buffer
        total_preheat = heating_duration + DEFAULTS.safety_buffer_minutes
        reasoning.append(f"Heating duration: {heating_duration}min + {DEFAULTS.safety_buffer_minutes}min buffer")

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
        # Use average daytime temp for setpoint, not overnight min
        optimal_setpoint = self._calculate_setpoint(
            outside_temp=avg_daytime_temp,  # Use forecast daytime avg, not current!
            target_room_temp=target_temp,
        )

        # Adjust for solar (reduce setpoint if significant solar gain expected)
        if solar_contribution > 0.5:
            optimal_setpoint -= 0.5
            reasoning.append(f"Reduced setpoint by 0.5°C for solar gain")

        # Clamp to configured bounds
        optimal_setpoint = round(
            max(DEFAULTS.min_setpoint, min(DEFAULTS.max_setpoint, optimal_setpoint)), 1
        )
        reasoning.append(f"Optimal setpoint: {optimal_setpoint}°C")

        # 6. Calculate switch-off time with minimum off duration constraint
        # Use overnight forecast for cooling prediction
        switch_off_time = self._calculate_switch_off_time(
            preferred_off_time=target_night_time,
            bedroom_temp=bedroom_temp,
            min_temp=min_bedroom_temp,
            outside_temp=min_overnight,  # Overnight forecast min for cooling calc
            target_warm_time=target_warm_time,
            switch_on_time=switch_on_time,
        )

        if switch_off_time is None:
            reasoning.append("Switch-off: SKIPPED (off period too short, continuous heating)")
        else:
            reasoning.append(f"Switch-off time: {switch_off_time.strftime('%H:%M')}")

        # 7. Build hourly plan
        hours = self._build_hourly_plan(
            switch_on_time=switch_on_time,
            switch_off_time=switch_off_time,
            optimal_setpoint=optimal_setpoint,
            bedroom_temp=bedroom_temp,
            outside_temp=outside_temp,
        )

        # 8. Estimate expected temperatures and gas usage
        expected_min, expected_max = self._estimate_temp_range(hours, bedroom_temp)
        expected_gas = self._estimate_gas_usage(hours, outside_temp)

        return DailyHeatingSchedule(
            date=current_time,
            hours=hours,
            switch_on_time=switch_on_time,
            switch_off_time=switch_off_time,
            optimal_setpoint=optimal_setpoint,
            cycles_per_day=1,  # Always 1 cycle with this approach
            expected_gas_usage=expected_gas,
            expected_min_temp=expected_min,
            expected_max_temp=expected_max,
            solar_contribution=solar_contribution,
            reasoning=reasoning,
        )

    def apply_safety_overrides(
        self,
        schedule: DailyHeatingSchedule,
        current_temps: dict[str, float],
        outside_temp: float,
        min_bedroom_temp: float,
    ) -> tuple[DailyHeatingSchedule, list[str]]:
        """Apply safety overrides to a schedule.

        Returns modified schedule and list of overrides applied.
        """
        overrides = []
        bedroom_temp = current_temps.get("bedroom", 20.0)

        # Extreme cold override
        if outside_temp < DEFAULTS.extreme_cold_threshold:
            if schedule.switch_on_time.hour > 4:
                # Move switch-on earlier
                new_on_time = time(4, 0)
                schedule.switch_on_time = new_on_time
                overrides.append(f"Extreme cold ({outside_temp}°C): moved switch-on to 04:00")

            if schedule.optimal_setpoint < 22:
                schedule.optimal_setpoint = 22
                overrides.append("Extreme cold: increased setpoint to 22°C")

        # Bedroom too cold override
        if bedroom_temp < min_bedroom_temp:
            overrides.append(f"Bedroom below {min_bedroom_temp}°C: recommend immediate heating")

        return schedule, overrides

    def _hours_until(self, current: time, target: time) -> float:
        """Calculate hours from current time until target time."""
        current_mins = current.hour * 60 + current.minute
        target_mins = target.hour * 60 + target.minute

        if target_mins <= current_mins:
            # Target is tomorrow
            target_mins += 24 * 60

        return (target_mins - current_mins) / 60

    def _calculate_setpoint(self, outside_temp: float, target_room_temp: float) -> float:
        """Calculate setpoint based on outside temperature.

        Matches user's manual approach:
        - Normal days: 20°C setpoint
        - Cold days (0 to -5°C): Scale up to 22°C
        - Extreme cold (<-5°C): Use max setpoint (22°C)
        """
        base_setpoint = DEFAULTS.default_setpoint  # 20°C

        if outside_temp < DEFAULTS.extreme_cold_threshold:
            # Extreme cold: use max setpoint
            setpoint = DEFAULTS.max_setpoint
            logger.debug(f"Extreme cold ({outside_temp}°C): using max setpoint {setpoint}°C")
        elif outside_temp < 0:
            # Cold: scale from 20 to 22 as temp drops from 0 to -5
            # +0.4°C per degree below 0
            adjustment = (abs(outside_temp) / 5) * 2
            setpoint = base_setpoint + adjustment
            logger.debug(
                f"Cold ({outside_temp}°C): base {base_setpoint} + {adjustment:.1f} = {setpoint:.1f}°C"
            )
        elif outside_temp < 5:
            # Cool: slight increase (+0.5°C)
            setpoint = base_setpoint + 0.5
            logger.debug(f"Cool ({outside_temp}°C): using setpoint {setpoint}°C")
        else:
            # Mild: use base setpoint
            setpoint = base_setpoint
            logger.debug(f"Mild ({outside_temp}°C): using base setpoint {setpoint}°C")

        return setpoint

    def _extract_overnight_temps(
        self,
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

        logger.debug(f"Extracted overnight temps from forecast: {temps}")
        return temps

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
        if not forecast:
            logger.debug(f"No forecast for {target_time}, using default: {default}°C")
            return default

        target_hour = target_time.hour
        now = datetime.now()

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
                        logger.debug(f"Forecast at {target_time}: {temp}°C (from {dt_str})")
                        return temp
            except ValueError:
                continue

        logger.debug(f"No forecast match for {target_time}, using default: {default}°C")
        return default

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
            logger.debug(f"Daytime forecast temps ({morning_time}-{evening_time}): min={min(temps)}, max={max(temps)}, avg={sum(temps)/len(temps):.1f}")
            return temps

        return [default]

    def _estimate_solar_contribution(
        self,
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
            temp = entry.get("temperature", 10)

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

    def _calculate_switch_off_time(
        self,
        preferred_off_time: time,
        bedroom_temp: float,
        min_temp: float,
        outside_temp: float,
        target_warm_time: time,
        switch_on_time: time,
    ) -> time | None:
        """Calculate safe switch-off time ensuring min temp until morning.

        Returns None if off period would be too short (skip turn-off entirely).
        """
        # Calculate hours of off-period (from preferred_off to next switch_on)
        hours_off = self._hours_until(preferred_off_time, switch_on_time)

        logger.debug(
            f"Switch-off calc: preferred={preferred_off_time}, "
            f"switch_on={switch_on_time}, hours_off={hours_off:.1f}"
        )

        # Check minimum off duration constraint
        if hours_off < DEFAULTS.min_off_duration_hours:
            logger.info(
                f"Off period too short ({hours_off:.1f}h < {DEFAULTS.min_off_duration_hours}h), "
                "skipping switch-off"
            )
            return None  # Don't turn off - continuous heating

        # Predict cooling overnight from preferred_off_time
        hours_overnight = self._hours_until(preferred_off_time, target_warm_time)

        cooling = self.thermal_model.predict_cooling_curve(
            start_temp=bedroom_temp,
            outside_temp=outside_temp,
            hours=int(hours_overnight) + 1,
        )

        # Check if temp drops below minimum
        for i, temp in enumerate(cooling.temperatures):
            if temp < min_temp:
                # Need to stay on longer - calculate delayed switch-off time
                delay_hours = max(0, hours_overnight - i) * 0.5
                new_hour = (preferred_off_time.hour + int(delay_hours)) % 24
                new_minute = preferred_off_time.minute + int((delay_hours % 1) * 60)
                if new_minute >= 60:
                    new_hour = (new_hour + 1) % 24
                    new_minute -= 60

                delayed_off = time(new_hour, new_minute)

                # Re-check min off duration with delayed time
                new_hours_off = self._hours_until(delayed_off, switch_on_time)
                if new_hours_off < DEFAULTS.min_off_duration_hours:
                    logger.info(
                        f"Delayed off ({delayed_off}) would create short off period "
                        f"({new_hours_off:.1f}h), skipping switch-off"
                    )
                    return None

                return delayed_off

        return preferred_off_time

    def _build_hourly_plan(
        self,
        switch_on_time: time,
        switch_off_time: time | None,
        optimal_setpoint: float,
        bedroom_temp: float,
        outside_temp: float,
    ) -> list[HourlyHeatingPlan]:
        """Build 24-hour heating plan."""
        plans = []
        current_temp = bedroom_temp

        for hour in range(24):
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

            if system_on:
                # Predict modulation
                modulation = self.thermal_model.predict_modulation(
                    outside_temp=outside_temp,
                    setpoint=optimal_setpoint,
                    room_temp=current_temp,
                )
                # Estimate temp increase
                current_temp += self.thermal_model.mean_heating_rate / 4
            else:
                modulation = 0
                # Estimate cooling
                current_temp -= self.thermal_model.mean_cooling_rate / 4

            current_temp = max(15, min(25, current_temp))

            plans.append(HourlyHeatingPlan(
                hour=hour,
                system_state="on" if system_on else "off",
                setpoint=optimal_setpoint if system_on else None,
                expected_modulation=round(modulation, 1),
                expected_room_temp=round(current_temp, 1),
            ))

        return plans

    def _estimate_temp_range(
        self,
        hours: list[HourlyHeatingPlan],
        starting_temp: float,
    ) -> tuple[float, float]:
        """Estimate min and max temperatures from hourly plan."""
        temps = [h.expected_room_temp for h in hours]
        return min(temps), max(temps)

    def _estimate_gas_usage(
        self,
        hours: list[HourlyHeatingPlan],
        outside_temp: float,
    ) -> float:
        """Estimate gas usage based on modulation and hours."""
        # Rough estimate: modulation % correlates with gas usage
        # Base consumption ~1.5 kWh at 50% modulation

        total_kwh = 0
        for h in hours:
            if h.system_state == "on":
                # Scale by modulation (higher modulation = more gas)
                hourly_kwh = 1.5 * (h.expected_modulation / 50)
                total_kwh += hourly_kwh

        return round(total_kwh, 1)

    def generate_schedule_summary(self, schedule: DailyHeatingSchedule) -> str:
        """Generate human-readable schedule summary."""
        off_time_str = (
            schedule.switch_off_time.strftime("%H:%M")
            if schedule.switch_off_time
            else "CONTINUOUS"
        )
        lines = [
            f"Heating Schedule for {schedule.date.strftime('%Y-%m-%d')}",
            f"",
            f"Switch ON:  {schedule.switch_on_time.strftime('%H:%M')}",
            f"Switch OFF: {off_time_str}",
            f"Setpoint:   {schedule.optimal_setpoint}°C",
            f"",
            f"Expected temperatures:",
            f"  Min: {schedule.expected_min_temp}°C",
            f"  Max: {schedule.expected_max_temp}°C",
            f"",
            f"Solar contribution: +{schedule.solar_contribution}°C",
            f"Expected gas usage: ~{schedule.expected_gas_usage} kWh",
            f"",
            f"Reasoning:",
        ]

        for reason in schedule.reasoning:
            lines.append(f"  - {reason}")

        return "\n".join(lines)
