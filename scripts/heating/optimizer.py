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
        warm_h = target_warm_time.hour
        off_h = preferred_off_time.hour
        if warm_h <= off_h:
            return warm_h <= hour < off_h
        else:
            return hour >= warm_h or hour < off_h

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
        # Use average daytime temp for setpoint, not overnight min
        optimal_setpoint = self._calculate_setpoint(
            outside_temp=avg_daytime_temp,  # Use forecast daytime avg, not current!
        )

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
        )

        # 8. Estimate expected temperatures and gas usage
        expected_min, expected_max = self._estimate_temp_range(hours)
        expected_gas = self._estimate_gas_usage(hours)

        # 9. Calculate additional prediction metrics for tracking
        # Expected temp at target warm time
        target_hour = target_warm_time.hour
        expected_target_time_temp = None
        for hp in hours:
            if hp.hour == target_hour:
                expected_target_time_temp = hp.expected_room_temp
                break

        # Expected temp at switch-off time
        expected_switch_off_temp = None
        if switch_off_time is not None:
            off_hour = switch_off_time.hour
            for hp in hours:
                if hp.hour == off_hour:
                    expected_switch_off_temp = hp.expected_room_temp
                    break

        # Expected burner hours (from on/off times)
        expected_burner_hours = self._calculate_burner_hours(
            switch_on_time, switch_off_time
        )

        # Expected average modulation (from hourly plan when ON)
        expected_avg_modulation = self._calculate_avg_modulation(hours)

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
            expected_switch_on_temp=morning_start_temp,
            expected_target_time_temp=expected_target_time_temp,
            expected_switch_off_temp=expected_switch_off_temp,
            expected_burner_hours=expected_burner_hours,
            expected_avg_modulation=expected_avg_modulation,
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
        original_setpoint: float | None = None,
        original_hourly_plans: list[HourlyHeatingPlan] | None = None,
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
        daytime_temps = self._extract_daytime_temps(
            weather_forecast, target_warm_time, target_night_time, outside_temp
        )
        avg_daytime_temp = sum(daytime_temps) / len(daytime_temps)

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

        # Setpoint adjustment based on actual vs predicted temp
        setpoint = original_setpoint or self._calculate_setpoint(avg_daytime_temp)
        current_hour = current_time.hour

        if original_hourly_plans:
            predicted_temp = None
            for hp in original_hourly_plans:
                if hp.hour == current_hour:
                    predicted_temp = hp.expected_room_temp
                    break

            if predicted_temp is not None:
                if bedroom_temp > predicted_temp:
                    setpoint -= 1.0
                    reasoning.append(
                        f"Setpoint -1°C: actual {bedroom_temp:.1f}°C > "
                        f"predicted {predicted_temp:.1f}°C"
                    )
                elif bedroom_temp < predicted_temp - 1.0:
                    setpoint += 1.0
                    reasoning.append(
                        f"Setpoint +1°C: actual {bedroom_temp:.1f}°C < "
                        f"predicted {predicted_temp:.1f}°C - 1.0"
                    )
                else:
                    reasoning.append(
                        f"Setpoint unchanged: actual {bedroom_temp:.1f}°C "
                        f"~ predicted {predicted_temp:.1f}°C"
                    )

        # Clamp setpoint
        setpoint = round(
            max(DEFAULTS.min_setpoint, min(DEFAULTS.max_setpoint, setpoint))
        )
        reasoning.append(f"Adjusted setpoint: {setpoint}°C")

        # Solar contribution
        solar_contribution = self._estimate_solar_contribution(
            weather_forecast, target_warm_time, target_night_time
        )

        # Build hourly plan from current hour onward
        hours = self._build_hourly_plan(
            switch_on_time=switch_on_time,
            switch_off_time=switch_off_time,
            optimal_setpoint=setpoint,
            bedroom_temp=bedroom_temp,
            outside_temp=outside_temp,
        )

        expected_min, expected_max = self._estimate_temp_range(hours)
        expected_gas = self._estimate_gas_usage(hours)

        # Expected temps at key times
        expected_target_time_temp = None
        for hp in hours:
            if hp.hour == target_warm_time.hour:
                expected_target_time_temp = hp.expected_room_temp
                break

        expected_switch_off_temp = None
        if switch_off_time is not None:
            for hp in hours:
                if hp.hour == switch_off_time.hour:
                    expected_switch_off_temp = hp.expected_room_temp
                    break

        expected_burner_hours = self._calculate_burner_hours(
            switch_on_time, switch_off_time
        )
        expected_avg_modulation = self._calculate_avg_modulation(hours)

        return DailyHeatingSchedule(
            date=current_time,
            hours=hours,
            switch_on_time=switch_on_time,
            switch_off_time=switch_off_time,
            optimal_setpoint=setpoint,
            cycles_per_day=1,
            expected_gas_usage=expected_gas,
            expected_min_temp=expected_min,
            expected_max_temp=expected_max,
            solar_contribution=solar_contribution,
            reasoning=reasoning,
            expected_switch_on_temp=bedroom_temp,
            expected_target_time_temp=expected_target_time_temp,
            expected_switch_off_temp=expected_switch_off_temp,
            expected_burner_hours=expected_burner_hours,
            expected_avg_modulation=expected_avg_modulation,
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
        current_mins = current.hour * 60 + current.minute
        target_mins = target.hour * 60 + target.minute

        if target_mins <= current_mins:
            # Target is tomorrow
            target_mins += 24 * 60

        return (target_mins - current_mins) / 60

    def _calculate_setpoint(self, outside_temp: float) -> float:
        """Calculate setpoint based on outside temperature.

        Matches user's manual approach:
        - Normal days: 20°C setpoint
        - Cold days (0 to -5°C): Scale up to 22°C
        - Extreme cold (<-5°C): Use max setpoint (22°C)
        """
        base_setpoint = DEFAULTS.default_setpoint  # 20°C
        logger.debug(f"Mild ({outside_temp}°C): using base setpoint {base_setpoint}°C")

        return base_setpoint

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
                        logger.debug(
                            f"Forecast at {target_time}: {temp}°C (from {dt_str})"
                        )
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
            logger.debug(
                f"Daytime forecast temps ({morning_time}-{evening_time}): min={min(temps)}, max={max(temps)}, avg={sum(temps) / len(temps):.1f}"
            )
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

        Args:
            preferred_off_time: User's preferred off time (used as soft maximum)
            target_temp: Target room temperature (assumed room temp when heating ON)
            min_overnight_temp: Hard minimum bedroom temperature overnight
            min_daytime_temp: Comfort minimum bedroom temperature during the day
            outside_temp: Forecast overnight outside temperature
            target_warm_time: When user wants the house warm
            switch_on_time: Next morning's computed switch-on time

        Returns:
            Optimal switch-off time, or None for continuous heating
        """
        logger.debug(
            f"Switch-off calc: preferred={preferred_off_time}, "
            f"target={target_temp}°C, min_day={min_daytime_temp}°C, "
            f"min_night={min_overnight_temp}°C, "
            f"outside={outside_temp}°C, switch_on={switch_on_time}"
        )

        # Build list of candidate hours from target_warm_time to preferred_off_time
        start_hour = target_warm_time.hour
        end_hour = preferred_off_time.hour

        if start_hour <= end_hour:
            candidate_hours = list(range(start_hour, end_hour + 1))
        else:
            # Wrap around midnight (e.g., warm at 22:00, preferred off at 02:00)
            candidate_hours = list(range(start_hour, 24)) + list(range(0, end_hour + 1))

        # Search for earliest safe switch-off time
        for candidate_hour in candidate_hours:
            candidate_time = time(candidate_hour, 0)

            # Hours from candidate off time until heating restarts
            hours_off = self._hours_until(candidate_time, switch_on_time)

            # Skip if off period would be too short
            if hours_off < DEFAULTS.min_off_duration_hours:
                continue

            # Simulate cooling from target_temp until switch_on_time
            # (after switch-on, the room is being heated so cooling curve doesn't apply)
            cooling = self.thermal_model.predict_cooling_curve(
                start_temp=target_temp,
                outside_temp=outside_temp,
                hours=int(hours_off),
            )

            # Check if temp stays above the time-aware threshold
            stays_warm = True
            for i, t in enumerate(cooling.temperatures):
                sim_hour = (candidate_hour + i) % 24
                if self._is_daytime(sim_hour, target_warm_time, preferred_off_time):
                    threshold = min_daytime_temp
                else:
                    threshold = min_overnight_temp
                if t < threshold:
                    stays_warm = False
                    break

            if stays_warm:
                logger.info(
                    f"Earliest safe switch-off: {candidate_time.strftime('%H:%M')} "
                    f"(cooling {target_temp}°C -> {cooling.temperatures[-1]:.1f}°C "
                    f"over {hours_off:.0f}h, stays above daytime={min_daytime_temp}°C / "
                    f"overnight={min_overnight_temp}°C)"
                )
                return candidate_time

        # No safe time found before preferred_off_time — search beyond it
        # (extend up to 2 hours past preferred, then give up)
        logger.debug("No safe off time before preferred time, searching later...")
        for extra_hour in range(1, 3):
            candidate_hour = (end_hour + extra_hour) % 24
            candidate_time = time(candidate_hour, 0)

            hours_off = self._hours_until(candidate_time, switch_on_time)
            if hours_off < DEFAULTS.min_off_duration_hours:
                continue

            cooling = self.thermal_model.predict_cooling_curve(
                start_temp=target_temp,
                outside_temp=outside_temp,
                hours=int(hours_off),
            )

            stays_warm = True
            for i, t in enumerate(cooling.temperatures):
                sim_hour = (candidate_hour + i) % 24
                if self._is_daytime(sim_hour, target_warm_time, preferred_off_time):
                    threshold = min_daytime_temp
                else:
                    threshold = min_overnight_temp
                if t < threshold:
                    stays_warm = False
                    break

            if stays_warm:
                logger.info(
                    f"Extended switch-off: {candidate_time.strftime('%H:%M')} "
                    f"({extra_hour}h past preferred)"
                )
                return candidate_time

        # Nothing works — continuous heating
        logger.info("No safe switch-off time found, using continuous heating")
        return None

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

            plans.append(
                HourlyHeatingPlan(
                    hour=hour,
                    system_state="on" if system_on else "off",
                    setpoint=optimal_setpoint if system_on else None,
                    expected_modulation=round(modulation, 1),
                    expected_room_temp=round(current_temp, 1),
                )
            )

        return plans

    def _estimate_temp_range(
        self,
        hours: list[HourlyHeatingPlan],
    ) -> tuple[float, float]:
        """Estimate min and max temperatures from hourly plan."""
        temps = [h.expected_room_temp for h in hours]
        return min(temps), max(temps)

    def _estimate_gas_usage(
        self,
        hours: list[HourlyHeatingPlan],
    ) -> float:
        """Estimate gas usage based on modulation and hours."""
        # Gas base rate is learned/calibrated (kWh/hour at 50% modulation)
        # Vitodens 100-W: ~10 kWh/h at 50% modulation (~19 kW nominal input)
        base_rate = self.thermal_model.gas_base_rate_kwh

        total_kwh = 0
        for h in hours:
            if h.system_state == "on":
                # Scale by modulation (higher modulation = more gas)
                hourly_kwh = base_rate * (h.expected_modulation / 50)
                total_kwh += hourly_kwh

        return round(total_kwh, 1)

    def _calculate_burner_hours(
        self,
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

    def _calculate_avg_modulation(self, hours: list[HourlyHeatingPlan]) -> float:
        """Calculate average modulation when system is ON."""
        on_hours = [h for h in hours if h.system_state == "on"]
        if not on_hours:
            return 0.0

        total_mod = sum(h.expected_modulation for h in on_hours)
        return round(total_mod / len(on_hours), 1)

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
