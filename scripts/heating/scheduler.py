#!/usr/bin/env python3
"""Main scheduler for adaptive heating optimization.

Orchestrates data collection, model training, optimization, and HA updates.
Can run at any time of day — determines the correct action based on current
time relative to the heating cycle (Cases A-E).
"""

import argparse
import logging
import sys
from datetime import datetime, time, timedelta
from typing import Any

from .config import (
    DEFAULTS,
    HELPERS,
    MODEL_CONFIG,
    NOTIFICATION_SERVICE,
    PREDICTION_CONFIG,
)
from .data_collector import DataCollector
from .ha_client import HAClient
from .optimizer import DailyHeatingSchedule, HeatingOptimizer, HourlyHeatingPlan
from .prediction_tracker import (
    PredictionTracker,
    format_error_summary,
    format_review_report,
)
from .thermal_model import ThermalModel

# Sentinel for cases where we only update switch-off and setpoint
_SWITCH_OFF_AND_SETPOINT_ONLY = "switch_off_and_setpoint_only"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class HeatingScheduler:
    """Main scheduler that orchestrates the heating optimization system."""

    def __init__(self):
        self.client = HAClient()
        self.collector = DataCollector(self.client)
        self.model = ThermalModel()
        self.optimizer: HeatingOptimizer | None = None
        self.tracker = PredictionTracker(self.client)

        # Try to load existing model
        if self.model.load():
            logger.info(f"Loaded thermal model (trained: {self.model.last_trained})")
            self.optimizer = HeatingOptimizer(self.model)
        else:
            logger.info("No existing thermal model found")

    def run(
        self,
        force_train: bool = False,
        dry_run: bool = False,
        shadow: bool = False,
        recommend_only: bool = False,
    ) -> dict[str, Any]:
        """Run the optimization cycle.

        Can be run at any time of day. Determines what to do based on
        current time relative to the heating cycle.

        Args:
            force_train: Force model retraining even if recent model exists
            dry_run: Calculate schedule but don't write anything
            shadow: Save predictions locally but don't update HA or adjust model
            recommend_only: Save prediction, print recommendation, don't update HA

        Returns:
            Dictionary with run results
        """
        results: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "errors": [],
        }

        try:
            # 1. Get user settings
            settings = self._get_user_settings()
            results["settings"] = settings
            logger.info(
                f"User settings: warm by {settings['target_warm_time']}, "
                f"off at {settings['preferred_off_time']}, target {settings['target_temp']}°C"
            )
            logger.info(
                f"Min temps: daytime {settings['min_daytime_temp']}°C, "
                f"overnight {settings['min_bedroom_temp']}°C"
            )

            # 2. Collect current state
            current_state = self.collector.get_current_state()
            results["current_state"] = {
                "bedroom_temp": current_state.get("room_temps", {}).get("bedroom"),
                "outside_temp": current_state.get("outside_temp"),
                "hvac_mode": current_state.get("hvac_mode"),
            }
            logger.info(
                f"Current state: bedroom {current_state['room_temps'].get('bedroom', 'N/A')}°C, "
                f"outside {current_state.get('outside_temp', 'N/A')}°C, "
                f"hvac {current_state.get('hvac_mode', 'N/A')}"
            )

            # Log forecast availability
            forecast = current_state.get("forecast", [])
            if forecast:
                logger.info(f"Weather forecast available: {len(forecast)} hours ahead")
            else:
                logger.warning(
                    "No weather forecast available - using current temps as fallback"
                )

            # 3. Train/update model if needed
            if force_train or self._should_retrain():
                logger.info("Training thermal model...")
                training_result = self._train_model()
                results["training"] = training_result
                if "error" in training_result:
                    logger.warning(f"Training warning: {training_result['error']}")
                else:
                    logger.info(
                        f"Model trained on {training_result.get('samples', 0)} samples"
                    )

            if self.optimizer is None:
                self.optimizer = HeatingOptimizer(self.model)

            # 4. Collect yesterday's actuals (once per day)
            if not dry_run:
                yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
                yesterday_record = self.tracker._load_record(yesterday)
                if yesterday_record and yesterday_record.actuals:
                    logger.info("Actuals already collected for yesterday, skipping")
                else:
                    logger.info(f"Collecting actuals for {yesterday}...")
                    actuals = self.tracker.collect_actuals(yesterday)
                    if actuals:
                        results["actuals_collected"] = True
                        logger.info(f"Collected actuals for {yesterday}")
                    else:
                        logger.info(f"No actuals available for {yesterday}")

            # 5. Apply coefficient adjustments (once per day, live mode only)
            if not dry_run and not shadow:
                today_str = datetime.now().strftime("%Y-%m-%d")
                today_record = self.tracker._load_record(today_str)
                if today_record and today_record.coefficients_adjusted:
                    logger.info(
                        "Coefficient adjustments already applied today, skipping"
                    )
                else:
                    error_summary = self.tracker.get_error_summary(
                        days=PREDICTION_CONFIG.min_sample_days
                    )
                    if (
                        error_summary["sample_count"]
                        >= PREDICTION_CONFIG.min_sample_days
                    ):
                        adjustments = self.tracker.suggest_coefficient_adjustments(
                            error_summary,
                            self.model.k,
                            self.model.mean_heating_rate,
                            self.model.gas_base_rate_kwh,
                        )
                        if adjustments:
                            applied = self.model.apply_adjustments(adjustments)
                            for adj in applied:
                                logger.info(f"Applied adjustment: {adj}")
                            results["adjustments_applied"] = applied
                            self.model.save()

                        # Mark coefficients as adjusted for today
                        if today_record:
                            today_record.coefficients_adjusted = True
                            self.tracker._save_record(today_record)
            elif shadow:
                logger.info("SHADOW MODE - skipping coefficient adjustments")

            # 6. Determine current heating state and calculate schedule
            now = datetime.now()
            today_str = now.strftime("%Y-%m-%d")
            existing_record = self.tracker._load_record(today_str)
            heating_is_on = current_state.get("hvac_mode") in ("heat", "auto")

            target_warm_time = self._parse_time(settings["target_warm_time"])
            target_night_time = self._parse_time(settings["preferred_off_time"])

            schedule, update_mode, case_label = self._determine_and_calculate(
                now=now,
                existing_record=existing_record,
                heating_is_on=heating_is_on,
                target_warm_time=target_warm_time,
                target_night_time=target_night_time,
                settings=settings,
                current_state=current_state,
            )

            logger.info(f"Case {case_label}")

            # Apply safety overrides
            schedule, overrides = self.optimizer.apply_safety_overrides(
                schedule=schedule,
                current_temps=current_state.get("room_temps", {}),
                min_overnight_temp=settings["min_bedroom_temp"],
                min_daytime_temp=settings["min_daytime_temp"],
                target_warm_time=target_warm_time,
                preferred_off_time=target_night_time,
            )

            off_time_str = (
                schedule.switch_off_time.strftime("%H:%M")
                if schedule.switch_off_time
                else "CONTINUOUS"
            )
            results["schedule"] = {
                "switch_on_time": schedule.switch_on_time.strftime("%H:%M"),
                "switch_off_time": off_time_str,
                "optimal_setpoint": schedule.optimal_setpoint,
                "expected_switch_on_temp": schedule.expected_switch_on_temp,
                "expected_min_temp": schedule.expected_min_temp,
                "expected_max_temp": schedule.expected_max_temp,
                "expected_gas_usage": schedule.expected_gas_usage,
                "solar_contribution": schedule.solar_contribution,
                "reasoning": schedule.reasoning,
                "overrides": overrides,
                "case": case_label,
            }

            switch_on_temp_str = (
                f"{schedule.expected_switch_on_temp:.1f}°C"
                if schedule.expected_switch_on_temp is not None
                else "N/A"
            )
            logger.info(
                f"Schedule: ON at {schedule.switch_on_time.strftime('%H:%M')}, "
                f"OFF at {off_time_str}, setpoint {schedule.optimal_setpoint}°C"
            )
            logger.info(f"Expected room temp at switch-on: {switch_on_temp_str}")
            if schedule.expected_target_time_temp is not None:
                logger.info(
                    f"Expected temp at target warm time "
                    f"({settings['target_warm_time']}): "
                    f"{schedule.expected_target_time_temp:.1f}°C"
                )
            if schedule.expected_switch_off_temp is not None:
                logger.info(
                    f"Expected temp at switch-off: "
                    f"{schedule.expected_switch_off_temp:.1f}°C"
                )
            if (
                schedule.expected_min_temp is not None
                and schedule.expected_max_temp is not None
            ):
                logger.info(
                    f"Expected temp range: "
                    f"{schedule.expected_min_temp:.1f} - {schedule.expected_max_temp:.1f}°C"
                )
            if schedule.solar_contribution > 0.5:
                logger.info(f"Solar gain: +{schedule.solar_contribution:.1f}°C")
            logger.info(f"Expected gas usage: ~{schedule.expected_gas_usage:.1f} kWh")

            # Log hourly forecast split by heating ON/OFF periods
            self._log_forecast_segments(
                forecast=current_state.get("forecast", []),
                switch_on_time=schedule.switch_on_time,
                switch_off_time=schedule.switch_off_time,
            )

            if overrides:
                for override in overrides:
                    logger.warning(f"Safety override: {override}")

            # 7. Save prediction/adjustment
            if not dry_run:
                if existing_record is None or case_label == "A":
                    # First run of the day — save as primary prediction
                    logger.info("Saving primary prediction...")
                    self.tracker.save_prediction(schedule, settings["target_warm_time"])
                    results["prediction_saved"] = True
                else:
                    # Subsequent run — save as adjustment
                    bedroom_temp = current_state.get("room_temps", {}).get(
                        "bedroom", 0.0
                    )
                    logger.info("Saving adjustment...")
                    self.tracker.save_adjustment(
                        date_str=today_str,
                        switch_off_time=off_time_str,
                        setpoint=schedule.optimal_setpoint,
                        actual_bedroom_temp=bedroom_temp,
                        reasoning=schedule.reasoning,
                    )
                    results["adjustment_saved"] = True
            else:
                logger.info("DRY RUN - skipping prediction/adjustment save")

            # 8. Update HA helpers
            if not dry_run and not shadow and not recommend_only:
                logger.info("Updating Home Assistant helpers...")
                if update_mode == _SWITCH_OFF_AND_SETPOINT_ONLY:
                    # Case D: only update switch-off and setpoint
                    self._update_ha_switch_off_and_setpoint(schedule)
                else:
                    self._update_ha_helpers(schedule)
                results["ha_updated"] = True

                # 9. Send notification
                self._send_schedule_notification(schedule)
                results["notification_sent"] = True
            elif recommend_only:
                logger.info("RECOMMEND ONLY - prediction saved, skipping HA updates")
                results["recommend_only"] = True
            elif shadow:
                logger.info("SHADOW MODE - predictions saved, skipping HA updates")
                results["shadow"] = True
            else:
                logger.info("DRY RUN - skipping HA updates")
                results["dry_run"] = True

            results["success"] = True

        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            results["errors"].append(str(e))

        return results

    def _determine_and_calculate(
        self,
        now: datetime,
        existing_record,
        heating_is_on: bool,
        target_warm_time: time,
        target_night_time: time,
        settings: dict[str, Any],
        current_state: dict[str, Any],
    ) -> tuple[Any, str, str]:
        """Determine the current case and calculate the appropriate schedule.

        Returns:
            (schedule, update_mode, case_label) where update_mode is either
            "all" or _SWITCH_OFF_AND_SETPOINT_ONLY
        """
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
            schedule = self.optimizer.calculate_optimal_schedule(**common_kwargs)
            return schedule, "all", "A"

        # We have an existing prediction — determine sub-case
        pred = existing_record.prediction
        switch_on_time = self._parse_time(pred.switch_on_time)
        switch_off_time = (
            self._parse_time(pred.switch_off_time)
            if pred.switch_off_time != "CONTINUOUS"
            else None
        )

        current_time = now.time()

        if heating_is_on:
            # CASE D: Heating is ON — mid-day recalculation
            logger.info("Case D: Heating ON — recalculating switch-off and setpoint")
            original_hourly_plans = self._load_original_hourly_plans(existing_record)
            schedule = self.optimizer.recalculate_mid_day(
                **common_kwargs,
                current_time=now,
                original_switch_on_time=switch_on_time,
                original_setpoint=pred.setpoint,
                original_hourly_plans=original_hourly_plans,
            )
            return schedule, _SWITCH_OFF_AND_SETPOINT_ONLY, "D"

        # Heating is OFF
        if switch_off_time is not None and self._time_ge(current_time, switch_off_time):
            # Past switch-off time — could be after today's cycle or before tomorrow's
            # Check if we're past the preferred off time too
            if self._time_ge(current_time, target_night_time):
                # CASE E: Heating done for today — this is effectively tomorrow's calc
                logger.info(
                    "Case E: Past switch-off and night time — "
                    "calculating tomorrow's schedule"
                )
                tomorrow = now + timedelta(days=1)
                schedule = self.optimizer.calculate_optimal_schedule(
                    **common_kwargs,
                    current_time=tomorrow,
                )
                return schedule, "all", "E"

        # Heating OFF, before switch-on time
        if not self._time_ge(current_time, switch_on_time):
            # CASE B: Before heating starts — recalculate with fresh conditions
            logger.info("Case B: Before switch-on — recalculating with fresh data")
            schedule = self.optimizer.calculate_optimal_schedule(**common_kwargs)
            return schedule, "all", "B"

        # Heating OFF but past switch-on time and before switch-off
        if self._time_ge(current_time, switch_on_time) and (
            switch_off_time is None or not self._time_ge(current_time, switch_off_time)
        ):
            if not self._time_ge(current_time, target_warm_time):
                # CASE C: Should have started but hasn't — trigger immediately
                logger.info(
                    "Case C: Past switch-on but heating OFF — triggering now+2min"
                )
                immediate_on = (now + timedelta(minutes=2)).time()
                original_hourly_plans = self._load_original_hourly_plans(
                    existing_record
                )
                schedule = self.optimizer.recalculate_mid_day(
                    **common_kwargs,
                    current_time=now,
                    original_switch_on_time=immediate_on,
                    original_setpoint=pred.setpoint,
                    original_hourly_plans=original_hourly_plans,
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
                    + [
                        f"Late start: switch-on set to {immediate_on.strftime('%H:%M')}"
                    ],
                    expected_switch_on_temp=schedule.expected_switch_on_temp,
                    expected_target_time_temp=schedule.expected_target_time_temp,
                    expected_switch_off_temp=schedule.expected_switch_off_temp,
                    expected_burner_hours=schedule.expected_burner_hours,
                    expected_avg_modulation=schedule.expected_avg_modulation,
                )
                return schedule, "all", "C"

        # Fallback: heating OFF, past warm time but before switch-off
        # This could happen if heating was turned off manually
        # CASE E: treat as done for today
        logger.info(
            "Case E: Heating OFF post warm-time — calculating tomorrow's schedule"
        )
        tomorrow = now + timedelta(days=1)
        schedule = self.optimizer.calculate_optimal_schedule(
            **common_kwargs,
            current_time=tomorrow,
        )
        return schedule, "all", "E"

    def _time_ge(self, a: time, b: time) -> bool:
        """Check if time a >= time b (simple comparison, no midnight wrapping)."""
        return (a.hour, a.minute) >= (b.hour, b.minute)

    def _load_original_hourly_plans(self, record) -> list[HourlyHeatingPlan] | None:
        """Try to reconstruct hourly plans from a saved prediction.

        Since we don't persist hourly plans in JSONL, we return None.
        The optimizer handles None gracefully (skips setpoint adjustment).
        """
        return None

    def _get_user_settings(self) -> dict[str, Any]:
        """Get user-configured settings from HA helpers."""
        settings = {}

        # Target warm time
        state = self.client.get_state(HELPERS.target_warm_time)
        if state and state.state not in ("unknown", "unavailable"):
            settings["target_warm_time"] = state.state
        else:
            settings["target_warm_time"] = DEFAULTS.target_warm_time

        # Preferred off time
        state = self.client.get_state(HELPERS.preferred_off_time)
        if state and state.state not in ("unknown", "unavailable"):
            settings["preferred_off_time"] = state.state
        else:
            settings["preferred_off_time"] = DEFAULTS.preferred_off_time

        # Target temperature
        state = self.client.get_state(HELPERS.target_temp)
        if state and state.state not in ("unknown", "unavailable"):
            try:
                settings["target_temp"] = float(state.state)
            except ValueError:
                settings["target_temp"] = DEFAULTS.target_temp
        else:
            settings["target_temp"] = DEFAULTS.target_temp

        # Minimum bedroom temperature (overnight)
        state = self.client.get_state(HELPERS.min_bedroom_temp)
        if state and state.state not in ("unknown", "unavailable"):
            try:
                settings["min_bedroom_temp"] = float(state.state)
            except ValueError:
                settings["min_bedroom_temp"] = DEFAULTS.min_bedroom_temp
        else:
            settings["min_bedroom_temp"] = DEFAULTS.min_bedroom_temp

        # Minimum daytime temperature
        state = self.client.get_state(HELPERS.min_daytime_temp)
        if state and state.state not in ("unknown", "unavailable"):
            try:
                settings["min_daytime_temp"] = float(state.state)
            except ValueError:
                settings["min_daytime_temp"] = DEFAULTS.min_daytime_temp
        else:
            settings["min_daytime_temp"] = DEFAULTS.min_daytime_temp

        # Daytime min must be >= overnight min
        settings["min_daytime_temp"] = max(
            settings["min_daytime_temp"], settings["min_bedroom_temp"]
        )

        return settings

    def _should_retrain(self) -> bool:
        """Determine if model should be retrained."""
        if self.model.last_trained is None:
            return True

        # Retrain if model is older than 7 days
        age = datetime.now() - self.model.last_trained
        return age.days >= 7

    def _train_model(self) -> dict[str, Any]:
        """Train the thermal model on historical data."""
        # Collect training data
        data = self.collector.build_training_dataset(days=MODEL_CONFIG.history_days)

        if data.empty:
            return {"error": "No training data available"}

        if len(data) < 100:
            return {
                "error": f"Insufficient data ({len(data)} samples)",
                "samples": len(data),
            }

        # Train model
        metrics = self.model.train(data)

        # Save model
        self.model.save()

        return {"samples": len(data), "metrics": metrics}

    def _update_ha_helpers(self, schedule) -> None:
        """Update HA helper entities with computed schedule."""
        # Update switch-on time
        success = self.client.set_input_datetime(
            HELPERS.switch_on_time,
            schedule.switch_on_time.strftime("%H:%M"),
        )
        if not success:
            logger.warning("Failed to update switch-on time helper")

        # Update switch-off time (use switch-on time if continuous heating)
        # When switch_off_time is None, we set it equal to switch_on_time
        # to indicate continuous heating (the automation won't trigger a real off)
        if schedule.switch_off_time is not None:
            off_time_str = schedule.switch_off_time.strftime("%H:%M")
        else:
            # Continuous heating - set off time to same as on time
            # This effectively disables the off automation
            off_time_str = schedule.switch_on_time.strftime("%H:%M")
            logger.info("Continuous heating mode - setting off time = on time")

        success = self.client.set_input_datetime(
            HELPERS.switch_off_time,
            off_time_str,
        )
        if not success:
            logger.warning("Failed to update switch-off time helper")

        # Update optimal setpoint
        success = self.client.set_input_number(
            HELPERS.optimal_setpoint,
            schedule.optimal_setpoint,
        )
        if not success:
            logger.warning("Failed to update optimal setpoint helper")

    def _update_ha_switch_off_and_setpoint(self, schedule) -> None:
        """Update only switch-off time and setpoint (for mid-day adjustments)."""
        if schedule.switch_off_time is not None:
            off_time_str = schedule.switch_off_time.strftime("%H:%M")
        else:
            off_time_str = schedule.switch_on_time.strftime("%H:%M")
            logger.info("Continuous heating mode - setting off time = on time")

        success = self.client.set_input_datetime(
            HELPERS.switch_off_time,
            off_time_str,
        )
        if not success:
            logger.warning("Failed to update switch-off time helper")

        success = self.client.set_input_number(
            HELPERS.optimal_setpoint,
            schedule.optimal_setpoint,
        )
        if not success:
            logger.warning("Failed to update optimal setpoint helper")

    def _send_schedule_notification(self, schedule) -> None:
        """Send notification with today's heating schedule."""
        off_time_str = (
            schedule.switch_off_time.strftime("%H:%M")
            if schedule.switch_off_time
            else "CONTINUOUS"
        )
        switch_on_temp_str = (
            f"{schedule.expected_switch_on_temp:.1f}°C"
            if schedule.expected_switch_on_temp is not None
            else "N/A"
        )
        message = (
            f"Heating Schedule:\n"
            f"ON: {schedule.switch_on_time.strftime('%H:%M')} (expect {switch_on_temp_str})\n"
            f"OFF: {off_time_str}\n"
            f"Setpoint: {schedule.optimal_setpoint}°C\n"
            f"Expected range: {schedule.expected_min_temp}-{schedule.expected_max_temp}°C"
        )

        if schedule.solar_contribution > 0.5:
            message += f"\nSolar bonus: +{schedule.solar_contribution:.1f}°C"

        self.client.send_notification(
            message=message,
            title="Heating Plan",
            service=NOTIFICATION_SERVICE,
        )

    def _parse_time(self, time_str: str) -> time:
        """Parse time string (HH:MM or HH:MM:SS) to time object."""
        parts = time_str.split(":")
        return time(int(parts[0]), int(parts[1]))

    def _log_forecast_segments(
        self,
        forecast: list[dict],
        switch_on_time: time,
        switch_off_time: time | None,
    ) -> None:
        """Log hourly forecast split by heating ON and OFF periods."""
        if not forecast:
            return

        on_entries: list[str] = []
        off_entries: list[str] = []

        on_minutes = switch_on_time.hour * 60 + switch_on_time.minute
        # If no switch-off, treat entire day as ON
        off_minutes = (
            (switch_off_time.hour * 60 + switch_off_time.minute)
            if switch_off_time
            else 24 * 60
        )

        for entry in forecast[:24]:  # Next 24 hours only
            dt_str = entry.get("datetime", "")
            temp = entry.get("temperature")
            if not dt_str or temp is None:
                continue

            try:
                dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                hour = dt.hour
                hour_minutes = hour * 60
                label = f"{hour:02d}:00={temp:.0f}°C"

                if on_minutes <= off_minutes:
                    is_on = on_minutes <= hour_minutes < off_minutes
                else:
                    is_on = hour_minutes >= on_minutes or hour_minutes < off_minutes

                if is_on:
                    on_entries.append(label)
                else:
                    off_entries.append(label)
            except ValueError:
                continue

        on_time_str = switch_on_time.strftime("%H:%M")
        off_time_str = switch_off_time.strftime("%H:%M") if switch_off_time else "N/A"

        if on_entries:
            logger.info(
                f"Forecast (heating ON {on_time_str}-{off_time_str}): "
                f"{' '.join(on_entries)}"
            )
        if off_entries:
            logger.info(
                f"Forecast (heating OFF {off_time_str}-{on_time_str}): "
                f"{' '.join(off_entries)}"
            )

    def show_model_info(self) -> None:
        """Display information about the current thermal model."""
        info = self.model.get_model_info()

        print("\nThermal Model Information:")
        print(f"  Trained: {info['trained']}")
        if info["last_trained"]:
            print(f"  Last trained: {info['last_trained']}")
        print(f"  Training samples: {info['training_samples']}")
        print("\nLearned parameters:")
        print(f"  Mean cooling rate: {info['mean_cooling_rate']:.3f} °C/hour")
        print(f"  Mean heating rate: {info['mean_heating_rate']:.3f} °C/hour")
        print(f"  Solar gain coefficient: {info['solar_gain_coefficient']:.4f}")
        print(
            f"  Cooling rate k: {info['k']:.6f} (τ = {info['time_constant_hours']} hours)"
        )
        print(f"  Gas base rate: {info['gas_base_rate_kwh']:.1f} kWh/h @50% modulation")
        print("\nModels available:")
        print(f"  Heating rate model: {info['has_heating_model']}")
        print(f"  Cooling rate model: {info['has_cooling_model']}")
        print(f"  Modulation model: {info['has_modulation_model']}")

    def show_review(self, date_str: str | None = None) -> None:
        """Show prediction review for a specific date (defaults to yesterday)."""
        if date_str is None:
            date_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        # Ensure we have actuals for this date
        record = self.tracker._load_record(date_str)
        if not record:
            print(f"No prediction record found for {date_str}")
            return

        if not record.actuals:
            print(f"Collecting actuals for {date_str}...")
            self.tracker.collect_actuals(date_str)
            record = self.tracker._load_record(date_str)

        if record:
            print()
            print(format_review_report(record))
            print()

            # Show error summary
            summary = self.tracker.get_error_summary()
            print(format_error_summary(summary))

    def show_history(self, days: int = 7) -> None:
        """Show prediction history for the last N days."""
        records = self.tracker.get_history(days)

        if not records:
            print("No prediction history available.")
            return

        print(f"\nPrediction History (last {days} days):\n")
        print(
            f"{'Date':<12} {'On Time':<8} {'Off Time':<10} {'Setpoint':<9} "
            f"{'Pred On°C':<10} {'Act On°C':<10} {'Error':<8}"
        )
        print("-" * 75)

        for record in records:
            p = record.prediction
            a = record.actuals
            e = record.errors

            act_on = (
                f"{a.actual_switch_on_temp:.1f}"
                if a and a.actual_switch_on_temp
                else "N/A"
            )
            error = (
                f"{e.switch_on_temp_error:+.1f}"
                if e and e.switch_on_temp_error
                else "N/A"
            )

            print(
                f"{p.date:<12} {p.switch_on_time:<8} {p.switch_off_time:<10} "
                f"{p.setpoint:<9.1f} {p.expected_switch_on_temp:<10.1f} "
                f"{act_on:<10} {error:<8}"
            )

        print()

        # Show summary
        summary = self.tracker.get_error_summary(days)
        print(format_error_summary(summary))

    def show_current_state(self) -> None:
        """Display current state of all heating-related entities."""
        state = self.collector.get_current_state()

        print("\nCurrent State:")
        print("\nRoom Temperatures:")
        for room, temp in state.get("room_temps", {}).items():
            print(f"  {room}: {temp}°C")

        print(f"\nOutside: {state.get('outside_temp', 'N/A')}°C")
        print("\nHeating:")
        print(f"  Mode: {state.get('hvac_mode', 'N/A')}")
        print(f"  Setpoint: {state.get('setpoint', 'N/A')}°C")
        print(f"  Burner modulation: {state.get('burner_modulation', 'N/A')}%")

        print("\nSun:")
        print(f"  Elevation: {state.get('sun_elevation', 'N/A')}°")
        print(f"  Azimuth: {state.get('sun_azimuth', 'N/A')}°")

        print(f"\nWeather: {state.get('weather_condition', 'N/A')}")
        print(f"Cloud coverage: {state.get('cloud_coverage', 'N/A')}%")


def main() -> None:
    """Main entry point for the heating scheduler."""
    parser = argparse.ArgumentParser(
        description="Adaptive heating optimization scheduler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run daily optimization:
    python -m scripts.heating.scheduler run

  Force model retraining:
    python -m scripts.heating.scheduler run --train

  Dry run (calculate only, no writes):
    python -m scripts.heating.scheduler run --dry-run

  Shadow mode (save predictions locally, don't touch HA):
    python -m scripts.heating.scheduler run --shadow

  Recommend only (save prediction, print result, don't update HA):
    python -m scripts.heating.scheduler run --recommend-only

  Show model information:
    python -m scripts.heating.scheduler info

  Show current state:
    python -m scripts.heating.scheduler state

  Review yesterday's predictions vs actuals:
    python -m scripts.heating.scheduler review

  Review a specific date:
    python -m scripts.heating.scheduler review 2026-02-01

  Show 7-day prediction history:
    python -m scripts.heating.scheduler history

  Show 14-day prediction history:
    python -m scripts.heating.scheduler history --days 14

Crontab example (can run at any time; multiple runs per day are safe):
  0 */4 * * * cd /path/to/home_assistant && .venv/bin/python -m scripts.heating.scheduler run
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Run command
    run_parser = subparsers.add_parser("run", help="Run the optimization cycle")
    run_parser.add_argument(
        "--train", action="store_true", help="Force model retraining"
    )
    run_parser.add_argument(
        "--dry-run", action="store_true", help="Don't write anything (pure calculation)"
    )
    run_parser.add_argument(
        "--shadow",
        action="store_true",
        help="Save predictions locally but don't update HA or adjust model",
    )
    run_parser.add_argument(
        "--recommend-only",
        action="store_true",
        help="Print recommendation without updating HA helpers",
    )
    run_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )

    # Info command
    subparsers.add_parser("info", help="Show thermal model information")

    # State command
    subparsers.add_parser("state", help="Show current heating state")

    # Review command
    review_parser = subparsers.add_parser(
        "review", help="Review prediction vs actuals for a date"
    )
    review_parser.add_argument(
        "date", nargs="?", help="Date to review (YYYY-MM-DD), defaults to yesterday"
    )

    # History command
    history_parser = subparsers.add_parser("history", help="Show prediction history")
    history_parser.add_argument(
        "--days", "-d", type=int, default=7, help="Number of days to show (default: 7)"
    )

    args = parser.parse_args()

    if args.command == "run":
        if getattr(args, "verbose", False):
            logging.getLogger().setLevel(logging.DEBUG)

        scheduler = HeatingScheduler()
        result = scheduler.run(
            force_train=getattr(args, "train", False),
            dry_run=getattr(args, "dry_run", False),
            shadow=getattr(args, "shadow", False),
            recommend_only=getattr(args, "recommend_only", False),
        )

        if result["success"]:
            logger.info("Scheduler run completed successfully")
            if "schedule" in result:
                off_display = result["schedule"]["switch_off_time"]
                switch_on_temp = result["schedule"].get("expected_switch_on_temp")
                temp_display = f"{switch_on_temp:.1f}°C" if switch_on_temp else "N/A"
                print(
                    f"\nSchedule: ON {result['schedule']['switch_on_time']} / "
                    f"OFF {off_display} / "
                    f"Setpoint {result['schedule']['optimal_setpoint']}°C"
                )
                print(f"Expected room temp at switch-on: {temp_display}")
        else:
            logger.error(f"Scheduler run failed: {result.get('errors', [])}")
            sys.exit(1)

    elif args.command == "info":
        scheduler = HeatingScheduler()
        scheduler.show_model_info()

    elif args.command == "state":
        scheduler = HeatingScheduler()
        scheduler.show_current_state()

    elif args.command == "review":
        scheduler = HeatingScheduler()
        scheduler.show_review(getattr(args, "date", None))

    elif args.command == "history":
        scheduler = HeatingScheduler()
        scheduler.show_history(getattr(args, "days", 7))


if __name__ == "__main__":
    main()
