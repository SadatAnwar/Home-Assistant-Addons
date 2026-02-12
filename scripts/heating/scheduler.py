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

from .config import PREDICTION_CONFIG
from .data_collector import DataCollector
from .ha_client import HAClient
from .optimizer import HeatingOptimizer
from .prediction_tracker import PredictionTracker
from .scheduler_case_engine import UpdateMode, determine_and_calculate, time_ge
from .scheduler_model_ops import should_retrain, train_model
from .scheduler_publish import SchedulerPublisher
from .scheduler_reporting import SchedulerReporter
from .scheduler_settings import SchedulerSettings
from .thermal_model import ThermalModel

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

        self.settings = SchedulerSettings(self.client)
        self.publisher = SchedulerPublisher(self.client)
        self.reporter = SchedulerReporter(self.model, self.collector, self.tracker)

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
                if existing_record is None or case_label in ("A", "E"):
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
                self._update_ha_helpers(
                    schedule,
                    include_switch_on=(update_mode == UpdateMode.ALL),
                )
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
    ) -> tuple[Any, UpdateMode, str]:
        """Determine the current case and calculate the appropriate schedule."""
        return determine_and_calculate(
            now=now,
            existing_record=existing_record,
            heating_is_on=heating_is_on,
            target_warm_time=target_warm_time,
            target_night_time=target_night_time,
            settings=settings,
            current_state=current_state,
            optimizer=self.optimizer,
            parse_time=self._parse_time,
        )

    def _time_ge(self, a: time, b: time) -> bool:
        """Check if time a >= time b (simple comparison, no midnight wrapping)."""
        return time_ge(a, b)

    def _get_helper_value(self, entity_id: str, default, cast=str):
        """Read a helper entity, returning default if unavailable."""
        return self.settings.get_helper_value(entity_id, default, cast)

    def _get_user_settings(self) -> dict[str, Any]:
        """Get user-configured settings from HA helpers."""
        return self.settings.get_user_settings()

    def _should_retrain(self) -> bool:
        """Determine if model should be retrained."""
        return should_retrain(self.model)

    def _train_model(self) -> dict[str, Any]:
        """Train the thermal model on historical data."""
        return train_model(self.collector, self.model)

    def _update_ha_helpers(self, schedule, include_switch_on: bool = True) -> None:
        """Update HA helper entities with computed schedule."""
        self.publisher.update_ha_helpers(schedule, include_switch_on=include_switch_on)

    def _send_schedule_notification(self, schedule) -> None:
        """Send notification with today's heating schedule."""
        self.publisher.send_schedule_notification(schedule)

    def _parse_time(self, time_str: str) -> time:
        """Parse time string (HH:MM or HH:MM:SS) to time object."""
        return self.settings.parse_time(time_str)

    def _log_forecast_segments(
        self,
        forecast: list[dict],
        switch_on_time: time,
        switch_off_time: time | None,
    ) -> None:
        """Log hourly forecast split by heating ON and OFF periods."""
        self.publisher.log_forecast_segments(forecast, switch_on_time, switch_off_time)

    def show_model_info(self) -> None:
        """Display information about the current thermal model."""
        self.reporter.show_model_info()

    def show_review(self, date_str: str | None = None) -> None:
        """Show prediction review for a specific date (defaults to yesterday)."""
        self.reporter.show_review(date_str)

    def show_history(self, days: int = 7) -> None:
        """Show prediction history for the last N days."""
        self.reporter.show_history(days)

    def show_current_state(self) -> None:
        """Display current state of all heating-related entities."""
        self.reporter.show_current_state()


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
