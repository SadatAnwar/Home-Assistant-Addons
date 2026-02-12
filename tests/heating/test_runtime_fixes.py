import unittest
from datetime import datetime, time, timedelta
from types import SimpleNamespace

from scripts.heating.optimizer import DailyHeatingSchedule, HourlyHeatingPlan
from scripts.heating.prediction_tracker import PredictionTracker
from scripts.heating.scheduler import HeatingScheduler
from scripts.heating.scheduler_case_engine import UpdateMode


class PredictionAdjustmentDirectionTests(unittest.TestCase):
    def test_target_temp_positive_error_decreases_heating_rate(self):
        tracker = PredictionTracker.__new__(PredictionTracker)
        adjustments = tracker.suggest_coefficient_adjustments(
            {
                "sample_count": 7,
                "avg_switch_on_temp_error": None,
                "avg_target_time_temp_error": 1.0,
                "avg_gas_pct_error": None,
            },
            current_k=0.0064,
            current_heating_rate=1.0,
            current_gas_base_rate=10.0,
        )
        self.assertIn("mean_heating_rate", adjustments)
        self.assertLess(adjustments["mean_heating_rate"], 1.0)

    def test_target_temp_negative_error_increases_heating_rate(self):
        tracker = PredictionTracker.__new__(PredictionTracker)
        adjustments = tracker.suggest_coefficient_adjustments(
            {
                "sample_count": 7,
                "avg_switch_on_temp_error": None,
                "avg_target_time_temp_error": -1.0,
                "avg_gas_pct_error": None,
            },
            current_k=0.0064,
            current_heating_rate=1.0,
            current_gas_base_rate=10.0,
        )
        self.assertIn("mean_heating_rate", adjustments)
        self.assertGreater(adjustments["mean_heating_rate"], 1.0)


class CaseEPersistenceTests(unittest.TestCase):
    def test_case_e_saves_primary_prediction_not_adjustment(self):
        scheduler = HeatingScheduler.__new__(HeatingScheduler)

        # Keep runtime isolated from HA/network side effects.
        scheduler._get_user_settings = lambda: {
            "target_warm_time": "08:00:00",
            "preferred_off_time": "23:00:00",
            "target_temp": 20.0,
            "min_bedroom_temp": 18.0,
            "min_daytime_temp": 19.5,
        }
        scheduler._should_retrain = lambda: False
        scheduler._parse_time = lambda value: time(int(value.split(":")[0]), int(value.split(":")[1]))
        scheduler._log_forecast_segments = lambda **kwargs: None
        scheduler._update_ha_helpers = lambda *args, **kwargs: None
        scheduler._send_schedule_notification = lambda *args, **kwargs: None

        scheduler.collector = SimpleNamespace(
            get_current_state=lambda: {
                "room_temps": {"bedroom": 19.6},
                "outside_temp": 2.0,
                "hvac_mode": "off",
                "forecast": [],
            }
        )

        tomorrow = datetime.now() + timedelta(days=1)
        schedule = DailyHeatingSchedule(
            date=tomorrow,
            hours=[
                HourlyHeatingPlan(
                    hour=0,
                    system_state="off",
                    setpoint=None,
                    expected_modulation=0.0,
                    expected_room_temp=19.6,
                )
            ],
            switch_on_time=time(6, 0),
            switch_off_time=time(18, 0),
            optimal_setpoint=20.0,
            cycles_per_day=1,
            expected_gas_usage=12.0,
            expected_min_temp=18.0,
            expected_max_temp=21.0,
            solar_contribution=0.0,
            reasoning=["test"],
            expected_switch_on_temp=19.0,
            expected_target_time_temp=20.0,
            expected_switch_off_temp=20.5,
            expected_burner_hours=12.0,
            expected_avg_modulation=30.0,
        )

        scheduler.optimizer = SimpleNamespace(
            apply_safety_overrides=lambda **kwargs: (kwargs["schedule"], []),
        )
        scheduler._determine_and_calculate = lambda **kwargs: (schedule, UpdateMode.ALL, "E")

        call_state = {"save_prediction": 0, "save_adjustment": 0}
        today_key = datetime.now().strftime("%Y-%m-%d")
        yesterday_key = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        existing = SimpleNamespace(prediction=SimpleNamespace(switch_on_time="06:00", switch_off_time="18:00"))

        class _Tracker:
            def _load_record(self, date_str):
                if date_str == today_key:
                    return existing
                if date_str == yesterday_key:
                    return SimpleNamespace(actuals=True)
                return None

            def save_prediction(self, schedule_obj, target_warm_time):
                call_state["save_prediction"] += 1

            def save_adjustment(self, **kwargs):
                call_state["save_adjustment"] += 1

        scheduler.tracker = _Tracker()
        scheduler.model = SimpleNamespace(k=0.0064, mean_heating_rate=1.0, gas_base_rate_kwh=10.0)

        result = scheduler.run(dry_run=False, shadow=True)
        self.assertTrue(result["success"])
        self.assertEqual(call_state["save_prediction"], 1)
        self.assertEqual(call_state["save_adjustment"], 0)


if __name__ == "__main__":
    unittest.main()
