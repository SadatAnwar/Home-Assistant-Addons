import unittest
from datetime import datetime, time
from types import SimpleNamespace

from scripts.heating.optimizer import DailyHeatingSchedule, HourlyHeatingPlan
from scripts.heating.scheduler import HeatingScheduler
from scripts.heating.scheduler_case_engine import UpdateMode, determine_and_calculate


class _FakeOptimizer:
    def __init__(self):
        self.calls = []

    def _schedule(self, current_time: datetime | None = None) -> DailyHeatingSchedule:
        dt = current_time or datetime(2026, 2, 12, 0, 0)
        return DailyHeatingSchedule(
            date=dt,
            hours=[
                HourlyHeatingPlan(
                    hour=0,
                    system_state="off",
                    setpoint=None,
                    expected_modulation=0.0,
                    expected_room_temp=19.5,
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

    def calculate_optimal_schedule(self, **kwargs):
        self.calls.append(("calculate", kwargs))
        return self._schedule(kwargs.get("current_time"))

    def recalculate_mid_day(self, **kwargs):
        self.calls.append(("recalculate", kwargs))
        return self._schedule(kwargs.get("current_time"))


class SchedulerCaseEngineTests(unittest.TestCase):
    def _settings(self):
        return {
            "target_temp": 20.0,
            "min_bedroom_temp": 18.0,
            "min_daytime_temp": 19.5,
        }

    def _state(self):
        return {
            "room_temps": {"bedroom": 19.6},
            "outside_temp": 2.0,
            "forecast": [],
        }

    def _parse_time(self, value: str) -> time:
        hh, mm = value.split(":")[:2]
        return time(int(hh), int(mm))

    def _existing_record(self, on: str = "06:00", off: str = "18:00"):
        return SimpleNamespace(prediction=SimpleNamespace(switch_on_time=on, switch_off_time=off))

    def test_case_a_first_run(self):
        optimizer = _FakeOptimizer()
        schedule, mode, case = determine_and_calculate(
            now=datetime(2026, 2, 12, 5, 0),
            existing_record=None,
            heating_is_on=False,
            target_warm_time=time(8, 0),
            target_night_time=time(23, 0),
            settings=self._settings(),
            current_state=self._state(),
            optimizer=optimizer,
            parse_time=self._parse_time,
        )
        self.assertEqual(case, "A")
        self.assertEqual(mode, UpdateMode.ALL)
        self.assertEqual(schedule.switch_on_time, time(6, 0))

    def test_case_d_midday_when_heating_on(self):
        optimizer = _FakeOptimizer()
        schedule, mode, case = determine_and_calculate(
            now=datetime(2026, 2, 12, 10, 0),
            existing_record=self._existing_record(),
            heating_is_on=True,
            target_warm_time=time(8, 0),
            target_night_time=time(23, 0),
            settings=self._settings(),
            current_state=self._state(),
            optimizer=optimizer,
            parse_time=self._parse_time,
        )
        self.assertEqual(case, "D")
        self.assertEqual(mode, UpdateMode.SWITCH_OFF_AND_SETPOINT)
        self.assertEqual(schedule.switch_off_time, time(18, 0))

    def test_case_c_overrides_switch_on_to_now_plus_2_minutes(self):
        optimizer = _FakeOptimizer()
        now = datetime(2026, 2, 12, 6, 30)
        schedule, mode, case = determine_and_calculate(
            now=now,
            existing_record=self._existing_record(on="06:00", off="18:00"),
            heating_is_on=False,
            target_warm_time=time(8, 0),
            target_night_time=time(23, 0),
            settings=self._settings(),
            current_state=self._state(),
            optimizer=optimizer,
            parse_time=self._parse_time,
        )
        self.assertEqual(case, "C")
        self.assertEqual(mode, UpdateMode.ALL)
        self.assertEqual(schedule.switch_on_time, time(6, 32))

    def test_case_e_after_night_time_calculates_tomorrow(self):
        optimizer = _FakeOptimizer()
        schedule, mode, case = determine_and_calculate(
            now=datetime(2026, 2, 12, 23, 30),
            existing_record=self._existing_record(on="06:00", off="18:00"),
            heating_is_on=False,
            target_warm_time=time(8, 0),
            target_night_time=time(23, 0),
            settings=self._settings(),
            current_state=self._state(),
            optimizer=optimizer,
            parse_time=self._parse_time,
        )
        self.assertEqual(case, "E")
        self.assertEqual(mode, UpdateMode.ALL)
        self.assertEqual(schedule.date.date().isoformat(), "2026-02-13")


class SchedulerFacadeTests(unittest.TestCase):
    def test_run_returns_top_level_schedule_keys_in_dry_run(self):
        scheduler = HeatingScheduler.__new__(HeatingScheduler)

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

        schedule = DailyHeatingSchedule(
            date=datetime(2026, 2, 12, 6, 0),
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
        scheduler._determine_and_calculate = lambda **kwargs: (schedule, UpdateMode.ALL, "A")

        scheduler.tracker = SimpleNamespace(
            _load_record=lambda _date: None,
            get_error_summary=lambda days=7: {"sample_count": 0},
        )
        scheduler.model = SimpleNamespace(k=0.0064, mean_heating_rate=1.0, gas_base_rate_kwh=10.0)

        result = scheduler.run(dry_run=True)
        self.assertTrue(result["success"])
        self.assertIn("schedule", result)
        self.assertEqual(result["schedule"]["switch_on_time"], "06:00")
        self.assertEqual(result["schedule"]["case"], "A")


if __name__ == "__main__":
    unittest.main()
