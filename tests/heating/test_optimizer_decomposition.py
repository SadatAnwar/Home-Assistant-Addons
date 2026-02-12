import unittest
from datetime import datetime, timedelta, time

from scripts.heating.optimizer import DailyHeatingSchedule, HeatingOptimizer
from scripts.heating.thermal_model import ThermalModel


class OptimizerDecompositionTests(unittest.TestCase):
    def setUp(self):
        model = ThermalModel()
        model.k = 0.0064
        model.mean_heating_rate = 1.0
        model.gas_base_rate_kwh = 10.0
        self.optimizer = HeatingOptimizer(model)

    def _forecast(self):
        start = datetime(2026, 2, 12, 0, 0)
        forecast = []
        for h in range(24):
            dt = start + timedelta(hours=h)
            forecast.append(
                {
                    "datetime": dt.isoformat(),
                    "temperature": 2.0 + (h % 6) * 0.3,
                    "condition": "cloudy",
                }
            )
        return forecast

    def test_calculate_optimal_schedule_public_api_stable(self):
        schedule = self.optimizer.calculate_optimal_schedule(
            target_warm_time=time(8, 0),
            target_night_time=time(23, 0),
            target_temp=20.0,
            min_overnight_temp=18.0,
            min_daytime_temp=19.5,
            current_temps={"bedroom": 19.6},
            outside_temp=2.0,
            weather_forecast=self._forecast(),
            current_time=datetime(2026, 2, 12, 4, 0),
        )

        self.assertIsInstance(schedule, DailyHeatingSchedule)
        self.assertIsNotNone(schedule.switch_on_time)
        self.assertGreaterEqual(schedule.expected_max_temp, schedule.expected_min_temp)

    def test_recalculate_mid_day_public_api_stable(self):
        schedule = self.optimizer.recalculate_mid_day(
            target_warm_time=time(8, 0),
            target_night_time=time(23, 0),
            target_temp=20.0,
            min_overnight_temp=18.0,
            min_daytime_temp=19.5,
            current_temps={"bedroom": 20.1},
            outside_temp=3.0,
            weather_forecast=self._forecast(),
            current_time=datetime(2026, 2, 12, 11, 0),
            original_switch_on_time=time(6, 0),
        )

        self.assertIsInstance(schedule, DailyHeatingSchedule)
        self.assertIsNotNone(schedule.optimal_setpoint)
        self.assertGreater(len(schedule.hours), 0)

    def test_apply_safety_overrides_signature_and_return_shape(self):
        schedule = self.optimizer.calculate_optimal_schedule(
            target_warm_time=time(8, 0),
            target_night_time=time(23, 0),
            target_temp=20.0,
            min_overnight_temp=18.0,
            min_daytime_temp=19.5,
            current_temps={"bedroom": 19.2},
            outside_temp=2.0,
            weather_forecast=self._forecast(),
            current_time=datetime(2026, 2, 12, 4, 0),
        )
        updated, overrides = self.optimizer.apply_safety_overrides(
            schedule=schedule,
            current_temps={"bedroom": 17.0},
            min_overnight_temp=18.0,
            min_daytime_temp=19.5,
            target_warm_time=time(8, 0),
            preferred_off_time=time(23, 0),
        )

        self.assertIsInstance(updated, DailyHeatingSchedule)
        self.assertIsInstance(overrides, list)

    def test_generate_schedule_summary_still_contains_key_fields(self):
        schedule = self.optimizer.calculate_optimal_schedule(
            target_warm_time=time(8, 0),
            target_night_time=time(23, 0),
            target_temp=20.0,
            min_overnight_temp=18.0,
            min_daytime_temp=19.5,
            current_temps={"bedroom": 19.6},
            outside_temp=2.0,
            weather_forecast=self._forecast(),
            current_time=datetime(2026, 2, 12, 4, 0),
        )
        summary = self.optimizer.generate_schedule_summary(schedule)

        self.assertIn("Switch ON:", summary)
        self.assertIn("Switch OFF:", summary)
        self.assertIn("Setpoint:", summary)

    def test_hourly_plan_starts_from_current_hour(self):
        current_time = datetime(2026, 2, 12, 11, 37)
        schedule = self.optimizer.calculate_optimal_schedule(
            target_warm_time=time(8, 0),
            target_night_time=time(23, 0),
            target_temp=20.0,
            min_overnight_temp=18.0,
            min_daytime_temp=19.5,
            current_temps={"bedroom": 19.6},
            outside_temp=2.0,
            weather_forecast=self._forecast(),
            current_time=current_time,
        )
        self.assertEqual(schedule.hours[0].hour, current_time.hour)


if __name__ == "__main__":
    unittest.main()
