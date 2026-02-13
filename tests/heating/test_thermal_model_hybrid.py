import unittest
from datetime import datetime, time

from scripts.heating.optimizer import HourlyHeatingPlan
from scripts.heating.optimizer_hourly_plan import build_hourly_plan
from scripts.heating.thermal_model import ThermalModel


class _ConstantModel:
    def __init__(self, value: float, n_features: int):
        self.value = value
        self.n_features_in_ = n_features

    def predict(self, features):
        return [self.value] * len(features)


class ThermalModelHybridTests(unittest.TestCase):
    def test_cooling_curve_blends_ml_and_physics(self):
        model = ThermalModel()
        model.k = 0.01
        model.mean_cooling_rate = 0.2

        baseline = model.predict_cooling_curve(
            start_temp=20.0,
            outside_temp=0.0,
            hours=1,
        )

        # Positive ML rate should reduce cooling vs pure physics baseline.
        model.cooling_rate_model = _ConstantModel(value=0.2, n_features=3)
        hybrid = model.predict_cooling_curve(
            start_temp=20.0,
            outside_temp=0.0,
            hours=1,
        )

        self.assertGreater(hybrid.temperatures[-1], baseline.temperatures[-1])

    def test_predict_heating_duration_uses_learned_rate_without_hard_one_floor(self):
        model = ThermalModel()
        model.heating_rate_model = None
        model.mean_heating_rate = 0.6

        duration = model.predict_heating_duration(
            start_temp=18.0,
            target_temp=19.0,
            outside_temp=0.0,
            setpoint=21.0,
        )

        self.assertEqual(duration, 100)

    def test_hourly_plan_uses_configured_min_not_hard_one_floor(self):
        model = ThermalModel()
        model.mean_heating_rate = 0.6

        plans = build_hourly_plan(
            thermal_model=model,
            switch_on_time=time(0, 0),
            switch_off_time=None,
            optimal_setpoint=20.0,
            bedroom_temp=18.0,
            outside_temp=0.0,
            weather_forecast=None,
            current_time=datetime(2026, 2, 13, 0, 0),
            plan_factory=HourlyHeatingPlan,
        )

        # With base_rate=0.6 (not forced to 1.0), first-hour rise is +0.6°C.
        self.assertAlmostEqual(plans[0].expected_room_temp, 18.6, places=1)


if __name__ == "__main__":
    unittest.main()
