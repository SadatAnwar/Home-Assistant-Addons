import tempfile
import unittest

from scripts.heating.thermal_model import ThermalModel


class _ConstantModel:
    def __init__(self, value: float, n_features: int):
        self.value = value
        self.n_features_in_ = n_features

    def predict(self, features):
        return [self.value] * len(features)


class _ErrorModel:
    def __init__(self, n_features: int):
        self.n_features_in_ = n_features

    def predict(self, _features):
        raise RuntimeError("forced predict failure")


class ThermalModelRegressionTests(unittest.TestCase):
    def test_heating_duration_uses_model_rate_when_above_floor(self):
        model = ThermalModel()
        model.heating_rate_model = _ConstantModel(value=1.2, n_features=4)
        model.mean_heating_rate = 0.6

        minutes = model.predict_heating_duration(
            start_temp=18.0,
            target_temp=19.0,
            outside_temp=2.0,
            setpoint=21.0,
        )

        self.assertEqual(minutes, 50)

    def test_heating_duration_uses_dynamic_floor_when_model_too_low(self):
        model = ThermalModel()
        model.heating_rate_model = _ConstantModel(value=0.1, n_features=4)
        model.mean_heating_rate = 0.8

        minutes = model.predict_heating_duration(
            start_temp=18.0,
            target_temp=19.0,
            outside_temp=2.0,
            setpoint=21.0,
        )

        # floor = max(mean*0.5=0.4, config min=0.5) = 0.5 => 120 minutes
        self.assertEqual(minutes, 120)

    def test_heating_duration_without_model_uses_config_floor(self):
        model = ThermalModel()
        model.heating_rate_model = None
        model.mean_heating_rate = 0.2

        minutes = model.predict_heating_duration(
            start_temp=18.0,
            target_temp=19.0,
            outside_temp=2.0,
            setpoint=21.0,
        )

        self.assertEqual(minutes, 120)

    def test_modulation_old_pickle_two_feature_model_supported(self):
        model = ThermalModel()
        model.modulation_model = _ConstantModel(value=42.0, n_features=2)
        model.modulation_feature_cols = []

        modulation = model.predict_modulation(
            outside_temp=1.0,
            setpoint=20.0,
            room_temp=19.0,
        )

        self.assertEqual(modulation, 42.0)

    def test_modulation_prediction_is_clamped(self):
        model = ThermalModel()
        model.modulation_model = _ConstantModel(value=150.0, n_features=3)
        model.modulation_feature_cols = ["outside_temp", "setpoint", "bedroom"]

        modulation = model.predict_modulation(
            outside_temp=1.0,
            setpoint=20.0,
            room_temp=19.0,
        )

        self.assertEqual(modulation, 100.0)

    def test_cooling_curve_old_pickle_two_feature_model_supported(self):
        model = ThermalModel()
        model.cooling_rate_model = _ConstantModel(value=-0.05, n_features=2)
        model.cooling_feature_cols = []

        prediction = model.predict_cooling_curve(
            start_temp=20.0,
            outside_temp=0.0,
            hours=2,
        )

        self.assertEqual(len(prediction.temperatures), 3)
        self.assertLess(prediction.temperatures[-1], 20.0)

    def test_cooling_curve_model_failure_falls_back_to_physics(self):
        model = ThermalModel()
        model.k = 0.01
        model.mean_cooling_rate = 0.2

        baseline = model.predict_cooling_curve(
            start_temp=20.0,
            outside_temp=0.0,
            hours=2,
        )

        model.cooling_rate_model = _ErrorModel(n_features=3)
        model.cooling_feature_cols = ["outside_temp", "bedroom", "sun_elevation"]
        fallback = model.predict_cooling_curve(
            start_temp=20.0,
            outside_temp=0.0,
            hours=2,
        )

        self.assertEqual(fallback.temperatures, baseline.temperatures)

    def test_save_load_persists_feature_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = ThermalModel(model_dir=tmpdir)
            model.heating_feature_cols = ["outside_temp", "bedroom", "setpoint"]
            model.cooling_feature_cols = ["outside_temp", "bedroom"]
            model.modulation_feature_cols = ["outside_temp", "setpoint", "bedroom"]

            model.save("test.pkl")

            restored = ThermalModel(model_dir=tmpdir)
            self.assertTrue(restored.load("test.pkl"))
            self.assertEqual(restored.heating_feature_cols, model.heating_feature_cols)
            self.assertEqual(restored.cooling_feature_cols, model.cooling_feature_cols)
            self.assertEqual(
                restored.modulation_feature_cols, model.modulation_feature_cols
            )


if __name__ == "__main__":
    unittest.main()
