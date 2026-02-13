import unittest

import pandas as pd

from scripts.heating.thermal_model import ThermalModel


class MLValidationTests(unittest.TestCase):
    def _base_frame(self, rows: int, freq: str = "h") -> pd.DataFrame:
        ts = pd.date_range("2026-01-01", periods=rows, freq=freq)
        idx = range(rows)
        return pd.DataFrame(
            {
                "timestamp": ts,
                "bedroom": [19.0 + (i % 12) * 0.05 for i in idx],
                "outside_temp": [2.0 + (i % 24) * 0.1 for i in idx],
                "setpoint": [20.0 + (i % 3) * 0.5 for i in idx],
                "burner_modulation": [25.0 + (i % 10) * 3.0 for i in idx],
                "sun_elevation": [max(0.0, (i % 24) - 8) for i in idx],
            }
        )

    def test_heating_training_uses_time_holdout_when_span_is_long_enough(self):
        model = ThermalModel()
        df = self._base_frame(240, "h")
        df["heating_on"] = True

        metrics = model._train_heating_model(df)

        self.assertNotIn("error", metrics)
        self.assertEqual(metrics.get("validation_split"), "time_holdout_last_2d")
        self.assertIn("test_mae", metrics)
        self.assertIn("test_rmse", metrics)
        self.assertIn("wf_mae", metrics)
        self.assertIn("wf_rmse", metrics)

    def test_heating_training_falls_back_to_tail_split_when_not_enough_holdout_span(self):
        model = ThermalModel()
        df = self._base_frame(60, "15min")
        df["heating_on"] = True

        metrics = model._train_heating_model(df)

        self.assertNotIn("error", metrics)
        self.assertEqual(metrics.get("validation_split"), "chronological_tail_20pct")
        self.assertIn("test_mae", metrics)
        self.assertIn("test_rmse", metrics)

    def test_cooling_and_modulation_training_report_time_aware_metrics(self):
        model = ThermalModel()

        cooling_df = self._base_frame(240, "h")
        cooling_df["heating_on"] = False
        cooling_metrics = model._train_cooling_model(cooling_df)
        self.assertNotIn("error", cooling_metrics)
        self.assertIn("validation_split", cooling_metrics)
        self.assertIn("test_mae", cooling_metrics)
        self.assertIn("test_rmse", cooling_metrics)

        modulation_df = self._base_frame(240, "h")
        modulation_metrics = model._train_modulation_model(modulation_df)
        self.assertNotIn("error", modulation_metrics)
        self.assertIn("validation_split", modulation_metrics)
        self.assertIn("test_mae", modulation_metrics)
        self.assertIn("test_rmse", modulation_metrics)


if __name__ == "__main__":
    unittest.main()
