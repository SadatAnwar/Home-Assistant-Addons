"""CLI/report rendering helpers for scheduler."""

from __future__ import annotations

from datetime import datetime, timedelta

from .data_collector import DataCollector
from .prediction_tracker import PredictionTracker, format_error_summary, format_review_report
from .thermal_model import ThermalModel


class SchedulerReporter:
    """Render scheduler-related reports and state output."""

    def __init__(
        self,
        model: ThermalModel,
        collector: DataCollector,
        tracker: PredictionTracker,
    ):
        self.model = model
        self.collector = collector
        self.tracker = tracker

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
