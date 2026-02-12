"""Standalone simulator for the heating optimizer.

Runs the optimizer with synthetic inputs -- no Home Assistant connection needed.
Useful for testing "what-if" scenarios: what schedule would the optimizer produce
given specific weather conditions, room temperatures, and clock times?

Usage:
    python -m scripts.heating.simulator --clock 04:00 --temp-min -5 --temp-max 3
    python -m scripts.heating.simulator --clock 06:00 --temp-min 0 --temp-max 8 --condition sunny
    python -m scripts.heating.simulator --clock 04:00 --temp-min -8 --temp-max 0 --k 0.008
"""

import argparse
import logging
import math
from datetime import datetime, time
from pathlib import Path

from .optimizer import DailyHeatingSchedule, HeatingOptimizer
from .thermal_model import ThermalModel

logger = logging.getLogger(__name__)


def diurnal_temperature(hour_float: float, temp_min: float, temp_max: float) -> float:
    """Calculate outside temperature at a given hour using asymmetric sinusoidal model.

    Minimum at 06:00, maximum at 15:00. Warming phase is shorter (9h)
    than cooling phase (15h), matching real meteorological patterns.
    """
    MIN_HOUR = 6.0
    MAX_HOUR = 15.0
    amplitude = (temp_max - temp_min) / 2.0
    midpoint = (temp_max + temp_min) / 2.0

    h = hour_float % 24.0

    if MIN_HOUR <= h <= MAX_HOUR:
        # Warming phase: 06:00 to 15:00
        phase = math.pi * (h - MIN_HOUR) / (MAX_HOUR - MIN_HOUR)
        return round(midpoint - amplitude * math.cos(phase), 1)
    else:
        # Cooling phase: 15:00 to 06:00 next day
        hours_past_max = (h - MAX_HOUR) % 24.0
        cooling_duration = 24.0 - (MAX_HOUR - MIN_HOUR)
        phase = math.pi * hours_past_max / cooling_duration
        return round(midpoint + amplitude * math.cos(phase), 1)


def generate_forecast(
    date: datetime,
    temp_min: float,
    temp_max: float,
    condition: str = "cloudy",
) -> list[dict]:
    """Generate a 24-hour hourly forecast from min/max daily temperatures."""
    forecast = []
    for hour in range(24):
        temp = diurnal_temperature(float(hour), temp_min, temp_max)
        dt = date.replace(hour=hour, minute=0, second=0, microsecond=0)
        # Night hours get "clear-night" condition
        cond = condition if 7 <= hour <= 20 else "clear-night"
        forecast.append(
            {
                "datetime": dt.isoformat(),
                "temperature": temp,
                "condition": cond,
            }
        )
    return forecast


def estimate_current_room_temp(
    model: ThermalModel,
    clock_time: time,
    temp_min: float,
    target_temp: float,
    target_warm_time: time,
    target_off_time: time,
) -> float:
    """Estimate room temperature at clock_time based on daily heating cycle.

    Models the full day: heating is ON between warm_time and off_time (room
    near target_temp), cooling occurs from off_time through the night until
    the next warm_time.
    """
    clock_h = clock_time.hour + clock_time.minute / 60
    warm_h = target_warm_time.hour + target_warm_time.minute / 60
    off_h = target_off_time.hour + target_off_time.minute / 60

    # During heating hours: room is at or near target temp
    if warm_h <= clock_h < off_h:
        return target_temp

    # After off time (same day): cool from target_temp since off_time
    if clock_h >= off_h:
        hours_cooling = clock_h - off_h
    else:
        # Before warm time (next morning): cool from target_temp since off_time yesterday
        hours_cooling = (24.0 - off_h) + clock_h

    if hours_cooling <= 0:
        return target_temp

    cooling = model.predict_cooling_curve(
        start_temp=target_temp,
        outside_temp=temp_min,
        hours=int(hours_cooling) + 1,
    )

    idx = min(int(hours_cooling), len(cooling.temperatures) - 1)
    return round(cooling.temperatures[idx], 1)


def parse_time(time_str: str) -> time:
    """Parse HH:MM string to time object."""
    parts = time_str.split(":")
    return time(int(parts[0]), int(parts[1]))


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Simulate the heating optimizer under different conditions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Cold winter night, early morning run
  python -m scripts.heating.simulator --clock 04:00 --temp-min -5 --temp-max 3

  # Mild day
  python -m scripts.heating.simulator --clock 04:00 --temp-min 5 --temp-max 12

  # Sunny day (shows solar contribution)
  python -m scripts.heating.simulator --clock 04:00 --temp-min -2 --temp-max 6 --condition sunny

  # House started cold
  python -m scripts.heating.simulator --clock 06:00 --temp-min -5 --temp-max 3 --current-temp 16.5

  # What-if with leakier house (higher k)
  python -m scripts.heating.simulator --clock 04:00 --temp-min -5 --temp-max 3 --k 0.008
        """,
    )

    parser.add_argument(
        "--clock", type=str, required=True, help="Simulated time of day (HH:MM)"
    )
    parser.add_argument(
        "--temp-min", type=float, required=True, help="Daily minimum outside temp (C)"
    )
    parser.add_argument(
        "--temp-max", type=float, required=True, help="Daily maximum outside temp (C)"
    )
    parser.add_argument(
        "--current-temp",
        type=float,
        default=None,
        help="Current bedroom temp (C). If omitted, derived from overnight cooling",
    )
    parser.add_argument(
        "--hvac-mode",
        type=str,
        default="off",
        choices=["heat", "off"],
        help="Current HVAC mode (default: off)",
    )
    parser.add_argument(
        "--target-warm-time",
        type=str,
        default="08:00",
        help="When house should be warm (default: 08:00)",
    )
    parser.add_argument(
        "--target-off-time",
        type=str,
        default="23:00",
        help="Preferred heating off time (default: 23:00)",
    )
    parser.add_argument(
        "--target-temp",
        type=float,
        default=20.0,
        help="Desired room temperature (default: 20.0)",
    )
    parser.add_argument(
        "--min-bedroom-temp",
        type=float,
        default=18.0,
        help="Minimum overnight bedroom temp (default: 18.0)",
    )
    parser.add_argument(
        "--min-daytime-temp",
        type=float,
        default=20.0,
        help="Minimum daytime bedroom temp (default: 20.0)",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="cloudy",
        choices=[
            "sunny",
            "clear",
            "partlycloudy",
            "cloudy",
            "overcast",
            "rainy",
            "snowy",
        ],
        help="Weather condition for solar gain (default: cloudy)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/heating/thermal_model.pkl",
        help="Path to thermal model pickle file",
    )
    parser.add_argument(
        "--k", type=float, default=None, help="Override cooling rate constant k"
    )
    parser.add_argument(
        "--heating-rate",
        type=float,
        default=None,
        help="Override mean heating rate (C/hour)",
    )
    parser.add_argument(
        "--show-forecast",
        action="store_true",
        help="Include the generated forecast in output",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show debug logging"
    )

    return parser.parse_args()


def simulate(args: argparse.Namespace) -> DailyHeatingSchedule:
    """Run the heating optimizer simulation and return the schedule."""
    # Load thermal model
    model_path = Path(args.model_path)
    model = ThermalModel(model_dir=str(model_path.parent))
    model_loaded = model.load(filename=model_path.name)
    if not model_loaded:
        print(f"  No model at {args.model_path}, using defaults")

    # Apply coefficient overrides
    if args.k is not None:
        model.k = args.k
    if args.heating_rate is not None:
        model.mean_heating_rate = args.heating_rate

    # Parse times
    clock_time = parse_time(args.clock)
    target_warm_time = parse_time(args.target_warm_time)
    target_night_time = parse_time(args.target_off_time)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    simulated_now = today.replace(hour=clock_time.hour, minute=clock_time.minute)

    # Generate forecast
    forecast = generate_forecast(today, args.temp_min, args.temp_max, args.condition)

    # Current outside temp from diurnal curve
    outside_temp = diurnal_temperature(
        clock_time.hour + clock_time.minute / 60, args.temp_min, args.temp_max
    )

    optimizer = HeatingOptimizer(model)

    if args.current_temp is not None:
        bedroom_temp = args.current_temp
    else:
        # Two-pass estimation: the optimizer often switches off well before the
        # preferred off time.  Pass 1 uses the preferred off time to get an
        # initial schedule, then pass 2 re-derives the room temp using the
        # optimizer's actual computed switch-off time (as if yesterday had the
        # same weather).
        bedroom_temp = estimate_current_room_temp(
            model,
            clock_time,
            args.temp_min,
            args.target_temp,
            target_warm_time,
            target_night_time,
        )
        pass1 = optimizer.calculate_optimal_schedule(
            target_warm_time=target_warm_time,
            target_night_time=target_night_time,
            target_temp=args.target_temp,
            min_overnight_temp=args.min_bedroom_temp,
            min_daytime_temp=args.min_daytime_temp,
            current_temps={"bedroom": bedroom_temp},
            outside_temp=outside_temp,
            weather_forecast=forecast,
            current_time=simulated_now,
        )
        computed_off = pass1.switch_off_time or target_night_time
        if computed_off != target_night_time:
            bedroom_temp = estimate_current_room_temp(
                model,
                clock_time,
                args.temp_min,
                args.target_temp,
                target_warm_time,
                computed_off,
            )

    current_temps = {"bedroom": bedroom_temp}

    # Final optimizer run with corrected room temperature
    schedule = optimizer.calculate_optimal_schedule(
        target_warm_time=target_warm_time,
        target_night_time=target_night_time,
        target_temp=args.target_temp,
        min_overnight_temp=args.min_bedroom_temp,
        min_daytime_temp=args.min_daytime_temp,
        current_temps=current_temps,
        outside_temp=outside_temp,
        weather_forecast=forecast,
        current_time=simulated_now,
    )

    print_results(schedule, args, forecast, bedroom_temp, outside_temp, model)
    return schedule


def print_results(
    schedule: DailyHeatingSchedule,
    args: argparse.Namespace,
    forecast: list[dict],
    bedroom_temp: float,
    outside_temp: float,
    model: ThermalModel,
) -> None:
    """Print formatted simulation results."""
    clock_time = parse_time(args.clock)
    off_str = (
        schedule.switch_off_time.strftime("%H:%M")
        if schedule.switch_off_time
        else "CONTINUOUS"
    )

    print()
    print("=" * 65)
    print("  HEATING OPTIMIZER SIMULATION")
    print("=" * 65)

    print()
    print("  Inputs:")
    print(f"    Simulated clock:       {args.clock}")
    print(f"    Outside temp range:    {args.temp_min}C to {args.temp_max}C")
    print(f"    Outside temp at clock: {outside_temp}C")
    print(f"    Bedroom temp at clock: {bedroom_temp}C", end="")
    if args.current_temp is None:
        print("  (derived from overnight cooling)")
    else:
        print("  (user-provided)")
    print(f"    Weather condition:     {args.condition}")

    print()
    print("  User Settings:")
    print(f"    Target warm time:      {args.target_warm_time}")
    print(f"    Preferred off time:    {args.target_off_time}")
    print(f"    Target room temp:      {args.target_temp}C")
    print(f"    Min overnight temp:    {args.min_bedroom_temp}C")
    print(f"    Min daytime temp:      {args.min_daytime_temp}C")

    print()
    print("  Model Coefficients:")
    print(f"    k (cooling constant):  {model.k:.6f}  (tau = {1 / model.k:.0f} hours)")
    print(f"    Mean heating rate:     {model.mean_heating_rate:.3f} C/hour")
    print(f"    Mean cooling rate:     {model.mean_cooling_rate:.3f} C/hour")
    print(f"    Gas base rate:         {model.gas_base_rate_kwh:.1f} kWh/h @50%")

    print()
    print("-" * 65)
    print("  COMPUTED SCHEDULE")
    print("-" * 65)
    print()
    print(f"    Switch ON:             {schedule.switch_on_time.strftime('%H:%M')}")
    print(f"    Switch OFF:            {off_str}")
    print(f"    Optimal setpoint:      {schedule.optimal_setpoint}C")
    if schedule.expected_switch_on_temp is not None:
        print(f"    Temp at switch-on:     {schedule.expected_switch_on_temp:.1f}C")
    if schedule.expected_target_time_temp is not None:
        print(
            f"    Temp at {args.target_warm_time}:       "
            f"{schedule.expected_target_time_temp:.1f}C"
        )
    if schedule.expected_switch_off_temp is not None:
        print(f"    Temp at switch-off:    {schedule.expected_switch_off_temp:.1f}C")
    print(
        f"    Expected temp range:   {schedule.expected_min_temp:.1f}C - "
        f"{schedule.expected_max_temp:.1f}C"
    )
    print(f"    Solar contribution:    +{schedule.solar_contribution:.1f}C")
    print(f"    Expected gas usage:    ~{schedule.expected_gas_usage:.1f} kWh")
    if schedule.expected_burner_hours is not None:
        print(f"    Expected burner hours: {schedule.expected_burner_hours:.1f} hrs")
    if schedule.expected_avg_modulation is not None:
        print(f"    Expected avg modul.:   {schedule.expected_avg_modulation:.0f}%")

    print()
    print("  Reasoning:")
    for reason in schedule.reasoning:
        print(f"    - {reason}")

    # Hourly plan
    print()
    print("-" * 65)
    print("  HOURLY PLAN")
    print("-" * 65)
    print()
    print(
        f"    {'Hour':<6} {'State':<6} {'Setpt':>6} "
        f"{'Room':>7} {'Modul':>7} {'Outside':>8}"
    )
    print(
        f"    {'----':<6} {'-----':<6} {'-----':>6} "
        f"{'----':>7} {'-----':>7} {'-------':>8}"
    )

    for hp in schedule.hours:
        setpt = f"{hp.setpoint}C" if hp.setpoint else "-"
        modul = f"{hp.expected_modulation:.0f}%" if hp.system_state == "on" else "-"
        outside_h = diurnal_temperature(hp.hour, args.temp_min, args.temp_max)
        marker = ""
        if hp.hour == schedule.switch_on_time.hour:
            marker = " << ON"
        elif schedule.switch_off_time and hp.hour == schedule.switch_off_time.hour:
            marker = " << OFF"
        if hp.hour == clock_time.hour:
            marker += " (now)"
        print(
            f"    {hp.hour:02d}:00  {hp.system_state:<6} {setpt:>6} "
            f"{hp.expected_room_temp:>6.1f}C {modul:>7} {outside_h:>7.1f}C{marker}"
        )

    if args.show_forecast:
        print()
        print("-" * 65)
        print("  GENERATED FORECAST")
        print("-" * 65)
        print()
        print(f"    {'Hour':<8} {'Temp':>6} {'Condition':<15}")
        print(f"    {'----':<8} {'----':>6} {'---------':<15}")
        for entry in forecast:
            dt = datetime.fromisoformat(entry["datetime"])
            print(
                f"    {dt.strftime('%H:%M'):<8} {entry['temperature']:>5.1f}C "
                f"{entry['condition']:<15}"
            )

    print()
    print("=" * 65)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s - %(name)s - %(message)s")
    simulate(args)


if __name__ == "__main__":
    main()
