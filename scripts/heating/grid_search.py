"""Grid search for optimal heating configuration.

Searches across target_temp, min_bedroom_temp, and min_daytime_temp under
multiple weather scenarios to find the minimum-gas-usage configuration.
The thermal model (k, heating rate) is fixed — only user-configurable
parameters are varied.

Usage:
    python -m scripts.heating.grid_search
    python -m scripts.heating.grid_search --csv results.csv
    python -m scripts.heating.grid_search --scenario cold
"""

import argparse
import csv
import logging
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path

from .optimizer import HeatingOptimizer
from .simulator import estimate_current_room_temp, generate_forecast
from .thermal_model import ThermalModel

logger = logging.getLogger(__name__)

# --- Parameter grid ---

TARGET_TEMPS = [19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0]
MIN_BEDROOM_TEMPS = [18.0, 18.5, 19.0]
MIN_DAYTIME_TEMPS = [19.0, 19.5, 20.0, 20.5, 21.0]

# --- Weather scenarios (all cloudy, no solar) ---

SCENARIOS: dict[str, tuple[float, float]] = {
    "very_cold": (-10.0, -2.0),
    "cold": (-5.0, 3.0),
    "cool": (0.0, 8.0),
    "mild": (5.0, 12.0),
}

# --- Fixed parameters ---

CLOCK_TIME = time(4, 0)
WARM_TIME = time(8, 0)
OFF_TIME = time(23, 0)
CONDITION = "cloudy"
MODEL_PATH = "models/heating/thermal_model.pkl"


@dataclass
class SimResult:
    """Result of a single grid search simulation."""

    target_temp: float
    min_bedroom_temp: float
    min_daytime_temp: float
    scenario: str
    gas_kwh: float
    burner_hours: float
    min_temp: float
    max_temp: float
    switch_on: str
    switch_off: str
    violations: int


def is_valid_combo(target_temp: float, min_bedroom: float, min_daytime: float) -> bool:
    """Check if parameter combination is valid."""
    if min_daytime > target_temp:
        return False
    if min_bedroom > min_daytime:
        return False
    return True


def check_violations(schedule, min_bedroom: float, min_daytime: float) -> int:
    """Count hours where room temp drops below the active threshold."""
    count = 0
    for hp in schedule.hours:
        # Daytime: warm_time (08) to off_time (23)
        if WARM_TIME.hour <= hp.hour < OFF_TIME.hour:
            threshold = min_daytime
        else:
            threshold = min_bedroom
        if hp.expected_room_temp < threshold - 0.05:  # tiny tolerance
            count += 1
    return count


def run_single(
    optimizer: HeatingOptimizer,
    model: ThermalModel,
    scenario_name: str,
    temp_min: float,
    temp_max: float,
    target_temp: float,
    min_bedroom: float,
    min_daytime: float,
) -> SimResult:
    """Run one optimizer simulation and return the result."""
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    simulated_now = today.replace(hour=CLOCK_TIME.hour, minute=CLOCK_TIME.minute)

    forecast = generate_forecast(today, temp_min, temp_max, CONDITION)

    from .simulator import diurnal_temperature

    outside_temp = diurnal_temperature(
        CLOCK_TIME.hour + CLOCK_TIME.minute / 60, temp_min, temp_max
    )

    # Two-pass bedroom temp estimation (mirrors simulator.simulate)
    bedroom_temp = estimate_current_room_temp(
        model, CLOCK_TIME, temp_min, target_temp, WARM_TIME, OFF_TIME
    )
    pass1 = optimizer.calculate_optimal_schedule(
        target_warm_time=WARM_TIME,
        target_night_time=OFF_TIME,
        target_temp=target_temp,
        min_overnight_temp=min_bedroom,
        min_daytime_temp=min_daytime,
        current_temps={"bedroom": bedroom_temp},
        outside_temp=outside_temp,
        weather_forecast=forecast,
        current_time=simulated_now,
    )
    computed_off = pass1.switch_off_time or OFF_TIME
    if computed_off != OFF_TIME:
        bedroom_temp = estimate_current_room_temp(
            model, CLOCK_TIME, temp_min, target_temp, WARM_TIME, computed_off
        )

    # Final run
    schedule = optimizer.calculate_optimal_schedule(
        target_warm_time=WARM_TIME,
        target_night_time=OFF_TIME,
        target_temp=target_temp,
        min_overnight_temp=min_bedroom,
        min_daytime_temp=min_daytime,
        current_temps={"bedroom": bedroom_temp},
        outside_temp=outside_temp,
        weather_forecast=forecast,
        current_time=simulated_now,
    )

    violations = check_violations(schedule, min_bedroom, min_daytime)

    off_str = (
        schedule.switch_off_time.strftime("%H:%M")
        if schedule.switch_off_time
        else "CONT"
    )

    return SimResult(
        target_temp=target_temp,
        min_bedroom_temp=min_bedroom,
        min_daytime_temp=min_daytime,
        scenario=scenario_name,
        gas_kwh=schedule.expected_gas_usage,
        burner_hours=schedule.expected_burner_hours or 0.0,
        min_temp=schedule.expected_min_temp,
        max_temp=schedule.expected_max_temp,
        switch_on=schedule.switch_on_time.strftime("%H:%M"),
        switch_off=off_str,
        violations=violations,
    )


def run_grid_search(
    scenarios: dict[str, tuple[float, float]],
) -> list[SimResult]:
    """Run full grid search across all parameter combos and scenarios."""
    model_path = Path(MODEL_PATH)
    model = ThermalModel(model_dir=str(model_path.parent))
    if not model.load(filename=model_path.name):
        print(f"WARNING: No model at {MODEL_PATH}, using defaults")

    print(
        f"Model: k={model.k:.6f} (tau={1 / model.k:.0f}h), "
        f"heating_rate={model.mean_heating_rate:.3f} C/h, "
        f"gas_base={model.gas_base_rate_kwh:.1f} kWh/h"
    )

    optimizer = HeatingOptimizer(model)
    results: list[SimResult] = []

    # Count valid combos
    combos = [
        (tt, mb, md)
        for tt in TARGET_TEMPS
        for mb in MIN_BEDROOM_TEMPS
        for md in MIN_DAYTIME_TEMPS
        if is_valid_combo(tt, mb, md)
    ]
    total = len(combos) * len(scenarios)
    print(
        f"Running {len(combos)} valid combos x {len(scenarios)} scenarios = {total} simulations\n"
    )

    for scenario_name, (temp_min, temp_max) in scenarios.items():
        for tt, mb, md in combos:
            result = run_single(
                optimizer, model, scenario_name, temp_min, temp_max, tt, mb, md
            )
            results.append(result)

    return results


def print_scenario_table(scenario: str, results: list[SimResult]) -> None:
    """Print a sorted table for one weather scenario."""
    scenario_results = [r for r in results if r.scenario == scenario]
    # Sort: feasible first (0 violations), then by gas usage
    scenario_results.sort(key=lambda r: (r.violations > 0, r.gas_kwh))

    temps = SCENARIOS[scenario]
    print(f"\n{'=' * 95}")
    print(f"  {scenario.upper().replace('_', ' ')} ({temps[0]}C to {temps[1]}C)")
    print(f"{'=' * 95}")
    print(
        f"  {'Target':>6} {'MinBed':>6} {'MinDay':>6} "
        f"{'Gas kWh':>8} {'Burner':>7} {'MinT':>5} {'MaxT':>5} "
        f"{'ON':>5} {'OFF':>5} {'Viol':>4}"
    )
    print(
        f"  {'------':>6} {'------':>6} {'------':>6} "
        f"{'-------':>8} {'------':>7} {'----':>5} {'----':>5} "
        f"{'--':>5} {'---':>5} {'----':>4}"
    )

    for r in scenario_results:
        flag = " *" if r.violations > 0 else ""
        print(
            f"  {r.target_temp:>6.1f} {r.min_bedroom_temp:>6.1f} {r.min_daytime_temp:>6.1f} "
            f"{r.gas_kwh:>8.1f} {r.burner_hours:>6.1f}h {r.min_temp:>5.1f} {r.max_temp:>5.1f} "
            f"{r.switch_on:>5} {r.switch_off:>5} {r.violations:>4}{flag}"
        )


def print_overall_ranking(results: list[SimResult], sort_by: str = "gas") -> None:
    """Print overall ranking averaged across all scenarios, feasible-first."""
    # Group by (target_temp, min_bedroom, min_daytime)
    combos: dict[tuple[float, float, float], list[SimResult]] = {}
    for r in results:
        key = (r.target_temp, r.min_bedroom_temp, r.min_daytime_temp)
        combos.setdefault(key, []).append(r)

    @dataclass
    class Ranking:
        target_temp: float
        min_bedroom: float
        min_daytime: float
        avg_gas: float
        max_gas: float
        avg_burner: float
        total_violations: int
        scenario_details: dict[str, float]

    rankings: list[Ranking] = []
    for (tt, mb, md), scenario_results in combos.items():
        avg_gas = sum(r.gas_kwh for r in scenario_results) / len(scenario_results)
        max_gas = max(r.gas_kwh for r in scenario_results)
        avg_burner = sum(r.burner_hours for r in scenario_results) / len(
            scenario_results
        )
        total_viol = sum(r.violations for r in scenario_results)
        details = {r.scenario: r.gas_kwh for r in scenario_results}
        rankings.append(
            Ranking(tt, mb, md, avg_gas, max_gas, avg_burner, total_viol, details)
        )

    # Sort: feasible first, then by chosen key
    if sort_by == "target":
        rankings.sort(key=lambda r: (r.total_violations > 0, -r.target_temp, r.avg_gas))
    elif sort_by == "burner":
        rankings.sort(key=lambda r: (r.total_violations > 0, r.avg_burner))
    else:
        rankings.sort(key=lambda r: (r.total_violations > 0, r.avg_gas))

    scenario_names = sorted(SCENARIOS.keys())
    scen_header = " ".join(f"{s[:8]:>8}" for s in scenario_names)

    print(f"\n{'=' * 105}")
    print("  OVERALL RANKING (averaged across scenarios)")
    print(f"{'=' * 105}")
    print(
        f"  {'Target':>6} {'MinBed':>6} {'MinDay':>6} "
        f"{'AvgGas':>7} {'MaxGas':>7} {'Burner':>7} {'Viol':>4}  {scen_header}"
    )
    print(
        f"  {'------':>6} {'------':>6} {'------':>6} "
        f"{'------':>7} {'------':>7} {'------':>7} {'----':>4}  "
        + " ".join("--------" for _ in scenario_names)
    )

    for rank in rankings:
        flag = " *" if rank.total_violations > 0 else ""
        scen_vals = " ".join(
            f"{rank.scenario_details.get(s, 0):>8.1f}" for s in scenario_names
        )
        print(
            f"  {rank.target_temp:>6.1f} {rank.min_bedroom:>6.1f} {rank.min_daytime:>6.1f} "
            f"{rank.avg_gas:>7.1f} {rank.max_gas:>7.1f} {rank.avg_burner:>6.1f}h {rank.total_violations:>4}{flag}  "
            f"{scen_vals}"
        )

    # Recommendation
    feasible = [r for r in rankings if r.total_violations == 0]
    if feasible:
        best = feasible[0]
        print(
            f"\n  RECOMMENDATION: target={best.target_temp}, "
            f"min_bed={best.min_bedroom}, min_day={best.min_daytime}"
        )
        print(
            f"  Average gas: {best.avg_gas:.1f} kWh/day, "
            f"worst-case: {best.max_gas:.1f} kWh/day, "
            f"0 violations across all scenarios"
        )
    else:
        print(
            "\n  WARNING: No fully feasible configuration found across all scenarios."
        )
        best = rankings[0]
        print(
            f"  Least-violated: target={best.target_temp}, "
            f"min_bed={best.min_bedroom}, min_day={best.min_daytime} "
            f"({best.total_violations} violations)"
        )


def export_csv(results: list[SimResult], path: str) -> None:
    """Export all results to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scenario",
                "target_temp",
                "min_bedroom_temp",
                "min_daytime_temp",
                "gas_kwh",
                "burner_hours",
                "min_temp",
                "max_temp",
                "switch_on",
                "switch_off",
                "violations",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.scenario,
                    r.target_temp,
                    r.min_bedroom_temp,
                    r.min_daytime_temp,
                    r.gas_kwh,
                    r.burner_hours,
                    r.min_temp,
                    r.max_temp,
                    r.switch_on,
                    r.switch_off,
                    r.violations,
                ]
            )
    print(f"\nExported {len(results)} rows to {path}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Grid search for optimal heating configuration",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        choices=list(SCENARIOS.keys()),
        help="Run a single scenario (default: all)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Export results to CSV file",
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="gas",
        choices=["gas", "target", "burner"],
        help="Sort overall ranking by: gas (default), target (desc), burner",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show debug logging",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s - %(name)s - %(message)s")

    if args.scenario:
        scenarios = {args.scenario: SCENARIOS[args.scenario]}
    else:
        scenarios = SCENARIOS

    results = run_grid_search(scenarios)

    # Per-scenario tables
    for scenario in scenarios:
        print_scenario_table(scenario, results)

    # Overall ranking (only if multiple scenarios)
    if len(scenarios) > 1:
        print_overall_ranking(results, sort_by=args.sort)

    # CSV export
    if args.csv:
        export_csv(results, args.csv)

    print()


if __name__ == "__main__":
    main()
