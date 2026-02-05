"""Prediction tracking and feedback system for heating optimizer.

Saves daily predictions, compares with actuals, calculates errors,
and provides coefficient adjustments to improve the model over time.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .config import (
    BOILER_SENSORS,
    PREDICTION_CONFIG,
    ROOM_SENSORS,
)
from .ha_client import HAClient

logger = logging.getLogger(__name__)


@dataclass
class DailyPrediction:
    """Predicted values for a single day."""

    date: str  # YYYY-MM-DD
    switch_on_time: str  # HH:MM
    switch_off_time: str  # HH:MM or "CONTINUOUS"
    setpoint: float
    expected_switch_on_temp: float
    expected_target_time_temp: float  # Temp at target warm time (e.g., 08:00)
    expected_switch_off_temp: float  # Temp at switch-off time
    expected_gas_kwh: float
    expected_burner_hours: float
    expected_avg_modulation: float
    target_warm_time: str  # HH:MM - user's target warm time
    timestamp: str = ""  # ISO format — when this prediction was made


@dataclass
class MidDayAdjustment:
    """Record of a mid-day schedule recalculation."""

    timestamp: str  # ISO format
    switch_off_time: str  # HH:MM or "CONTINUOUS"
    setpoint: float
    actual_bedroom_temp: float
    reasoning: list[str]


@dataclass
class DailyActuals:
    """Actual measured values for a single day."""

    date: str
    actual_switch_on_temp: float | None
    actual_target_time_temp: float | None
    actual_switch_off_temp: float | None
    actual_gas_kwh: float | None
    actual_burner_hours: float | None
    actual_avg_modulation: float | None
    actual_burner_starts: int | None


@dataclass
class PredictionErrors:
    """Calculated errors between predictions and actuals."""

    switch_on_temp_error: (
        float | None
    )  # Predicted - Actual (positive = predicted too warm)
    target_time_temp_error: float | None
    switch_off_temp_error: float | None
    gas_kwh_error: float | None  # Predicted - Actual (negative = underestimated)
    gas_kwh_error_pct: float | None
    burner_hours_error: float | None
    avg_modulation_error: float | None


@dataclass
class DailyRecord:
    """Complete record for a single day."""

    date: str
    prediction: DailyPrediction
    adjustments: list[MidDayAdjustment] | None = None
    actuals: DailyActuals | None = None
    errors: PredictionErrors | None = None
    coefficients_adjusted: bool = False


class PredictionTracker:
    """Tracks predictions, collects actuals, and calculates errors."""

    def __init__(self, client: HAClient | None = None):
        self.client = client or HAClient()
        self.data_dir = Path(PREDICTION_CONFIG.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_file = self.data_dir / PREDICTION_CONFIG.predictions_file

    def save_prediction(self, schedule: Any, target_warm_time: str) -> DailyPrediction:
        """Save today's prediction from the computed schedule.

        Args:
            schedule: DailyHeatingSchedule from optimizer
            target_warm_time: User's target warm time (HH:MM)

        Returns:
            The saved DailyPrediction
        """
        date_str = schedule.date.strftime("%Y-%m-%d")

        off_time_str = (
            schedule.switch_off_time.strftime("%H:%M")
            if schedule.switch_off_time
            else "CONTINUOUS"
        )

        prediction = DailyPrediction(
            date=date_str,
            switch_on_time=schedule.switch_on_time.strftime("%H:%M"),
            switch_off_time=off_time_str,
            setpoint=schedule.optimal_setpoint,
            expected_switch_on_temp=schedule.expected_switch_on_temp or 0.0,
            expected_target_time_temp=getattr(
                schedule, "expected_target_time_temp", 0.0
            ),
            expected_switch_off_temp=getattr(schedule, "expected_switch_off_temp", 0.0),
            expected_gas_kwh=schedule.expected_gas_usage,
            expected_burner_hours=getattr(schedule, "expected_burner_hours", 0.0),
            expected_avg_modulation=getattr(schedule, "expected_avg_modulation", 0.0),
            target_warm_time=target_warm_time,
            timestamp=datetime.now().isoformat(),
        )

        # Load existing record for today or create new one
        record = self._load_record(date_str)
        if record:
            record.prediction = prediction
        else:
            record = DailyRecord(date=date_str, prediction=prediction)

        self._save_record(record)
        logger.info(f"Saved prediction for {date_str}")
        return prediction

    def save_adjustment(
        self,
        date_str: str,
        switch_off_time: str,
        setpoint: float,
        actual_bedroom_temp: float,
        reasoning: list[str],
    ) -> MidDayAdjustment:
        """Save a mid-day schedule adjustment for an existing prediction.

        Args:
            date_str: Date in YYYY-MM-DD format
            switch_off_time: New switch-off time (HH:MM or "CONTINUOUS")
            setpoint: Adjusted setpoint
            actual_bedroom_temp: Current measured bedroom temperature
            reasoning: List of reasoning strings

        Returns:
            The saved MidDayAdjustment
        """
        record = self._load_record(date_str)
        if not record:
            raise ValueError(f"No primary prediction exists for {date_str}")

        adjustment = MidDayAdjustment(
            timestamp=datetime.now().isoformat(),
            switch_off_time=switch_off_time,
            setpoint=setpoint,
            actual_bedroom_temp=actual_bedroom_temp,
            reasoning=reasoning,
        )

        if record.adjustments is None:
            record.adjustments = []
        record.adjustments.append(adjustment)

        self._save_record(record)
        logger.info(
            f"Saved adjustment for {date_str}: off={switch_off_time}, sp={setpoint}"
        )
        return adjustment

    def collect_actuals(self, date_str: str) -> DailyActuals | None:
        """Collect actual values from HA history for a specific date.

        Args:
            date_str: Date in YYYY-MM-DD format

        Returns:
            DailyActuals if data found, None otherwise
        """
        record = self._load_record(date_str)
        if not record:
            logger.warning(f"No prediction record found for {date_str}")
            return None

        prediction = record.prediction
        date = datetime.strptime(date_str, "%Y-%m-%d")

        # Query time range: from midnight to midnight
        start_time = datetime.combine(date, datetime.min.time())
        end_time = start_time + timedelta(days=1)

        try:
            # Get bedroom temperature history
            bedroom_entity = ROOM_SENSORS["bedroom"]
            temp_history = self.client.get_history(
                [bedroom_entity],
                start_time=start_time,
                end_time=end_time,
            )

            # Get burner metrics history
            burner_hours_entity = BOILER_SENSORS["burner_hours"]
            burner_starts_entity = BOILER_SENSORS["burner_starts"]
            modulation_entity = BOILER_SENSORS["burner_modulation"]

            burner_history = self.client.get_history(
                [burner_hours_entity, burner_starts_entity, modulation_entity],
                start_time=start_time,
                end_time=end_time,
            )

            # Extract temps at specific times
            temp_at_switch_on = self._get_temp_at_time(
                temp_history.get(bedroom_entity, []),
                prediction.switch_on_time,
                date,
            )
            temp_at_target = self._get_temp_at_time(
                temp_history.get(bedroom_entity, []),
                prediction.target_warm_time,
                date,
            )
            temp_at_switch_off = None
            if prediction.switch_off_time != "CONTINUOUS":
                temp_at_switch_off = self._get_temp_at_time(
                    temp_history.get(bedroom_entity, []),
                    prediction.switch_off_time,
                    date,
                )

            # Calculate burner hours delta
            burner_hours = self._calc_sensor_delta(
                burner_history.get(burner_hours_entity, [])
            )

            # Calculate burner starts delta
            burner_starts = self._calc_sensor_delta(
                burner_history.get(burner_starts_entity, [])
            )
            if burner_starts is not None:
                burner_starts = int(burner_starts)

            # Calculate average modulation (when burner was active)
            avg_modulation = self._calc_avg_modulation(
                burner_history.get(modulation_entity, [])
            )

            # Get gas usage from SmartNetz sensor (more accurate)
            gas_kwh = self._get_gas_usage(date_str)

            actuals = DailyActuals(
                date=date_str,
                actual_switch_on_temp=temp_at_switch_on,
                actual_target_time_temp=temp_at_target,
                actual_switch_off_temp=temp_at_switch_off,
                actual_gas_kwh=gas_kwh,
                actual_burner_hours=burner_hours,
                actual_avg_modulation=avg_modulation,
                actual_burner_starts=burner_starts,
            )

            # Calculate errors and save
            errors = self.calculate_errors(prediction, actuals)
            record.actuals = actuals
            record.errors = errors
            self._save_record(record)

            logger.info(f"Collected actuals for {date_str}")
            return actuals

        except Exception as e:
            logger.error(f"Failed to collect actuals for {date_str}: {e}")
            return None

    def calculate_errors(
        self, prediction: DailyPrediction, actuals: DailyActuals
    ) -> PredictionErrors:
        """Calculate prediction errors (Predicted - Actual)."""

        def safe_diff(pred: float | None, actual: float | None) -> float | None:
            if pred is None or actual is None or pred == 0:
                return None
            return round(pred - actual, 2)

        gas_error = safe_diff(prediction.expected_gas_kwh, actuals.actual_gas_kwh)
        gas_error_pct = None
        if (
            gas_error is not None
            and actuals.actual_gas_kwh
            and actuals.actual_gas_kwh > 0
        ):
            gas_error_pct = round((gas_error / actuals.actual_gas_kwh) * 100, 1)

        return PredictionErrors(
            switch_on_temp_error=safe_diff(
                prediction.expected_switch_on_temp, actuals.actual_switch_on_temp
            ),
            target_time_temp_error=safe_diff(
                prediction.expected_target_time_temp, actuals.actual_target_time_temp
            ),
            switch_off_temp_error=safe_diff(
                prediction.expected_switch_off_temp, actuals.actual_switch_off_temp
            ),
            gas_kwh_error=gas_error,
            gas_kwh_error_pct=gas_error_pct,
            burner_hours_error=safe_diff(
                prediction.expected_burner_hours, actuals.actual_burner_hours
            ),
            avg_modulation_error=safe_diff(
                prediction.expected_avg_modulation, actuals.actual_avg_modulation
            ),
        )

    def get_history(self, days: int = 7) -> list[DailyRecord]:
        """Get recent prediction records.

        Args:
            days: Number of days to look back

        Returns:
            List of DailyRecord, most recent first
        """
        records = []
        today = datetime.now().date()

        for i in range(days):
            date = today - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            record = self._load_record(date_str)
            if record:
                records.append(record)

        return records

    def get_error_summary(self, days: int = 7) -> dict[str, Any]:
        """Calculate aggregate error statistics.

        Args:
            days: Number of days to include in summary

        Returns:
            Dictionary with error averages and counts
        """
        records = self.get_history(days)
        records_with_errors = [r for r in records if r.errors is not None]

        if not records_with_errors:
            return {"sample_count": 0}

        # Collect non-None errors
        switch_on_errors = [
            r.errors.switch_on_temp_error
            for r in records_with_errors
            if r.errors.switch_on_temp_error is not None
        ]
        target_time_errors = [
            r.errors.target_time_temp_error
            for r in records_with_errors
            if r.errors.target_time_temp_error is not None
        ]
        switch_off_errors = [
            r.errors.switch_off_temp_error
            for r in records_with_errors
            if r.errors.switch_off_temp_error is not None
        ]
        gas_errors = [
            r.errors.gas_kwh_error
            for r in records_with_errors
            if r.errors.gas_kwh_error is not None
        ]
        gas_pct_errors = [
            r.errors.gas_kwh_error_pct
            for r in records_with_errors
            if r.errors.gas_kwh_error_pct is not None
        ]
        burner_hours_errors = [
            r.errors.burner_hours_error
            for r in records_with_errors
            if r.errors.burner_hours_error is not None
        ]
        modulation_errors = [
            r.errors.avg_modulation_error
            for r in records_with_errors
            if r.errors.avg_modulation_error is not None
        ]

        def safe_avg(values: list[float]) -> float | None:
            return round(sum(values) / len(values), 2) if values else None

        return {
            "sample_count": len(records_with_errors),
            "avg_switch_on_temp_error": safe_avg(switch_on_errors),
            "avg_target_time_temp_error": safe_avg(target_time_errors),
            "avg_switch_off_temp_error": safe_avg(switch_off_errors),
            "avg_gas_kwh_error": safe_avg(gas_errors),
            "avg_gas_pct_error": safe_avg(gas_pct_errors),
            "avg_burner_hours_error": safe_avg(burner_hours_errors),
            "avg_modulation_error": safe_avg(modulation_errors),
        }

    def suggest_coefficient_adjustments(
        self,
        error_summary: dict[str, Any],
        current_k: float,
        current_heating_rate: float,
    ) -> dict[str, float]:
        """Suggest model coefficient adjustments based on error patterns.

        Args:
            error_summary: From get_error_summary()
            current_k: Current cooling rate constant
            current_heating_rate: Current mean heating rate

        Returns:
            Dictionary of suggested new coefficient values (empty if no change needed)
        """
        adjustments = {}

        # Only adjust if we have enough data
        if error_summary["sample_count"] < PREDICTION_CONFIG.min_sample_days:
            logger.info(
                f"Insufficient data for adjustment ({error_summary['sample_count']} < "
                f"{PREDICTION_CONFIG.min_sample_days} days)"
            )
            return adjustments

        # Cooling rate adjustment based on switch-on temp error
        switch_on_error = error_summary.get("avg_switch_on_temp_error")
        if (
            switch_on_error is not None
            and abs(switch_on_error) > PREDICTION_CONFIG.error_threshold
        ):
            # Positive error = predicted too warm = house cools faster = increase k
            k_adjustment = switch_on_error * PREDICTION_CONFIG.k_adjustment_factor
            new_k = current_k + k_adjustment
            new_k = max(PREDICTION_CONFIG.k_min, min(PREDICTION_CONFIG.k_max, new_k))

            if new_k != current_k:
                adjustments["k"] = round(new_k, 6)
                logger.info(
                    f"Suggesting k adjustment: {current_k:.6f} -> {new_k:.6f} "
                    f"(switch-on temp error: {switch_on_error:+.2f}°C)"
                )

        # Heating rate adjustment based on target time temp error
        target_error = error_summary.get("avg_target_time_temp_error")
        if (
            target_error is not None
            and abs(target_error) > PREDICTION_CONFIG.error_threshold
        ):
            # Negative error = predicted too cold = heating slower than expected
            rate_factor = 1 + (
                target_error * PREDICTION_CONFIG.heating_rate_adjustment_factor
            )
            new_rate = current_heating_rate * rate_factor
            new_rate = max(
                PREDICTION_CONFIG.heating_rate_min,
                min(PREDICTION_CONFIG.heating_rate_max, new_rate),
            )

            if abs(new_rate - current_heating_rate) > 0.01:
                adjustments["mean_heating_rate"] = round(new_rate, 3)
                logger.info(
                    f"Suggesting heating rate adjustment: {current_heating_rate:.3f} -> "
                    f"{new_rate:.3f} (target time temp error: {target_error:+.2f}°C)"
                )

        return adjustments

    def _get_temp_at_time(
        self, history: list[dict], time_str: str, date: datetime
    ) -> float | None:
        """Get temperature from history closest to the specified time."""
        if not history or time_str == "CONTINUOUS":
            return None

        parts = time_str.split(":")
        target_dt = datetime.combine(
            date.date(),
            datetime.min.time().replace(hour=int(parts[0]), minute=int(parts[1])),
        )

        closest_temp = None
        min_diff = timedelta(hours=1)  # Max tolerance: 1 hour

        for entry in history:
            state = entry.get("state")
            if state in ("unknown", "unavailable"):
                continue

            try:
                temp = float(state)
                dt_str = entry.get("last_changed", entry.get("last_updated", ""))
                if not dt_str:
                    continue

                entry_dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                # Make naive for comparison
                if entry_dt.tzinfo:
                    entry_dt = entry_dt.replace(tzinfo=None)

                diff = abs(entry_dt - target_dt)
                if diff < min_diff:
                    min_diff = diff
                    closest_temp = temp

            except (ValueError, TypeError):
                continue

        return round(closest_temp, 1) if closest_temp is not None else None

    def _calc_sensor_delta(self, history: list[dict]) -> float | None:
        """Calculate the delta (last - first) of a cumulative sensor."""
        if not history or len(history) < 2:
            return None

        first_val = None
        last_val = None

        for entry in history:
            state = entry.get("state")
            if state in ("unknown", "unavailable"):
                continue
            try:
                val = float(state)
                if first_val is None:
                    first_val = val
                last_val = val
            except (ValueError, TypeError):
                continue

        if first_val is not None and last_val is not None:
            return round(last_val - first_val, 2)
        return None

    def _calc_avg_modulation(self, history: list[dict]) -> float | None:
        """Calculate average modulation when burner was active (>0)."""
        if not history:
            return None

        active_mods = []
        for entry in history:
            state = entry.get("state")
            if state in ("unknown", "unavailable"):
                continue
            try:
                mod = float(state)
                if mod > 0:
                    active_mods.append(mod)
            except (ValueError, TypeError):
                continue

        if active_mods:
            return round(sum(active_mods) / len(active_mods), 1)
        return None

    def _get_gas_usage(self, date_str: str) -> float | None:
        """Get gas usage for a specific date from SmartNetz sensor."""
        # If checking today, use today's sensor
        today_str = datetime.now().strftime("%Y-%m-%d")
        yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        if date_str == today_str:
            entity = "sensor.today_s_gas_usage_energy"
        elif date_str == yesterday_str:
            entity = "sensor.yesterday_s_gas_usage_energy"
        else:
            # For older dates, we'd need to query history
            # For now, return None for dates older than yesterday
            logger.debug(
                f"Gas usage not available for {date_str} (older than yesterday)"
            )
            return None

        state = self.client.get_state(entity)
        if state and state.state not in ("unknown", "unavailable"):
            try:
                return float(state.state)
            except (ValueError, TypeError):
                pass
        return None

    def _load_record(self, date_str: str) -> DailyRecord | None:
        """Load a record for a specific date from the JSONL file."""
        if not self.predictions_file.exists():
            return None

        with open(self.predictions_file) as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if data.get("date") == date_str:
                        return self._dict_to_record(data)
                except json.JSONDecodeError:
                    continue

        return None

    def _save_record(self, record: DailyRecord) -> None:
        """Save or update a record in the JSONL file."""
        records = []

        # Load all existing records
        if self.predictions_file.exists():
            with open(self.predictions_file) as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if data.get("date") != record.date:
                            records.append(data)
                    except json.JSONDecodeError:
                        continue

        # Add/update the current record
        records.append(self._record_to_dict(record))

        # Sort by date and write back
        records.sort(key=lambda x: x.get("date", ""))

        with open(self.predictions_file, "w") as f:
            for data in records:
                f.write(json.dumps(data) + "\n")

    def _record_to_dict(self, record: DailyRecord) -> dict:
        """Convert a DailyRecord to a dictionary for JSON serialization."""
        data = {
            "date": record.date,
            "prediction": asdict(record.prediction),
        }
        if record.adjustments:
            data["adjustments"] = [asdict(a) for a in record.adjustments]
        if record.actuals:
            data["actuals"] = asdict(record.actuals)
        if record.errors:
            data["errors"] = asdict(record.errors)
        if record.coefficients_adjusted:
            data["coefficients_adjusted"] = True
        return data

    def _dict_to_record(self, data: dict) -> DailyRecord:
        """Convert a dictionary back to a DailyRecord."""
        prediction_data = data["prediction"]
        # Handle old records missing timestamp field
        if "timestamp" not in prediction_data:
            prediction_data["timestamp"] = ""
        prediction = DailyPrediction(**prediction_data)

        adjustments = None
        if data.get("adjustments"):
            adjustments = [MidDayAdjustment(**a) for a in data["adjustments"]]

        actuals = DailyActuals(**data["actuals"]) if data.get("actuals") else None
        errors = PredictionErrors(**data["errors"]) if data.get("errors") else None
        return DailyRecord(
            date=data["date"],
            prediction=prediction,
            adjustments=adjustments,
            actuals=actuals,
            errors=errors,
            coefficients_adjusted=data.get("coefficients_adjusted", False),
        )


def format_review_report(record: DailyRecord) -> str:
    """Format a single day's prediction review as a readable report."""
    lines = [f"Prediction Review for {record.date}:", ""]

    p = record.prediction
    a = record.actuals
    e = record.errors

    lines.append("Temperatures:")

    # Switch-on temp
    if p.expected_switch_on_temp and a and a.actual_switch_on_temp:
        error_str = (
            f"{e.switch_on_temp_error:+.1f}°C"
            if e and e.switch_on_temp_error
            else "N/A"
        )
        lines.append(
            f"  At switch-on ({p.switch_on_time}):  "
            f"Predicted: {p.expected_switch_on_temp:.1f}°C  "
            f"Actual: {a.actual_switch_on_temp:.1f}°C  "
            f"Error: {error_str}"
        )

    # Target time temp
    if p.expected_target_time_temp and a and a.actual_target_time_temp:
        error_str = (
            f"{e.target_time_temp_error:+.1f}°C"
            if e and e.target_time_temp_error
            else "N/A"
        )
        lines.append(
            f"  At {p.target_warm_time}:              "
            f"Predicted: {p.expected_target_time_temp:.1f}°C  "
            f"Actual: {a.actual_target_time_temp:.1f}°C  "
            f"Error: {error_str}"
        )

    # Switch-off temp
    if p.switch_off_time != "CONTINUOUS" and p.expected_switch_off_temp:
        if a and a.actual_switch_off_temp:
            error_str = (
                f"{e.switch_off_temp_error:+.1f}°C"
                if e and e.switch_off_temp_error
                else "N/A"
            )
            lines.append(
                f"  At switch-off ({p.switch_off_time}): "
                f"Predicted: {p.expected_switch_off_temp:.1f}°C  "
                f"Actual: {a.actual_switch_off_temp:.1f}°C  "
                f"Error: {error_str}"
            )

    # Mid-day adjustments
    if record.adjustments:
        lines.append("")
        lines.append(f"Mid-day Adjustments ({len(record.adjustments)}):")
        for adj in record.adjustments:
            ts = (
                adj.timestamp.split("T")[1][:5]
                if "T" in adj.timestamp
                else adj.timestamp
            )
            lines.append(
                f"  [{ts}] Off: {adj.switch_off_time}, Setpoint: {adj.setpoint}°C, "
                f"Bedroom: {adj.actual_bedroom_temp:.1f}°C"
            )
            for reason in adj.reasoning:
                lines.append(f"    - {reason}")

    lines.append("")
    lines.append("Energy:")

    # Gas usage
    if a and a.actual_gas_kwh:
        error_str = ""
        if e and e.gas_kwh_error is not None:
            pct_str = f" ({e.gas_kwh_error_pct:+.0f}%)" if e.gas_kwh_error_pct else ""
            error_str = f"Error: {e.gas_kwh_error:+.1f} kWh{pct_str}"
        lines.append(
            f"  Gas usage:      Predicted: {p.expected_gas_kwh:.1f} kWh  "
            f"Actual: {a.actual_gas_kwh:.1f} kWh  {error_str}"
        )

    # Burner hours
    if a and a.actual_burner_hours is not None:
        error_str = ""
        if e and e.burner_hours_error is not None:
            error_str = f"Error: {e.burner_hours_error:+.1f} hrs"
        lines.append(
            f"  Burner hours:   Predicted: {p.expected_burner_hours:.1f} hrs   "
            f"Actual: {a.actual_burner_hours:.1f} hrs   {error_str}"
        )

    # Avg modulation
    if a and a.actual_avg_modulation is not None:
        error_str = ""
        if e and e.avg_modulation_error is not None:
            error_str = f"Error: {e.avg_modulation_error:+.0f}%"
        lines.append(
            f"  Avg modulation: Predicted: {p.expected_avg_modulation:.0f}%       "
            f"Actual: {a.actual_avg_modulation:.0f}%       {error_str}"
        )

    # Burner starts
    if a and a.actual_burner_starts is not None:
        lines.append(f"  Burner starts:  Actual: {a.actual_burner_starts}")

    return "\n".join(lines)


def format_error_summary(summary: dict[str, Any]) -> str:
    """Format error summary as a readable report."""
    if summary["sample_count"] == 0:
        return "No prediction data available for error summary."

    lines = [
        f"{summary['sample_count']}-day average errors:",
        "",
    ]

    if summary.get("avg_switch_on_temp_error") is not None:
        bias = (
            "predicts warm"
            if summary["avg_switch_on_temp_error"] > 0
            else "predicts cold"
        )
        lines.append(
            f"  Temp at switch-on: {summary['avg_switch_on_temp_error']:+.2f}°C (model {bias})"
        )

    if summary.get("avg_target_time_temp_error") is not None:
        status = "accurate" if abs(summary["avg_target_time_temp_error"]) < 0.3 else ""
        lines.append(
            f"  Temp at target time: {summary['avg_target_time_temp_error']:+.2f}°C {status}"
        )

    if summary.get("avg_gas_pct_error") is not None:
        direction = (
            "underestimates" if summary["avg_gas_pct_error"] < 0 else "overestimates"
        )
        lines.append(
            f"  Gas usage: {summary['avg_gas_pct_error']:+.0f}% (model {direction})"
        )

    return "\n".join(lines)
