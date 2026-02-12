"""Publishing helpers for scheduler outputs to Home Assistant and logs."""

from __future__ import annotations

import logging
from datetime import datetime, time

from .config import HELPERS, NOTIFICATION_SERVICE
from .ha_client import HAClient

logger = logging.getLogger(__name__)


class SchedulerPublisher:
    """Publish computed schedules to HA helpers and notifications."""

    def __init__(self, client: HAClient):
        self.client = client

    def update_ha_helpers(self, schedule, include_switch_on: bool = True) -> None:
        """Update HA helper entities with computed schedule."""
        if include_switch_on:
            success = self.client.set_input_datetime(
                HELPERS.switch_on_time,
                schedule.switch_on_time.strftime("%H:%M"),
            )
            if not success:
                logger.warning("Failed to update switch-on time helper")

        # When switch_off_time is None (continuous heating), set off = on
        # to effectively disable the off automation
        if schedule.switch_off_time is not None:
            off_time_str = schedule.switch_off_time.strftime("%H:%M")
        else:
            off_time_str = schedule.switch_on_time.strftime("%H:%M")
            logger.info("Continuous heating mode - setting off time = on time")

        success = self.client.set_input_datetime(
            HELPERS.switch_off_time,
            off_time_str,
        )
        if not success:
            logger.warning("Failed to update switch-off time helper")

        success = self.client.set_input_number(
            HELPERS.optimal_setpoint,
            schedule.optimal_setpoint,
        )
        if not success:
            logger.warning("Failed to update optimal setpoint helper")

    def send_schedule_notification(self, schedule) -> None:
        """Send notification with today's heating schedule."""
        off_time_str = (
            schedule.switch_off_time.strftime("%H:%M")
            if schedule.switch_off_time
            else "CONTINUOUS"
        )
        switch_on_temp_str = (
            f"{schedule.expected_switch_on_temp:.1f}°C"
            if schedule.expected_switch_on_temp is not None
            else "N/A"
        )
        message = (
            f"Heating Schedule:\n"
            f"ON: {schedule.switch_on_time.strftime('%H:%M')} (expect {switch_on_temp_str})\n"
            f"OFF: {off_time_str}\n"
            f"Setpoint: {schedule.optimal_setpoint}°C\n"
            f"Expected range: {schedule.expected_min_temp}-{schedule.expected_max_temp}°C"
        )

        if schedule.solar_contribution > 0.5:
            message += f"\nSolar bonus: +{schedule.solar_contribution:.1f}°C"

        self.client.send_notification(
            message=message,
            title="Heating Plan",
            service=NOTIFICATION_SERVICE,
        )

    def log_forecast_segments(
        self,
        forecast: list[dict],
        switch_on_time: time,
        switch_off_time: time | None,
    ) -> None:
        """Log hourly forecast split by heating ON and OFF periods."""
        if not forecast:
            return

        on_entries: list[str] = []
        off_entries: list[str] = []

        on_minutes = switch_on_time.hour * 60 + switch_on_time.minute
        # If no switch-off, treat entire day as ON
        off_minutes = (
            (switch_off_time.hour * 60 + switch_off_time.minute)
            if switch_off_time
            else 24 * 60
        )

        for entry in forecast[:24]:  # Next 24 hours only
            dt_str = entry.get("datetime", "")
            temp = entry.get("temperature")
            if not dt_str or temp is None:
                continue

            try:
                dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                hour = dt.hour
                hour_minutes = hour * 60
                label = f"{hour:02d}:00={temp:.0f}°C"

                if on_minutes <= off_minutes:
                    is_on = on_minutes <= hour_minutes < off_minutes
                else:
                    is_on = hour_minutes >= on_minutes or hour_minutes < off_minutes

                if is_on:
                    on_entries.append(label)
                else:
                    off_entries.append(label)
            except ValueError:
                continue

        on_time_str = switch_on_time.strftime("%H:%M")
        off_time_str = switch_off_time.strftime("%H:%M") if switch_off_time else "N/A"

        if on_entries:
            logger.info(
                f"Forecast (heating ON {on_time_str}-{off_time_str}): "
                f"{' '.join(on_entries)}"
            )
        if off_entries:
            logger.info(
                f"Forecast (heating OFF {off_time_str}-{on_time_str}): "
                f"{' '.join(off_entries)}"
            )
