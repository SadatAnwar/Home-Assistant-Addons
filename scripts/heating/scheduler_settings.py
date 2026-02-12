"""Settings and parsing helpers for heating scheduler."""

from __future__ import annotations

from typing import Any

from .config import DEFAULTS, HELPERS
from .ha_client import HAClient


class SchedulerSettings:
    """Read and normalize user settings from HA helpers."""

    def __init__(self, client: HAClient):
        self.client = client

    def get_helper_value(self, entity_id: str, default: Any, cast=str):
        """Read a helper entity, returning default if unavailable."""
        state = self.client.get_state(entity_id)
        if state and state.state not in ("unknown", "unavailable"):
            try:
                return cast(state.state)
            except (ValueError, TypeError):
                pass
        return default

    def get_user_settings(self) -> dict[str, Any]:
        """Get user-configured settings from HA helpers."""
        settings = {
            "target_warm_time": self.get_helper_value(
                HELPERS.target_warm_time, DEFAULTS.target_warm_time
            ),
            "preferred_off_time": self.get_helper_value(
                HELPERS.preferred_off_time, DEFAULTS.preferred_off_time
            ),
            "target_temp": self.get_helper_value(
                HELPERS.target_temp, DEFAULTS.target_temp, float
            ),
            "min_bedroom_temp": self.get_helper_value(
                HELPERS.min_bedroom_temp, DEFAULTS.min_bedroom_temp, float
            ),
            "min_daytime_temp": self.get_helper_value(
                HELPERS.min_daytime_temp, DEFAULTS.min_daytime_temp, float
            ),
        }

        # Daytime min must be >= overnight min
        settings["min_daytime_temp"] = max(
            settings["min_daytime_temp"], settings["min_bedroom_temp"]
        )

        return settings

    def parse_time(self, time_str: str):
        """Parse time string (HH:MM or HH:MM:SS) to time object."""
        parts = time_str.split(":")
        from datetime import time

        return time(int(parts[0]), int(parts[1]))
