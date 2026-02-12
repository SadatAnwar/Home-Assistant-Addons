"""Data collector for heating optimization system.

Pulls historical data from Home Assistant and converts to pandas DataFrames
for analysis and model training.
"""

from datetime import datetime
from typing import Any

import pandas as pd

from .config import (
    BOILER_SENSORS,
    CLIMATE_ENTITY,
    MODEL_CONFIG,
    ROOM_SENSORS,
    WEATHER_ENTITIES,
)
from .ha_client import HAClient


class DataCollector:
    """Collects and processes historical data from Home Assistant."""

    def __init__(self, client: HAClient | None = None):
        self.client = client or HAClient()

    def collect_temperature_data(
        self,
        days: int = MODEL_CONFIG.history_days,
    ) -> pd.DataFrame:
        """Collect room temperature history for all rooms."""
        entity_ids = list(ROOM_SENSORS.values())
        history = self.client.get_history(entity_ids, days=days)

        records = []
        for entity_id, states in history.items():
            room = self._entity_to_room(entity_id, ROOM_SENSORS)
            for state in states:
                if state["state"] not in ("unknown", "unavailable"):
                    try:
                        records.append(
                            {
                                "timestamp": self._parse_timestamp(
                                    state["last_changed"]
                                ),
                                "room": room,
                                "temperature": float(state["state"]),
                            }
                        )
                    except (ValueError, KeyError):
                        continue

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("timestamp")
            df = df.drop_duplicates(subset=["timestamp", "room"])

        return df

    def collect_outside_temperature(
        self,
        days: int = MODEL_CONFIG.history_days,
    ) -> pd.DataFrame:
        """Collect outside temperature history."""
        entity_id = BOILER_SENSORS["outside_temp"]
        history = self.client.get_history([entity_id], days=days)

        records = self._parse_numeric_history(
            history.get(entity_id, []), "outside_temp"
        )

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("timestamp")
            df = df.drop_duplicates(subset=["timestamp"])

        return df

    def collect_heating_data(
        self,
        days: int = MODEL_CONFIG.history_days,
    ) -> pd.DataFrame:
        """Collect heating state, modulation, and climate data."""
        entity_ids = [
            CLIMATE_ENTITY,
            BOILER_SENSORS["burner_modulation"],
            BOILER_SENSORS["burner_active"],
            BOILER_SENSORS["supply_temp"],
        ]
        history = self.client.get_history(entity_ids, days=days)

        # Process climate state (on/off/auto)
        climate_records = []
        for state in history.get(CLIMATE_ENTITY, []):
            try:
                attrs = state.get("attributes", {})
                climate_records.append(
                    {
                        "timestamp": self._parse_timestamp(state["last_changed"]),
                        "hvac_mode": state["state"],
                        "setpoint": attrs.get("temperature"),
                    }
                )
            except (ValueError, KeyError):
                continue

        climate_df = pd.DataFrame(climate_records)

        # Process burner modulation
        modulation_records = self._parse_numeric_history(
            history.get(BOILER_SENSORS["burner_modulation"], []),
            "burner_modulation",
        )
        modulation_df = pd.DataFrame(modulation_records)

        # Process burner active state
        burner_records = []
        for state in history.get(BOILER_SENSORS["burner_active"], []):
            try:
                burner_records.append(
                    {
                        "timestamp": self._parse_timestamp(state["last_changed"]),
                        "burner_active": state["state"] == "on",
                    }
                )
            except (ValueError, KeyError):
                continue

        burner_df = pd.DataFrame(burner_records)

        # Process supply temperature
        supply_records = self._parse_numeric_history(
            history.get(BOILER_SENSORS["supply_temp"], []), "supply_temp"
        )
        supply_df = pd.DataFrame(supply_records)

        # Merge all DataFrames
        if climate_df.empty:
            return pd.DataFrame()

        climate_df = climate_df.sort_values("timestamp")
        result = climate_df

        for df in [modulation_df, burner_df, supply_df]:
            if not df.empty:
                df = df.sort_values("timestamp")
                result = pd.merge_asof(
                    result,
                    df,
                    on="timestamp",
                    direction="nearest",
                    tolerance=pd.Timedelta("5min"),
                )

        return result

    def collect_weather_data(
        self,
        days: int = MODEL_CONFIG.history_days,
    ) -> pd.DataFrame:
        """Collect sun elevation and weather condition history."""
        entity_ids = [WEATHER_ENTITIES["sun"]]
        history = self.client.get_history(entity_ids, days=days)

        records = []
        for state in history.get(WEATHER_ENTITIES["sun"], []):
            try:
                attrs = state.get("attributes", {})
                records.append(
                    {
                        "timestamp": self._parse_timestamp(state["last_changed"]),
                        "sun_state": state["state"],
                        "sun_elevation": attrs.get("elevation"),
                        "sun_azimuth": attrs.get("azimuth"),
                    }
                )
            except (ValueError, KeyError):
                continue

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("timestamp")

        return df

    def collect_all_data(
        self,
        days: int = MODEL_CONFIG.history_days,
    ) -> dict[str, pd.DataFrame]:
        """Collect all historical data needed for model training."""
        return {
            "temperatures": self.collect_temperature_data(days),
            "outside_temp": self.collect_outside_temperature(days),
            "heating": self.collect_heating_data(days),
            "weather": self.collect_weather_data(days),
        }

    def build_training_dataset(
        self,
        days: int = MODEL_CONFIG.history_days,
    ) -> pd.DataFrame:
        """Build a unified training dataset with all features.

        Creates a time-series dataset at regular intervals with:
        - Room temperatures (bedroom as primary)
        - Outside temperature
        - Heating state and setpoint
        - Burner modulation
        - Sun elevation and weather
        """
        data = self.collect_all_data(days)

        # Pivot room temperatures to have one column per room
        temps_df = data["temperatures"]
        if temps_df.empty:
            return pd.DataFrame()

        temps_pivot = temps_df.pivot_table(
            index="timestamp",
            columns="room",
            values="temperature",
            aggfunc="mean",
        ).reset_index()

        # Resample to regular intervals (15 minutes)
        temps_pivot = temps_pivot.set_index("timestamp")
        temps_pivot = temps_pivot.resample("15min").mean().interpolate(method="time")
        temps_pivot = temps_pivot.reset_index()

        # Merge with outside temperature
        if not data["outside_temp"].empty:
            outside_df = data["outside_temp"].set_index("timestamp")
            outside_df = outside_df.resample("15min").mean().interpolate(method="time")
            outside_df = outside_df.reset_index()
            temps_pivot = pd.merge_asof(
                temps_pivot,
                outside_df,
                on="timestamp",
                direction="nearest",
                tolerance=pd.Timedelta("30min"),
            )

        # Merge with heating data
        if not data["heating"].empty:
            heating_df = data["heating"].set_index("timestamp")
            heating_df = heating_df.resample("15min").ffill()
            heating_df = heating_df.reset_index()
            temps_pivot = pd.merge_asof(
                temps_pivot,
                heating_df,
                on="timestamp",
                direction="nearest",
                tolerance=pd.Timedelta("30min"),
            )

        # Merge with weather/sun data
        if not data["weather"].empty:
            weather_df = data["weather"].set_index("timestamp")
            weather_df = weather_df.resample("15min").ffill()
            weather_df = weather_df.reset_index()
            temps_pivot = pd.merge_asof(
                temps_pivot,
                weather_df,
                on="timestamp",
                direction="nearest",
                tolerance=pd.Timedelta("30min"),
            )

        # Add derived features
        if "bedroom" in temps_pivot.columns and "outside_temp" in temps_pivot.columns:
            temps_pivot["temp_diff"] = (
                temps_pivot["bedroom"] - temps_pivot["outside_temp"]
            )

        # Add heating_on flag
        if "hvac_mode" in temps_pivot.columns:
            temps_pivot["heating_on"] = temps_pivot["hvac_mode"].isin(["heat", "auto"])

        # Add hour of day
        temps_pivot["hour"] = temps_pivot["timestamp"].dt.hour

        return temps_pivot

    def get_current_state(self) -> dict[str, Any]:
        """Get current state of all relevant entities."""
        # Room temperatures
        room_temps = {}
        for room, entity_id in ROOM_SENSORS.items():
            state = self.client.get_state(entity_id)
            if state and state.state not in ("unknown", "unavailable"):
                try:
                    room_temps[room] = float(state.state)
                except ValueError:
                    pass

        # Outside temperature
        outside_state = self.client.get_state(BOILER_SENSORS["outside_temp"])
        outside_temp = None
        if outside_state and outside_state.state not in ("unknown", "unavailable"):
            try:
                outside_temp = float(outside_state.state)
            except ValueError:
                pass

        # Heating state
        climate_state = self.client.get_state(CLIMATE_ENTITY)
        hvac_mode = None
        setpoint = None
        if climate_state:
            hvac_mode = climate_state.state
            setpoint = climate_state.attributes.get("temperature")

        # Burner modulation
        modulation_state = self.client.get_state(BOILER_SENSORS["burner_modulation"])
        burner_modulation = None
        if modulation_state and modulation_state.state not in (
            "unknown",
            "unavailable",
        ):
            try:
                burner_modulation = float(modulation_state.state)
            except ValueError:
                pass

        # Sun state
        sun_state = self.client.get_sun_state()

        # Weather
        weather = self.client.get_weather_forecast()

        return {
            "room_temps": room_temps,
            "outside_temp": outside_temp,
            "hvac_mode": hvac_mode,
            "setpoint": setpoint,
            "burner_modulation": burner_modulation,
            "sun_elevation": sun_state.get("elevation"),
            "sun_azimuth": sun_state.get("azimuth"),
            "weather_condition": weather.get("current_condition"),
            "cloud_coverage": weather.get("cloud_coverage"),
            "forecast": weather.get("forecast", []),
        }

    def _parse_numeric_history(self, states: list[dict], field_name: str) -> list[dict]:
        """Parse HA states into [{timestamp, field_name: float}, ...]."""
        records = []
        for state in states:
            if state["state"] not in ("unknown", "unavailable"):
                try:
                    records.append(
                        {
                            "timestamp": self._parse_timestamp(state["last_changed"]),
                            field_name: float(state["state"]),
                        }
                    )
                except (ValueError, KeyError):
                    continue
        return records

    def _entity_to_room(self, entity_id: str, mapping: dict) -> str | None:
        """Convert entity ID to room name using mapping."""
        for room, eid in mapping.items():
            if eid == entity_id:
                return room
        return None

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse HA timestamp string to datetime."""
        # Handle both 'Z' suffix and +00:00 format
        if timestamp_str.endswith("Z"):
            timestamp_str = timestamp_str[:-1] + "+00:00"
        return datetime.fromisoformat(timestamp_str)
