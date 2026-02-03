"""Home Assistant API client for heating optimization system."""

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import requests
import websockets
from dotenv import load_dotenv


def load_config() -> tuple[str, str]:
    """Load Home Assistant configuration from environment."""
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path)

    ha_url = os.getenv("HA_URL")
    ha_token = os.getenv("HA_TOKEN")

    if not ha_url or not ha_token:
        raise RuntimeError("HA_URL and HA_TOKEN must be set in .env file")

    return ha_url.rstrip("/"), ha_token


def get_ws_url(ha_url: str) -> str:
    """Convert HTTP URL to WebSocket URL."""
    if ha_url.startswith("https://"):
        return ha_url.replace("https://", "wss://") + "/api/websocket"
    return ha_url.replace("http://", "ws://") + "/api/websocket"


@dataclass
class EntityState:
    """Represents a Home Assistant entity state."""

    entity_id: str
    state: str
    attributes: dict
    last_changed: datetime
    last_updated: datetime


class HAClient:
    """Home Assistant API client."""

    def __init__(self):
        self.ha_url, self.token = load_config()
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def get_state(self, entity_id: str) -> EntityState | None:
        """Get current state of an entity."""
        response = requests.get(
            f"{self.ha_url}/api/states/{entity_id}",
            headers=self.headers,
        )
        if response.status_code != 200:
            return None

        data = response.json()
        return EntityState(
            entity_id=data["entity_id"],
            state=data["state"],
            attributes=data.get("attributes", {}),
            last_changed=datetime.fromisoformat(
                data["last_changed"].replace("Z", "+00:00")
            ),
            last_updated=datetime.fromisoformat(
                data["last_updated"].replace("Z", "+00:00")
            ),
        )

    def get_states(self, entity_ids: list[str]) -> dict[str, EntityState]:
        """Get current states of multiple entities."""
        states = {}
        for entity_id in entity_ids:
            state = self.get_state(entity_id)
            if state:
                states[entity_id] = state
        return states

    def get_history(
        self,
        entity_ids: list[str],
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        days: int = 7,
    ) -> dict[str, list[dict]]:
        """Get historical state data for entities."""
        if end_time is None:
            end_time = datetime.now()
        if start_time is None:
            start_time = end_time - timedelta(days=days)

        # Format timestamps for API
        start_str = start_time.strftime("%Y-%m-%dT%H:%M:%S")
        end_str = end_time.strftime("%Y-%m-%dT%H:%M:%S")

        # Build filter entity list
        filter_entity_id = ",".join(entity_ids)

        response = requests.get(
            f"{self.ha_url}/api/history/period/{start_str}",
            headers=self.headers,
            params={
                "filter_entity_id": filter_entity_id,
                "end_time": end_str,
                "minimal_response": "false",
                "significant_changes_only": "false",
            },
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Failed to get history: {response.status_code} - {response.text}"
            )

        data = response.json()

        # Convert to dict keyed by entity_id
        result = {}
        for entity_history in data:
            if entity_history:
                entity_id = entity_history[0]["entity_id"]
                result[entity_id] = entity_history

        return result

    def get_weather_forecast(
        self, entity_id: str = "weather.forecast_home"
    ) -> dict[str, Any]:
        """Get weather entity state with hourly forecast.

        Uses the weather.get_forecasts service (HA 2023.12+) for proper forecast data.
        Falls back to attributes for older HA versions.
        """
        state = self.get_state(entity_id)
        if not state:
            return {}

        result = {
            "current_condition": state.state,
            "temperature": state.attributes.get("temperature"),
            "humidity": state.attributes.get("humidity"),
            "cloud_coverage": state.attributes.get("cloud_coverage"),
            "forecast": [],
        }

        # Try to get hourly forecast via service call (HA 2023.12+)
        hourly_forecast = self._fetch_hourly_forecast(entity_id)
        if hourly_forecast:
            result["forecast"] = hourly_forecast
        else:
            # Fallback to attributes (older HA or if service fails)
            result["forecast"] = state.attributes.get("forecast", [])

        return result

    def _fetch_hourly_forecast(self, entity_id: str) -> list[dict]:
        """Fetch hourly forecast via weather.get_forecasts service.

        Uses the HA 2023.12+ service with return_response parameter.
        Response format: {"changed_states": [...], "service_response": {"weather.xxx": {"forecast": [...]}}}
        """
        try:
            # HA requires ?return_response to get actual response data
            response = requests.post(
                f"{self.ha_url}/api/services/weather/get_forecasts?return_response",
                headers=self.headers,
                json={
                    "entity_id": entity_id,
                    "type": "hourly",
                },
            )

            if response.status_code != 200:
                return []

            data = response.json()
            if not isinstance(data, dict):
                return []

            # Response is under 'service_response' key
            service_response = data.get("service_response", {})
            if entity_id in service_response:
                return service_response[entity_id].get("forecast", [])

            # Fallback: try direct key (older format)
            if entity_id in data:
                return data[entity_id].get("forecast", [])

            return []
        except Exception:
            return []

    def get_sun_state(self, entity_id: str = "sun.sun") -> dict[str, Any]:
        """Get sun entity state with elevation and azimuth."""
        state = self.get_state(entity_id)
        if not state:
            return {}

        return {
            "state": state.state,  # above_horizon / below_horizon
            "elevation": state.attributes.get("elevation"),
            "azimuth": state.attributes.get("azimuth"),
            "rising": state.attributes.get("rising"),
            "next_dawn": state.attributes.get("next_dawn"),
            "next_sunrise": state.attributes.get("next_sunrise"),
            "next_noon": state.attributes.get("next_noon"),
            "next_sunset": state.attributes.get("next_sunset"),
            "next_dusk": state.attributes.get("next_dusk"),
        }

    def call_service(
        self,
        domain: str,
        service: str,
        entity_id: str | None = None,
        data: dict | None = None,
    ) -> bool:
        """Call a Home Assistant service."""
        payload = data or {}
        if entity_id:
            payload["entity_id"] = entity_id

        response = requests.post(
            f"{self.ha_url}/api/services/{domain}/{service}",
            headers=self.headers,
            json=payload,
        )

        return response.status_code == 200

    def set_input_datetime(self, entity_id: str, time_value: str) -> bool:
        """Set an input_datetime helper value (time format: HH:MM or HH:MM:SS)."""
        return self.call_service(
            "input_datetime",
            "set_datetime",
            entity_id=entity_id,
            data={"time": time_value},
        )

    def set_input_number(self, entity_id: str, value: float) -> bool:
        """Set an input_number helper value."""
        return self.call_service(
            "input_number",
            "set_value",
            entity_id=entity_id,
            data={"value": value},
        )

    def set_input_boolean(self, entity_id: str, value: bool) -> bool:
        """Set an input_boolean helper value."""
        service = "turn_on" if value else "turn_off"
        return self.call_service("input_boolean", service, entity_id=entity_id)

    def set_climate_mode(self, entity_id: str, hvac_mode: str) -> bool:
        """Set climate entity HVAC mode (off, heat, auto, etc.)."""
        return self.call_service(
            "climate",
            "set_hvac_mode",
            entity_id=entity_id,
            data={"hvac_mode": hvac_mode},
        )

    def set_climate_temperature(self, entity_id: str, temperature: float) -> bool:
        """Set climate entity target temperature."""
        return self.call_service(
            "climate",
            "set_temperature",
            entity_id=entity_id,
            data={"temperature": temperature},
        )

    def set_number(self, entity_id: str, value: float) -> bool:
        """Set a number entity value (like Viessmann setpoints)."""
        return self.call_service(
            "number",
            "set_value",
            entity_id=entity_id,
            data={"value": value},
        )

    def send_notification(
        self,
        message: str,
        title: str | None = None,
        service: str = "notify.mobile_app_sadats_iphone_15_olx",
    ) -> bool:
        """Send a notification via mobile app."""
        domain, service_name = service.split(".", 1)
        data = {"message": message}
        if title:
            data["title"] = title

        return self.call_service(domain, service_name, data=data)


class HAWebSocketClient:
    """Home Assistant WebSocket client for creating helpers."""

    def __init__(self):
        self.ha_url, self.token = load_config()
        self.ws_url = get_ws_url(self.ha_url)
        self.msg_id = 0

    async def _send_command(self, ws, msg_type: str, **kwargs) -> dict:
        """Send a command and wait for response."""
        self.msg_id += 1
        cmd = {"id": self.msg_id, "type": msg_type, **kwargs}
        await ws.send(json.dumps(cmd))

        while True:
            response = json.loads(await ws.recv())
            if response.get("id") == self.msg_id:
                return response

    async def create_input_boolean(
        self, name: str, icon: str = "mdi:toggle-switch"
    ) -> dict:
        """Create an input_boolean helper."""
        async with websockets.connect(self.ws_url) as ws:
            # Authenticate
            await ws.recv()  # auth_required
            await ws.send(json.dumps({"type": "auth", "access_token": self.token}))
            msg = json.loads(await ws.recv())
            if msg.get("type") != "auth_ok":
                raise RuntimeError(f"Authentication failed: {msg}")

            response = await self._send_command(
                ws, "input_boolean/create", name=name, icon=icon
            )
            if not response.get("success"):
                error = response.get("error", {})
                raise RuntimeError(
                    f"Failed to create input_boolean: {error.get('message', response)}"
                )
            return response.get("result", {})

    async def create_input_datetime(
        self,
        name: str,
        has_date: bool = False,
        has_time: bool = True,
        icon: str = "mdi:clock",
    ) -> dict:
        """Create an input_datetime helper."""
        async with websockets.connect(self.ws_url) as ws:
            # Authenticate
            await ws.recv()
            await ws.send(json.dumps({"type": "auth", "access_token": self.token}))
            msg = json.loads(await ws.recv())
            if msg.get("type") != "auth_ok":
                raise RuntimeError(f"Authentication failed: {msg}")

            response = await self._send_command(
                ws,
                "input_datetime/create",
                name=name,
                has_date=has_date,
                has_time=has_time,
                icon=icon,
            )
            if not response.get("success"):
                error = response.get("error", {})
                raise RuntimeError(
                    f"Failed to create input_datetime: {error.get('message', response)}"
                )
            return response.get("result", {})

    async def create_input_number(
        self,
        name: str,
        min_value: float,
        max_value: float,
        step: float = 0.5,
        unit: str = "°C",
        mode: str = "slider",
        icon: str = "mdi:thermometer",
    ) -> dict:
        """Create an input_number helper."""
        async with websockets.connect(self.ws_url) as ws:
            # Authenticate
            await ws.recv()
            await ws.send(json.dumps({"type": "auth", "access_token": self.token}))
            msg = json.loads(await ws.recv())
            if msg.get("type") != "auth_ok":
                raise RuntimeError(f"Authentication failed: {msg}")

            response = await self._send_command(
                ws,
                "input_number/create",
                name=name,
                min=min_value,
                max=max_value,
                step=step,
                unit_of_measurement=unit,
                mode=mode,
                icon=icon,
            )
            if not response.get("success"):
                error = response.get("error", {})
                raise RuntimeError(
                    f"Failed to create input_number: {error.get('message', response)}"
                )
            return response.get("result", {})


def create_helper(helper_type: str, **kwargs) -> dict:
    """Synchronous wrapper to create a helper entity."""
    client = HAWebSocketClient()

    if helper_type == "input_boolean":
        return asyncio.run(client.create_input_boolean(**kwargs))
    elif helper_type == "input_datetime":
        return asyncio.run(client.create_input_datetime(**kwargs))
    elif helper_type == "input_number":
        return asyncio.run(client.create_input_number(**kwargs))
    else:
        raise ValueError(f"Unknown helper type: {helper_type}")
