#!/usr/bin/env python3
"""Create helpers and automations in Home Assistant via WebSocket API."""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import requests
import websockets
import yaml
from dotenv import load_dotenv


def load_config() -> tuple[str, str]:
    """Load Home Assistant configuration from environment."""
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)

    ha_url = os.getenv("HA_URL")
    ha_token = os.getenv("HA_TOKEN")

    if not ha_url or not ha_token:
        print("Error: HA_URL and HA_TOKEN must be set in .env file")
        sys.exit(1)

    return ha_url.rstrip("/"), ha_token


def get_ws_url(ha_url: str) -> str:
    """Convert HTTP URL to WebSocket URL."""
    if ha_url.startswith("https://"):
        return ha_url.replace("https://", "wss://") + "/api/websocket"
    return ha_url.replace("http://", "ws://") + "/api/websocket"


class HAWebSocket:
    """Home Assistant WebSocket client."""

    def __init__(self, ws):
        self.ws = ws
        self.msg_id = 0

    async def send_command(self, msg_type: str, **kwargs) -> dict:
        """Send a command and wait for response."""
        self.msg_id += 1
        cmd = {"id": self.msg_id, "type": msg_type, **kwargs}
        await self.ws.send(json.dumps(cmd))

        while True:
            response = json.loads(await self.ws.recv())
            if response.get("id") == self.msg_id:
                return response

    async def create_input_boolean(self, name: str, entity_id: str, icon: str | None = None) -> dict:
        """Create an input_boolean helper."""
        data = {"name": name, "icon": icon or "mdi:toggle-switch"}
        response = await self.send_command("input_boolean/create", **data)
        if not response.get("success"):
            error = response.get("error", {})
            raise RuntimeError(f"Failed to create input_boolean: {error.get('message', response)}")
        return response.get("result", {})

    async def create_input_datetime(self, name: str, has_date: bool = False, has_time: bool = True, icon: str | None = None) -> dict:
        """Create an input_datetime helper."""
        data = {"name": name, "has_date": has_date, "has_time": has_time, "icon": icon or "mdi:clock"}
        response = await self.send_command("input_datetime/create", **data)
        if not response.get("success"):
            error = response.get("error", {})
            raise RuntimeError(f"Failed to create input_datetime: {error.get('message', response)}")
        return response.get("result", {})



async def create_delayed_stop_helpers() -> None:
    """Create the helper entities for delayed stop automation."""
    ha_url, token = load_config()
    ws_url = get_ws_url(ha_url)

    async with websockets.connect(ws_url) as ws:
        msg = json.loads(await ws.recv())
        await ws.send(json.dumps({"type": "auth", "access_token": token}))
        msg = json.loads(await ws.recv())
        if msg.get("type") != "auth_ok":
            raise RuntimeError(f"Authentication failed: {msg}")

        client = HAWebSocket(ws)

        # Create input_boolean for delay stop toggle
        print("Creating input_boolean.central_heating_delay_stop...")
        try:
            result = await client.create_input_boolean(
                name="Central Heating Delay Stop",
                entity_id="central_heating_delay_stop",
                icon="mdi:timer-off-outline"
            )
            print(f"  Created: {result}")
        except RuntimeError as e:
            if "already exists" in str(e).lower():
                print("  Already exists, skipping...")
            else:
                raise

        # Create input_datetime for stop time
        print("Creating input_datetime.central_heating_stop...")
        try:
            result = await client.create_input_datetime(
                name="Central Heating Stop",
                has_date=False,
                has_time=True,
                icon="mdi:clock-end"
            )
            print(f"  Created: {result}")
        except RuntimeError as e:
            if "already exists" in str(e).lower():
                print("  Already exists, skipping...")
            else:
                raise

        print("Done creating helpers!")


def create_delayed_stop_automation() -> None:
    """Create the delayed stop automation via REST API."""
    ha_url, token = load_config()

    # Generate a proper numeric ID (timestamp-based like HA does)
    automation_id = str(int(time.time() * 1000))

    automation_config = {
        "id": automation_id,
        "alias": "Automation Central Heating Delayed Stop",
        "description": "Turns off heating at a scheduled time when delay stop is enabled",
        "triggers": [
            {
                "trigger": "time",
                "at": "input_datetime.central_heating_stop"
            }
        ],
        "conditions": [
            {
                "condition": "state",
                "entity_id": "input_boolean.central_heating_delay_stop",
                "state": "on"
            }
        ],
        "actions": [
            {
                "device_id": "33e4a91a3466d1d40cecf1fbff622934",
                "domain": "climate",
                "entity_id": "ae3aa52d06ebf1a386a3cbf4fc461fae",
                "type": "set_hvac_mode",
                "hvac_mode": "off"
            },
            {
                "action": "input_boolean.turn_off",
                "metadata": {},
                "data": {},
                "target": {
                    "entity_id": "input_boolean.central_heating_delay_stop"
                }
            },
            {
                "action": "notify.mobile_app_sadats_iphone_15_olx",
                "metadata": {},
                "data": {
                    "message": "Heating Switched off",
                    "title": "Heating"
                }
            }
        ],
        "mode": "single"
    }

    print("Creating automation: Automation Central Heating Delayed Stop...")

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    response = requests.post(
        f"{ha_url}/api/config/automation/config/{automation_id}",
        headers=headers,
        json=automation_config
    )

    if response.status_code == 200:
        print(f"  Created automation with ID: {automation_id}")
        print("Done!")
    else:
        raise RuntimeError(f"Failed to create automation: {response.status_code} - {response.text}")


def _find_automations_by_alias(ha_url: str, token: str, alias: str) -> list[dict]:
    """Find existing automations matching an alias."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    response = requests.get(f"{ha_url}/api/states", headers=headers)
    if response.status_code != 200:
        return []

    matches = []
    for entity in response.json():
        entity_id = entity.get("entity_id", "")
        if not entity_id.startswith("automation."):
            continue
        attrs = entity.get("attributes", {})
        if attrs.get("friendly_name") == alias:
            # Extract the automation config ID from the entity
            matches.append({
                "entity_id": entity_id,
                "friendly_name": attrs.get("friendly_name"),
                "id": attrs.get("id"),
            })
    return matches


def _delete_automation(ha_url: str, token: str, automation_id: str) -> bool:
    """Delete an automation by its config ID."""
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    response = requests.delete(
        f"{ha_url}/api/config/automation/config/{automation_id}",
        headers=headers,
    )
    return response.status_code == 200


def delete_automation_by_alias(alias: str) -> None:
    """Delete all automations matching an alias."""
    ha_url, token = load_config()
    matches = _find_automations_by_alias(ha_url, token, alias)
    if not matches:
        print(f"No automation found with alias: {alias}")
        sys.exit(1)
    for auto in matches:
        auto_id = auto.get("id")
        if auto_id and _delete_automation(ha_url, token, auto_id):
            print(f"  Deleted: {auto['entity_id']} (id={auto_id})")
        else:
            print(f"  Failed to delete: {auto['entity_id']}")
    print("Done!")


def upload_automation_file(file_path: str) -> None:
    """Upload an automation from a YAML file via REST API.

    If an automation with the same alias already exists, it is deleted first
    to prevent duplicates.
    """
    ha_url, token = load_config()
    path = Path(file_path)

    if not path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    with open(path) as f:
        config = yaml.safe_load(f)

    alias = config.get("alias", path.stem)

    # Delete existing automations with the same alias
    existing = _find_automations_by_alias(ha_url, token, alias)
    for auto in existing:
        auto_id = auto.get("id")
        if auto_id:
            print(f"  Deleting existing automation: {auto['entity_id']} (id={auto_id})")
            if _delete_automation(ha_url, token, auto_id):
                print(f"  Deleted.")
            else:
                print(f"  Warning: failed to delete {auto_id}, continuing...")

    # Generate a proper numeric ID (timestamp-based like HA does)
    automation_id = str(int(time.time() * 1000))

    # HA API expects: triggers, conditions, actions (plural)
    # with trigger: "type" format (not platform: "type")
    api_config = {
        "id": automation_id,
        "alias": alias,
        "description": config.get("description", ""),
        "mode": config.get("mode", "single"),
        "triggers": config.get("triggers", []),
        "conditions": config.get("conditions", []),
        "actions": config.get("actions", []),
    }

    print(f"Creating automation: {alias}...")

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    response = requests.post(
        f"{ha_url}/api/config/automation/config/{automation_id}",
        headers=headers,
        json=api_config
    )

    if response.status_code == 200:
        print(f"  Created automation with ID: {automation_id}")
        print("Done!")
    else:
        raise RuntimeError(f"Failed to create automation: {response.status_code} - {response.text}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create Home Assistant helpers and automations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Create delayed stop helpers:
    python ha_automation.py create-helpers

  Create delayed stop automation:
    python ha_automation.py create-automation

  Create both helpers and automation:
    python ha_automation.py setup-delayed-stop

  Upload automation from YAML file:
    python ha_automation.py upload automations/my_automation.yaml

  Delete automation by alias:
    python ha_automation.py delete "Sunset night light"
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("create-helpers", help="Create input_boolean and input_datetime helpers for delayed stop")
    subparsers.add_parser("create-automation", help="Create the delayed stop automation")
    subparsers.add_parser("setup-delayed-stop", help="Create both helpers and automation")

    upload_parser = subparsers.add_parser("upload", help="Upload an automation from YAML file")
    upload_parser.add_argument("file", help="Path to the automation YAML file")

    delete_parser = subparsers.add_parser("delete", help="Delete an automation by alias")
    delete_parser.add_argument("alias", help="Friendly name (alias) of the automation to delete")

    args = parser.parse_args()

    if args.command == "create-helpers":
        asyncio.run(create_delayed_stop_helpers())
    elif args.command == "create-automation":
        create_delayed_stop_automation()
    elif args.command == "setup-delayed-stop":
        asyncio.run(create_delayed_stop_helpers())
        create_delayed_stop_automation()
    elif args.command == "upload":
        upload_automation_file(args.file)
    elif args.command == "delete":
        delete_automation_by_alias(args.alias)


if __name__ == "__main__":
    main()
