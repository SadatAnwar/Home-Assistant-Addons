#!/usr/bin/env python3
"""Setup script to create HA helper entities for heating optimization."""

import argparse
import asyncio
import logging
import sys

from .config import DEFAULTS, HELPERS
from .ha_client import HAClient, HAWebSocketClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def create_helpers_async(skip_existing: bool = True) -> dict[str, bool]:
    """Create all required helper entities in Home Assistant."""
    ws_client = HAWebSocketClient()
    rest_client = HAClient()
    results = {}

    # User-configured helpers
    user_helpers = [
        {
            "type": "input_datetime",
            "name": "Heating Target Warm Time",
            "entity_id": HELPERS.target_warm_time,
            "has_date": False,
            "has_time": True,
            "icon": "mdi:weather-sunset-up",
        },
        {
            "type": "input_datetime",
            "name": "Heating Preferred Off Time",
            "entity_id": HELPERS.preferred_off_time,
            "has_date": False,
            "has_time": True,
            "icon": "mdi:weather-sunset-down",
        },
        {
            "type": "input_number",
            "name": "Heating Target Temp",
            "entity_id": HELPERS.target_temp,
            "min": 16,
            "max": 24,
            "step": 0.5,
            "unit": "°C",
            "mode": "slider",
            "icon": "mdi:thermometer",
        },
        {
            "type": "input_number",
            "name": "Heating Min Bedroom Temp",
            "entity_id": HELPERS.min_bedroom_temp,
            "min": 15,
            "max": 22,
            "step": 0.5,
            "unit": "°C",
            "mode": "slider",
            "icon": "mdi:thermometer-low",
        },
        {
            "type": "input_number",
            "name": "Heating Min Daytime Temp",
            "entity_id": HELPERS.min_daytime_temp,
            "min": 16,
            "max": 24,
            "step": 0.5,
            "unit": "°C",
            "mode": "slider",
            "icon": "mdi:thermometer-high",
        },
        {
            "type": "input_boolean",
            "name": "Heating Optimization Enabled",
            "entity_id": HELPERS.optimization_enabled,
            "icon": "mdi:robot",
        },
    ]

    # ML-computed helpers
    ml_helpers = [
        {
            "type": "input_datetime",
            "name": "Heating Switch On Time",
            "entity_id": HELPERS.switch_on_time,
            "has_date": False,
            "has_time": True,
            "icon": "mdi:power-on",
        },
        {
            "type": "input_datetime",
            "name": "Heating Switch Off Time",
            "entity_id": HELPERS.switch_off_time,
            "has_date": False,
            "has_time": True,
            "icon": "mdi:power-off",
        },
        {
            "type": "input_number",
            "name": "Heating Optimal Setpoint",
            "entity_id": HELPERS.optimal_setpoint,
            "min": 16,
            "max": 26,
            "step": 1,
            "unit": "°C",
            "mode": "slider",
            "icon": "mdi:thermometer-auto",
        },
    ]

    all_helpers = user_helpers + ml_helpers

    for helper in all_helpers:
        entity_id = helper["entity_id"]

        # Check if helper already exists
        if skip_existing:
            state = rest_client.get_state(entity_id)
            if state is not None:
                logger.info(f"Skipping {entity_id} (already exists)")
                results[entity_id] = True
                continue

        try:
            if helper["type"] == "input_datetime":
                await ws_client.create_input_datetime(
                    name=helper["name"],
                    has_date=helper.get("has_date", False),
                    has_time=helper.get("has_time", True),
                    icon=helper.get("icon", "mdi:clock"),
                )
                logger.info(f"Created {entity_id}")
                results[entity_id] = True

            elif helper["type"] == "input_number":
                await ws_client.create_input_number(
                    name=helper["name"],
                    min_value=helper["min"],
                    max_value=helper["max"],
                    step=helper.get("step", 0.5),
                    unit=helper.get("unit", "°C"),
                    mode=helper.get("mode", "slider"),
                    icon=helper.get("icon", "mdi:thermometer"),
                )
                logger.info(f"Created {entity_id}")
                results[entity_id] = True

            elif helper["type"] == "input_boolean":
                await ws_client.create_input_boolean(
                    name=helper["name"],
                    icon=helper.get("icon", "mdi:toggle-switch"),
                )
                logger.info(f"Created {entity_id}")
                results[entity_id] = True

        except RuntimeError as e:
            if "already exists" in str(e).lower():
                logger.info(f"Skipping {entity_id} (already exists)")
                results[entity_id] = True
            else:
                logger.error(f"Failed to create {entity_id}: {e}")
                results[entity_id] = False

    return results


def set_default_values() -> dict[str, bool]:
    """Set default values for helper entities."""
    client = HAClient()
    results = {}

    defaults = [
        (HELPERS.target_warm_time, "set_input_datetime", DEFAULTS.target_warm_time),
        (HELPERS.preferred_off_time, "set_input_datetime", DEFAULTS.preferred_off_time),
        (HELPERS.target_temp, "set_input_number", DEFAULTS.target_temp),
        (HELPERS.min_bedroom_temp, "set_input_number", DEFAULTS.min_bedroom_temp),
        (HELPERS.min_daytime_temp, "set_input_number", DEFAULTS.min_daytime_temp),
        (HELPERS.switch_on_time, "set_input_datetime", DEFAULTS.default_switch_on_time),
        (
            HELPERS.switch_off_time,
            "set_input_datetime",
            DEFAULTS.default_switch_off_time,
        ),
        (HELPERS.optimal_setpoint, "set_input_number", DEFAULTS.default_setpoint),
        (HELPERS.optimization_enabled, "set_input_boolean", True),
    ]

    for entity_id, method, value in defaults:
        try:
            if method == "set_input_datetime":
                success = client.set_input_datetime(entity_id, value)
            elif method == "set_input_number":
                success = client.set_input_number(entity_id, value)
            elif method == "set_input_boolean":
                success = client.set_input_boolean(entity_id, value)
            else:
                success = False

            if success:
                logger.info(f"Set {entity_id} = {value}")
            else:
                logger.warning(f"Failed to set {entity_id}")

            results[entity_id] = success

        except Exception as e:
            logger.error(f"Error setting {entity_id}: {e}")
            results[entity_id] = False

    return results


def create_helpers(skip_existing: bool = True) -> dict[str, bool]:
    """Synchronous wrapper to create helpers."""
    return asyncio.run(create_helpers_async(skip_existing))


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create HA helper entities for heating optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Create all helpers (skip existing):
    python -m scripts.heating.setup_helpers create

  Force recreate all helpers:
    python -m scripts.heating.setup_helpers create --force

  Set default values:
    python -m scripts.heating.setup_helpers defaults

  Create helpers and set defaults:
    python -m scripts.heating.setup_helpers setup
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create helper entities")
    create_parser.add_argument(
        "--force", action="store_true", help="Recreate even if exists"
    )

    subparsers.add_parser("defaults", help="Set default values for helpers")

    subparsers.add_parser("setup", help="Create helpers and set defaults")

    args = parser.parse_args()

    if args.command == "create":
        results = create_helpers(skip_existing=not getattr(args, "force", False))
        success = all(results.values())
        if success:
            logger.info("All helpers created successfully")
        else:
            failed = [k for k, v in results.items() if not v]
            logger.error(f"Failed to create: {failed}")
            sys.exit(1)

    elif args.command == "defaults":
        results = set_default_values()
        success = all(results.values())
        if success:
            logger.info("All defaults set successfully")
        else:
            failed = [k for k, v in results.items() if not v]
            logger.error(f"Failed to set: {failed}")
            sys.exit(1)

    elif args.command == "setup":
        logger.info("Creating helpers...")
        create_results = create_helpers(skip_existing=True)

        logger.info("Setting defaults...")
        default_results = set_default_values()

        all_results = {**create_results, **default_results}
        success = all(all_results.values())

        if success:
            logger.info("Setup completed successfully")
        else:
            failed = [k for k, v in all_results.items() if not v]
            logger.error(f"Setup had failures: {failed}")
            sys.exit(1)


if __name__ == "__main__":
    main()
