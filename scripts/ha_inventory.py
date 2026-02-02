#!/usr/bin/env python3
"""Query Home Assistant for all entities, automations, services, and integrations."""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import requests
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


def get_headers(token: str) -> dict:
    """Get HTTP headers for HA API requests."""
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def fetch_states(ha_url: str, headers: dict) -> list[dict]:
    """Fetch all entity states."""
    response = requests.get(f"{ha_url}/api/states", headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_services(ha_url: str, headers: dict) -> list[dict]:
    """Fetch all available services."""
    response = requests.get(f"{ha_url}/api/services", headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_config(ha_url: str, headers: dict) -> dict:
    """Fetch HA configuration."""
    response = requests.get(f"{ha_url}/api/config", headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_events(ha_url: str, headers: dict) -> list[dict]:
    """Fetch available event types."""
    response = requests.get(f"{ha_url}/api/events", headers=headers, timeout=30)
    response.raise_for_status()
    return response.json()


def organize_entities(states: list[dict]) -> dict[str, list[dict]]:
    """Organize entities by domain (sensor, switch, light, etc.)."""
    by_domain = defaultdict(list)

    for state in states:
        entity_id = state.get("entity_id", "")
        domain = entity_id.split(".")[0] if "." in entity_id else "unknown"

        entity_info = {
            "entity_id": entity_id,
            "state": state.get("state"),
            "friendly_name": state.get("attributes", {}).get("friendly_name", ""),
            "device_class": state.get("attributes", {}).get("device_class"),
            "unit": state.get("attributes", {}).get("unit_of_measurement"),
        }

        # Add domain-specific attributes
        attrs = state.get("attributes", {})
        if domain == "sensor":
            entity_info["state_class"] = attrs.get("state_class")
        elif domain == "light":
            entity_info["supported_color_modes"] = attrs.get("supported_color_modes")
            entity_info["brightness"] = attrs.get("brightness")
        elif domain == "climate":
            entity_info["hvac_modes"] = attrs.get("hvac_modes")
            entity_info["current_temperature"] = attrs.get("current_temperature")
            entity_info["target_temperature"] = attrs.get("temperature")
        elif domain == "automation":
            entity_info["last_triggered"] = attrs.get("last_triggered")
            entity_info["mode"] = attrs.get("mode")
        elif domain == "input_boolean":
            entity_info["editable"] = attrs.get("editable")
        elif domain == "input_datetime":
            entity_info["has_date"] = attrs.get("has_date")
            entity_info["has_time"] = attrs.get("has_time")

        by_domain[domain].append(entity_info)

    # Sort entities within each domain
    for domain in by_domain:
        by_domain[domain].sort(key=lambda x: x["entity_id"])

    return dict(sorted(by_domain.items()))


def organize_services(services: list[dict]) -> dict[str, list[str]]:
    """Organize services by domain."""
    result = {}
    for service in services:
        domain = service.get("domain", "unknown")
        service_names = list(service.get("services", {}).keys())
        result[domain] = sorted(service_names)
    return dict(sorted(result.items()))


def generate_inventory(output_dir: Path) -> None:
    """Generate complete inventory of Home Assistant setup."""
    ha_url, token = load_config()
    headers = get_headers(token)

    print("Fetching Home Assistant inventory...")

    # Fetch all data
    print("  - Fetching entity states...")
    states = fetch_states(ha_url, headers)

    print("  - Fetching services...")
    services = fetch_services(ha_url, headers)

    print("  - Fetching config...")
    config = fetch_config(ha_url, headers)

    print("  - Fetching events...")
    events = fetch_events(ha_url, headers)

    # Organize data
    entities_by_domain = organize_entities(states)
    services_by_domain = organize_services(services)

    # Count entities
    entity_counts = {domain: len(entities) for domain, entities in entities_by_domain.items()}
    total_entities = sum(entity_counts.values())

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate summary
    summary = {
        "generated_at": datetime.now().isoformat(),
        "ha_version": config.get("version"),
        "location_name": config.get("location_name"),
        "total_entities": total_entities,
        "entity_counts": entity_counts,
        "components": sorted(config.get("components", [])),
    }

    # Write summary
    summary_path = output_dir / "summary.yaml"
    with open(summary_path, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"  - Wrote {summary_path}")

    # Write entities by domain
    entities_path = output_dir / "entities.yaml"
    with open(entities_path, "w") as f:
        yaml.dump(entities_by_domain, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"  - Wrote {entities_path}")

    # Write services
    services_path = output_dir / "services.yaml"
    with open(services_path, "w") as f:
        yaml.dump(services_by_domain, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"  - Wrote {services_path}")

    # Write events
    events_path = output_dir / "events.yaml"
    event_list = sorted([e.get("event") for e in events if e.get("event")])
    with open(events_path, "w") as f:
        yaml.dump(event_list, f, default_flow_style=False, allow_unicode=True)
    print(f"  - Wrote {events_path}")

    # Generate a more readable entity reference organized by type
    entity_reference = {}

    # Sensors by device class
    if "sensor" in entities_by_domain:
        sensors_by_class = defaultdict(list)
        for sensor in entities_by_domain["sensor"]:
            device_class = sensor.get("device_class") or "other"
            sensors_by_class[device_class].append({
                "entity_id": sensor["entity_id"],
                "name": sensor["friendly_name"],
                "unit": sensor.get("unit"),
            })
        entity_reference["sensors"] = dict(sorted(sensors_by_class.items()))

    # Lights
    if "light" in entities_by_domain:
        entity_reference["lights"] = [
            {"entity_id": e["entity_id"], "name": e["friendly_name"]}
            for e in entities_by_domain["light"]
        ]

    # Switches
    if "switch" in entities_by_domain:
        entity_reference["switches"] = [
            {"entity_id": e["entity_id"], "name": e["friendly_name"]}
            for e in entities_by_domain["switch"]
        ]

    # Climate
    if "climate" in entities_by_domain:
        entity_reference["climate"] = [
            {
                "entity_id": e["entity_id"],
                "name": e["friendly_name"],
                "hvac_modes": e.get("hvac_modes"),
            }
            for e in entities_by_domain["climate"]
        ]

    # Automations
    if "automation" in entities_by_domain:
        entity_reference["automations"] = [
            {
                "entity_id": e["entity_id"],
                "name": e["friendly_name"],
                "state": e["state"],
                "last_triggered": e.get("last_triggered"),
            }
            for e in entities_by_domain["automation"]
        ]

    # Input helpers
    helpers = {}
    for domain in ["input_boolean", "input_datetime", "input_number", "input_select", "input_text"]:
        if domain in entities_by_domain:
            helpers[domain] = [
                {"entity_id": e["entity_id"], "name": e["friendly_name"], "state": e["state"]}
                for e in entities_by_domain[domain]
            ]
    if helpers:
        entity_reference["input_helpers"] = helpers

    # Binary sensors
    if "binary_sensor" in entities_by_domain:
        binary_by_class = defaultdict(list)
        for sensor in entities_by_domain["binary_sensor"]:
            device_class = sensor.get("device_class") or "other"
            binary_by_class[device_class].append({
                "entity_id": sensor["entity_id"],
                "name": sensor["friendly_name"],
                "state": sensor["state"],
            })
        entity_reference["binary_sensors"] = dict(sorted(binary_by_class.items()))

    # Write entity reference
    reference_path = output_dir / "entity_reference.yaml"
    with open(reference_path, "w") as f:
        yaml.dump(entity_reference, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"  - Wrote {reference_path}")

    # Print summary
    print("\n" + "=" * 60)
    print(f"Home Assistant v{config.get('version')}")
    print(f"Total entities: {total_entities}")
    print("=" * 60)
    print("\nEntity counts by domain:")
    for domain, count in sorted(entity_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {domain:25} {count:5}")
    if len(entity_counts) > 20:
        print(f"  ... and {len(entity_counts) - 20} more domains")

    print(f"\nInventory saved to: {output_dir}/")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate Home Assistant entity inventory")
    parser.add_argument(
        "-o", "--output",
        default="inventory",
        help="Output directory (default: inventory)",
    )

    args = parser.parse_args()
    output_dir = Path(__file__).parent.parent / args.output

    generate_inventory(output_dir)


if __name__ == "__main__":
    main()
