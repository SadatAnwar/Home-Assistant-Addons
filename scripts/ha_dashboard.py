#!/usr/bin/env python3
"""Upload and manage Lovelace dashboards in Home Assistant via WebSocket API."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

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


def ensure_valid_url_path(url_path: str) -> str:
    """Ensure URL path contains a hyphen (HA requirement)."""
    if "-" not in url_path:
        return f"{url_path}-dashboard"
    return url_path


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

    async def list_dashboards(self) -> list[dict]:
        """List all custom dashboards."""
        response = await self.send_command("lovelace/dashboards/list")
        if not response.get("success"):
            error = response.get("error", {})
            raise RuntimeError(f"Failed to list dashboards: {error.get('message', response)}")
        return response.get("result", [])

    async def create_dashboard(self, url_path: str, title: str, **kwargs) -> dict:
        """Create a new dashboard."""
        response = await self.send_command(
            "lovelace/dashboards/create",
            url_path=url_path,
            title=title,
            icon=kwargs.get("icon", "mdi:view-dashboard"),
            show_in_sidebar=kwargs.get("show_in_sidebar", True),
            require_admin=kwargs.get("require_admin", False),
        )
        if not response.get("success"):
            error = response.get("error", {})
            raise RuntimeError(f"Failed to create dashboard: {error.get('message', response)}")
        return response.get("result", {})

    async def delete_dashboard(self, dashboard_id: str) -> None:
        """Delete a dashboard."""
        response = await self.send_command("lovelace/dashboards/delete", dashboard_id=dashboard_id)
        if not response.get("success"):
            error = response.get("error", {})
            raise RuntimeError(f"Failed to delete dashboard: {error.get('message', response)}")

    async def get_config(self, url_path: str | None = None) -> dict | None:
        """Get dashboard configuration."""
        kwargs = {}
        if url_path:
            kwargs["url_path"] = url_path
        response = await self.send_command("lovelace/config", **kwargs)
        if not response.get("success"):
            return None
        return response.get("result")

    async def save_config(self, config: dict, url_path: str | None = None) -> None:
        """Save dashboard configuration."""
        kwargs = {"config": config}
        if url_path:
            kwargs["url_path"] = url_path
        response = await self.send_command("lovelace/config/save", **kwargs)
        if not response.get("success"):
            error = response.get("error", {})
            raise RuntimeError(f"Failed to save config: {error.get('message', response)}")

    async def delete_config(self, url_path: str | None = None) -> None:
        """Delete dashboard configuration."""
        kwargs = {}
        if url_path:
            kwargs["url_path"] = url_path
        response = await self.send_command("lovelace/config/delete", **kwargs)
        if not response.get("success"):
            error = response.get("error", {})
            raise RuntimeError(f"Failed to delete config: {error.get('message', response)}")


async def connect(ws_url: str, token: str):
    """Connect to HA WebSocket and authenticate."""
    ws = await websockets.connect(ws_url)

    # Wait for auth_required message
    msg = json.loads(await ws.recv())
    if msg.get("type") != "auth_required":
        raise RuntimeError(f"Expected auth_required, got: {msg}")

    # Send auth
    await ws.send(json.dumps({"type": "auth", "access_token": token}))

    # Wait for auth result
    msg = json.loads(await ws.recv())
    if msg.get("type") != "auth_ok":
        raise RuntimeError(f"Authentication failed: {msg}")

    return HAWebSocket(ws)


def load_yaml_file(file_path: Path) -> dict:
    """Load and parse a YAML file."""
    with open(file_path) as f:
        return yaml.safe_load(f)


async def upload_dashboard_async(file_path: str, url_path: str | None = None) -> None:
    """Upload a dashboard YAML file to Home Assistant."""
    ha_url, token = load_config()
    ws_url = get_ws_url(ha_url)
    path = Path(file_path)

    if not path.exists():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    config = load_yaml_file(path)
    title = config.get("title", path.stem.replace("_", " ").title())

    # Use filename as url_path if not provided
    if url_path is None:
        url_path = path.stem

    # HA requires url_path to contain a hyphen
    url_path = ensure_valid_url_path(url_path)

    print(f"Dashboard: {title}")
    print(f"URL path: {url_path}")
    print(f"Source: {file_path}")

    async with websockets.connect(ws_url) as ws:
        # Authenticate
        msg = json.loads(await ws.recv())
        if msg.get("type") != "auth_required":
            raise RuntimeError(f"Expected auth_required, got: {msg}")

        await ws.send(json.dumps({"type": "auth", "access_token": token}))
        msg = json.loads(await ws.recv())
        if msg.get("type") != "auth_ok":
            raise RuntimeError(f"Authentication failed: {msg}")

        client = HAWebSocket(ws)

        # Check if dashboard exists
        dashboards = await client.list_dashboards()
        existing = next((d for d in dashboards if d.get("url_path") == url_path), None)

        if existing:
            print(f"Updating existing dashboard '{url_path}'...")
        else:
            print(f"Creating new dashboard '{url_path}'...")
            await client.create_dashboard(url_path, title)

        # Save the configuration
        await client.save_config(config, url_path)
        print("Done!")


async def list_dashboards_async() -> None:
    """List all dashboards in Home Assistant."""
    ha_url, token = load_config()
    ws_url = get_ws_url(ha_url)

    async with websockets.connect(ws_url) as ws:
        # Authenticate
        msg = json.loads(await ws.recv())
        await ws.send(json.dumps({"type": "auth", "access_token": token}))
        msg = json.loads(await ws.recv())
        if msg.get("type") != "auth_ok":
            raise RuntimeError(f"Authentication failed: {msg}")

        client = HAWebSocket(ws)
        dashboards = await client.list_dashboards()

        print("Lovelace Dashboards:")
        print("-" * 60)
        print(f"  {'(default)':20} - Default dashboard (built-in)")

        if dashboards:
            for dash in dashboards:
                url_path = dash.get("url_path", "unknown")
                title = dash.get("title", url_path)
                mode = dash.get("mode", "storage")
                print(f"  {url_path:20} - {title} (mode: {mode})")
        else:
            print("  (no custom dashboards)")


async def fetch_dashboard_async(url_path: str, output_file: str | None = None) -> None:
    """Fetch a dashboard configuration from Home Assistant."""
    ha_url, token = load_config()
    ws_url = get_ws_url(ha_url)

    async with websockets.connect(ws_url) as ws:
        # Authenticate
        msg = json.loads(await ws.recv())
        await ws.send(json.dumps({"type": "auth", "access_token": token}))
        msg = json.loads(await ws.recv())
        if msg.get("type") != "auth_ok":
            raise RuntimeError(f"Authentication failed: {msg}")

        client = HAWebSocket(ws)

        # Verify dashboard exists
        dashboards = await client.list_dashboards()
        existing = next((d for d in dashboards if d.get("url_path") == url_path), None)

        if not existing:
            print(f"Dashboard '{url_path}' not found")
            sys.exit(1)

        # Fetch the configuration
        config = await client.get_config(url_path)
        if config is None:
            print(f"Could not fetch configuration for '{url_path}'")
            sys.exit(1)

        yaml_output = yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True)

        if output_file:
            with open(output_file, "w") as f:
                f.write(yaml_output)
            print(f"Saved dashboard '{url_path}' to {output_file}")
        else:
            print(yaml_output)


async def delete_dashboard_async(url_path: str) -> None:
    """Delete a dashboard."""
    ha_url, token = load_config()
    ws_url = get_ws_url(ha_url)

    async with websockets.connect(ws_url) as ws:
        # Authenticate
        msg = json.loads(await ws.recv())
        await ws.send(json.dumps({"type": "auth", "access_token": token}))
        msg = json.loads(await ws.recv())
        if msg.get("type") != "auth_ok":
            raise RuntimeError(f"Authentication failed: {msg}")

        client = HAWebSocket(ws)

        # Find the dashboard
        dashboards = await client.list_dashboards()
        existing = next((d for d in dashboards if d.get("url_path") == url_path), None)

        if not existing:
            print(f"Dashboard '{url_path}' not found")
            sys.exit(1)

        # Delete the dashboard
        dashboard_id = existing.get("id")
        await client.delete_dashboard(dashboard_id)
        print(f"Deleted dashboard '{url_path}'")


def upload_dashboard(file_path: str, url_path: str | None = None, title: str | None = None) -> None:
    """Sync wrapper for upload."""
    asyncio.run(upload_dashboard_async(file_path, url_path))


def list_all_dashboards() -> None:
    """Sync wrapper for list."""
    asyncio.run(list_dashboards_async())


def delete_dashboard(url_path: str) -> None:
    """Sync wrapper for delete."""
    asyncio.run(delete_dashboard_async(url_path))


def fetch_dashboard(url_path: str, output_file: str | None = None) -> None:
    """Sync wrapper for fetch."""
    asyncio.run(fetch_dashboard_async(url_path, output_file))


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Manage Home Assistant Lovelace dashboards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Upload a dashboard:
    python ha_dashboard.py upload dashboards/lights.yaml

  Upload with custom URL path:
    python ha_dashboard.py upload dashboards/lights.yaml --url-path my-lights

  List all dashboards:
    python ha_dashboard.py list

  Delete a dashboard:
    python ha_dashboard.py delete lights

  Fetch a dashboard:
    python ha_dashboard.py fetch heating-overview

  Fetch and save to file:
    python ha_dashboard.py fetch heating-overview -o dashboards/heating.yaml
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload a dashboard YAML file")
    upload_parser.add_argument("file", help="Path to the dashboard YAML file")
    upload_parser.add_argument("--url-path", help="Custom URL path (default: filename)")
    upload_parser.add_argument("--title", help="Dashboard title (default: from YAML or filename)")

    # List command
    subparsers.add_parser("list", help="List all dashboards")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a dashboard")
    delete_parser.add_argument("url_path", help="URL path of the dashboard to delete")

    # Fetch command
    fetch_parser = subparsers.add_parser("fetch", help="Fetch a dashboard configuration")
    fetch_parser.add_argument("url_path", help="URL path of the dashboard to fetch")
    fetch_parser.add_argument("-o", "--output", help="Output file path (prints to stdout if not specified)")

    args = parser.parse_args()

    if args.command == "upload":
        upload_dashboard(args.file, args.url_path, args.title)
    elif args.command == "list":
        list_all_dashboards()
    elif args.command == "delete":
        delete_dashboard(args.url_path)
    elif args.command == "fetch":
        fetch_dashboard(args.url_path, args.output)


if __name__ == "__main__":
    main()
