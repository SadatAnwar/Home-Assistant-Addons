# Home Assistant Workspace

A flexible workspace for Home Assistant related code including dashboards, automations, Python scripts, and ML models for interfacing with Home Assistant.

## Project Structure

```
home_assistant/
├── dashboards/           # Lovelace YAML configurations
│   ├── lights.yaml       # Lights control dashboard
│   ├── heating-beautiful.yaml  # Heating/climate control dashboard
│   ├── heating-optimizer.yaml  # ML heating optimizer monitoring dashboard
│   ├── temperature-glass.yaml  # Temperature & humidity dashboard (glass style)
│   └── security.yaml          # Lock controls & smoke alarm status (glass style)
├── automations/          # HA automation YAML files
│   ├── central_heating_delayed_stop.yaml  # Manual delayed stop automation
│   └── heating/          # ML-driven heating automations
│       ├── morning_switch_on.yaml    # Auto switch-on at ML time
│       ├── night_switch_off.yaml     # Auto switch-off at ML time
│       ├── bedroom_protection.yaml   # Bedroom Temperature Guard
│       └── setpoint_sync.yaml        # Sync optimal setpoint to boiler (1-min debounce)
├── scripts/              # Python scripts for HA interaction
│   ├── ha_dashboard.py   # Dashboard upload/management tool
│   ├── ha_automation.py  # Automation and helper creation tool
│   ├── ha_inventory.py   # HA entity/service inventory generator
│   └── heating/          # Adaptive heating optimization system
│       ├── __init__.py
│       ├── config.py             # Entity IDs, thresholds, settings
│       ├── ha_client.py          # HA REST/WebSocket API client
│       ├── data_collector.py     # Historical data collection
│       ├── thermal_model.py      # ML model for thermal learning
│       ├── optimizer.py          # Schedule optimization logic
│       ├── scheduler.py          # Main entry point (run daily)
│       ├── setup_helpers.py      # Create HA helper entities
│       └── prediction_tracker.py # Prediction tracking & feedback
├── data/                 # Data storage
│   └── heating/          # Heating prediction history
│       └── predictions.jsonl  # Daily predictions & actuals
├── inventory/            # Generated HA entity inventory (auto-generated)
│   ├── summary.yaml      # HA version, entity counts, components
│   ├── entities.yaml     # All entities by domain
│   ├── entity_reference.yaml  # Organized reference for development
│   ├── services.yaml     # Available services by domain
│   └── events.yaml       # Available event types
├── models/               # ML models and inference code
│   └── heating/          # Heating optimization models
│       └── thermal_model.pkl  # Trained thermal model
├── LEARNINGS.md          # Perpetual knowledge from analysis & experiments
├── .venv/                # Python virtual environment (gitignored)
├── groups.yaml           # Light groups (include in HA config)
├── requirements.txt      # Python dependencies
└── .env                  # Secrets (never commit)
```

## Home Assistant Connection

- **HA_URL**: `http://homeassistant.local:8123` (configure in .env)
- **HA_TOKEN**: Long-lived access token (configure in .env)
- Interface via REST API for commands, WebSocket API for subscriptions
- **Long-Term Statistics**: For historical data older than ~10 days, use the WebSocket `recorder/statistics_during_period` command (see LEARNINGS.md for details). The REST `/api/history/period/` endpoint only returns short-term recorder data.

## Home Assistant Configuration Reference

The target HA instance uses these key configurations in `configuration.yaml`:
- `default_config` - Standard HA integrations including Lovelace (storage mode)
- `frontend.themes` - Custom themes from themes folder
- `yahoofinance` - Stock/commodity tracking
- `ffmpeg` / `stream` - Media streaming support
- `mqtt` - MQTT integration (config in mqtt.yaml)
- Standard includes: automations.yaml, scripts.yaml, scenes.yaml

## Key Integrations (from inventory)

| Integration | Purpose | Entity Domains |
|-------------|---------|----------------|
| `vicare` | Viessmann W100 boiler control | climate, sensor, binary_sensor |
| `fritz` | FRITZ!Box router & devices | sensor, switch, device_tracker |
| `eufy_security` | Eufy cameras & doorbell | camera, binary_sensor, switch |
| `dreame_vacuum` | L10s Ultra Gen 2 robot vacuum | vacuum, sensor, switch, camera |
| `home_connect` | Siemens dishwasher & oven | sensor, switch, select |
| `shelly` | Shelly relays & power monitor | switch, sensor, light |
| `meross_lan` | Meross smart plugs & covers | switch, sensor, cover |
| `matter` | Matter-enabled devices | light, climate, lock, sensor |
| `mqtt` / `zigbee2mqtt` | Zigbee sensors via MQTT | sensor, binary_sensor, light |
| `switchbot` | SwitchBot locks & sensors | lock, sensor, binary_sensor |
| `xsense` | X-Sense smoke alarms | binary_sensor, sensor |
| `islamic_prayer_times` | Prayer time sensors | sensor |
| `midea_dehumidifier_lan` | Midea dehumidifier | humidifier, sensor, climate |

## Key Entity Quick Reference

**Lights (actual controllable):**
- `light.bathroomlight` - Bathroom main light
- `light.kitchenspot` / `light.kitchencounter` - Kitchen lights
- `light.h6072` / `light.h6072_2` - Govee lamps (left/right)
- `switch.bedroom_lamp` - Bedroom lamp (via switch)
- `switch.relay_switch_2pm_ce2a_channel_1` - Dining table light
- `switch.relay_switch_2pm_ce2a_channel_2` - Drawing room light

**Climate:**
- `climate.e3_vitodens_100_0421_1_heating` - Viessmann boiler (off/auto modes)

**Covers:**
- `cover.smart_roller_shutter_*` - Window blinds & garden shade

**Vacuum:**
- `vacuum.l10s_ultra_gen_2` - Dreame robot vacuum

**Locks:**
- `lock.door_lock` - Main door lock
- `lock.lock_ultra_4c` - Lock Ultra 4C

**Power Monitoring:**
- `sensor.shelly_power_monitor_phase_*` - 3-phase power monitor
- `sensor.daily_electricity_consumed` - Daily energy usage
- `sensor.total_power_consumed` - Current total power

## Python Conventions

- **Version**: Python 3.11+
- **Style**: PEP 8 compliant with type hints on all functions
- **Docstrings**: Minimal one-line descriptions
- **Formatting**: Use `ruff` for linting and formatting

Example:
```python
def get_entity_state(entity_id: str) -> dict:
    """Fetch current state of a Home Assistant entity."""
    ...
```

## YAML Conventions

- Use 2-space indentation
- Prefer explicit keys over shortcuts
- Group related automations/scripts logically

## Lovelace Card Tips

- **Use `type: tile` for mixed entity types** - The `type: light` card only works with `light.*` entities. For `switch.*` entities that control lights, use `type: tile` which works with both.
- Add custom icons for switch entities (e.g., `icon: mdi:ceiling-light`) since they don't have default light icons.

## Lovelace Dashboard Layout

**View Types and When to Use Them:**

- **`type: sections`** - Uses a masonry/flow layout. Cards are distributed across columns automatically. Good for simple dashboards but causes alignment issues when you need precise card placement.
- **`type: panel`** - Single card fills the view. Use with `stack-in-card` for full layout control. **Best for complex dashboards requiring precise alignment.**
- **`type: masonry`** - Similar to sections, auto-arranges cards.

**For Precise Card Alignment (Recommended Pattern):**

Use `type: panel` with nested stacks:
```yaml
views:
- title: My Dashboard
  type: panel
  cards:
  - type: custom:stack-in-card
    mode: vertical
    cards:
    # Section header
    - type: custom:bubble-card
      card_type: separator
      name: Section Name
      icon: mdi:icon-name
    # Row of cards (evenly distributed)
    - type: horizontal-stack
      cards:
      - type: some-card
      - type: some-card
      - type: some-card
    # Next row...
    - type: horizontal-stack
      cards:
      - type: another-card
```

This pattern gives explicit control over rows and card distribution.

## HACS Custom Cards

The following HACS (Home Assistant Community Store) cards are used in dashboards:

| Card | Purpose | Install From |
|------|---------|--------------|
| `mushroom` | Modern template cards, chips, weather cards | HACS > Frontend |
| `mini-graph-card` | Compact historical graphs with multiple entities | HACS > Frontend |
| `stack-in-card` | Combine multiple cards seamlessly (removes borders) | HACS > Frontend |
| `card-mod` | Custom CSS styling for any card | HACS > Frontend |
| `bubble-card` | Glassmorphism cards, separators | HACS > Frontend |

**Glass/Frosted Effect CSS (use with card-mod):**
```yaml
card_mod:
  style: |
    ha-card {
      background: rgba(255, 255, 255, 0.1) !important;
      backdrop-filter: blur(10px);
      -webkit-backdrop-filter: blur(10px);
      border: 1px solid rgba(255, 255, 255, 0.2);
      border-radius: 16px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    }
```

## Energy Monitoring (Shelly 3EM)

The home has a **Shelly 3EM** energy meter measuring whole-home consumption across 3 phases.

**Per-Phase Sensors:**

| Measurement | Phase A | Phase B | Phase C |
|-------------|---------|---------|---------|
| Power (W) | `sensor.shelly_power_monitor_phase_a_power` | `sensor.shelly_power_monitor_phase_b_power` | `sensor.shelly_power_monitor_phase_c_power` |
| Current (A) | `sensor.shelly_power_monitor_phase_a_current` | `sensor.shelly_power_monitor_phase_b_current` | `sensor.shelly_power_monitor_phase_c_current` |
| Voltage (V) | `sensor.shelly_power_monitor_phase_a_voltage` | `sensor.shelly_power_monitor_phase_b_voltage` | `sensor.shelly_power_monitor_phase_c_voltage` |
| Energy (kWh) | `sensor.shelly_power_monitor_phase_a_energy` | `sensor.shelly_power_monitor_phase_b_energy` | `sensor.shelly_power_monitor_phase_c_energy` |
| Energy Returned (kWh) | `sensor.shelly_power_monitor_phase_a_energy_returned` | `sensor.shelly_power_monitor_phase_b_energy_returned` | `sensor.shelly_power_monitor_phase_c_energy_returned` |
| Power Factor | `sensor.shelly_power_monitor_phase_a_power_factor` | `sensor.shelly_power_monitor_phase_b_power_factor` | `sensor.shelly_power_monitor_phase_c_power_factor` |

**Aggregate/Utility Sensors:**

| Sensor | Description |
|--------|-------------|
| `sensor.total_power_consumed` | Current total power draw (W) |
| `sensor.total_energy` | Cumulative energy (kWh) |
| `sensor.daily_electricity_consumed` | Today's energy usage (kWh) |

**Other:**
- `switch.shelly_power_monitor` - Main switch
- `binary_sensor.shelly_power_monitor_overpowering` - Overpowering alert

**Outlet-Level Monitors (secondary):**
- Freezer: `sensor.smart_plug_*_freezer_*` (power, energy, current, voltage)
- Washing Machine: `sensor.smart_plug_*_washing_machine_*` (power, energy, current, voltage)

## Gas Monitoring (SmartNetz Gas Reader)

⚠️ **Important:** There are two gas data sources. **Always use the SmartNetz gas reader** - it's more accurate and granular. Avoid the Viessmann boiler's gas sensor.

**Preferred Sensors (SmartNetz Gas Reader):**

| Sensor | Description | Unit |
|--------|-------------|------|
| `sensor.gas_meter_total_reading` | Cumulative meter reading | m³ |
| `sensor.gas_used_since_reset` | Usage since last reset | m³ |
| `sensor.today_s_gas_usage_volume` | Today's consumption | m³ |
| `sensor.today_s_gas_usage_energy` | Today's consumption (energy) | kWh |
| `sensor.yesterday_s_gas_usage_volume` | Yesterday's consumption | m³ |
| `sensor.yesterday_s_gas_usage_energy` | Yesterday's consumption (energy) | kWh |
| `sensor.day_before_yesterday_s_usage_volume` | Day before yesterday | m³ |
| `sensor.day_before_yesterday_s_usage_energy` | Day before yesterday (energy) | kWh |

**Avoid (inaccurate):**
- ~~`sensor.e3_vitodens_100_0421_1_heating_gas_consumption_today`~~ - Viessmann boiler estimate, not reliable

## Temperature Sensors Reference

| Room | Temperature Entity | Humidity Entity |
|------|-------------------|-----------------|
| Parents Bedroom | `sensor.bedroom_thermo_temperature` | `sensor.bedroom_thermo_humidity` |
| Kids Room | `sensor.aayat_room_temp_temperature` | `sensor.aayat_room_temp_humidity` |
| Bathroom | `sensor.bathroom_temp_temperature` | `sensor.bathroom_temp_humidity` |
| Living Room | `sensor.living_room_temp_temperature` | `sensor.living_room_temp_humidity` |
| Kitchen | `sensor.kitchen_temperature` | `sensor.kitchen_humidity` |
| Top Floor | `sensor.top_floor_thermo_temperature` | `sensor.top_floor_thermo_humidity` |
| Basement | `sensor.basement_temperature` | `sensor.basement_humidity` |
| Outside | `sensor.e3_vitodens_100_0421_1_outside_temperature` | - |

## Home Assistant Automation API

When creating automations via the REST API, follow these requirements:

- **Use numeric IDs**: Automations require a proper numeric ID (timestamp-based). Generate with `str(int(time.time() * 1000))`. Do NOT use the `/api/config/automation/config/new` endpoint literally - it creates a broken automation with `id: "new"`.
- **Correct endpoint**: `POST /api/config/automation/config/{automation_id}` where `{automation_id}` is your generated numeric ID.
- **Use plural keys**: The API expects `triggers`, `conditions`, `actions` (plural, not singular).
- **Trigger format**: Use `trigger: "time"` not `platform: "time"` for time-based triggers.
- **Action format**: Use `action: "service.name"` not `service: "service.name"`.

Example automation config:
```python
{
    "id": "1769990052349",  # Generated timestamp ID
    "alias": "My Automation",
    "description": "",
    "triggers": [{"trigger": "time", "at": "input_datetime.my_time"}],
    "conditions": [{"condition": "state", "entity_id": "input_boolean.my_toggle", "state": "on"}],
    "actions": [{"action": "light.turn_on", "target": {"entity_id": "light.my_light"}}],
    "mode": "single"
}
```

**Helper creation** uses WebSocket API:
- `input_boolean/create` - Create toggle helpers
- `input_datetime/create` - Create date/time helpers

## Secrets Management

- Store secrets in `.env` file (never commit)
- Use `python-dotenv` to load environment variables
- Required env vars: `HA_URL`, `HA_TOKEN`
- **CRITICAL**: Never send secrets, tokens, or credentials over the internet or include them in any API calls, logs, or outputs

## Commands

**Virtual Environment Setup (first time only):**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Activate Virtual Environment:**
```bash
# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

# Deactivate when done
deactivate
```

**Dashboard Management:**
```bash
# Upload a dashboard (creates new or updates existing)
python scripts/ha_dashboard.py upload dashboards/lights.yaml

# Upload with custom URL path
python scripts/ha_dashboard.py upload dashboards/lights.yaml --url-path my-lights

# List all dashboards in HA
python scripts/ha_dashboard.py list

# Delete a dashboard
python scripts/ha_dashboard.py delete lights-dashboard
```

> **Note:** The script uses Home Assistant's WebSocket API. Dashboard URL paths must contain a hyphen (HA requirement). Single-word filenames like `lights.yaml` become `lights-dashboard`. Re-uploading the same file updates the existing dashboard.

**Automation & Helper Management:**
```bash
# Create input_boolean and input_datetime helpers for delayed stop feature
python scripts/ha_automation.py create-helpers

# Create the delayed stop automation
python scripts/ha_automation.py create-automation

# Create both helpers and automation in one command
python scripts/ha_automation.py setup-delayed-stop

# Upload an automation from a YAML file
python scripts/ha_automation.py upload automations/my_automation.yaml
```

**Code Quality:**
```bash
ruff check .   # Lint Python code
ruff format .  # Format Python code
```

**Entity Inventory:**
```bash
# Generate/refresh full HA inventory (entities, services, events)
python scripts/ha_inventory.py

# Output to custom directory
python scripts/ha_inventory.py -o custom_inventory
```

> **Note:** Run `ha_inventory.py` periodically to refresh the inventory when new devices or integrations are added to Home Assistant. The generated files in `inventory/` provide a complete reference of available entities for dashboard and automation development.

**Heating Optimization System:**
```bash
# Initial setup - create HA helpers and set defaults
python -m scripts.heating.setup_helpers setup

# Run daily optimization (calculates switch times, updates HA)
python -m scripts.heating.scheduler run

# Force model retraining
python -m scripts.heating.scheduler run --train

# Dry run (calculate only, no writes at all)
python -m scripts.heating.scheduler run --dry-run

# Shadow mode (save predictions locally, don't touch HA or adjust model)
python -m scripts.heating.scheduler run --shadow

# Recommend only (save prediction, print result, don't update HA)
python -m scripts.heating.scheduler run --recommend-only

# Show thermal model information
python -m scripts.heating.scheduler info

# Show current heating state
python -m scripts.heating.scheduler state

# Review yesterday's predictions vs actuals
python -m scripts.heating.scheduler review

# Review a specific date
python -m scripts.heating.scheduler review 2026-02-01

# Show 7-day prediction history
python -m scripts.heating.scheduler history

# Show 14-day prediction history
python -m scripts.heating.scheduler history --days 14

# Upload heating automations to HA
python scripts/ha_automation.py upload automations/heating/morning_switch_on.yaml
python scripts/ha_automation.py upload automations/heating/night_switch_off.yaml
python scripts/ha_automation.py upload automations/heating/bedroom_protection.yaml
```

**Crontab example (can run at any time; multiple runs per day are safe):**
```bash
# Run heating optimizer every 4 hours
0 */4 * * * cd /path/to/home_assistant && .venv/bin/python -m scripts.heating.scheduler run >> /var/log/heating_optimizer.log 2>&1
```

## Adaptive Heating Optimization System

The heating optimization system learns your home's thermal characteristics and automatically calculates optimal heating schedules.

**How it works:**
1. **Data Collection**: Pulls 14 days of temperature, heating state, and weather data from HA
2. **Thermal Model**: Learns cooling/heating rates, solar gain coefficients
3. **Optimizer**: Calculates optimal switch-on/off times and setpoints
4. **Prediction Tracking**: Saves predictions, collects actuals, calculates errors
5. **Coefficient Adjustment**: Auto-adjusts model coefficients based on 7+ days of error data
6. **Scheduler**: Updates HA helpers (can run at any time; determines correct action based on current heating state)
7. **Automations**: Simple HA automations execute the ML-computed schedule

**User-Configured Helpers (set these in HA):**
| Helper | Purpose | Default |
|--------|---------|---------|
| `input_datetime.heating_target_warm_time` | When you want house warm | 08:00 |
| `input_datetime.heating_preferred_off_time` | Preferred heating off time | 23:00 |
| `input_number.heating_target_temp` | Desired room temperature | 20°C |
| `input_number.heating_min_bedroom_temp` | Overnight hard floor - never below | 18°C |
| `input_number.heating_min_daytime_temp` | Daytime comfort floor | 20°C |
| `input_boolean.heating_optimization_enabled` | Master switch | on |

**ML-Computed Helpers (updated by optimizer):**
| Helper | Purpose |
|--------|---------|
| `input_datetime.heating_switch_on_time` | Optimal switch-on time |
| `input_datetime.heating_switch_off_time` | Optimal switch-off time |
| `input_number.heating_optimal_setpoint` | Optimal boiler setpoint |

**Safety Features:**
- Bedroom protection: Emergency heat if below min temp
- Single ON/OFF cycle per day (minimizes boiler wear)
- Solar gain prediction (delays heating on sunny days)

**Viessmann Boiler Control Notes:**
- Control via HVAC mode (off/heat) + normal_temperature setpoint
- No indoor sensor - room temp achieved indirectly via setpoint
- Lower setpoint = lower modulation = less gas
- Night = system OFF (not reduced temp - Viessmann quirk)

## Production Deployment (Mac Mini)

The heating optimization system runs in production on a remote Ubuntu machine (Mac Mini). This ensures the scheduler runs reliably every 4 hours without depending on a local development machine.

**Remote Machine:**
| Property | Value |
|----------|-------|
| Host | `macmini.fritz.box` |
| User | `shadman` |
| Install Path | `~/dev/home_assistant` |
| Python | 3.12.3 |
| Cron Schedule | Every 4 hours (`0 */4 * * *`) |
| Log File | `~/dev/home_assistant/logs/heating.log` |

**What's Deployed:**
```
~/dev/home_assistant/
├── .venv/                    # Python virtual environment
├── .env                      # HA credentials (same as local)
├── requirements.txt
├── scripts/heating/          # All heating automation code
├── models/heating/           # Trained thermal model
├── data/heating/             # Prediction history
└── logs/heating.log          # Cron output logs
```

### Syncing Code Updates

After making changes locally, sync to the remote machine:
```bash
rsync -avz --exclude='.venv' --exclude='.env' --exclude='__pycache__' \
  --exclude='.git' --exclude='*.pyc' --exclude='inventory/' \
  /Users/sadat.anwar/dev/home_assistant/ \
  shadman@macmini.fritz.box:~/dev/home_assistant/
```

If dependencies changed (requirements.txt), also run:
```bash
ssh shadman@macmini.fritz.box "cd ~/dev/home_assistant && .venv/bin/pip install -r requirements.txt"
```

### Checking Logs

```bash
# View recent logs
ssh shadman@macmini.fritz.box "tail -50 ~/dev/home_assistant/logs/heating.log"

# Follow logs in real-time
ssh shadman@macmini.fritz.box "tail -f ~/dev/home_assistant/logs/heating.log"

# View full log
ssh shadman@macmini.fritz.box "cat ~/dev/home_assistant/logs/heating.log"
```

### Manual Operations

```bash
# Run scheduler manually
ssh shadman@macmini.fritz.box "cd ~/dev/home_assistant && .venv/bin/python -m scripts.heating.scheduler run"

# Dry run (no changes to HA)
ssh shadman@macmini.fritz.box "cd ~/dev/home_assistant && .venv/bin/python -m scripts.heating.scheduler run --dry-run"

# Check current state
ssh shadman@macmini.fritz.box "cd ~/dev/home_assistant && .venv/bin/python -m scripts.heating.scheduler state"

# Review predictions
ssh shadman@macmini.fritz.box "cd ~/dev/home_assistant && .venv/bin/python -m scripts.heating.scheduler history"
```

### Cron Job Management

```bash
# View current cron jobs
ssh shadman@macmini.fritz.box "crontab -l"

# Edit cron jobs
ssh shadman@macmini.fritz.box "crontab -e"

# Current cron entry:
# 0 */4 * * * cd ~/dev/home_assistant && .venv/bin/python -m scripts.heating.scheduler run >> ~/dev/home_assistant/logs/heating.log 2>&1
```

### Troubleshooting

**If scheduler fails:**
1. Check logs: `tail -50 ~/dev/home_assistant/logs/heating.log`
2. Verify .env exists and has correct credentials
3. Test HA connectivity: `ssh shadman@macmini.fritz.box "cd ~/dev/home_assistant && .venv/bin/python -c 'from scripts.heating.ha_client import HAClient; c = HAClient(); print(c.get_state(\"sensor.bedroom_thermo_temperature\"))'"`

**If cron not running:**
1. Check cron service: `ssh shadman@macmini.fritz.box "systemctl status cron"`
2. Check cron logs: `ssh shadman@macmini.fritz.box "grep CRON /var/log/syslog | tail -20"`

**Full redeployment (if needed):**
```bash
# 1. Sync code
rsync -avz --exclude='.venv' --exclude='.env' --exclude='__pycache__' \
  --exclude='.git' --exclude='*.pyc' --exclude='inventory/' \
  /Users/sadat.anwar/dev/home_assistant/ \
  shadman@macmini.fritz.box:~/dev/home_assistant/

# 2. Recreate venv (if Python issues)
ssh shadman@macmini.fritz.box "cd ~/dev/home_assistant && rm -rf .venv && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"

# 3. Copy .env
scp /Users/sadat.anwar/dev/home_assistant/.env shadman@macmini.fritz.box:~/dev/home_assistant/.env

# 4. Test
ssh shadman@macmini.fritz.box "cd ~/dev/home_assistant && .venv/bin/python -m scripts.heating.scheduler run --dry-run"
```

## AI Agent Instructions

- **Keep this file updated**: When adding new project patterns, conventions, dependencies, or significant architectural decisions, update this CLAUDE.md file to reflect those changes
- **Never transmit secrets**: Do not send any secrets, tokens, API keys, or credentials over the network or include them in any API calls, logs, or external communications
- **Document learnings automatically**: When discovering new patterns, fixing issues, or finding better approaches (e.g., layout fixes, API quirks, entity naming conventions), add them to this file immediately so future sessions benefit from the knowledge
- **Update entity references**: When new sensors or entities are discovered or corrected, update the reference tables in this file
- **Add new dashboards to structure**: When creating new dashboard files, update the Project Structure section
- **Save perpetual knowledge to LEARNINGS.md**: When performing analysis, calculations, or experiments that produce valuable insights, save these findings to `LEARNINGS.md`. This includes:
  - Measured values and calibration parameters (e.g., time constants, cooling rates, sensor biases)
  - Comparisons to standards or benchmarks
  - Explanations of why certain approaches work or don't work
  - Home-specific characteristics discovered through data analysis
  - Any knowledge that would be lost between sessions but is valuable long-term

  **Do NOT save to LEARNINGS.md**:
  - Bug fixes or implementation details (these belong in code comments or commit messages)
  - API code snippets or workarounds (these belong in CLAUDE.md under relevant sections)
  - Temporary issues or one-time problems

- **Consult LEARNINGS.md for relevant work**: Before working on heating optimization, thermal calculations, or energy-related tasks, read `LEARNINGS.md` to leverage existing knowledge about the home's thermal characteristics, sensor behaviors, and proven approaches
