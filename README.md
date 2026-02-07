# Home Assistant Addons

A smart home workspace featuring an **adaptive ML-driven heating optimizer** that learns your home's thermal characteristics, along with custom Lovelace dashboards, automations, and management tools for Home Assistant.

## Highlights

- **Adaptive Heating Optimization** -- ML system that learns cooling/heating rates, solar gain, and gas consumption patterns to compute optimal daily heating schedules
- **Prediction Feedback Loop** -- Logs predictions, collects actuals, calculates errors, and auto-adjusts model coefficients over time
- **5 Custom Dashboards** -- Heating optimizer, climate monitoring, lights, security, and occupancy
- **Safety Automations** -- Emergency bedroom temperature guard, morning switch-on, night switch-off, and setpoint sync
- **Runs Anywhere** -- Scheduler can execute at any time; determines the correct action based on current heating state

## Project Structure

```
home_assistant/
├── dashboards/              # Lovelace YAML dashboards
│   ├── heating-optimizer.yaml   # ML optimizer monitoring & control
│   ├── heating-beautiful.yaml   # Climate system overview
│   ├── lights.yaml              # Room-by-room light control
│   ├── security.yaml            # Locks & smoke alarms
│   └── occupants.yaml           # Household presence tracking
├── automations/             # HA automation YAML files
│   ├── central_heating_delayed_stop.yaml
│   └── heating/
│       ├── morning_switch_on.yaml     # Turn on heating at ML-computed time
│       ├── night_switch_off.yaml      # Turn off heating at ML-computed time
│       ├── bedroom_protection.yaml    # Emergency heat if temp drops below minimum
│       └── setpoint_sync.yaml         # Push optimal setpoint to boiler (debounced)
├── scripts/
│   ├── heating/             # Adaptive heating optimization system
│   │   ├── scheduler.py         # Main entry point & orchestrator
│   │   ├── optimizer.py         # Schedule calculation engine
│   │   ├── thermal_model.py     # ML models (Gradient Boosting)
│   │   ├── prediction_tracker.py# Prediction logging & feedback
│   │   ├── data_collector.py    # Historical data from HA
│   │   ├── ha_client.py         # HA REST/WebSocket client
│   │   ├── config.py            # Entity IDs & thresholds
│   │   └── setup_helpers.py     # Create HA helper entities
│   ├── ha_dashboard.py      # Dashboard upload/management
│   ├── ha_automation.py     # Automation & helper creation
│   └── ha_inventory.py      # Entity/service inventory generator
├── models/heating/          # Trained ML models
├── data/heating/            # Prediction history (JSONL)
├── inventory/               # Auto-generated HA entity reference
└── LEARNINGS.md             # Thermal analysis & home insights
```

## Adaptive Heating Optimizer

The flagship feature is a complete ML pipeline that replaces manual thermostat scheduling with data-driven optimization.

### How It Works

```
┌─────────────┐    ┌──────────────┐    ┌────────────┐    ┌──────────┐
│  HA Sensors  │───>│  Data        │───>│  Thermal   │───>│ Optimizer│
│  (7 rooms,   │    │  Collector   │    │  Model     │    │          │
│   weather,   │    │  (14 days)   │    │  (GBR)     │    │ Schedule │
│   boiler)    │    └──────────────┘    └────────────┘    └────┬─────┘
└─────────────┘                                                │
                                                               v
┌─────────────┐    ┌──────────────┐    ┌────────────┐    ┌──────────┐
│  HA Helpers  │<───│  Scheduler   │<───│ Prediction │<───│ Daily    │
│  (times,     │    │  (5-case     │    │ Tracker    │    │ Schedule │
│   setpoint)  │    │   logic)     │    │ (feedback) │    │          │
└──────┬──────┘    └──────────────┘    └────────────┘    └──────────┘
       │
       v
┌─────────────┐
│  HA Automations  │
│  (execute schedule)│
└─────────────┘
```

### Learned Parameters

The thermal model learns these home-specific characteristics:

| Parameter | Description | Default |
|-----------|-------------|---------|
| Cooling rate (k) | Heat loss rate constant | 0.0064 (tau = 156h) |
| Heating rate | Room temp rise when heating ON | ~1.0 C/hour |
| Solar gain coefficient | Free heating from sunlight | 0.05 C/degree elevation |
| Gas base rate | Consumption at 50% modulation | 10 kWh/hour |

All parameters auto-adjust based on prediction errors after 7+ days of data.

### ML Models (Gradient Boosting)

- **Heating rate model** -- Predicts warming speed from outside temp, setpoint, initial temp, modulation
- **Cooling rate model** -- Predicts cooling speed from outside temp, sun elevation, cloud cover
- **Modulation model** -- Predicts burner modulation needed for a target setpoint
- **Solar gain model** -- Predicts solar contribution by room and time of day

### Scheduler Cases

The scheduler can run at any time and determines the correct action:

| Case | Condition | Action |
|------|-----------|--------|
| A | First run of day | Full schedule calculation |
| B | Before heating starts | Recalculate with fresh data |
| C | Past switch-on, heating OFF | Trigger heating immediately |
| D | Heating currently ON | Recalculate switch-off & setpoint |
| E | Past switch-off time | Prepare tomorrow's schedule |

### Safety Features

- **Bedroom temperature guard** -- Emergency heating if temp drops below configured minimum
- **Single ON/OFF cycle** per day to minimize boiler wear
- **Hard temperature floors** -- Separate overnight and daytime minimums
- **Graceful fallback** to manual control if optimization is disabled

## Dashboards

### Heating Optimizer
Glass-morphism styled dashboard showing ML-computed schedule, live temperatures, user settings, gas usage, and automation status.

### Climate Overview
Boiler gauges (flow temp, burner load, pressure), 6-room temperature history, and gas consumption tracking.

### Lights
Room-by-room light control with power monitoring overlays using Mushroom cards.

### Security
Lock control with battery status, door state indicators, and smoke alarm monitoring (3 X-Sense detectors).

### Occupancy
Household presence tracking using WiFi AP location and person entities.

**Required HACS cards:** `mushroom`, `mini-graph-card`, `stack-in-card`, `card-mod`, `bubble-card`

## Setup

### Prerequisites

- Python 3.11+
- Home Assistant instance with REST API access
- Long-lived access token

### Installation

```bash
git clone https://github.com/SadatAnwar/Home-Assistant-Addons.git
cd Home-Assistant-Addons

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure credentials
cp .env.example .env  # Edit with your HA_URL and HA_TOKEN
```

### Running the Heating Optimizer

```bash
# Initial setup -- create HA helper entities
python -m scripts.heating.setup_helpers setup

# Run optimization (safe to run multiple times per day)
python -m scripts.heating.scheduler run

# Dry run (no changes to HA)
python -m scripts.heating.scheduler run --dry-run

# View model info
python -m scripts.heating.scheduler info

# Review yesterday's predictions vs actuals
python -m scripts.heating.scheduler review

# Show prediction history
python -m scripts.heating.scheduler history --days 14
```

### Uploading Dashboards & Automations

```bash
# Upload a dashboard
python scripts/ha_dashboard.py upload dashboards/heating-optimizer.yaml

# Upload an automation
python scripts/ha_automation.py upload automations/heating/morning_switch_on.yaml
```

### Production Deployment (Cron)

```bash
# Run every 4 hours
0 */4 * * * cd /path/to/home_assistant && .venv/bin/python -m scripts.heating.scheduler run >> logs/heating.log 2>&1
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Smart Home | Home Assistant |
| ML Models | scikit-learn (Gradient Boosting) |
| Data Processing | pandas, numpy |
| HA Integration | REST API + WebSocket |
| Dashboards | Lovelace YAML + HACS cards |
| Boiler | Viessmann Vitodens 100-W (ViCare) |
| Sensors | Zigbee (via Zigbee2MQTT), Matter, SwitchBot |
| Energy Monitor | Shelly 3EM (3-phase) |

## License

Personal project. Use at your own risk.
