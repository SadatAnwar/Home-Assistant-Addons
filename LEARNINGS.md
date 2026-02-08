# Home & Heating Learnings

Perpetual knowledge gained from analysis, experiments, and observations. This file preserves insights that would otherwise be lost between sessions.

---

## Thermal Performance Analysis (February 2026)

### Time Constant (τ) Measurements

The time constant τ represents how many hours it takes for the home to lose 63% of its heat advantage over outside temperature.

**Bedroom (parents room) -- original analysis using short-term recorder data:**

| Measurement Period | Conditions | Calculated τ | Notes |
|-------------------|------------|--------------|-------|
| Overnight (Feb 1-2, 2026) | Normal living, bathroom kippen | 156 hours | Includes ventilation losses |
| Christmas vacation (Dec 23-28, 2025) | All windows sealed, heating off | 333 hours | True envelope performance |
| January vacation (Jan 7-23, 2026) | All windows sealed, heating off | 450-600 hours | Extended measurement |

**Conclusion**: True building envelope τ ≈ 400+ hours. Daily living (kippen ventilation) effectively halves this to ~156 hours.

### Room Comparison: Kids Room vs Bedroom (February 2026)

Analysis repeated using kids room sensor (`sensor.aayat_room_temp_temperature`) and compared side-by-side with bedroom (`sensor.bedroom_thermo_temperature`). Vacation periods used **Long-Term Statistics** (hourly mean values) since short-term recorder data only retains ~10 days.

**Sealed house (vacation, no ventilation):**

| Period | Kids Room τ | Bedroom τ | Kids faster by |
|--------|------------|-----------|----------------|
| Christmas (Dec 23-28, 5 days) | 504 hours | 620 hours | 19% |
| January week 1 (Jan 7-14) | 860 hours | 903 hours | 5% |
| January full (Jan 7-21, 14 days) | 906 hours | 1060 hours | 15% |

**With daily kippen ventilation (Feb 2026 overnights):**

| Period | Kids Room τ | Bedroom τ | Kids faster by |
|--------|------------|-----------|----------------|
| Night Feb 5-6 (13.5h, avg -0.7°C) | 135 hours | 131 hours | kids 3% slower |
| Night Feb 6-7 (12.0h, avg 1.7°C) | 124 hours | 115 hours | kids 8% slower |
| **Overnight average** | **130 hours** | **123 hours** | kids 6% slower |

**Key findings:**
- **Sealed house**: Kids room cools 5-19% faster than bedroom, likely due to more exterior wall exposure or window orientation
- **With ventilation**: Kids room actually retains heat slightly *better* (6%) than bedroom -- the bathroom kippen is physically closer to the bedroom, explaining why ventilation affects it more
- **Ventilation impact**: τ drops from 500-900h (sealed) to ~130h (with kippen) -- a 4-7x reduction
- **LTS vs recorder**: τ values from Long-Term Statistics (hourly means) are higher than from short-term recorder (point samples) because averaging smooths fluctuations. The LTS-based bedroom τ of 620h (Christmas) is higher than the original 333h from point samples for the same period
- **Equilibrium**: Both rooms approached ~13°C after 14 days with heating off and outside averaging -1°C, consistent with internal gains of ~300-500W

**Temperature progression during January vacation (daily snapshots):**

| Date | Kids Room | Bedroom | Outside |
|------|-----------|---------|---------|
| Jan 7 | 18.9°C | 19.0°C | -4.2°C |
| Jan 10 | 16.3°C | 16.4°C | -4.6°C |
| Jan 14 | 14.9°C | 15.2°C | 3.0°C |
| Jan 18 | 13.1°C | 13.7°C | 3.0°C |
| Jan 21 | 12.7°C | 13.6°C | -4.6°C |

### Comparison to German Building Standards

| Building Standard | Typical τ | This Home |
|------------------|-----------|-----------|
| Altbau (pre-1977) | 20-50 hours | |
| WSchV 1995 | 50-80 hours | |
| EnEV 2014 | 80-120 hours | |
| KfW Effizienzhaus 55 | 120-180 hours | |
| **KfW Effizienzhaus 40** | **180-300 hours** | **✓ Matches** |
| Passivhaus | 300-500+ hours | Near this level |

The home (built 1995) performs at KfW 40 / near-Passivhaus level despite being 30 years old.

### Equilibrium Temperature

When heating is off, the home stabilizes at **15-16°C** with outside temperatures around 0-5°C.

**Formula**: `T_equilibrium = T_outside + (Internal_Gains ÷ Heat_Loss_Coefficient)`

**Internal heat gains (no occupants):**
- Fridge/freezer: ~60-100W
- Router, standby devices: ~50-100W
- Solar gain (varies): 0-500W
- **Total**: ~300-500W

With heat loss coefficient of ~50-80 W/K, this produces 6-10°C elevation above outside temperature.

### Cooling Coefficient

Derived from measurements:
- **k = 0.0064 per hour** (used in thermal model)
- This means the home loses 0.64% of its temperature difference per hour

---

## Building Characteristics

### Construction
- **Year built**: 1995
- **Type**: Endhaus (end-of-terrace) - 3 exposed sides
- **Windows**: Double-glazed, good air-tightness
- **Neighbor**: Mittelhaus (2 exposed sides) - likely has even better τ

### Heat Loss by House Type

| House Type | Exposed Sides | Relative Heat Loss |
|------------|---------------|-------------------|
| Detached (freistehend) | 4 walls + roof | 100% |
| **Endhaus (this home)** | 3 walls + roof | ~75-80% |
| Mittelhaus (neighbor) | 2 walls + roof | ~55-65% |
| Middle apartment | 0-1 walls | ~30-40% |

---

## Humidity Management

### The Problem
Well-insulated homes trap moisture. Daily moisture production:

| Source | Moisture |
|--------|----------|
| Breathing (2 adults, overnight) | 1-2 liters |
| Showering | 1-2 liters each |
| Cooking | 0.5-1 liter |
| Drying clothes indoors | 2-5 liters per load |

### The Solution
**Kipplüften (tilted bathroom window)** works better than Stoßlüften (burst ventilation) for this home because:
1. Removes moisture continuously as it's produced
2. Prevents overnight humidity buildup while sleeping
3. Keeps RH at ~50-55% instead of 65-70%

### Condensation Formula
```
Condensation occurs when: Surface Temperature < Dew Point
```

| Indoor RH | Dew Point (at 20°C) | Condensation on 10°C window? |
|-----------|---------------------|------------------------------|
| 70% | 14.4°C | YES |
| 55% | 10.7°C | Borderline |
| 50% | 9.3°C | NO |

### Trade-off
Kippen ventilation costs ~20-30% more in heating but prevents:
- Mold growth (starts at sustained >60% RH)
- Window condensation and frame damage
- Morning window wiping

**This is the correct trade-off for health and building protection.**

---

## Heating System Insights

### Boiler Heating Curve (Viessmann Vitodens 100-W)

The heating curve maps outside temperature to flow (supply) temperature. Current settings: **Slope 1.3, Level 0.0**.

| Outside Temp | Flow Temp |
|-------------|-----------|
| +20°C | 20°C |
| +10°C | 36°C |
| 0°C | 49°C |
| -10°C | 60°C |
| -20°C | 72°C |
| -30°C | 75°C |

**How it works**: The boiler adjusts its output water temperature based on how cold it is outside. Colder outside = hotter flow water = more energy needed = higher gas consumption. The slope (1.3) determines how aggressively flow temp rises as outside temp drops.

**Gas usage implications**: Higher flow temperatures require more burner modulation and gas. At 0°C outside (49°C flow), the boiler operates at moderate modulation. At -10°C (60°C flow), it runs significantly harder.

**Note on gas metering**: The SmartNetz gas meter reads total household gas. This home has no DHW (domestic hot water) from gas -- gas is used only for central heating and cooking. Cooking usage is marginal, typically <0.5 m³/day.

### Weather Forecast Availability

The weather forecast entity `weather.forecast_home` provides **48 hours of hourly forecasts** via the `weather.get_forecasts` service. This should be used for heating schedule optimization:
- **Overnight temps**: Used for cooling prediction during sleep hours
- **Morning temp (at target_warm_time)**: Used for heating duration calculation
- **Daytime temps**: Used for setpoint calculation and solar gain prediction

### Outdoor Sensor vs Weather Forecast Comparison

**Purpose**: Determine if there's a consistent bias between the physical sensor and weather forecast to improve heating predictions.

**Data Sources**:
- **Sensor**: `sensor.e3_vitodens_100_0421_1_outside_temperature` (Viessmann boiler sensor)
- **Forecast**: `weather.forecast_home` hourly forecast

**Observations (February 2026)**:

| Time | Sensor Reading | Forecast | Delta (Sensor - Forecast) |
|------|---------------|----------|---------------------------|
| 2026-02-02 20:45 UTC | -6.8°C | -7.8°C | **+1.0°C** |

**Analysis**:
The boiler's outdoor sensor reads approximately **1°C warmer** than the weather forecast. Possible reasons:
1. **Sensor placement**: The boiler sensor is likely mounted on the building wall, receiving radiant heat from the building
2. **Microclimate**: The sensor location may be sheltered from wind or exposed to warmer air from the basement/utility area
3. **Forecast location**: Weather forecasts are for the general area, not this specific property

**Recommendation for Optimizer**:
- When using forecast data for heating calculations, consider that actual outdoor temps may be **0.5-1.5°C warmer** than forecast
- This delta appears more significant in cold weather (below 0°C) - needs more data points to confirm
- For conservative heating (avoid under-heating), use forecast as-is
- For energy optimization (avoid over-heating), add +1°C to forecast temps

**Methodology for Ongoing Observation**:
Since HA doesn't store historical forecasts, compare sensor vs forecast at decision points:
1. At scheduler run time: Compare overnight forecast accuracy
2. At target_warm_time: Compare morning forecast accuracy
3. Log deltas in scheduler output for pattern analysis

### Thermal Model Parameters
Based on actual measurements, the heating optimizer uses:
- **Cooling coefficient k**: 0.0064 per hour (physics-based, not ML)
- **Minimum heating rate**: 1.0°C/hour (floor for safety)
- **Setpoint range**: 19-22°C (base 20°C, scaled by outside temp)

### Actual vs Model Performance
| Parameter | ML Model Learned | Actual Measured |
|-----------|------------------|-----------------|
| Cooling rate | 0.046°C/hour | 0.22-0.27°C/hour |
| τ (time constant) | ~2000 hours | 156-333 hours |

The ML model was over-optimistic. Physics-based model with k=0.0064 is more reliable.

---

## Future Reference

### Useful Calculations

**Time constant from temperature drop:**
```
τ = -t / ln((T_final - T_outside) / (T_initial - T_outside))
```

**Heat loss coefficient from τ:**
```
Heat_Loss_Coefficient (W/K) = Thermal_Mass (J/K) / τ (seconds)
```

**Equilibrium temperature:**
```
T_eq = T_outside + (Internal_Gains_W / Heat_Loss_Coefficient_W_per_K)
```

### Retrieving Long-Term Statistics from Home Assistant

The standard `/api/history/period/` REST endpoint only returns short-term recorder data (typically last ~10 days). For older data, use the **Long-Term Statistics** (LTS) via the WebSocket API. LTS stores hourly aggregated values (mean, min, max) indefinitely.

**WebSocket command to fetch LTS data:**
```python
{
    "id": 1,
    "type": "recorder/statistics_during_period",
    "start_time": "2025-12-23T00:00:00",  # ISO format
    "end_time": "2025-12-28T00:00:00",
    "statistic_ids": [
        "sensor.aayat_room_temp_temperature",
        "sensor.bedroom_thermo_temperature",
        "sensor.e3_vitodens_100_0421_1_outside_temperature"
    ],
    "period": "hour",  # Options: "5minute", "hour", "day", "week", "month"
    "types": ["mean", "min", "max", "state"]
}
```

**Response format:** Result is a dict keyed by `statistic_id`, each containing a list of entries:
```python
{
    "start": 1735002000000,  # Epoch milliseconds (divide by 1000 for datetime.fromtimestamp)
    "end": 1735005600000,
    "mean": 19.4,
    "min": 19.2,
    "max": 19.6,
    "state": 19.4
}
```

**To list available statistics:**
```python
{"id": 1, "type": "recorder/list_statistic_ids", "statistic_type": "mean"}
```

**Important notes:**
- Timestamps in results are **epoch milliseconds** (not seconds) -- divide by 1000 for Python's `datetime.fromtimestamp()`
- LTS values are hourly **means**, not point samples -- this produces smoother curves and higher τ estimates compared to raw recorder data
- Requires WebSocket connection with authentication (same as helper creation)
- The `period` parameter controls granularity: use `"hour"` for multi-day analysis, `"5minute"` for detailed short periods

---

## Device Entity Naming

### SwitchBot Lock

The main door lock uses these entities (not the `lock_ultra_4c` naming):

| Purpose | Entity ID |
|---------|-----------|
| Lock control | `lock.door_lock` |
| Battery | `sensor.door_lock_battery` |
| Door state | `binary_sensor.door_lock` |
| Unclosed alarm | `binary_sensor.door_lock_unclosed_alarm` |
| Unlocked alarm | `binary_sensor.door_lock_unlocked_alarm` |

**Note**: The `lock_ultra_4c` entities are stale/incorrect references.

---

## Dashboard Card Patterns

### Mushroom Cards for Lights

Mushroom cards provide better flexibility than tile cards for light controls:

| Card Type | Use Case |
|-----------|----------|
| `mushroom-template-card` | Switches with power monitoring - allows templated secondary text (e.g., "{{ states('sensor.power') }} W") |
| `mushroom-light-card` | Actual `light.*` entities - has built-in brightness slider |
| `mushroom-entity-card` | Simple switches without extra info |

**Common settings for consistency:**
```yaml
layout: horizontal
fill_container: true
tap_action:
  action: toggle
icon_tap_action:
  action: more-info
```

**Note:** Tile cards can't show values from other entities in `state_content` - only attributes of the main entity. Use mushroom-template-card when you need to display related sensor values.
