# Home & Heating Learnings

Perpetual knowledge gained from analysis, experiments, and observations. This file preserves insights that would otherwise be lost between sessions.

---

## Thermal Performance Analysis (February 2026)

### Time Constant (τ) Measurements

The time constant τ represents how many hours it takes for the home to lose 63% of its heat advantage over outside temperature.

| Measurement Period | Conditions | Calculated τ | Notes |
|-------------------|------------|--------------|-------|
| Overnight (Feb 1-2, 2026) | Normal living, bathroom kippen | 156 hours | Includes ventilation losses |
| Christmas vacation (Dec 23-28, 2025) | All windows sealed, heating off | 333 hours | True envelope performance |
| January vacation (Jan 7-23, 2026) | All windows sealed, heating off | 450-600 hours | Extended measurement |

**Conclusion**: True building envelope τ ≈ 400+ hours. Daily living (kippen ventilation) effectively halves this to ~156 hours.

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
