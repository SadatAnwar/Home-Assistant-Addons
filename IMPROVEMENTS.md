# Future Improvements

## GBM Model Audit (2026-02-11)

Four GBM models are trained in `thermal_model.py`. Here is what each does and whether it actually matters:

| GBM Model | Trained In | Used In | Affects Schedule? | Status |
|---|---|---|---|---|
| `heating_rate_model` | `_train_heating_model()` | `predict_heating_duration()` | **Yes** — determines switch-on time | Active |
| `modulation_model` | `_train_modulation_model()` | `predict_modulation()` | No — gas estimation only | Active |
| `cooling_rate_model` | `_train_cooling_model()` | Nowhere | No | Dead code |
| `solar_gain_model` | `_train_solar_model()` | Nowhere | No | Dead code (field always None) |

### heating_rate_model (ACTIVE)
- **Features:** outside_temp, bedroom, setpoint, burner_modulation
- **Predicts:** Temperature change rate (°C/hour) when heating is ON
- **Called by:** `predict_heating_duration()` (thermal_model.py:442-453)
- **Impact:** The optimizer calls `predict_heating_duration()` to calculate how long before target_warm_time to switch on. If the GBM exists, its prediction is used (floored at `max(mean_heating_rate * 0.5, 1.0)`). Without it, falls back to `max(mean_heating_rate, 1.0)`.
- **Schedule effect:** Directly controls switch-on time. A slower predicted rate = earlier switch-on.

### modulation_model (ACTIVE, low impact)
- **Features:** outside_temp, setpoint, bedroom
- **Predicts:** Burner modulation percentage (0-100%)
- **Called by:** `predict_modulation()` (thermal_model.py:498-503), which is called by `_build_hourly_plan()` in the optimizer
- **Impact:** Only affects the `expected_modulation` field in the hourly plan, which feeds gas usage estimation. Does NOT affect room temperature predictions or switch-on/off times.
- **Without it:** Falls back to a linear heuristic `30 + (setpoint - outside_temp) * 2`.

### cooling_rate_model (DEAD)
- **Features:** outside_temp, bedroom, sun_elevation
- **Predicts:** Temperature change rate (°C/hour) when heating is OFF
- **Called by:** Nothing
- **Why dead:** `predict_cooling_curve()` uses Newton's law `dT/dt = -k * (T_inside - T_outside)` with the simple `k` constant. The GBM is trained, pickled, loaded, and ignored.
- **Side effect of training:** Updates `mean_cooling_rate` as `abs(mean(y))`, but `mean_cooling_rate` itself is also unused — `k` is what matters.

### solar_gain_model (DEAD)
- **Features:** N/A — no GBM is actually fitted
- **What `_train_solar_model()` does:** Computes `solar_gain_coefficient = mean(temp_change / sun_elevation)` for sunny heating-off periods. This is a simple ratio, not a GBM. The `self.solar_gain_model` field stays None.
- **Where `solar_gain_coefficient` is used:** `predict_cooling_curve()` line 393 — but only if `sun_elevation_curve` is passed, which nobody does. Also `predict_solar_gain()` uses it, but that method is never called.

### What actually controls the heating schedule

The schedule (switch-on time, switch-off time, setpoint) is primarily driven by simple coefficients, not GBMs:

| Coefficient | Controls | Auto-adjusted? |
|---|---|---|
| `k` (cooling constant, default 0.0064) | Switch-off time via `predict_cooling_curve()` | Yes, by prediction_tracker after 7+ days |
| `mean_heating_rate` (default 1.0°C/hr) | Switch-on time as fallback when GBM unavailable | Yes, by prediction_tracker |
| `solar_gain_coefficient` (default 0.05) | Unused in practice (no sun elevation passed) | Yes, by `_train_solar_model()` |
| `gas_base_rate_kwh` (default 10.0) | Gas estimation only | Yes, by prediction_tracker |

## 1. Use learned solar_gain_coefficient in cooling curve predictions

**Current state:** `predict_cooling_curve()` accepts a `sun_elevation_curve` parameter and has code to apply `solar_gain_coefficient * sun_elevation` to the cooling rate (thermal_model.py:392-393). But no caller ever passes `sun_elevation_curve`, so solar gain is ignored in all cooling/overnight predictions.

**Improvement:** Pass hourly sun elevation data into `predict_cooling_curve()` from the optimizer. This would make daytime cooling predictions more accurate — on sunny winter days, rooms cool slower (or even warm) due to solar gain through south-facing windows.

**Requires:** Sun elevation data in the weather forecast, or computing it from latitude/longitude + time.

## 2. Replace hardcoded solar heuristic in optimizer with predict_solar_gain()

**Current state:** The optimizer's `_estimate_solar_contribution()` uses hardcoded values: +0.5°C per sunny hour, +0.2°C per partly cloudy hour, capped at 3°C. This is only used to reduce the setpoint by 1°C when solar contribution > 0.5°C.

**Improvement:** Use `thermal_model.predict_solar_gain()` which applies the learned `solar_gain_coefficient` with sun elevation and cloud cover. This would give a data-driven solar estimate instead of fixed heuristics.

**Requires:** Cloud cover percentage in the forecast (currently only condition strings like "sunny"/"cloudy" are available).

## 3. Remove dead cooling_rate_model GBM

**Current state:** A GBM `cooling_rate_model` is trained every cycle, pickled, loaded, and never queried. All cooling predictions use Newton's law with the simple `k` constant. The GBM is pure dead weight — training time, disk space, no benefit.

**Improvement:** Remove `_train_cooling_model()`, remove `cooling_rate_model` from save/load, simplify the training pipeline. The `k` constant + prediction tracker auto-adjustment is working well.

## 4. Remove dead solar_gain_model GBM field

**Current state:** `self.solar_gain_model` is declared as a GBM field but `_train_solar_model()` never actually fits a GBM — it just computes `solar_gain_coefficient` as a simple mean ratio. The field is always None despite being saved/loaded.

**Improvement:** Remove the `solar_gain_model` field entirely. Keep `_train_solar_model()` since it usefully computes `solar_gain_coefficient`, but stop pretending there's a GBM behind it.
