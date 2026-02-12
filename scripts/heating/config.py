"""Configuration for the adaptive heating optimization system."""

from dataclasses import dataclass


# Room temperature sensor mapping
ROOM_SENSORS = {
    "bedroom": "sensor.bedroom_thermo_temperature",
    "kids_room": "sensor.aayat_room_temp_temperature",
    "bathroom": "sensor.bathroom_temp_temperature",
    "living_room": "sensor.living_room_temp_temperature",
    "kitchen": "sensor.kitchen_temperature",
    "top_floor": "sensor.top_floor_thermo_temperature",
    "basement": "sensor.basement_temperature",
}

# Climate entity
CLIMATE_ENTITY = "climate.e3_vitodens_100_0421_1_heating"

# Boiler sensor entities
BOILER_SENSORS = {
    "outside_temp": "sensor.e3_vitodens_100_0421_1_outside_temperature",
    "supply_temp": "sensor.e3_vitodens_100_0421_1_supply_temperature",
    "boiler_supply_temp": "sensor.e3_vitodens_100_0421_1_boiler_supply_temperature",
    "burner_modulation": "sensor.e3_vitodens_100_0421_1_burner_modulation",
    "burner_active": "binary_sensor.e3_vitodens_100_0421_1_burner",
    "burner_hours": "sensor.e3_vitodens_100_0421_1_burner_hours",
    "burner_starts": "sensor.e3_vitodens_100_0421_1_burner_starts",
    "gas_consumption_today": "sensor.e3_vitodens_100_0421_1_heating_gas_consumption_today",
}

# Weather entities
WEATHER_ENTITIES = {
    "weather": "weather.forecast_home",
    "sun": "sun.sun",
}

# Notification service
NOTIFICATION_SERVICE = "notify.mobile_app_sadats_iphone_15_olx"


@dataclass
class HeatingHelpers:
    """HA helper entities for heating optimization."""

    # User-configured settings
    target_warm_time: str = "input_datetime.heating_target_warm_time"
    preferred_off_time: str = "input_datetime.heating_preferred_off_time"
    target_temp: str = "input_number.heating_target_temp"
    min_bedroom_temp: str = "input_number.heating_min_bedroom_temp"
    min_daytime_temp: str = "input_number.heating_min_daytime_temp"
    optimization_enabled: str = "input_boolean.heating_optimization_enabled"

    # ML-computed values
    switch_on_time: str = "input_datetime.heating_switch_on_time"
    switch_off_time: str = "input_datetime.heating_switch_off_time"
    optimal_setpoint: str = "input_number.heating_optimal_setpoint"


HELPERS = HeatingHelpers()


@dataclass
class DefaultSettings:
    """Default values for heating optimization."""

    # User defaults
    target_warm_time: str = "08:00"
    preferred_off_time: str = "23:00"
    target_temp: float = 20.0
    min_bedroom_temp: float = 18.0
    min_daytime_temp: float = 20.0

    # ML-computed defaults (used until model learns)
    default_switch_on_time: str = "06:00"
    default_switch_off_time: str = "22:00"
    default_setpoint: float = 20.0  # Base setpoint for normal days

    # Setpoint bounds (matching user's manual approach)
    min_setpoint: float = 19.0  # Never below this
    max_setpoint: float = 22.0  # Hard cap, even on coldest days

    # Safety thresholds
    extreme_cold_threshold: float = -5.0
    max_cycles_per_day: int = 2
    safety_buffer_minutes: int = 15
    min_off_duration_hours: float = 2.0  # Don't turn off if off period < 2 hours

    # Forecast bias correction (sensor reads warmer than forecast)
    forecast_temp_bias: float = 0.75  # °C added to forecast temps


DEFAULTS = DefaultSettings()


@dataclass
class ModelConfig:
    """Configuration for thermal model training."""

    # Data collection
    history_days: int = 14
    min_history_days: int = 3

    # Model persistence
    model_dir: str = "models/heating"
    model_file: str = "thermal_model.pkl"


MODEL_CONFIG = ModelConfig()


@dataclass
class PredictionConfig:
    """Configuration for prediction tracking and coefficient adjustment."""

    # Data storage
    data_dir: str = "data/heating"
    predictions_file: str = "predictions.jsonl"

    # Adjustment thresholds
    min_sample_days: int = 2  # Minimum days of data before adjusting coefficients
    error_threshold: float = 0.3  # Only adjust if avg error > 0.3°C

    # Coefficient bounds (physically realistic ranges)
    k_min: float = 0.003  # Min cooling rate constant (τ = 333 hours)
    k_max: float = 0.010  # Max cooling rate constant (τ = 100 hours)
    heating_rate_min: float = 0.5  # Min heating rate (°C/hour)
    heating_rate_max: float = 2.0  # Max heating rate (°C/hour)

    # Adjustment factors (incremental changes to avoid oscillation)
    k_adjustment_factor: float = 0.0005  # Adjustment per degree error
    heating_rate_adjustment_factor: float = 0.05  # 5% per degree error

    # Gas estimation
    gas_base_rate_kwh: float = 10.0  # kWh/hour at 50% modulation (Vitodens 100-W)
    gas_base_rate_min: float = 5.0  # Lower bound for auto-adjustment
    gas_base_rate_max: float = 15.0  # Upper bound for auto-adjustment
    gas_adjustment_factor: float = 0.05  # 5% adjustment per unit ratio error
    gas_error_threshold: float = 0.15  # Only adjust if avg % error > 15%


PREDICTION_CONFIG = PredictionConfig()
