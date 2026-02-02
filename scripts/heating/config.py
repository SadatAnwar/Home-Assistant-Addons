"""Configuration for the adaptive heating optimization system."""

from dataclasses import dataclass, field


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

# Room humidity sensor mapping
HUMIDITY_SENSORS = {
    "bedroom": "sensor.bedroom_thermo_humidity",
    "kids_room": "sensor.aayat_room_temp_humidity",
    "bathroom": "sensor.bathroom_temp_humidity",
    "living_room": "sensor.living_room_temp_humidity",
    "kitchen": "sensor.kitchen_humidity",
    "top_floor": "sensor.top_floor_thermo_humidity",
    "basement": "sensor.basement_humidity",
}

# Rooms that get significant solar gain (south-facing)
SOLAR_GAIN_ROOMS = ["bedroom", "living_room"]

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

# Boiler setpoint entities (number helpers)
BOILER_SETPOINTS = {
    "normal_temp": "number.e3_vitodens_100_0421_1_normal_temperature",
    "comfort_temp": "number.e3_vitodens_100_0421_1_comfort_temperature",
    "reduced_temp": "number.e3_vitodens_100_0421_1_reduced_temperature",
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
    target_warm_time: str = "07:00"
    preferred_off_time: str = "22:00"
    target_temp: float = 20.0
    min_bedroom_temp: float = 19.0

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


DEFAULTS = DefaultSettings()


# Primary room for optimization (most important room)
PRIMARY_ROOM = "bedroom"  # Parent's bedroom


@dataclass
class ModelConfig:
    """Configuration for thermal model training."""

    # Data collection
    history_days: int = 14
    min_history_days: int = 3

    # Model parameters
    heating_rate_features: list = field(default_factory=lambda: [
        "outside_temp",
        "setpoint",
        "start_temp",
        "burner_modulation",
    ])

    cooling_rate_features: list = field(default_factory=lambda: [
        "outside_temp",
        "start_temp",
        "sun_elevation",
        "cloud_cover",
    ])

    # Model persistence
    model_dir: str = "models/heating"
    model_file: str = "thermal_model.pkl"


MODEL_CONFIG = ModelConfig()
