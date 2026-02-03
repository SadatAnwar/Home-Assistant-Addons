"""Thermal model for learning home heating/cooling characteristics.

Learns:
- Cooling rate: How fast each room loses heat at different outside temps
- Heating rate: How fast the house warms when heating is on
- Solar gain: How much free heating south-facing rooms get from sun
- Setpoint-to-room-temp mapping: What setpoint achieves target room temp
"""

import logging
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from .config import MODEL_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class HeatingPrediction:
    """Prediction result from thermal model."""

    predicted_temp: float
    heating_duration_minutes: int
    expected_modulation: float
    solar_contribution: float
    confidence: float


@dataclass
class CoolingPrediction:
    """Cooling curve prediction."""

    hours: list[float]
    temperatures: list[float]
    time_to_target: float | None  # Hours until reaching target temp


class ThermalModel:
    """Machine learning model for home thermal characteristics."""

    def __init__(self, model_dir: str = MODEL_CONFIG.model_dir):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Models
        self.heating_rate_model: GradientBoostingRegressor | None = None
        self.cooling_rate_model: GradientBoostingRegressor | None = None
        self.modulation_model: GradientBoostingRegressor | None = None
        self.solar_gain_model: GradientBoostingRegressor | None = None

        # Learned parameters (defaults based on real-world measurements)
        # Actual measured cooling: 0.16-0.27°C/hour in extreme cold (-6 to -8°C)
        self.mean_cooling_rate: float = 0.15  # °C/hour (default, conservative)
        self.mean_heating_rate: float = 1.0  # °C/hour (default, measured 0.96-1.18)
        self.solar_gain_coefficient: float = 0.05  # °C per degree elevation

        # Cooling rate constant k (1/τ where τ is time constant in hours)
        # Based on measured τ = 156 hours (KfW Effizienzhaus 40 level)
        self.k: float = 0.0064  # Default: 1/156

        # Training metadata
        self.last_trained: datetime | None = None
        self.training_samples: int = 0

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        """Train all models on historical data.

        Args:
            data: DataFrame from DataCollector.build_training_dataset()

        Returns:
            Dictionary of training metrics
        """
        if data.empty or len(data) < 100:
            return {"error": "Insufficient data for training"}

        metrics = {}

        # Train cooling rate model (when heating is off)
        cooling_metrics = self._train_cooling_model(data)
        metrics.update({"cooling_" + k: v for k, v in cooling_metrics.items()})

        # Train heating rate model (when heating is on)
        heating_metrics = self._train_heating_model(data)
        metrics.update({"heating_" + k: v for k, v in heating_metrics.items()})

        # Train modulation prediction model
        modulation_metrics = self._train_modulation_model(data)
        metrics.update({"modulation_" + k: v for k, v in modulation_metrics.items()})

        # Train solar gain model
        solar_metrics = self._train_solar_model(data)
        metrics.update({"solar_" + k: v for k, v in solar_metrics.items()})

        self.last_trained = datetime.now()
        self.training_samples = len(data)

        return metrics

    def _train_cooling_model(self, data: pd.DataFrame) -> dict[str, float]:
        """Train model for predicting cooling rate when heating is off."""
        if "heating_on" not in data.columns or "bedroom" not in data.columns:
            return {"error": "missing required columns"}

        # Filter for heating off periods
        cooling_data = data[~data["heating_on"]].copy()
        if len(cooling_data) < 50:
            return {"samples": len(cooling_data), "error": "insufficient data"}

        # Calculate temperature change rate (°C per 15 min interval)
        cooling_data["temp_change"] = cooling_data["bedroom"].diff()
        cooling_data = cooling_data.dropna(subset=["temp_change"])

        # Filter out unrealistic changes (sensor noise)
        cooling_data = cooling_data[abs(cooling_data["temp_change"]) < 2]

        if len(cooling_data) < 30:
            return {"samples": len(cooling_data), "error": "insufficient valid data"}

        # Features for cooling prediction
        feature_cols = []
        if "outside_temp" in cooling_data.columns:
            feature_cols.append("outside_temp")
        if "bedroom" in cooling_data.columns:
            feature_cols.append("bedroom")
        if "sun_elevation" in cooling_data.columns:
            cooling_data["sun_elevation"] = cooling_data["sun_elevation"].fillna(0)
            feature_cols.append("sun_elevation")

        if len(feature_cols) < 2:
            return {"error": "insufficient features"}

        # Drop rows with NaN in any feature column
        cooling_data = cooling_data.dropna(subset=feature_cols)
        if len(cooling_data) < 30:
            return {
                "samples": len(cooling_data),
                "error": "insufficient data after NaN removal",
            }

        X = cooling_data[feature_cols].values
        y = cooling_data["temp_change"].values * 4  # Convert to °C/hour

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.cooling_rate_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        )
        self.cooling_rate_model.fit(X_train, y_train)

        # Calculate metrics
        train_score = self.cooling_rate_model.score(X_train, y_train)
        test_score = self.cooling_rate_model.score(X_test, y_test)

        # Update mean cooling rate
        self.mean_cooling_rate = abs(np.mean(y))

        return {
            "samples": len(cooling_data),
            "train_r2": round(train_score, 3),
            "test_r2": round(test_score, 3),
            "mean_rate": round(self.mean_cooling_rate, 3),
        }

    def _train_heating_model(self, data: pd.DataFrame) -> dict[str, float]:
        """Train model for predicting heating rate when heating is on."""
        if "heating_on" not in data.columns or "bedroom" not in data.columns:
            return {"error": "missing required columns"}

        # Filter for heating on periods
        heating_data = data[data["heating_on"]].copy()
        if len(heating_data) < 50:
            return {"samples": len(heating_data), "error": "insufficient data"}

        # Calculate temperature change rate
        heating_data["temp_change"] = heating_data["bedroom"].diff()
        heating_data = heating_data.dropna(subset=["temp_change"])

        # Filter out unrealistic changes
        heating_data = heating_data[abs(heating_data["temp_change"]) < 2]

        if len(heating_data) < 30:
            return {"samples": len(heating_data), "error": "insufficient valid data"}

        # Features for heating prediction
        feature_cols = []
        if "outside_temp" in heating_data.columns:
            feature_cols.append("outside_temp")
        if "bedroom" in heating_data.columns:
            feature_cols.append("bedroom")
        if "setpoint" in heating_data.columns:
            heating_data["setpoint"] = heating_data["setpoint"].fillna(21)
            feature_cols.append("setpoint")
        if "burner_modulation" in heating_data.columns:
            heating_data["burner_modulation"] = heating_data[
                "burner_modulation"
            ].fillna(50)
            feature_cols.append("burner_modulation")

        if len(feature_cols) < 2:
            return {"error": "insufficient features"}

        # Drop rows with NaN in any feature column
        heating_data = heating_data.dropna(subset=feature_cols)
        if len(heating_data) < 30:
            return {
                "samples": len(heating_data),
                "error": "insufficient data after NaN removal",
            }

        X = heating_data[feature_cols].values
        y = heating_data["temp_change"].values * 4  # Convert to °C/hour

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.heating_rate_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        )
        self.heating_rate_model.fit(X_train, y_train)

        train_score = self.heating_rate_model.score(X_train, y_train)
        test_score = self.heating_rate_model.score(X_test, y_test)

        # Update mean heating rate
        self.mean_heating_rate = np.mean(y[y > 0]) if np.any(y > 0) else 1.0

        return {
            "samples": len(heating_data),
            "train_r2": round(train_score, 3),
            "test_r2": round(test_score, 3),
            "mean_rate": round(self.mean_heating_rate, 3),
        }

    def _train_modulation_model(self, data: pd.DataFrame) -> dict[str, float]:
        """Train model for predicting burner modulation at given conditions."""
        required = ["burner_modulation", "outside_temp", "setpoint"]
        if not all(col in data.columns for col in required):
            return {"error": "missing required columns"}

        mod_data = data.dropna(subset=required).copy()
        mod_data = mod_data[
            mod_data["burner_modulation"] > 0
        ]  # Only when burner active

        if len(mod_data) < 50:
            return {"samples": len(mod_data), "error": "insufficient data"}

        feature_cols = ["outside_temp", "setpoint"]
        if "bedroom" in mod_data.columns:
            feature_cols.append("bedroom")

        # Drop rows with NaN in any feature column
        mod_data = mod_data.dropna(subset=feature_cols)
        if len(mod_data) < 30:
            return {
                "samples": len(mod_data),
                "error": "insufficient data after NaN removal",
            }

        X = mod_data[feature_cols].values
        y = mod_data["burner_modulation"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.modulation_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        )
        self.modulation_model.fit(X_train, y_train)

        train_score = self.modulation_model.score(X_train, y_train)
        test_score = self.modulation_model.score(X_test, y_test)

        return {
            "samples": len(mod_data),
            "train_r2": round(train_score, 3),
            "test_r2": round(test_score, 3),
        }

    def _train_solar_model(self, data: pd.DataFrame) -> dict[str, float]:
        """Train model for predicting solar heat gain in south-facing rooms."""
        required = ["sun_elevation", "bedroom"]
        if not all(col in data.columns for col in required):
            return {"error": "missing required columns"}

        # Filter for daytime, heating off, positive sun elevation
        solar_data = data.copy()
        if "heating_on" in solar_data.columns:
            solar_data = solar_data[~solar_data["heating_on"]]

        solar_data = solar_data.dropna(subset=required)
        solar_data = solar_data[solar_data["sun_elevation"] > 10]  # Sun is up

        if len(solar_data) < 30:
            return {"samples": len(solar_data), "error": "insufficient data"}

        # Calculate temp change rate during sunny periods
        solar_data["temp_change"] = solar_data["bedroom"].diff() * 4  # °C/hour
        solar_data = solar_data.dropna(subset=["temp_change"])

        # Simple linear estimate of solar gain coefficient
        # When cooling should occur but temp rises, that's solar gain
        sunny_warm = solar_data[
            (solar_data["sun_elevation"] > 20)
            & (solar_data["temp_change"] > -0.5)  # Not cooling much
        ]

        if len(sunny_warm) > 10:
            # Estimate: temp_change ~ solar_coeff * sun_elevation
            self.solar_gain_coefficient = np.mean(
                sunny_warm["temp_change"] / sunny_warm["sun_elevation"]
            )
            self.solar_gain_coefficient = max(0, min(0.1, self.solar_gain_coefficient))

        return {
            "samples": len(solar_data),
            "solar_coefficient": round(self.solar_gain_coefficient, 4),
        }

    def predict_cooling_curve(
        self,
        start_temp: float,
        outside_temp: float,
        hours: int = 8,
        sun_elevation_curve: list[float] | None = None,
    ) -> CoolingPrediction:
        """Predict temperature curve when heating is off.

        Args:
            start_temp: Starting room temperature
            outside_temp: Outside temperature (assumed constant if no forecast)
            hours: Number of hours to predict
            sun_elevation_curve: Optional hourly sun elevation values

        Returns:
            CoolingPrediction with hourly temperatures
        """
        # Maximum reasonable cooling rate based on thermal mass:
        # Well-insulated homes typically lose 0.1-0.3°C/hour
        # Use 3x mean cooling rate as max, or 0.3°C/hour, whichever is higher
        MAX_COOLING_RATE = max(self.mean_cooling_rate * 3, 0.3)

        hour_values = list(range(hours + 1))
        temps = [start_temp]
        current_temp = start_temp

        logger.debug(
            f"Cooling prediction: start={start_temp}°C, outside={outside_temp}°C, "
            f"hours={hours}, mean_rate={self.mean_cooling_rate:.3f}°C/h, max_rate={MAX_COOLING_RATE:.3f}°C/h"
        )

        for h in range(hours):
            sun_elev = sun_elevation_curve[h] if sun_elevation_curve else 0

            # Use physics-based model: Newton's law of cooling
            # dT/dt = -k * (T_inside - T_outside) where k = 1/τ
            # Measured from 7 nights of data (Jan 24 - Feb 2):
            #   - Time constant τ = 156 hours (KfW Effizienzhaus 40 level)
            #   - k = 1/τ = 0.0064 per hour
            # This gives ~0.17°C/hour cooling at 27°C temperature difference
            # Note: k is now adjustable based on prediction feedback
            temp_diff = current_temp - outside_temp
            rate = -self.k * temp_diff

            # Add solar gain during day
            if sun_elev > 10:
                rate += self.solar_gain_coefficient * sun_elev

            # Clamp to reasonable range: between -MAX_COOLING_RATE and slight warming
            rate = max(-MAX_COOLING_RATE, min(0.2, rate))

            current_temp += rate
            temps.append(round(current_temp, 2))

        logger.debug(
            f"Cooling prediction result: {temps[0]}°C -> {temps[-1]}°C over {hours}h"
        )

        return CoolingPrediction(
            hours=hour_values,
            temperatures=temps,
            time_to_target=None,
        )

    def predict_heating_duration(
        self,
        start_temp: float,
        target_temp: float,
        outside_temp: float,
        setpoint: float = 21.0,
    ) -> int:
        """Predict minutes needed to heat from start_temp to target_temp.

        Args:
            start_temp: Starting room temperature
            target_temp: Target room temperature
            outside_temp: Current outside temperature
            setpoint: Boiler setpoint temperature

        Returns:
            Estimated minutes to reach target
        """
        if start_temp >= target_temp:
            logger.debug(
                f"No heating needed: start {start_temp}°C >= target {target_temp}°C"
            )
            return 0

        temp_diff = target_temp - start_temp

        # Absolute minimum heating rate based on real-world experience:
        # User typically heats for 30-60 min to warm up 1-2°C, implying ~1-2°C/hour
        # Use 1.0°C/hour as a reasonable floor
        ABSOLUTE_MIN_RATE = 1.0

        if self.heating_rate_model is not None:
            # Use ML model to estimate average heating rate
            features = np.array(
                [[outside_temp, start_temp, setpoint, 50]]
            )  # 50% modulation
            ml_rate = self.heating_rate_model.predict(features)[0]

            # Apply minimum rate constraints:
            # 1. At least 50% of learned mean heating rate
            # 2. At least the absolute minimum (0.5°C/hour)
            min_rate = max(self.mean_heating_rate * 0.5, ABSOLUTE_MIN_RATE)
            rate = max(min_rate, ml_rate)

            if ml_rate < min_rate:
                logger.debug(
                    f"ML rate {ml_rate:.3f}°C/h too low, using minimum {rate:.3f}°C/h"
                )
            else:
                logger.debug(
                    f"ML heating rate: {rate:.3f}°C/hour (features: outside={outside_temp}, "
                    f"start={start_temp}, setpoint={setpoint})"
                )
        else:
            # Use simple model with minimum constraint
            rate = max(self.mean_heating_rate, ABSOLUTE_MIN_RATE)
            logger.debug(f"Using mean heating rate: {rate:.3f}°C/hour")

        # Time = temp_diff / rate (hours), convert to minutes
        hours = temp_diff / rate
        raw_minutes = hours * 60
        minutes = int(raw_minutes)

        logger.debug(
            f"Heating duration calc: {temp_diff:.1f}°C / {rate:.3f}°C/h = "
            f"{raw_minutes:.0f}min (capped to 15-180)"
        )

        # Cap at reasonable limits
        return max(15, min(180, minutes))

    def predict_modulation(
        self,
        outside_temp: float,
        setpoint: float,
        room_temp: float | None = None,
    ) -> float:
        """Predict expected burner modulation at given conditions.

        Args:
            outside_temp: Outside temperature
            setpoint: Boiler setpoint
            room_temp: Current room temperature (optional)

        Returns:
            Predicted modulation percentage (0-100)
        """
        if self.modulation_model is not None:
            if room_temp is not None:
                features = np.array([[outside_temp, setpoint, room_temp]])
            else:
                features = np.array([[outside_temp, setpoint]])
            modulation = self.modulation_model.predict(features)[0]
        else:
            # Simple linear estimate
            temp_diff = setpoint - outside_temp
            modulation = 30 + temp_diff * 2  # Higher diff = higher modulation

        return max(0, min(100, modulation))

    def predict_solar_gain(
        self,
        sun_elevation: float,
        cloud_cover: float = 0,
        is_south_facing: bool = True,
    ) -> float:
        """Predict solar heat contribution in °C.

        Args:
            sun_elevation: Current sun elevation in degrees
            cloud_cover: Cloud cover percentage (0-100)
            is_south_facing: Whether the room faces south

        Returns:
            Predicted temperature contribution from solar gain
        """
        if sun_elevation <= 0 or not is_south_facing:
            return 0.0

        # Base solar gain
        gain = self.solar_gain_coefficient * sun_elevation

        # Reduce by cloud cover
        clear_sky_factor = 1 - (cloud_cover / 100) * 0.8
        gain *= clear_sky_factor

        return round(gain, 2)

    def find_optimal_setpoint(
        self,
        target_room_temp: float,
        outside_temp: float,
        max_modulation: float = 70,
    ) -> float:
        """Find minimum setpoint to achieve target room temperature.

        Uses learned relationship between setpoint, outside temp, and room temp.

        Args:
            target_room_temp: Desired room temperature
            outside_temp: Current outside temperature
            max_modulation: Maximum acceptable modulation (for efficiency)

        Returns:
            Recommended setpoint
        """
        # Start with a baseline offset
        base_offset = 1.0  # Room is typically 1°C below setpoint

        # Adjust for outside temperature
        # Colder outside = need higher setpoint
        if outside_temp < 0:
            temp_adjustment = 0.5 + abs(outside_temp) * 0.1
        elif outside_temp < 5:
            temp_adjustment = 0.3
        else:
            temp_adjustment = 0

        setpoint = target_room_temp + base_offset + temp_adjustment

        # If we have a modulation model, verify and adjust
        if self.modulation_model is not None:
            predicted_mod = self.predict_modulation(
                outside_temp, setpoint, target_room_temp - 0.5
            )
            if predicted_mod > max_modulation:
                # Reduce setpoint to lower modulation
                setpoint -= 0.5

        # Clamp to reasonable range
        return round(max(18, min(24, setpoint)), 1)

    def save(self, filename: str = MODEL_CONFIG.model_file) -> None:
        """Save model to disk."""
        filepath = self.model_dir / filename
        model_data = {
            "heating_rate_model": self.heating_rate_model,
            "cooling_rate_model": self.cooling_rate_model,
            "modulation_model": self.modulation_model,
            "solar_gain_model": self.solar_gain_model,
            "mean_cooling_rate": self.mean_cooling_rate,
            "mean_heating_rate": self.mean_heating_rate,
            "solar_gain_coefficient": self.solar_gain_coefficient,
            "k": self.k,
            "last_trained": self.last_trained,
            "training_samples": self.training_samples,
        }
        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

    def load(self, filename: str = MODEL_CONFIG.model_file) -> bool:
        """Load model from disk. Returns True if successful."""
        filepath = self.model_dir / filename
        if not filepath.exists():
            return False

        try:
            with open(filepath, "rb") as f:
                model_data = pickle.load(f)

            self.heating_rate_model = model_data.get("heating_rate_model")
            self.cooling_rate_model = model_data.get("cooling_rate_model")
            self.modulation_model = model_data.get("modulation_model")
            self.solar_gain_model = model_data.get("solar_gain_model")
            self.mean_cooling_rate = model_data.get("mean_cooling_rate", 0.3)
            self.mean_heating_rate = model_data.get("mean_heating_rate", 1.0)
            self.solar_gain_coefficient = model_data.get("solar_gain_coefficient", 0.05)
            self.k = model_data.get("k", 0.0064)
            self.last_trained = model_data.get("last_trained")
            self.training_samples = model_data.get("training_samples", 0)
            return True
        except Exception:
            return False

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model state."""
        return {
            "trained": self.last_trained is not None,
            "last_trained": self.last_trained.isoformat()
            if self.last_trained
            else None,
            "training_samples": self.training_samples,
            "mean_cooling_rate": self.mean_cooling_rate,
            "mean_heating_rate": self.mean_heating_rate,
            "solar_gain_coefficient": self.solar_gain_coefficient,
            "k": self.k,
            "time_constant_hours": round(1 / self.k, 1) if self.k > 0 else None,
            "has_heating_model": self.heating_rate_model is not None,
            "has_cooling_model": self.cooling_rate_model is not None,
            "has_modulation_model": self.modulation_model is not None,
        }

    def apply_adjustments(self, adjustments: dict[str, float]) -> list[str]:
        """Apply coefficient adjustments from prediction tracker.

        Args:
            adjustments: Dictionary of coefficient names to new values

        Returns:
            List of applied adjustment descriptions
        """
        applied = []

        if "k" in adjustments:
            old_k = self.k
            self.k = adjustments["k"]
            applied.append(
                f"Cooling rate k: {old_k:.6f} -> {self.k:.6f} "
                f"(τ: {1 / old_k:.0f}h -> {1 / self.k:.0f}h)"
            )

        if "mean_heating_rate" in adjustments:
            old_rate = self.mean_heating_rate
            self.mean_heating_rate = adjustments["mean_heating_rate"]
            applied.append(
                f"Heating rate: {old_rate:.3f} -> {self.mean_heating_rate:.3f} °C/hour"
            )

        return applied
