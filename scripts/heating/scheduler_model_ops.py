"""Model retraining operations for heating scheduler."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .config import MODEL_CONFIG
from .data_collector import DataCollector
from .thermal_model import ThermalModel


def should_retrain(model: ThermalModel) -> bool:
    """Determine if model should be retrained."""
    if model.last_trained is None:
        return True

    # Retrain if model is older than 7 days
    age = datetime.now() - model.last_trained
    return age.days >= 7


def train_model(collector: DataCollector, model: ThermalModel) -> dict[str, Any]:
    """Train the thermal model on historical data."""
    # Collect training data
    data = collector.build_training_dataset(days=MODEL_CONFIG.history_days)

    if data.empty:
        return {"error": "No training data available"}

    if len(data) < 100:
        return {
            "error": f"Insufficient data ({len(data)} samples)",
            "samples": len(data),
        }

    # Train model
    metrics = model.train(data)

    # Save model
    model.save()

    return {"samples": len(data), "metrics": metrics}
