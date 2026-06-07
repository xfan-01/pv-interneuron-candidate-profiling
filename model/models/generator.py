"""Backward-compatible aliases for forecasting model classes.

Deprecated: import from model.models.forecasting or model.models (Forecaster API).
"""

from .forecasting import (
    AblationForecaster,
    Forecaster,
    ForecasterConfig,
    AblationGenerator,
    Generator,
    GeneratorConfig,
)

__all__ = [
    "ForecasterConfig",
    "Forecaster",
    "AblationForecaster",
    "GeneratorConfig",
    "Generator",
    "AblationGenerator",
]
