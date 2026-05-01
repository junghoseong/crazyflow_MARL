"""Tests of the parametrization of the models."""

from __future__ import annotations

from typing import Callable

import pytest

from drone_models import available_models
from drone_models.core import parametrize
from drone_models.drones import available_drones


@pytest.mark.unit
@pytest.mark.parametrize("model_name, model", available_models.items())
@pytest.mark.parametrize("drone_type", available_drones)
def test_model_parametrization(model_name: str, model: Callable, drone_type: str):
    """TODO."""
    parametrize(model, drone_type)
