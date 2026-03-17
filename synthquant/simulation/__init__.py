"""Monte Carlo simulation engines and utilities."""

from synthquant.simulation.correlation import CopulaSimulator
from synthquant.simulation.engines.gbm import GBMEngine
from synthquant.simulation.engines.heston import HestonEngine
from synthquant.simulation.engines.merton_jd import MertonJDEngine
from synthquant.simulation.engines.regime_switching import RegimeSwitchingEngine
from synthquant.simulation.engines.rough_bergomi import RoughBergomiEngine
from synthquant.simulation.gpu_engine import GPUEngine
from synthquant.simulation.variance_reduction import (
    antithetic_variates,
    control_variates,
    importance_sampling,
    stratified_sampling,
)

__all__ = [
    "GBMEngine",
    "MertonJDEngine",
    "HestonEngine",
    "RegimeSwitchingEngine",
    "RoughBergomiEngine",
    "CopulaSimulator",
    "GPUEngine",
    "antithetic_variates",
    "control_variates",
    "importance_sampling",
    "stratified_sampling",
]
