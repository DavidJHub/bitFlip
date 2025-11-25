"""Core package for bit flip simulation."""

from .simulation import SimulationResult, simulate_bit_flips, probability_curve

__all__ = [
    "SimulationResult",
    "simulate_bit_flips",
    "probability_curve",
]
