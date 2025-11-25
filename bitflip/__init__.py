"""Core package for bit flip simulation."""

from .simulation import (
    AnimationFrame,
    SimulationResult,
    probability_curve,
    simulate_bit_flips,
    simulate_bit_flips_over_time,
)

__all__ = [
    "AnimationFrame",
    "SimulationResult",
    "simulate_bit_flips",
    "simulate_bit_flips_over_time",
    "probability_curve",
]
