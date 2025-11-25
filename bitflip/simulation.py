from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from PIL import Image


def altitude_scale(altitude_m: float) -> float:
    """Return the exponential scaling factor for the error rate at a given altitude."""
    return 2 ** (altitude_m / 1000.0)


def total_error_rate_per_hour(base_rate_per_bit: float, altitude_m: float, num_bits: int) -> float:
    """Compute the total error rate for a memory of ``num_bits`` at a given altitude."""
    return base_rate_per_bit * altitude_scale(altitude_m) * num_bits


@dataclass
class SimulationResult:
    corrupted_image: Image.Image
    expected_errors: float
    observed_errors: int
    probability_at_least_one: float
    total_bits: int


@dataclass
class AnimationFrame:
    """Snapshot of the simulation state at a specific time."""

    image: Image.Image
    time_hours: float
    expected_errors: float
    observed_errors: int
    probability_at_least_one: float


def _flip_bits(bytes_view: np.ndarray, bit_indices: Iterable[int]) -> None:
    for idx in bit_indices:
        byte_index = idx // 8
        bit_in_byte = idx % 8
        mask = 1 << bit_in_byte
        bytes_view[byte_index] ^= mask


def simulate_bit_flips(
    image: Image.Image,
    altitude_m: float,
    hours: float,
    base_rate_per_bit: float = 1e-12,
    seed: Optional[int] = None,
) -> SimulationResult:
    """
    Simulate bit flips on an image array using a Poisson process.

    Args:
        image: Source image to corrupt.
        altitude_m: Altitude in meters used for the exponential scaling of the rate.
        hours: Duration of the simulated exposure window, in hours.
        base_rate_per_bit: Base error rate per bit-hour at sea level.
        seed: Optional seed for reproducibility.

    Returns:
        SimulationResult containing the corrupted image and error statistics.
    """

    rgb_image = image.convert("RGB")
    arr = np.array(rgb_image)
    bytes_view = arr.view(np.uint8).reshape(-1)
    num_bits = int(bytes_view.size * 8)

    rng = np.random.default_rng(seed)
    total_rate = total_error_rate_per_hour(base_rate_per_bit, altitude_m, num_bits)
    mu = total_rate * hours

    observed_errors = int(rng.poisson(mu))
    if observed_errors > 0:
        bit_indices = rng.integers(0, num_bits, size=observed_errors)
        _flip_bits(bytes_view, bit_indices)

    corrupted_arr = bytes_view.reshape(arr.shape).astype(np.uint8)
    corrupted_image = Image.fromarray(corrupted_arr, mode="RGB")

    probability_at_least_one = 1 - float(np.exp(-mu))

    return SimulationResult(
        corrupted_image=corrupted_image,
        expected_errors=mu,
        observed_errors=observed_errors,
        probability_at_least_one=probability_at_least_one,
        total_bits=num_bits,
    )


def probability_curve(
    base_rate_per_bit: float,
    altitude_m: float,
    num_bits: int,
    hours_points: Iterable[float],
) -> np.ndarray:
    """Return probabilities of at least one error for each time value in ``hours_points``."""
    total_rate = total_error_rate_per_hour(base_rate_per_bit, altitude_m, num_bits)
    hours_array = np.array(list(hours_points), dtype=float)
    mu_values = total_rate * hours_array
    return 1 - np.exp(-mu_values)


def simulate_bit_flips_over_time(
    image: Image.Image,
    altitude_m: float,
    total_hours: float,
    steps: int,
    base_rate_per_bit: float = 1e-12,
    seed: Optional[int] = None,
) -> list[AnimationFrame]:
    """Simulate progressive corruption over time, returning snapshots for an animation."""

    if steps < 1:
        raise ValueError("steps must be >= 1")

    rgb_image = image.convert("RGB")
    arr = np.array(rgb_image)
    bytes_view = arr.view(np.uint8).reshape(-1)
    num_bits = int(bytes_view.size * 8)

    rng = np.random.default_rng(seed)
    total_rate = total_error_rate_per_hour(base_rate_per_bit, altitude_m, num_bits)
    step_hours = float(total_hours) / steps

    frames: list[AnimationFrame] = []
    cumulative_errors = 0

    for step in range(1, steps + 1):
        mu_step = total_rate * step_hours
        observed_step = int(rng.poisson(mu_step))
        cumulative_errors += observed_step

        if observed_step > 0:
            bit_indices = rng.integers(0, num_bits, size=observed_step)
            _flip_bits(bytes_view, bit_indices)

        cumulative_mu = total_rate * (step_hours * step)
        probability_at_least_one = 1 - float(np.exp(-cumulative_mu))

        frame_image = Image.fromarray(bytes_view.reshape(arr.shape).astype(np.uint8), mode="RGB")
        frames.append(
            AnimationFrame(
                image=frame_image,
                time_hours=step_hours * step,
                expected_errors=cumulative_mu,
                observed_errors=cumulative_errors,
                probability_at_least_one=probability_at_least_one,
            )
        )

    return frames
