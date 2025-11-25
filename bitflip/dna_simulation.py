from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from .simulation import altitude_scale


DNA_BASES: Sequence[str] = ("A", "C", "G", "T")


@dataclass
class DNASimulationFrame:
    """Snapshot of a DNA string after cosmic-ray–driven mutations."""

    sequence: str
    time_hours: float
    expected_mutations: float
    observed_mutations: int
    probability_at_least_one: float


def _mutate_base(original: str, rng: np.random.Generator) -> str:
    """Return a mutated nucleotide different from the original."""

    candidates = [b for b in DNA_BASES if b != original]
    return rng.choice(candidates)  # type: ignore[arg-type]


def simulate_dna_mutations_over_time(
    sequence: str,
    altitude_m: float,
    total_hours: float,
    steps: int,
    base_rate_per_base: float = 1e-12,
    seed: Optional[int] = None,
) -> list[DNASimulationFrame]:
    """
    Simulate mutations on a DNA sequence using a Poisson process scaled by altitude.

    Args:
        sequence: Initial DNA sequence to mutate.
        altitude_m: Altitude in meters (scales the mutation rate as ``2^(h/1000)``).
        total_hours: Duration to simulate in hours.
        steps: Number of frames to emit in the timeline.
        base_rate_per_base: Base mutation rate per base-hour at sea level.
        seed: Optional seed for reproducibility.

    Returns:
        List of frames containing the mutated sequence and metrics at each step.
    """

    if steps < 1:
        raise ValueError("steps must be >= 1")

    seq_array = np.array(list(sequence))
    num_bases = len(seq_array)
    rng = np.random.default_rng(seed)

    total_rate = base_rate_per_base * altitude_scale(altitude_m) * num_bases
    step_hours = float(total_hours) / steps

    frames: list[DNASimulationFrame] = []
    cumulative_mutations = 0

    for step in range(1, steps + 1):
        mu_step = total_rate * step_hours
        observed_step = int(rng.poisson(mu_step))
        cumulative_mutations += observed_step

        if observed_step > 0:
            indices = rng.integers(0, num_bases, size=observed_step)
            for idx in indices:
                seq_array[idx] = _mutate_base(str(seq_array[idx]), rng)

        cumulative_mu = total_rate * (step_hours * step)
        probability_at_least_one = 1 - float(np.exp(-cumulative_mu))

        frames.append(
            DNASimulationFrame(
                sequence="".join(seq_array.tolist()),
                time_hours=step_hours * step,
                expected_mutations=cumulative_mu,
                observed_mutations=cumulative_mutations,
                probability_at_least_one=probability_at_least_one,
            )
        )

    return frames


def mutation_probability_curve(
    base_rate_per_base: float,
    altitude_m: float,
    num_bases: int,
    hours_points: Iterable[float],
) -> np.ndarray:
    """Return probability of at least one mutation across time points."""

    total_rate = base_rate_per_base * altitude_scale(altitude_m) * num_bases
    hours_array = np.array(list(hours_points), dtype=float)
    mu_values = total_rate * hours_array
    return 1 - np.exp(-mu_values)
