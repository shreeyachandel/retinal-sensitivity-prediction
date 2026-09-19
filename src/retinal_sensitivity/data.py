"""Synthetic data generation for the public demonstration.

The generator recreates the *shape* of a repeated-measures retinal dataset,
not any real participant or measurement from the thesis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DemoConfig:
    """Configuration for a deterministic synthetic cohort."""

    participants: int = 30
    visits: int = 2
    eyes: tuple[str, ...] = ("L", "R")
    points_per_eye_visit: int = 37
    random_state: int = 42


def generate_synthetic_cohort(config: DemoConfig | None = None) -> pd.DataFrame:
    """Create synthetic pointwise retinal observations with grouped structure.

    Participant effects, repeat visits and paired eyes deliberately induce
    correlation. This makes participant-grouped validation necessary and lets
    the public demo exercise the central statistical safeguard from the thesis.
    """

    config = config or DemoConfig()
    rng = np.random.default_rng(config.random_state)
    records: list[dict[str, float | int | str]] = []

    for participant_number in range(config.participants):
        participant_id = f"SYN-{participant_number + 1:03d}"
        participant_effect = rng.normal(0.0, 2.2)
        severity = rng.uniform(0.0, 1.0)

        for visit in range(config.visits):
            progression = visit * rng.uniform(0.2, 1.4)
            for eye in config.eyes:
                eye_effect = rng.normal(0.0, 0.7)
                angles = np.linspace(-np.pi, np.pi, config.points_per_eye_visit, endpoint=False)
                eccentricity = np.linspace(0.05, 1.0, config.points_per_eye_visit)
                rng.shuffle(eccentricity)

                thickness = (
                    105
                    - 30 * severity
                    - 14 * eccentricity
                    - 3 * progression
                    + rng.normal(0, 5, config.points_per_eye_visit)
                )
                reflectivity = (
                    0.72
                    - 0.20 * severity
                    - 0.08 * eccentricity
                    + rng.normal(0, 0.045, config.points_per_eye_visit)
                )
                roughness = (
                    2.5
                    + 3.5 * severity
                    + 1.3 * eccentricity
                    + rng.normal(0, 0.7, config.points_per_eye_visit)
                )

                latent_sensitivity = (
                    8.0
                    + 0.13 * thickness
                    + 11.0 * reflectivity
                    - 0.75 * roughness
                    - 5.0 * eccentricity
                    + participant_effect
                    + eye_effect
                    - progression
                    + rng.normal(0, 3.0, config.points_per_eye_visit)
                )
                sensitivity = np.clip(latent_sensitivity, -1.0, 30.0)
                sensitivity[sensitivity < 1.0] = -1.0

                for point_index in range(config.points_per_eye_visit):
                    records.append(
                        {
                            "participant_id": participant_id,
                            "visit": f"V{visit + 1}",
                            "eye": eye,
                            "point": point_index + 1,
                            "eccentricity": float(eccentricity[point_index]),
                            "angle_sin": float(np.sin(angles[point_index])),
                            "angle_cos": float(np.cos(angles[point_index])),
                            "retinal_thickness": float(thickness[point_index]),
                            "reflectivity": float(reflectivity[point_index]),
                            "roughness": float(roughness[point_index]),
                            "sensitivity_db": float(sensitivity[point_index]),
                        }
                    )

    return pd.DataFrame.from_records(records)

