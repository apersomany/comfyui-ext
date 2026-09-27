from dataclasses import dataclass

from bench.measurement import Measurement


@dataclass(frozen=True)
class TargetOutcome:
    measurement: Measurement
    description: dict | None
    model_detached: bool
