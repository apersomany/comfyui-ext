from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkConfiguration:
    model: str
    width: int
    height: int
    batch_size: int
    context_length: int
    steps: int
    classifier_free_guidance: float
    sampler_name: str
    scheduler: str
    denoise: float
    seed: int
    warmup_evaluations: int
    data_type: str
    device: str | None
    attention: str
    target: str
    compile_backend: str | None
    compile_mode: str | None
    tunable_file: str | None
    profile_mode: str | None


@dataclass(frozen=True)
class OutputConfiguration:
    format: str
    color: str
    path: Path | None
    progress: bool
