import statistics
from datetime import UTC, datetime

from pydantic import BaseModel

from bench import __version__
from bench.profiling import PROFILE_BUCKETS, summarize_buckets


class ToolInfo(BaseModel):
    name: str
    version: str
    revision: str | None


class DeviceInfo(BaseModel):
    type: str
    index: int | None
    name: str


class EnvironmentInfo(BaseModel):
    python_version: str
    torch_version: str
    comfyui_version: str
    comfyui_revision: str | None
    accelerator_runtime: str | None
    device: DeviceInfo


class ModelInfo(BaseModel):
    source: str
    architecture: str
    parameters: int
    data_type: str
    synthetic_weights: bool = True


class ImageWorkload(BaseModel):
    width: int
    height: int
    batch_size: int


class SamplingWorkload(BaseModel):
    steps: int
    classifier_free_guidance: float
    sampler: str
    scheduler: str
    denoise: float
    seed: int


class ConditioningWorkload(BaseModel):
    token_count: int
    token_width: int
    pooled_width: int | None


class SdpaWorkload(BaseModel):
    attention_function: str
    query_shape: list[int]
    key_shape: list[int]
    value_shape: list[int]
    query_data_type: str
    key_data_type: str
    value_data_type: str
    heads: int
    mask_shape: list[int] | None
    mask_data_type: str | None
    attention_flops: int
    skip_reshape: bool
    skip_output_reshape: bool


class WorkloadInfo(BaseModel):
    target: str
    image: ImageWorkload
    sampling: SamplingWorkload
    conditioning: ConditioningWorkload
    sdpa: SdpaWorkload | None


class CompileInfo(BaseModel):
    backend: str
    mode: str | None


class TunableInfo(BaseModel):
    filename: str
    result_count: int


class RuntimeInfo(BaseModel):
    attention: str
    compile: CompileInfo | None
    tunable: TunableInfo | None


class LatencyMeasurement(BaseModel):
    unit: str = "seconds"
    samples: list[float]
    median: float
    mean: float
    standard_deviation: float | None
    minimum: float
    maximum: float


class MemoryMeasurement(BaseModel):
    baseline_allocated_bytes: int | None
    peak_allocated_bytes: int | None
    incremental_peak_bytes: int | None


class ProfileBucket(BaseModel):
    category: str
    duration_seconds: float
    share: float
    flops: float
    tflops: float | None


class ProfileOperation(BaseModel):
    category: str
    name: str
    duration_seconds: float
    calls: int
    flops: float


class ProfileMeasurement(BaseModel):
    buckets: list[ProfileBucket]
    operations: list[ProfileOperation]


class MeasurementsInfo(BaseModel):
    requested_warmup_evaluations: int
    observed_evaluations: int
    timed_evaluations: int
    latency: LatencyMeasurement
    evaluations_per_second: float
    memory: MemoryMeasurement
    profile: ProfileMeasurement | None


class BenchmarkResult(BaseModel):
    schema_version: int = 1
    created_at: datetime
    tool: ToolInfo
    environment: EnvironmentInfo
    model: ModelInfo
    workload: WorkloadInfo
    runtime: RuntimeInfo
    measurements: MeasurementsInfo


def create_latency(durations):
    median = statistics.median(durations)
    return LatencyMeasurement(
        samples=durations,
        median=median,
        mean=statistics.fmean(durations),
        standard_deviation=statistics.stdev(durations) if len(durations) > 1 else None,
        minimum=min(durations),
        maximum=max(durations),
    )


def create_profile(entries):
    if entries is None:
        return None
    buckets = summarize_buckets(entries)
    total_duration = sum(bucket["duration_seconds"] for bucket in buckets.values())
    bucket_models = []
    for category, _ in PROFILE_BUCKETS:
        duration = buckets[category]["duration_seconds"]
        flops = buckets[category]["flops"]
        bucket_models.append(
            ProfileBucket(
                category=category,
                duration_seconds=duration,
                share=duration / total_duration if total_duration else 0.0,
                flops=flops,
                tflops=flops / duration / 1e12 if flops and duration else None,
            )
        )
    operations = [
        ProfileOperation(
            category=entry.category,
            name=entry.name,
            duration_seconds=entry.duration_seconds,
            calls=entry.calls,
            flops=entry.flops,
        )
        for entry in entries
    ]
    return ProfileMeasurement(buckets=bucket_models, operations=operations)


def create_result(
    configuration,
    tool_revision,
    environment,
    model,
    conditioning,
    outcome,
):
    sdpa = None
    if outcome.description is not None:
        description = outcome.description
        sdpa = SdpaWorkload(
            attention_function=description["attention_function"],
            query_shape=description["query_shape"],
            key_shape=description["key_shape"],
            value_shape=description["value_shape"],
            query_data_type=description["query_dtype"],
            key_data_type=description["key_dtype"],
            value_data_type=description["value_dtype"],
            heads=description["heads"],
            mask_shape=description["mask_shape"],
            mask_data_type=description["mask_dtype"],
            attention_flops=description["attention_flops"],
            skip_reshape=description["skip_reshape"],
            skip_output_reshape=description["skip_output_reshape"],
        )
    measurement = outcome.measurement
    latency = create_latency(measurement.durations)
    return BenchmarkResult(
        created_at=datetime.now(UTC),
        tool=ToolInfo(name="comfy-bench", version=__version__, revision=tool_revision),
        environment=environment,
        model=model,
        workload=WorkloadInfo(
            target=configuration.target,
            image=ImageWorkload(
                width=configuration.width,
                height=configuration.height,
                batch_size=configuration.batch_size,
            ),
            sampling=SamplingWorkload(
                steps=configuration.steps,
                classifier_free_guidance=configuration.classifier_free_guidance,
                sampler=configuration.sampler_name,
                scheduler=configuration.scheduler,
                denoise=configuration.denoise,
                seed=configuration.seed,
            ),
            conditioning=ConditioningWorkload(
                token_count=configuration.context_length,
                token_width=conditioning.context_dimension,
                pooled_width=conditioning.pooled_dimension,
            ),
            sdpa=sdpa,
        ),
        runtime=RuntimeInfo(
            attention=configuration.attention,
            compile=None
            if configuration.compile_backend is None
            else CompileInfo(
                backend=configuration.compile_backend, mode=configuration.compile_mode
            ),
            tunable=None
            if configuration.tunable_file is None
            else TunableInfo(
                filename=environment_tunable_filename(),
                result_count=environment_tunable_result_count(),
            ),
        ),
        measurements=MeasurementsInfo(
            requested_warmup_evaluations=configuration.warmup_evaluations,
            observed_evaluations=measurement.observed_evaluations,
            timed_evaluations=len(measurement.durations),
            latency=latency,
            evaluations_per_second=1.0 / latency.median,
            memory=MemoryMeasurement(
                baseline_allocated_bytes=measurement.memory.baseline_allocated_bytes,
                peak_allocated_bytes=measurement.memory.peak_allocated_bytes,
                incremental_peak_bytes=measurement.memory.incremental_peak_bytes,
            ),
            profile=create_profile(measurement.profile_entries),
        ),
    )


def environment_tunable_filename():
    import torch

    return torch.cuda.tunable.get_filename()


def environment_tunable_result_count():
    import torch

    return len(torch.cuda.tunable.get_results())
