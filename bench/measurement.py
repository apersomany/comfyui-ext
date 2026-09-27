import time
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from bench.devices import MemoryMeasurement
from bench.exceptions import BenchmarkError
from bench.profiling import ProfileEntry, create_profiler, summarize_profile


@dataclass(frozen=True)
class Measurement:
    durations: list[float]
    observed_evaluations: int
    memory: MemoryMeasurement
    profile_entries: list[ProfileEntry] | None


class EvaluationTimer:
    def __init__(self, device_controller, configuration, reporter, task):
        self.device_controller = device_controller
        self.configuration = configuration
        self.reporter = reporter
        self.task = task
        self.durations = []
        self.evaluations = 0
        self.enabled = True

    def __call__(self, executor, *arguments, **keywords):
        if (
            self.configuration.tunable_file is not None
            and self.evaluations == self.configuration.warmup_evaluations
        ):
            torch.cuda.tunable.tuning_enable(False)
        self.device_controller.synchronize()
        start = time.perf_counter()
        output = executor(*arguments, **keywords)
        self.device_controller.synchronize()
        if self.enabled:
            self.durations.append(time.perf_counter() - start)
            self.evaluations += 1
            self.reporter.update(self.task, advance=1)
        return output


def measure_sampling(
    patcher,
    configuration,
    conditioning,
    latent,
    device_controller,
    reporter,
    run_workflow,
):
    import comfy.patcher_extension

    task = reporter.add_task(
        "Benchmarking model evaluations", total=configuration.steps
    )
    timer = EvaluationTimer(device_controller, configuration, reporter, task)
    patcher.add_wrapper(comfy.patcher_extension.WrappersMP.APPLY_MODEL, timer)
    baseline = device_controller.allocated_memory()
    device_controller.reset_peak_memory()
    run_workflow(
        patcher,
        configuration,
        conditioning.positive,
        conditioning.negative,
        latent,
    )
    timer.enabled = False
    reporter.finish(task)
    durations = timer.durations[configuration.warmup_evaluations :]
    if not durations:
        raise BenchmarkError(
            f"sampling produced no evaluations after {configuration.warmup_evaluations} warmup evaluations"
        )
    memory = device_controller.memory_measurement(baseline)
    return durations, timer.evaluations, memory


def measure_iterations(operation, configuration, device_controller, reporter):
    task = reporter.add_task("Benchmarking attention", total=configuration.steps)
    durations = []
    baseline = device_controller.allocated_memory()
    device_controller.reset_peak_memory()
    for evaluation in range(configuration.steps):
        if (
            configuration.tunable_file is not None
            and evaluation == configuration.warmup_evaluations
        ):
            torch.cuda.tunable.tuning_enable(False)
        device_controller.synchronize()
        start = time.perf_counter()
        output = operation()
        device_controller.synchronize()
        duration = time.perf_counter() - start
        if evaluation >= configuration.warmup_evaluations:
            durations.append(duration)
        del output
        reporter.update(task, advance=1)
    reporter.finish(task)
    if not durations:
        raise BenchmarkError(
            f"benchmark produced no evaluations after {configuration.warmup_evaluations} warmup evaluations"
        )
    return (
        durations,
        configuration.steps,
        device_controller.memory_measurement(baseline),
    )


def profile_operation(
    operation,
    attention_flop_counter,
    device_controller,
    reporter,
    record_as_sdpa=False,
):
    task = reporter.add_task("Profiling operations", total=1)
    profiler = create_profiler(device_controller)
    profiler.start()
    attention_flop_counter.enabled = True
    recording = (
        torch.profiler.record_function("sdpa") if record_as_sdpa else nullcontext()
    )
    try:
        with recording:
            operation()
        device_controller.synchronize()
    finally:
        attention_flop_counter.enabled = False
        profiler.stop()
        reporter.update(task, advance=1)
        reporter.finish(task)
    return summarize_profile(profiler, attention_flop_counter.total)
