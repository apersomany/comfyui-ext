from dataclasses import dataclass

import torch

from bench.exceptions import BenchmarkError


@dataclass(frozen=True)
class MemoryMeasurement:
    baseline_allocated_bytes: int | None
    peak_allocated_bytes: int | None
    incremental_peak_bytes: int | None


class DeviceController:
    def __init__(self, device):
        self.device = device

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "xpu":
            torch.xpu.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def allocated_memory(self):
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated(self.device)
        if self.device.type == "xpu":
            return torch.xpu.memory_allocated(self.device)
        return None

    def reset_peak_memory(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        elif self.device.type == "xpu":
            torch.xpu.reset_peak_memory_stats(self.device)

    def peak_memory(self):
        if self.device.type == "cuda":
            return torch.cuda.max_memory_allocated(self.device)
        if self.device.type == "xpu":
            return torch.xpu.max_memory_allocated(self.device)
        return None

    def memory_measurement(self, baseline):
        peak = self.peak_memory()
        incremental = (
            None if baseline is None or peak is None else max(0, peak - baseline)
        )
        return MemoryMeasurement(baseline, peak, incremental)

    def profiler_activity(self):
        if self.device.type == "cuda":
            return torch.profiler.ProfilerActivity.CUDA
        if self.device.type == "xpu":
            return torch.profiler.ProfilerActivity.XPU
        raise BenchmarkError("profiling requires a CUDA or XPU device")
