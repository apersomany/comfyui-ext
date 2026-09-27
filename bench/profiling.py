import math
from dataclasses import dataclass

import torch

GEMM_OPERATION_MARKERS = ("mm", "matmul", "linear", "conv", "gemm", "gemv")
PROFILE_BUCKETS = (("gemm", "GEMM"), ("sdpa", "SDPA"), ("other", "Other"))


@dataclass
class ProfileEntry:
    category: str
    name: str
    duration_seconds: float
    calls: int
    flops: float


class AttentionFlopCounter:
    def __init__(self):
        self.total = 0
        self.enabled = False
        self.capture_shapes = False
        self.shapes = []

    def record(self, arguments, keywords=None, function=None):
        if not self.enabled and not self.capture_shapes:
            return None
        if len(arguments) < 2:
            return None
        query = arguments[0]
        key = arguments[1]
        if not isinstance(query, torch.Tensor) or not isinstance(key, torch.Tensor):
            return None
        if query.ndim not in (3, 4) or key.ndim != query.ndim:
            return None
        key_count = key.shape[query.ndim - 2]
        flops = 4 * query.numel() * key_count
        if self.enabled:
            self.total += flops
        if not self.capture_shapes or len(arguments) < 3:
            return None
        value = arguments[2]
        if not isinstance(value, torch.Tensor):
            return None
        keywords = {} if keywords is None else keywords
        heads = (
            arguments[3]
            if len(arguments) > 3
            else keywords.get("heads", keywords.get("num_heads"))
        )
        if not isinstance(heads, int) or heads <= 0:
            return None
        mask = arguments[4] if len(arguments) > 4 else keywords.get("mask")
        attention_precision = (
            arguments[5] if len(arguments) > 5 else keywords.get("attn_precision")
        )
        skip_reshape = (
            arguments[6] if len(arguments) > 6 else keywords.get("skip_reshape", False)
        )
        skip_output_reshape = (
            arguments[7]
            if len(arguments) > 7
            else keywords.get("skip_output_reshape", False)
        )
        shape = {
            "function": function,
            "query_shape": tuple(query.shape),
            "key_shape": tuple(key.shape),
            "value_shape": tuple(value.shape),
            "query_dtype": query.dtype,
            "key_dtype": key.dtype,
            "value_dtype": value.dtype,
            "heads": heads,
            "mask_shape": tuple(mask.shape) if isinstance(mask, torch.Tensor) else None,
            "mask_dtype": mask.dtype if isinstance(mask, torch.Tensor) else None,
            "attn_precision": attention_precision,
            "skip_reshape": bool(skip_reshape),
            "skip_output_reshape": bool(skip_output_reshape),
            "scale": keywords.get("scale"),
            "enable_gqa": bool(keywords.get("enable_gqa", False)),
            "low_precision_attention": keywords.get("low_precision_attention"),
            "flops": flops,
        }
        self.shapes.append(shape)
        return shape


def create_attention_override(counter):
    def override(function, *arguments, **keywords):
        counter.record(arguments, keywords, function)
        with torch.profiler.record_function("sdpa"):
            return function(*arguments, **keywords)

    return override


def is_gemm(name):
    lowered_name = name.lower()
    return any(marker in lowered_name for marker in GEMM_OPERATION_MARKERS)


def is_numeric_shape(shape):
    return (
        isinstance(shape, list)
        and len(shape) > 0
        and all(isinstance(dimension, int) for dimension in shape)
    )


def estimate_flops(name, shapes):
    if not shapes:
        return None
    operation = name.split("::")[-1]
    if (
        operation == "linear"
        and len(shapes) >= 2
        and is_numeric_shape(shapes[0])
        and is_numeric_shape(shapes[1])
    ):
        input_shape, weight_shape = shapes[0], shapes[1]
        return 2 * math.prod(input_shape[:-1]) * input_shape[-1] * weight_shape[0]
    if operation in (
        "mm",
        "addmm",
        "bmm",
        "addbmm",
        "baddbmm",
        "matmul",
        "_scaled_mm",
        "_int_mm",
    ):
        operands = [
            shape for shape in shapes if is_numeric_shape(shape) and len(shape) >= 2
        ]
        if len(operands) >= 2:
            left_operand, right_operand = operands[-2], operands[-1]
            return (
                2
                * math.prod(left_operand[:-2])
                * left_operand[-2]
                * left_operand[-1]
                * right_operand[-1]
            )
    return None


def create_profiler(device_controller):
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            device_controller.profiler_activity(),
        ],
        record_shapes=True,
        with_flops=True,
    )


def summarize_profile(profiler, attention_flop_count):
    entries = {}

    def walk(event, inside_sdpa):
        if event.is_user_annotation:
            inside_sdpa = inside_sdpa or event.name == "sdpa"
        elif (
            (event.kernels or event.cpu_children)
            and event.self_device_time_total > 0
            and not event.name.startswith("hip")
        ):
            category = (
                "sdpa" if inside_sdpa else ("gemm" if is_gemm(event.name) else "other")
            )
            key = (category, event.name)
            entry = entries.setdefault(
                key, ProfileEntry(category, event.name, 0.0, 0, 0)
            )
            entry.duration_seconds += event.self_device_time_total / 1e6
            entry.calls += 1
            if category != "sdpa":
                per_call = event.flops or estimate_flops(event.name, event.input_shapes)
                if per_call:
                    entry.flops += per_call
        for child in event.cpu_children:
            walk(child, inside_sdpa)

    for event in profiler.events():
        if event.cpu_parent is None:
            walk(event, False)
    sdpa_entries = [entry for entry in entries.values() if entry.category == "sdpa"]
    sdpa_duration = sum(entry.duration_seconds for entry in sdpa_entries)
    if sdpa_duration > 0:
        for entry in sdpa_entries:
            entry.flops = attention_flop_count * entry.duration_seconds / sdpa_duration
    return sorted(
        entries.values(),
        key=lambda entry: (-entry.duration_seconds, entry.category, entry.name),
    )


def summarize_buckets(entries):
    buckets = {
        name: {"duration_seconds": 0.0, "flops": 0.0} for name, _ in PROFILE_BUCKETS
    }
    for entry in entries:
        bucket = buckets[entry.category]
        bucket["duration_seconds"] += entry.duration_seconds
        bucket["flops"] += entry.flops
    return buckets
