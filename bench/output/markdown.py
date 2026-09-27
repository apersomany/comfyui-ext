from bench.output.console import format_bytes, format_count, format_shape


def field(label, value):
    return f"**{label}:** {value}"


def render_markdown(result, profile_mode):
    image = result.workload.image
    sampling = result.workload.sampling
    latency = result.measurements.latency
    memory = result.measurements.memory
    lines = [
        "# Comfy Bench",
        "",
        "## Configuration",
        "",
        field(
            "Model",
            f"{result.model.architecture} · {format_count(result.model.parameters)} · "
            f"{result.model.data_type}",
        ),
        "",
        field("Device", result.environment.device.name),
        "",
        field("Target", result.workload.target.upper()),
        "",
        field("Image", f"{image.width}×{image.height} · batch {image.batch_size}"),
        "",
        field(
            "Sampling",
            f"{sampling.steps} steps · {sampling.sampler} · {sampling.scheduler}",
        ),
        "",
        field("Attention", result.runtime.attention),
    ]
    if result.workload.sdpa is not None:
        sdpa = result.workload.sdpa
        lines.extend(
            [
                "",
                field(
                    "Query",
                    f"{format_shape(sdpa.query_shape)} · {sdpa.query_data_type}",
                ),
                "",
                field("Key", f"{format_shape(sdpa.key_shape)} · {sdpa.key_data_type}"),
                "",
                field(
                    "Value",
                    f"{format_shape(sdpa.value_shape)} · {sdpa.value_data_type}",
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Performance",
            "",
            field("Median latency", f"{latency.median * 1000:.3f} ms"),
            "",
            field(
                "Latency range",
                f"{latency.minimum * 1000:.3f}–{latency.maximum * 1000:.3f} ms",
            ),
            "",
            field(
                "Throughput",
                f"{result.measurements.evaluations_per_second:.3f} evaluations/s",
            ),
            "",
            field("Timed evaluations", str(result.measurements.timed_evaluations)),
        ]
    )
    if memory.peak_allocated_bytes is not None:
        lines.extend(
            [
                "",
                field(
                    "Peak allocated memory", format_bytes(memory.peak_allocated_bytes)
                ),
                "",
                field(
                    "Incremental peak memory",
                    format_bytes(memory.incremental_peak_bytes),
                ),
            ]
        )
    profile = result.measurements.profile
    if profile is not None:
        lines.extend(["", "## Profile"])
        for bucket in profile.buckets:
            throughput = (
                "" if bucket.tflops is None else f" · {bucket.tflops:.2f} TFLOP/s"
            )
            lines.extend(
                [
                    "",
                    field(
                        bucket.category.upper(),
                        f"{bucket.share * 100:.1f}% · "
                        f"{bucket.duration_seconds * 1000:.3f} ms{throughput}",
                    ),
                ]
            )
        if profile_mode == "full":
            lines.extend(["", "### Operations"])
            for operation in profile.operations:
                lines.extend(
                    [
                        "",
                        field(
                            operation.name,
                            f"{operation.category.upper()} · {operation.calls} calls · "
                            f"{operation.duration_seconds * 1000:.3f} ms",
                        ),
                    ]
                )
    return "\n".join(lines) + "\n"
