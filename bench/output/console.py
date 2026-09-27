from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

CATEGORY_STYLES = {
    "gemm": "blue",
    "sdpa": "magenta",
    "other": "dim",
}


def format_count(value):
    for threshold, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if value >= threshold:
            return f"{value / threshold:.2f}{suffix}"
    return f"{value:,}"


def format_bytes(value):
    if value is None:
        return "unavailable"
    for threshold, suffix in (
        (1 << 40, "TiB"),
        (1 << 30, "GiB"),
        (1 << 20, "MiB"),
        (1 << 10, "KiB"),
    ):
        if value >= threshold:
            return f"{value / threshold:.2f} {suffix}"
    return f"{value} B"


def format_shape(shape):
    return "×".join(str(dimension) for dimension in shape)


def detail_table(result):
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right")
    table.add_column()
    runtime = result.runtime
    compile_description = "disabled"
    if runtime.compile is not None:
        compile_description = runtime.compile.backend
        if runtime.compile.mode is not None:
            compile_description += f" · {runtime.compile.mode}"
    table.add_row(
        "Model",
        f"{result.model.architecture} · {format_count(result.model.parameters)} · "
        f"{result.model.data_type} · synthetic weights",
    )
    accelerator = result.environment.accelerator_runtime
    device_description = result.environment.device.name
    if accelerator:
        device_description += f" · {accelerator}"
    table.add_row("Device", device_description)
    image = result.workload.image
    sampling = result.workload.sampling
    table.add_row(
        "Workload",
        f"{result.workload.target.upper()} · {image.width}×{image.height} · batch {image.batch_size} · "
        f"{sampling.steps} steps",
    )
    table.add_row(
        "Runtime",
        f"{runtime.attention} attention · compile {compile_description}",
    )
    conditioning = result.workload.conditioning
    context = f"{conditioning.token_count} tokens × {conditioning.token_width} width"
    if conditioning.pooled_width is not None:
        context += f" · pooled {conditioning.pooled_width}"
    table.add_row("Conditioning", context)
    if result.workload.sdpa is not None:
        sdpa = result.workload.sdpa
        table.add_row(
            "Attention",
            f"{sdpa.attention_function} · Q {format_shape(sdpa.query_shape)} · "
            f"K {format_shape(sdpa.key_shape)} · {sdpa.heads} heads",
        )
    return table


def performance_table(result):
    latency = result.measurements.latency
    memory = result.measurements.memory
    table = Table(title="Performance", box=None, header_style="bold cyan", expand=True)
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_column("Detail", style="dim")
    table.add_row(
        "Median latency",
        f"[bold cyan]{latency.median * 1000:.3f} ms[/]",
        f"{result.measurements.timed_evaluations} measured evaluations",
    )
    table.add_row(
        "Latency range",
        f"{latency.minimum * 1000:.3f}–{latency.maximum * 1000:.3f} ms",
        f"mean {latency.mean * 1000:.3f} ms",
    )
    table.add_row(
        "Throughput",
        f"[bold cyan]{result.measurements.evaluations_per_second:.3f}/s[/]",
        f"{result.measurements.requested_warmup_evaluations} warmup evaluations",
    )
    if memory.peak_allocated_bytes is not None:
        table.add_row(
            "Peak allocated memory",
            format_bytes(memory.peak_allocated_bytes),
            f"incremental {format_bytes(memory.incremental_peak_bytes)}",
        )
    return table


def profile_table(result, profile_mode):
    profile = result.measurements.profile
    if profile is None:
        return None
    table = Table(title="Profile", box=None, header_style="bold cyan", expand=True)
    table.add_column("Category")
    table.add_column("Share")
    table.add_column("Duration", justify="right")
    table.add_column("Throughput", justify="right")
    for bucket in profile.buckets:
        filled = min(20, max(0, round(bucket.share * 20)))
        bar = Text(
            "█" * filled + "░" * (20 - filled), style=CATEGORY_STYLES[bucket.category]
        )
        share = Text.assemble(bar, f" {bucket.share * 100:.1f}%")
        throughput = "" if bucket.tflops is None else f"{bucket.tflops:.2f} TFLOP/s"
        table.add_row(
            bucket.category.upper(),
            share,
            f"{bucket.duration_seconds * 1000:.3f} ms",
            throughput,
            style=CATEGORY_STYLES[bucket.category],
        )
    if profile_mode != "full":
        return table
    operations = Table(
        title="Profile operations", box=None, header_style="bold cyan", expand=True
    )
    operations.add_column("Category")
    operations.add_column("Operation", overflow="fold")
    operations.add_column("Average", justify="right")
    operations.add_column("Calls", justify="right")
    operations.add_column("Total", justify="right")
    for operation in profile.operations:
        operations.add_row(
            operation.category.upper(),
            operation.name,
            f"{operation.duration_seconds / operation.calls * 1000:.3f} ms",
            str(operation.calls),
            f"{operation.duration_seconds * 1000:.3f} ms",
            style=CATEGORY_STYLES[operation.category],
        )
    return Group(table, "", operations)


def render_console(result, console, profile_mode):
    console.print(
        Panel(detail_table(result), title="[bold cyan]Comfy Bench[/]", expand=False)
    )
    console.print(performance_table(result))
    profile = profile_table(result, profile_mode)
    if profile is not None:
        console.print(profile)
