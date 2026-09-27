import argparse
import math
import os
import sys
from pathlib import Path

from bench.configuration import BenchmarkConfiguration, OutputConfiguration
from bench.exceptions import BenchmarkError

DESCRIPTION = (
    "Benchmark a diffusion model through ComfyUI sampling using synthetic weights "
    "built from a safetensors header."
)
ATTENTION_CHOICES = ("auto", "split", "quad", "pytorch", "flex", "sage", "flash", "ck")


def create_parser():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("model", help="path or HTTP URL to a safetensors checkpoint")
    parser.add_argument("--width", type=int, default=512, help="image width in pixels")
    parser.add_argument(
        "--height", type=int, default=512, help="image height in pixels"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="latent batch size")
    parser.add_argument(
        "--context-length", type=int, default=64, help="conditioning token count"
    )
    parser.add_argument(
        "--steps", type=int, default=20, help="denoising steps or SDPA evaluations"
    )
    parser.add_argument(
        "--cfg", type=float, default=1.0, help="classifier free guidance scale"
    )
    parser.add_argument("--sampler-name", default="euler", metavar="NAME")
    parser.add_argument("--scheduler", default="normal", metavar="NAME")
    parser.add_argument(
        "--denoise", type=float, default=1.0, help="amount of denoising applied"
    )
    parser.add_argument("--seed", type=int, default=0, help="noise seed")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1,
        help="leading evaluations excluded from timing",
    )
    parser.add_argument(
        "--dtype", choices=("auto", "fp32", "fp16", "bf16"), default="auto"
    )
    parser.add_argument(
        "--device", default=None, help="override device, such as cpu or cuda:1"
    )
    parser.add_argument(
        "--attention",
        choices=ATTENTION_CHOICES,
        default="auto",
        help="attention implementation",
    )
    parser.add_argument(
        "--target",
        choices=("full", "sdpa"),
        default="full",
        help="benchmark the model or an extracted attention shape",
    )
    parser.add_argument(
        "--compile",
        nargs="?",
        const="default",
        default=None,
        metavar="BACKEND",
        help="compile the benchmark target",
    )
    parser.add_argument(
        "--compile-mode", default=None, metavar="MODE", help="torch.compile mode"
    )
    parser.add_argument(
        "--tunable",
        nargs="?",
        const="tunableop_results.csv",
        default=None,
        metavar="FILE",
        help="enable ROCm TunableOp",
    )
    parser.add_argument(
        "--profile",
        nargs="?",
        const="simple",
        choices=("simple", "full"),
        default=None,
        help="profile accelerator operations",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=("console", "json", "markdown"),
        default="console",
        help="result output format",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="write results to a file"
    )
    parser.add_argument(
        "--color",
        choices=("auto", "always", "never"),
        default="auto",
        help="control terminal color",
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="disable progress display"
    )
    parser.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    return parser


def validate_arguments(arguments, bootstrap_state, compile_backends):
    if arguments.attention not in bootstrap_state.attention_choices:
        choices = ", ".join(bootstrap_state.attention_choices)
        raise BenchmarkError(
            f"invalid attention implementation {arguments.attention!r}; choose from {choices}"
        )
    if arguments.compile is not None and arguments.compile not in (
        "default",
        *compile_backends,
    ):
        raise BenchmarkError(f"invalid torch.compile backend {arguments.compile!r}")
    if arguments.width < 16 or arguments.width % 8:
        raise BenchmarkError("width must be at least 16 and divisible by 8")
    if arguments.height < 16 or arguments.height % 8:
        raise BenchmarkError("height must be at least 16 and divisible by 8")
    if arguments.batch_size < 1:
        raise BenchmarkError("batch size must be at least 1")
    if arguments.context_length < 1:
        raise BenchmarkError("context length must be at least 1")
    if arguments.steps < 1:
        raise BenchmarkError("steps must be at least 1")
    if not math.isfinite(arguments.cfg) or arguments.cfg < 0:
        raise BenchmarkError("classifier free guidance must be finite and nonnegative")
    if not 0 <= arguments.denoise <= 1:
        raise BenchmarkError("denoise must be between 0 and 1")
    if not 0 <= arguments.seed <= 0xFFFFFFFFFFFFFFFF:
        raise BenchmarkError("seed must be between 0 and 18446744073709551615")
    if not 0 <= arguments.warmup_steps < arguments.steps:
        raise BenchmarkError("warmup steps must be nonnegative and less than steps")
    if arguments.tunable is not None and arguments.warmup_steps == 0:
        raise BenchmarkError("TunableOp requires at least one warmup evaluation")
    if arguments.compile_mode is not None and arguments.compile is None:
        arguments.compile = "default"
    if arguments.json and arguments.output_format != "console":
        raise BenchmarkError(
            "the legacy --json option cannot be combined with --format"
        )
    if arguments.json:
        arguments.output_format = "json"


def create_configuration(arguments):
    benchmark = BenchmarkConfiguration(
        model=arguments.model,
        width=arguments.width,
        height=arguments.height,
        batch_size=arguments.batch_size,
        context_length=arguments.context_length,
        steps=arguments.steps,
        classifier_free_guidance=arguments.cfg,
        sampler_name=arguments.sampler_name,
        scheduler=arguments.scheduler,
        denoise=arguments.denoise,
        seed=arguments.seed,
        warmup_evaluations=arguments.warmup_steps,
        data_type=arguments.dtype,
        device=arguments.device,
        attention=arguments.attention,
        target=arguments.target,
        compile_backend=arguments.compile,
        compile_mode=arguments.compile_mode,
        tunable_file=arguments.tunable,
        profile_mode=arguments.profile,
    )
    output = OutputConfiguration(
        format=arguments.output_format,
        color=arguments.color,
        path=arguments.output,
        progress=not arguments.no_progress and arguments.output_format != "json",
    )
    return benchmark, output


def create_console(stream, color):
    from rich.console import Console

    force_terminal = None
    if color == "always":
        force_terminal = True
    elif color == "never":
        force_terminal = False
    no_color = color == "never" or "NO_COLOR" in os.environ
    return Console(file=stream, force_terminal=force_terminal, no_color=no_color)


def main():
    os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
    parser = create_parser()
    arguments = parser.parse_args()
    output_stream = None
    try:
        if (
            not arguments.model.startswith(("http://", "https://"))
            and not Path(arguments.model).exists()
        ):
            raise BenchmarkError(f"model not found: {arguments.model}")
        from bench.bootstrap import configure_attention, prepare_comfyui

        bootstrap_state = prepare_comfyui()
        configure_attention(bootstrap_state, arguments.attention)
        import comfy.utils
        import torch

        if arguments.compile == "migraphx":
            import torch_migraphx

        validate_arguments(
            arguments,
            bootstrap_state,
            tuple(torch.compiler.list_backends()),
        )
        benchmark, output = create_configuration(arguments)
        from bench.comfy_adapter import validate_comfy_choices
        from bench.progress import ProgressReporter
        from bench.runner import run_benchmark

        validate_comfy_choices(benchmark)
        comfy.utils.set_progress_bar_enabled(False)
        with ProgressReporter(output.progress) as reporter:
            result = run_benchmark(benchmark, reporter)
        if output.path is None:
            output_stream = sys.stdout
        else:
            output.path.parent.mkdir(parents=True, exist_ok=True)
            output_stream = output.path.open("w", encoding="utf-8")
        if output.format == "console":
            from bench.output.console import render_console

            render_console(
                result,
                create_console(output_stream, output.color),
                benchmark.profile_mode,
            )
        elif output.format == "markdown":
            from bench.output.markdown import render_markdown

            output_stream.write(render_markdown(result, benchmark.profile_mode))
        else:
            from bench.output.json import render_json

            output_stream.write(render_json(result) + "\n")
    except BenchmarkError as error:
        from rich.console import Console

        Console(stderr=True).print(f"[bold red]error:[/] {error}")
        raise SystemExit(2) from error
    finally:
        if output_stream is not None and output_stream is not sys.stdout:
            output_stream.close()
