import torch

from bench.comfy_adapter import configure_tunable, run_workflow
from bench.exceptions import BenchmarkError
from bench.measurement import Measurement, measure_iterations, profile_operation
from bench.profiling import AttentionFlopCounter
from bench.targets.base import TargetOutcome


def attention_placeholder(shape, arguments):
    query = arguments[0]
    heads = shape["heads"]
    if shape["skip_reshape"]:
        if query.ndim != 4:
            return None
        if shape["skip_output_reshape"]:
            return query.new_zeros(query.shape)
        return query.new_zeros((query.shape[0], query.shape[2], heads * query.shape[3]))
    if query.ndim != 3:
        return None
    if shape["skip_output_reshape"]:
        if query.shape[-1] % heads:
            return None
        return query.new_zeros(
            (query.shape[0], heads, query.shape[1], query.shape[-1] // heads)
        )
    return query.new_zeros(query.shape)


def create_shape_extraction_override(counter):
    def override(function, *arguments, **keywords):
        shape = counter.record(arguments, keywords, function)
        if shape is None:
            return function(*arguments, **keywords)
        output = attention_placeholder(shape, arguments)
        if output is None:
            return function(*arguments, **keywords)
        return output

    return override


def extract_sdpa_shape(patcher, configuration, conditioning, latent, reporter):
    counter = AttentionFlopCounter()
    counter.enabled = True
    counter.capture_shapes = True
    transformer_options = patcher.model_options["transformer_options"]
    previous_override = transformer_options.get("optimized_attention_override")
    transformer_options["optimized_attention_override"] = (
        create_shape_extraction_override(counter)
    )
    task = reporter.add_task("Extracting attention shapes", total=1)
    try:
        run_workflow(
            patcher,
            configuration,
            conditioning.positive,
            conditioning.negative,
            latent,
            steps=1,
        )
    finally:
        reporter.update(task, advance=1)
        reporter.finish(task)
        if previous_override is None:
            transformer_options.pop("optimized_attention_override", None)
        else:
            transformer_options["optimized_attention_override"] = previous_override
    if not counter.shapes:
        raise BenchmarkError("no compatible SDPA call was observed")
    return max(counter.shapes, key=lambda shape: shape["flops"])


def create_attention_tensor(shape, data_type, device):
    if data_type in (torch.float64, torch.float32, torch.float16, torch.bfloat16):
        return torch.randn(shape, device=device, dtype=data_type)
    return torch.randn(shape, device=device, dtype=torch.float32).to(data_type)


def create_attention_operation(shape, configuration, device):
    query = create_attention_tensor(shape["query_shape"], shape["query_dtype"], device)
    key = create_attention_tensor(shape["key_shape"], shape["key_dtype"], device)
    value = create_attention_tensor(shape["value_shape"], shape["value_dtype"], device)
    mask = None
    if shape["mask_shape"] is not None:
        if shape["mask_dtype"] == torch.bool:
            mask = torch.ones(shape["mask_shape"], device=device, dtype=torch.bool)
        else:
            mask = torch.zeros(
                shape["mask_shape"], device=device, dtype=shape["mask_dtype"]
            )
    attention_keywords = {
        "mask": mask,
        "skip_reshape": shape["skip_reshape"],
        "skip_output_reshape": shape["skip_output_reshape"],
    }
    if shape["attn_precision"] is not None:
        attention_keywords["attn_precision"] = shape["attn_precision"]
    if shape["scale"] is not None:
        attention_keywords["scale"] = shape["scale"]
    if shape["enable_gqa"]:
        attention_keywords["enable_gqa"] = True
    if shape["low_precision_attention"] is not None:
        attention_keywords["low_precision_attention"] = shape["low_precision_attention"]

    def run_attention(query_tensor, key_tensor, value_tensor):
        return shape["function"](
            query_tensor,
            key_tensor,
            value_tensor,
            shape["heads"],
            **attention_keywords,
        )

    attention_runner = run_attention
    if configuration.compile_backend is not None:
        compile_backend = (
            None
            if configuration.compile_backend == "default"
            else configuration.compile_backend
        )
        attention_runner = torch.compile(
            run_attention,
            backend=compile_backend,
            mode=configuration.compile_mode,
        )

    def operation():
        return attention_runner(query, key, value)

    return operation


def describe_shape(shape):
    attention_function = shape["function"]
    return {
        "attention_function": getattr(
            attention_function,
            "__name__",
            type(attention_function).__name__,
        ),
        "query_shape": list(shape["query_shape"]),
        "key_shape": list(shape["key_shape"]),
        "value_shape": list(shape["value_shape"]),
        "query_dtype": str(shape["query_dtype"]).removeprefix("torch."),
        "key_dtype": str(shape["key_dtype"]).removeprefix("torch."),
        "value_dtype": str(shape["value_dtype"]).removeprefix("torch."),
        "heads": shape["heads"],
        "mask_shape": list(shape["mask_shape"])
        if shape["mask_shape"] is not None
        else None,
        "mask_dtype": str(shape["mask_dtype"]).removeprefix("torch.")
        if shape["mask_dtype"] is not None
        else None,
        "attention_flops": shape["flops"],
        "skip_reshape": shape["skip_reshape"],
        "skip_output_reshape": shape["skip_output_reshape"],
    }


def run_sdpa_target(
    patcher,
    configuration,
    conditioning,
    latent,
    device_controller,
    reporter,
    lifecycle,
):
    shape = extract_sdpa_shape(patcher, configuration, conditioning, latent, reporter)
    lifecycle.detach()
    configure_tunable(configuration, device_controller.device)
    operation = create_attention_operation(
        shape, configuration, device_controller.device
    )
    durations, evaluations, memory = measure_iterations(
        operation,
        configuration,
        device_controller,
        reporter,
    )
    profile_entries = None
    if configuration.profile_mode is not None:
        counter = AttentionFlopCounter()
        counter.total = shape["flops"]
        profile_entries = profile_operation(
            operation,
            counter,
            device_controller,
            reporter,
            record_as_sdpa=True,
        )
    measurement = Measurement(durations, evaluations, memory, profile_entries)
    return TargetOutcome(measurement, describe_shape(shape), True)
