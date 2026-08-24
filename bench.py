import argparse
import http.client
import json
import math
import os
import statistics
import struct
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "comfyui"))

import comfy.cli_args


def attention_options():
    options = {}
    for action in comfy.cli_args.attn_group._group_actions:
        name = action.dest.removeprefix("use_").removesuffix("_attention").removesuffix("_cross")
        options[name.replace("_", "-")] = action.dest
    return options


ATTENTION_OPTIONS = attention_options()
ATTENTION_CHOICES = ("auto", *ATTENTION_OPTIONS)


def extract_attention():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--attention", choices=ATTENTION_CHOICES, default="auto")
    arguments, remaining_arguments = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining_arguments]
    return arguments.attention


def configure_attention(attention):
    if attention == "auto":
        return
    setattr(comfy.cli_args.args, ATTENTION_OPTIONS[attention], True)
    comfy.cli_args.args.disable_xformers = True


SELECTED_ATTENTION = extract_attention()
configure_attention(SELECTED_ATTENTION)

DESCRIPTION = "Benchmark a diffusion model through ComfyUI sampling using synthetic weights built from the safetensors header."

SAFETENSORS_DTYPES = None

METADATA_TENSOR_NAMES = ("comfy_quant",)

CONTEXT_DIMENSION_KEYS = ("caption_channels", "context_dim", "cross_attention_dim", "text_dim",
                          "cap_feat_dim", "encoder_hidden_states_dim", "c_dim")

KEY_PROJECTION_NAMES = ("k_proj", "to_k", "k", "kv_proj", "to_kv", "c_kv", "k_norm")

TEXT_EMBEDDER_NAMES = ("cap_embedder", "context_embedder", "txt_in", "caption_projection",
                       "text_embedder", "y_embedder", "context_refiner", "encoder_hid_proj")

POOLED_EMBEDDER_NAMES = ("vector_in", "y_embedder", "pooled_embedder")

DATA_TYPES = None

COMPILE_BACKENDS = ()

GEMM_OPERATION_MARKERS = ("mm", "matmul", "linear", "conv", "gemm", "gemv")

PROFILE_BUCKETS = (("gemm", "GEMM"), ("sdpa", "SDPA"), ("other", "OTHER"))

METADATA_READ_GAP = 1 << 20

MAXIMUM_HEADER_LENGTH = 64 << 20

ANSI_STYLE_CODES = {
    "heading": "1;36",
    "header": "1",
    "label": "2",
    "emphasis": "1;36",
    "gemm": "34",
    "sdpa": "35",
    "other": "2",
}


def load_dependencies():
    global torch, tqdm, comfy, nodes, SAFETENSORS_DTYPES, DATA_TYPES, COMPILE_BACKENDS
    import torch
    from tqdm.auto import tqdm
    import comfy.model_management
    import comfy.patcher_extension
    import comfy.samplers
    import comfy.sd
    import comfy.utils
    import nodes

    SAFETENSORS_DTYPES = {
        "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
        "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32,
        "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
        "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
    }
    DATA_TYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    COMPILE_BACKENDS = tuple(torch.compiler.list_backends())


def parse_arguments():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("model", help="path or http(s) URL to a .safetensors checkpoint")
    parser.add_argument("--width", type=int, default=512, help="image width in pixels")
    parser.add_argument("--height", type=int, default=512, help="image height in pixels")
    parser.add_argument("--batch-size", type=int, default=1, help="latent batch size")
    parser.add_argument("--context-length", type=int, default=64, help="number of text tokens in the conditioning")
    parser.add_argument("--steps", type=int, default=20, help="number of denoising steps")
    parser.add_argument("--cfg", type=float, default=1.0, help="classifier free guidance scale")
    parser.add_argument("--sampler-name", default="euler", metavar="NAME")
    parser.add_argument("--scheduler", default="normal", metavar="NAME")
    parser.add_argument("--denoise", type=float, default=1.0, help="amount of denoising applied")
    parser.add_argument("--seed", type=int, default=0, help="seed used for creating the noise")
    parser.add_argument("--warmup-steps", type=int, default=1, help="leading model evaluations excluded from the timings")
    parser.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto")
    parser.add_argument("--device", default=None, help="override device, such as cpu, cuda, cuda:1, xpu or mps")
    parser.add_argument("--attention", choices=ATTENTION_CHOICES, default=SELECTED_ATTENTION, help="attention implementation")
    parser.add_argument("--target", choices=("full", "sdpa"), default="full", help="benchmark the full model or an extracted attention shape")
    parser.add_argument("--compile", nargs="?", const="default", default=None, metavar="BACKEND", help="compile the benchmark target with torch.compile; use its default backend when none is given")
    parser.add_argument("--compile-mode", default=None, metavar="MODE", help="torch.compile mode; specifying a mode enables compilation with its default backend")
    parser.add_argument("--tunable", nargs="?", const="tunableop_results.csv", default=None, metavar="FILE", help="use ROCm TunableOp and tune missing GEMMs during at least one warmup step")
    parser.add_argument("--profile", nargs="?", const="simple", choices=("simple", "full"), default=None, help="profile gpu time split between gemm, sdpa and other ops; 'full' also lists every captured op (adds overhead)")
    parser.add_argument("--color", choices=("auto", "always", "never"), default="auto", help="control ANSI color in text output")
    parser.add_argument("--json", action="store_true", help="print a single json object and suppress progress bars")
    return parser.parse_args()


def validate_arguments(arguments):
    if arguments.sampler_name not in comfy.samplers.KSampler.SAMPLERS:
        raise SystemExit(f"error: argument --sampler-name: invalid choice: {arguments.sampler_name!r}")
    if arguments.scheduler not in comfy.samplers.KSampler.SCHEDULERS:
        raise SystemExit(f"error: argument --scheduler: invalid choice: {arguments.scheduler!r}")
    if arguments.compile is not None and arguments.compile not in ("default", *COMPILE_BACKENDS):
        raise SystemExit(f"error: argument --compile: invalid choice: {arguments.compile!r}")
    if arguments.width < 16 or arguments.width % 8:
        raise SystemExit("error: --width must be at least 16 and divisible by 8")
    if arguments.height < 16 or arguments.height % 8:
        raise SystemExit("error: --height must be at least 16 and divisible by 8")
    if arguments.batch_size < 1:
        raise SystemExit("error: --batch-size must be at least 1")
    if arguments.context_length < 1:
        raise SystemExit("error: --context-length must be at least 1")
    if arguments.steps < 1:
        raise SystemExit("error: --steps must be at least 1")
    if not math.isfinite(arguments.cfg) or arguments.cfg < 0:
        raise SystemExit("error: --cfg must be finite and not negative")
    if not 0 <= arguments.denoise <= 1:
        raise SystemExit("error: --denoise must be between 0 and 1")
    if not 0 <= arguments.seed <= 0xffffffffffffffff:
        raise SystemExit("error: --seed must be between 0 and 18446744073709551615")
    if not 0 <= arguments.warmup_steps < arguments.steps:
        raise SystemExit("error: --warmup-steps must be at least 0 and less than --steps")
    if arguments.tunable is not None and arguments.warmup_steps == 0:
        raise SystemExit("error: --tunable requires at least one warmup step")
    if arguments.compile_mode is not None and arguments.compile is None:
        arguments.compile = "default"


def is_network_source(source):
    return source.startswith("http://") or source.startswith("https://")


def read_range(source, offset, length):
    if is_network_source(source):
        request = urllib.request.Request(source, headers={"Range": f"bytes={offset}-{offset + length - 1}"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                status = response.status
                content_range = response.headers.get("Content-Range", "")
                data = response.read(length) if status == 206 else b""
        except (OSError, ValueError, http.client.HTTPException) as error:
            raise SystemExit(f"error: could not read {source}: {error}")
        if status != 206:
            raise SystemExit(f"error: {source} does not support HTTP range requests (status {status})")
        expected_range = f"bytes {offset}-{offset + length - 1}/"
        if not content_range.startswith(expected_range):
            raise SystemExit(f"error: unexpected Content-Range {content_range!r} from {source}")
        if len(data) != length:
            raise SystemExit(f"error: short read from {source} ({len(data)} of {length} bytes)")
        return data
    try:
        with open(source, "rb") as file:
            file.seek(offset)
            data = file.read(length)
    except OSError as error:
        raise SystemExit(f"error: could not read {source}: {error}")
    if len(data) != length:
        raise SystemExit(f"error: short read from {source} ({len(data)} of {length} bytes)")
    return data


def read_file_size(source):
    try:
        return os.path.getsize(source)
    except OSError as error:
        raise SystemExit(f"error: could not inspect {source}: {error}")


def read_checkpoint_header(source):
    header_length_bytes = read_range(source, 0, 8)
    header_length = struct.unpack("<Q", header_length_bytes)[0]
    if header_length <= 0:
        raise SystemExit(f"error: invalid safetensors header length {header_length} in {source}")
    if header_length > MAXIMUM_HEADER_LENGTH:
        raise SystemExit(f"error: safetensors header length {header_length} in {source} exceeds the maximum of {MAXIMUM_HEADER_LENGTH}")
    if not is_network_source(source) and 8 + header_length > read_file_size(source):
        raise SystemExit(f"error: safetensors header length {header_length} exceeds the file size of {source}")
    header_bytes = read_range(source, 8, header_length)
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"error: invalid safetensors header in {source}: {error}")
    if not isinstance(header, dict):
        raise SystemExit(f"error: safetensors header in {source} is not a json object")
    metadata = header.get("__metadata__")
    if metadata is not None and (not isinstance(metadata, dict) or any(not isinstance(value, str) for value in metadata.values())):
        raise SystemExit(f"error: invalid safetensors metadata in {source}")
    data_start = 8 + header_length
    return validate_tensor_entries(source, header, data_start), metadata, data_start


def validate_tensor_entries(source, header, data_start):
    entries = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            raise SystemExit(f"error: invalid safetensors entry for tensor {name} in {source}")
        dtype = SAFETENSORS_DTYPES.get(entry.get("dtype"))
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if dtype is None:
            raise SystemExit(f"error: unsupported data type {entry.get('dtype')!r} for tensor {name} in {source}")
        if not isinstance(shape, list) or any(type(dimension) is not int or dimension < 0 for dimension in shape):
            raise SystemExit(f"error: invalid shape for tensor {name} in {source}")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(offset) is not int for offset in offsets):
            raise SystemExit(f"error: invalid data offsets for tensor {name} in {source}")
        begin, end = offsets
        if begin < 0 or end < begin or end - begin != math.prod(shape) * dtype.itemsize:
            raise SystemExit(f"error: invalid data span for tensor {name} in {source}")
        entries[name] = entry

    spans = sorted((entry["data_offsets"][0], entry["data_offsets"][1], name) for name, entry in entries.items())
    previous_end = 0
    for begin, end, name in spans:
        if begin < previous_end:
            raise SystemExit(f"error: overlapping data span for tensor {name} in {source}")
        previous_end = end
    if not is_network_source(source) and spans and data_start + spans[-1][1] > read_file_size(source):
        raise SystemExit(f"error: tensor data in {source} exceeds the file size")
    return entries


def read_metadata_tensors(source, entries, data_start, names, show_progress):
    if not names:
        return {}
    spans = sorted(tuple(entries[name]["data_offsets"]) + (name,) for name in names)
    runs = [[spans[0][0], spans[0][1], [spans[0]]]]
    for begin, end, name in spans[1:]:
        run = runs[-1]
        if begin - run[1] <= METADATA_READ_GAP:
            run[1] = max(run[1], end)
            run[2].append((begin, end, name))
        else:
            runs.append([begin, end, [(begin, end, name)]])

    tensors = {}
    label = "downloading" if is_network_source(source) else "reading"
    for run_begin, run_end, members in tqdm(runs, desc=label, leave=False, disable=not show_progress):
        blob = read_range(source, data_start + run_begin, run_end - run_begin)
        for begin, end, name in members:
            entry = entries[name]
            buffer = bytearray(blob[begin - run_begin:end - run_begin])
            dtype = SAFETENSORS_DTYPES[entry["dtype"]]
            tensors[name] = torch.frombuffer(buffer, dtype=dtype).reshape(tuple(entry["shape"]))
    return tensors


def create_synthetic_tensor(name, shape, dtype, device):
    lowered = name.lower()
    if dtype == torch.bool:
        return torch.zeros(shape, dtype=dtype, device=device)
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return (torch.randn(shape, dtype=torch.float16, device=device) * 0.02).to(dtype)
    if not dtype.is_floating_point:
        if dtype.is_signed:
            return torch.randint(-8, 8, shape, dtype=dtype, device=device)
        return torch.randint(0, 16, shape, dtype=dtype, device=device)
    if name.endswith("bias"):
        return torch.zeros(shape, dtype=dtype, device=device)
    if "norm" in lowered:
        return torch.ones(shape, dtype=dtype, device=device)
    if "scale" in lowered:
        return torch.full(shape, 0.02, dtype=dtype, device=device)
    return torch.randn(shape, dtype=dtype, device=device) * 0.02


def create_model_state(source, entries, data_start, device, show_progress):
    metadata_names = [name for name in entries if name.split(".")[-1] in METADATA_TENSOR_NAMES]
    metadata = read_metadata_tensors(source, entries, data_start, metadata_names, show_progress)

    total = sum(max(1, math.prod(entry["shape"])) for entry in entries.values())
    torch.manual_seed(0)
    state_dict = {}
    with tqdm(total=total, desc="initializing", unit="weight", unit_scale=True, leave=False, disable=not show_progress) as progress:
        for name, entry in entries.items():
            if name in metadata:
                state_dict[name] = metadata[name]
            else:
                state_dict[name] = create_synthetic_tensor(
                    name, tuple(entry["shape"]), SAFETENSORS_DTYPES[entry["dtype"]], device)
            progress.update(max(1, state_dict[name].numel()))
    return state_dict


def resolve_device(arguments):
    if arguments.device is not None:
        try:
            return torch.device(arguments.device)
        except RuntimeError as error:
            raise SystemExit(f"error: invalid --device {arguments.device!r}: {error}")
    return comfy.model_management.get_torch_device()


def configure_tunable(arguments, device):
    if arguments.tunable is None:
        return
    if device.type != "cuda" or torch.version.hip is None:
        raise SystemExit("error: --tunable requires a ROCm device")
    torch.cuda.tunable.set_filename(arguments.tunable)
    torch.cuda.tunable.enable(True)
    torch.cuda.tunable.tuning_enable(True)


def load_model(state_dict, metadata, model_options, show_progress):
    parameters = comfy.utils.calculate_parameters(state_dict)
    with tqdm(total=1, desc="loading", leave=False, disable=not show_progress) as progress:
        patcher = comfy.sd.load_diffusion_model_state_dict(state_dict, model_options=model_options, metadata=metadata)
        progress.update(1)
    if patcher is None:
        raise SystemExit("error: ComfyUI could not detect the model architecture of this checkpoint")
    return patcher, parameters


def first_input_width(module):
    for submodule in module.modules():
        input_features = getattr(submodule, "in_features", None)
        if isinstance(input_features, int) and input_features > 0:
            return input_features
        weight = getattr(submodule, "weight", None)
        if weight is not None and weight.dim() in (1, 2):
            return weight.shape[-1]
    return None


def infer_context_dimension(diffusion_model, model_config):
    unet_config = getattr(model_config, "unet_config", None) or {}
    for key in CONTEXT_DIMENSION_KEYS:
        value = unet_config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    for name, module in diffusion_model.named_modules():
        if name.split(".")[-1] in KEY_PROJECTION_NAMES and ("cross" in name or "attn2" in name):
            width = first_input_width(module)
            if width:
                return width
    for name in TEXT_EMBEDDER_NAMES:
        module = getattr(diffusion_model, name, None)
        if module is not None:
            width = first_input_width(module)
            if width:
                return width
    return None


def infer_pooled_dimension(diffusion_model, model_config, model_class_name):
    for name in POOLED_EMBEDDER_NAMES:
        module = getattr(diffusion_model, name, None)
        if module is not None:
            width = first_input_width(module)
            if width:
                return width
    channels = (getattr(model_config, "unet_config", None) or {}).get("adm_in_channels")
    if channels:
        embedding_count = 5 if model_class_name == "SDXLRefiner" else 6
        return channels - embedding_count * 256 if channels > embedding_count * 256 else channels
    return None


def create_conditioning(patcher, context_length, device):
    model = patcher.model
    dtype = model.get_dtype_inference()
    context_dimension = infer_context_dimension(model.diffusion_model, model.model_config)
    if context_dimension is None:
        raise SystemExit("error: could not infer the text context width for this model")
    pooled_dimension = infer_pooled_dimension(
        model.diffusion_model, model.model_config, model.__class__.__name__)
    conditioning = [[torch.randn(1, context_length, context_dimension, device=device, dtype=dtype), {}]]
    if pooled_dimension is not None:
        conditioning[0][1]["pooled_output"] = torch.randn(1, pooled_dimension, device=device, dtype=dtype)
    if model.model_config.__class__.__name__ == "Anima":
        conditioning[0][1]["t5xxl_ids"] = torch.randint(0, 32128, (context_length,), device=device, dtype=torch.int32)
        conditioning[0][1]["t5xxl_weights"] = torch.ones(context_length, device=device, dtype=dtype)
    return conditioning, context_dimension, pooled_dimension


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "xpu":
        torch.xpu.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def reset_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "xpu":
        torch.xpu.reset_peak_memory_stats(device)


def read_peak_memory(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device)
    if device.type == "xpu":
        return torch.xpu.max_memory_allocated(device)
    return None


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
        if keywords is None:
            keywords = {}
        heads = arguments[3] if len(arguments) > 3 else keywords.get("heads", keywords.get("num_heads"))
        if not isinstance(heads, int) or heads <= 0:
            return None
        mask = arguments[4] if len(arguments) > 4 else keywords.get("mask")
        attn_precision = arguments[5] if len(arguments) > 5 else keywords.get("attn_precision")
        skip_reshape = arguments[6] if len(arguments) > 6 else keywords.get("skip_reshape", False)
        skip_output_reshape = arguments[7] if len(arguments) > 7 else keywords.get("skip_output_reshape", False)
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
            "attn_precision": attn_precision,
            "skip_reshape": bool(skip_reshape),
            "skip_output_reshape": bool(skip_output_reshape),
            "scale": keywords.get("scale"),
            "enable_gqa": bool(keywords.get("enable_gqa", False)),
            "low_precision_attention": keywords.get("low_precision_attention"),
            "flops": flops,
        }
        self.shapes.append(shape)
        return shape


class EvaluationTimer:
    def __init__(self, device, profiler, steps, attention_flop_counter, tuning_steps):
        self.device = device
        self.profiler = profiler
        self.steps = steps
        self.attention_flop_counter = attention_flop_counter
        self.tuning_steps = tuning_steps
        self.durations = []
        self.evaluations = 0

    def __call__(self, executor, *arguments, **keywords):
        if self.tuning_steps is not None and self.evaluations == self.tuning_steps:
            torch.cuda.tunable.tuning_enable(False)
        if self.profiler is not None and self.evaluations == self.steps - 1:
            self.profiler.start()
            self.attention_flop_counter.enabled = True
        synchronize(self.device)
        start = time.perf_counter()
        output = executor(*arguments, **keywords)
        synchronize(self.device)
        self.durations.append(time.perf_counter() - start)
        self.evaluations += 1
        if self.profiler is not None and self.evaluations == self.steps:
            self.attention_flop_counter.enabled = False
            self.profiler.stop()
        return output


def create_attention_override(attention_flop_counter):
    def override(function, *arguments, **keywords):
        attention_flop_counter.record(arguments, keywords, function)
        with torch.profiler.record_function("sdpa"):
            return function(*arguments, **keywords)
    return override


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
        return query.new_zeros((query.shape[0], heads, query.shape[1], query.shape[-1] // heads))
    return query.new_zeros(query.shape)


def create_shape_extraction_override(attention_flop_counter):
    def override(function, *arguments, **keywords):
        shape = attention_flop_counter.record(arguments, keywords, function)
        if shape is None:
            return function(*arguments, **keywords)
        output = attention_placeholder(shape, arguments)
        if output is None:
            return function(*arguments, **keywords)
        return output
    return override


def is_gemm(name):
    lowered = name.lower()
    return any(marker in lowered for marker in GEMM_OPERATION_MARKERS)


def is_numeric_shape(shape):
    return isinstance(shape, list) and len(shape) > 0 and all(isinstance(dimension, int) for dimension in shape)


def estimate_flops(name, shapes):
    if not shapes:
        return None
    operation = name.split("::")[-1]
    if operation == "linear" and len(shapes) >= 2 and is_numeric_shape(shapes[0]) and is_numeric_shape(shapes[1]):
        input_shape, weight_shape = shapes[0], shapes[1]
        return 2 * math.prod(input_shape[:-1]) * input_shape[-1] * weight_shape[0]
    if operation in ("mm", "addmm", "bmm", "addbmm", "baddbmm", "matmul", "_scaled_mm", "_int_mm"):
        operands = [shape for shape in shapes if is_numeric_shape(shape) and len(shape) >= 2]
        if len(operands) >= 2:
            left_operand, right_operand = operands[-2], operands[-1]
            return 2 * math.prod(left_operand[:-2]) * left_operand[-2] * left_operand[-1] * right_operand[-1]
        return None
    return None


def summarize_profile(profiler, attention_flop_count):
    entries = {}

    def walk(event, inside_sdpa):
        if event.is_user_annotation:
            inside_sdpa = inside_sdpa or event.name == "sdpa"
        elif (event.kernels or event.cpu_children) and event.self_device_time_total > 0:
            if not event.name.startswith("hip"):
                entry_type = "sdpa" if inside_sdpa else ("gemm" if is_gemm(event.name) else "other")
                entry = entries.setdefault((entry_type, event.name), {
                    "type": entry_type, "name": event.name, "time": 0.0, "calls": 0, "flops": 0})
                entry["time"] += event.self_device_time_total / 1e6
                entry["calls"] += 1
                if entry_type != "sdpa":
                    per_call = event.flops or estimate_flops(event.name, event.input_shapes)
                    if per_call:
                        entry["flops"] += per_call
        for child in event.cpu_children:
            walk(child, inside_sdpa)

    for event in profiler.events():
        if event.cpu_parent is None:
            walk(event, False)
    sdpa_entries = [entry for entry in entries.values() if entry["type"] == "sdpa"]
    sdpa_time = sum(entry["time"] for entry in sdpa_entries)
    if sdpa_time > 0:
        for entry in sdpa_entries:
            entry["flops"] = attention_flop_count * entry["time"] / sdpa_time
    return sorted(entries.values(), key=lambda entry: (-entry["time"], entry["type"], entry["name"]))


def run_workflow(patcher, arguments, positive, negative, latent, steps=None):
    if steps is None:
        steps = arguments.steps
    return nodes.KSampler().sample(
        model=patcher,
        seed=arguments.seed,
        steps=steps,
        cfg=arguments.cfg,
        sampler_name=arguments.sampler_name,
        scheduler=arguments.scheduler,
        positive=positive,
        negative=negative,
        latent_image=latent,
        denoise=arguments.denoise,
    )


def create_profiler(device):
    device_activity = (
        torch.profiler.ProfilerActivity.XPU
        if device.type == "xpu"
        else torch.profiler.ProfilerActivity.CUDA
    )
    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, device_activity],
        record_shapes=True,
        with_flops=True)


def measure_sampling(patcher, arguments, positive, negative, latent, load_device):
    profiler = None
    attention_flop_counter = AttentionFlopCounter()
    if arguments.profile is not None:
        profiler = create_profiler(load_device)
        patcher.model_options["transformer_options"]["optimized_attention_override"] = create_attention_override(
            attention_flop_counter)
    tuning_steps = arguments.warmup_steps if arguments.tunable is not None else None
    timer = EvaluationTimer(load_device, profiler, arguments.steps, attention_flop_counter, tuning_steps)
    patcher.add_wrapper(comfy.patcher_extension.WrappersMP.APPLY_MODEL, timer)
    reset_peak_memory(load_device)
    run_workflow(patcher, arguments, positive, negative, latent)
    durations = timer.durations[arguments.warmup_steps:]
    if not durations:
        raise SystemExit(f"error: sampling produced no evaluations after {arguments.warmup_steps} warmup steps")
    profile_entries = summarize_profile(profiler, attention_flop_counter.total) if profiler is not None else None
    return durations, read_peak_memory(load_device), profile_entries


def extract_sdpa_shape(patcher, arguments, positive, negative, latent):
    attention_flop_counter = AttentionFlopCounter()
    attention_flop_counter.enabled = True
    attention_flop_counter.capture_shapes = True
    transformer_options = patcher.model_options["transformer_options"]
    previous_override = transformer_options.get("optimized_attention_override")
    transformer_options["optimized_attention_override"] = create_shape_extraction_override(attention_flop_counter)
    try:
        run_workflow(patcher, arguments, positive, negative, latent, steps=1)
    finally:
        if previous_override is None:
            transformer_options.pop("optimized_attention_override", None)
        else:
            transformer_options["optimized_attention_override"] = previous_override
    if not attention_flop_counter.shapes:
        raise SystemExit("error: no compatible SDPA call was observed")
    return max(attention_flop_counter.shapes, key=lambda shape: shape["flops"])


def create_attention_tensor(shape, dtype, device):
    if dtype in (torch.float64, torch.float32, torch.float16, torch.bfloat16):
        return torch.randn(shape, device=device, dtype=dtype)
    return torch.randn(shape, device=device, dtype=torch.float32).to(dtype)


def create_attention_runner(shape, arguments, mask):
    attention_function = shape["function"]
    attention_kwargs = {
        "mask": mask,
        "skip_reshape": shape["skip_reshape"],
        "skip_output_reshape": shape["skip_output_reshape"],
    }
    if shape["attn_precision"] is not None:
        attention_kwargs["attn_precision"] = shape["attn_precision"]
    if shape["scale"] is not None:
        attention_kwargs["scale"] = shape["scale"]
    if shape["enable_gqa"]:
        attention_kwargs["enable_gqa"] = True
    if shape["low_precision_attention"] is not None:
        attention_kwargs["low_precision_attention"] = shape["low_precision_attention"]

    def run_attention(query, key, value):
        return attention_function(query, key, value, shape["heads"], **attention_kwargs)

    if arguments.compile is None:
        return run_attention
    compile_backend = None if arguments.compile == "default" else arguments.compile
    return torch.compile(run_attention, backend=compile_backend, mode=arguments.compile_mode)


def measure_sdpa(shape, arguments, load_device):
    query = create_attention_tensor(shape["query_shape"], shape["query_dtype"], load_device)
    key = create_attention_tensor(shape["key_shape"], shape["key_dtype"], load_device)
    value = create_attention_tensor(shape["value_shape"], shape["value_dtype"], load_device)
    mask = None
    if shape["mask_shape"] is not None:
        if shape["mask_dtype"] == torch.bool:
            mask = torch.ones(shape["mask_shape"], device=load_device, dtype=torch.bool)
        else:
            mask = torch.zeros(shape["mask_shape"], device=load_device, dtype=shape["mask_dtype"])
    attention_runner = create_attention_runner(shape, arguments, mask)
    profiler = create_profiler(load_device) if arguments.profile is not None else None
    reset_peak_memory(load_device)
    durations = []
    for evaluation in range(arguments.steps):
        if arguments.tunable is not None and evaluation == arguments.warmup_steps:
            torch.cuda.tunable.tuning_enable(False)
        if profiler is not None and evaluation == arguments.warmup_steps:
            profiler.start()
        synchronize(load_device)
        start = time.perf_counter()
        if profiler is None:
            output = attention_runner(query, key, value)
        else:
            with torch.profiler.record_function("sdpa"):
                output = attention_runner(query, key, value)
        synchronize(load_device)
        duration = time.perf_counter() - start
        if evaluation >= arguments.warmup_steps:
            durations.append(duration)
        del output
        if profiler is not None and evaluation == arguments.steps - 1:
            profiler.stop()
    if not durations:
        raise SystemExit(f"error: benchmark produced no evaluations after {arguments.warmup_steps} warmup steps")
    profile_entries = summarize_profile(profiler, shape["flops"]) if profiler is not None else None
    return durations, read_peak_memory(load_device), profile_entries


def create_result(arguments, architecture, data_type, parameters, context_dimension, pooled_dimension,
                  durations, peak_memory, load_device, profile_entries):
    median_seconds = statistics.median(durations)
    result = {
        "target": "full",
        "model": arguments.model,
        "architecture": architecture,
        "device": str(load_device),
        "device_name": comfy.model_management.get_torch_device_name(load_device),
        "data_type": data_type,
        "parameters": parameters,
        "width": arguments.width,
        "height": arguments.height,
        "batch_size": arguments.batch_size,
        "context_length": arguments.context_length,
        "context_dimension": context_dimension,
        "pooled_dimension": pooled_dimension,
        "steps": arguments.steps,
        "cfg": arguments.cfg,
        "sampler_name": arguments.sampler_name,
        "scheduler": arguments.scheduler,
        "denoise": arguments.denoise,
        "seed": arguments.seed,
        "attention": arguments.attention,
        "compile": arguments.compile,
        "compile_mode": arguments.compile_mode,
        "tunable": None if arguments.tunable is None else {
            "filename": torch.cuda.tunable.get_filename(),
            "results": len(torch.cuda.tunable.get_results()),
        },
        "warmup_steps": arguments.warmup_steps,
        "timed_evaluations": len(durations),
        "median_seconds": median_seconds,
        "minimum_seconds": min(durations),
        "maximum_seconds": max(durations),
        "evaluations_per_second": 1.0 / median_seconds,
        "peak_memory_bytes": peak_memory,
    }
    if profile_entries is not None:
        result["profile"] = profile_entries
    return result


def create_sdpa_result(arguments, architecture, data_type, parameters, context_dimension, pooled_dimension,
                       shape, durations, peak_memory, load_device, profile_entries):
    result = create_result(
        arguments, architecture, data_type, parameters, context_dimension, pooled_dimension,
        durations, peak_memory, load_device, profile_entries)
    attention_function = shape["function"]
    result.update({
        "target": "sdpa",
        "attention_function": getattr(attention_function, "__name__", type(attention_function).__name__),
        "query_shape": list(shape["query_shape"]),
        "key_shape": list(shape["key_shape"]),
        "value_shape": list(shape["value_shape"]),
        "query_dtype": str(shape["query_dtype"]),
        "key_dtype": str(shape["key_dtype"]),
        "value_dtype": str(shape["value_dtype"]),
        "heads": shape["heads"],
        "mask_shape": list(shape["mask_shape"]) if shape["mask_shape"] is not None else None,
        "mask_dtype": str(shape["mask_dtype"]) if shape["mask_dtype"] is not None else None,
        "attention_flops": shape["flops"],
        "skip_reshape": shape["skip_reshape"],
        "skip_output_reshape": shape["skip_output_reshape"],
    })
    return result


def color_enabled(arguments):
    if arguments.json or arguments.color == "never":
        return False
    if arguments.color == "always":
        return True
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"


def style_text(text, style, use_color):
    if not use_color or style is None:
        return text
    return f"\x1b[{ANSI_STYLE_CODES[style]}m{text}\x1b[0m"


def format_parameters(parameters):
    for threshold, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if parameters >= threshold:
            return f"{parameters / threshold:.2f}{suffix}"
    return f"{parameters:,}"


def format_bytes(byte_count):
    for threshold, suffix in ((1_000_000_000, "GB"), (1_000_000, "MB"), (1_000, "KB")):
        if byte_count >= threshold:
            return f"{byte_count / threshold:.2f} {suffix}"
    return f"{byte_count} B"


def render_share_bar(share, width=20):
    filled = min(width, max(0, round(share * width)))
    return "█" * filled + "░" * (width - filled)


def render_table(rows, alignments=None, styles=None, use_color=False):
    widths = []
    for row in rows:
        for index, cell in enumerate(row):
            if index >= len(widths):
                widths.append(0)
            widths[index] = max(widths[index], len(cell))
    lines = []
    for index, row in enumerate(rows):
        cells = []
        for cell_index, cell in enumerate(row):
            alignment = ">" if alignments is not None and cell_index < len(alignments) and alignments[cell_index] else "<"
            rendered = f"{cell:{alignment}{widths[cell_index]}}"
            style = styles[index][cell_index] if styles is not None and index < len(styles) and cell_index < len(styles[index]) else None
            cells.append(style_text(rendered, style, use_color))
        lines.append(" ".join(cells).rstrip())
    return "\n".join(lines)


def render_text(result, profile_mode, use_color):
    context_line = f"{result['context_length']} tokens × {result['context_dimension']} width"
    if result["pooled_dimension"]:
        context_line = f"{context_line} · pooled {result['pooled_dimension']}"
    warmup_label = "step" if result["warmup_steps"] == 1 else "steps"
    evaluation_label = "evaluation" if result["timed_evaluations"] == 1 else "evaluations"
    rows = [
        ("model", result["model"]),
        ("architecture", result["architecture"]),
        ("device", result["device_name"]),
        ("data type", result["data_type"].removeprefix("torch.")),
        ("parameters", format_parameters(result["parameters"])),
        ("workload", f"{result['width']}×{result['height']} · batch {result['batch_size']} · {result['steps']} steps"),
        ("sampling", f"{result['sampler_name']} / {result['scheduler']} · cfg {result['cfg']:g} · denoise {result['denoise']:g} · seed {result['seed']}"),
        ("attention", result["attention"]),
    ]
    if result["compile"]:
        rows.append(("compile", result["compile"]))
    if result["compile_mode"]:
        rows.append(("compile mode", result["compile_mode"]))
    if result["tunable"]:
        result_label = "result" if result["tunable"]["results"] == 1 else "results"
        rows.append(("tunable", f"{result['tunable']['filename']} · {result['tunable']['results']} {result_label}"))
    rows.extend([
        ("context", context_line),
        ("timing", f"{result['warmup_steps']} warmup {warmup_label} · {result['timed_evaluations']} measured {evaluation_label}"),
        ("median", f"{result['median_seconds'] * 1000:.2f} ms/evaluation"),
        ("range", f"{result['minimum_seconds'] * 1000:.2f}–{result['maximum_seconds'] * 1000:.2f} ms"),
        ("performance", f"{result['evaluations_per_second']:.3f} evaluations/second"),
    ])
    if result["peak_memory_bytes"] is not None:
        rows.append(("peak memory", format_bytes(result["peak_memory_bytes"])))
    styles = [["label", None] for _ in rows]
    for index, row in enumerate(rows):
        if row[0] in ("median", "performance"):
            styles[index][1] = "emphasis"
    lines = [style_text("Results", "heading", use_color),
             render_table(rows, styles=styles, use_color=use_color)]
    if "profile" in result:
        lines.extend(("", render_profile(result["profile"], profile_mode, use_color)))
    return "\n".join(lines)


def format_shape(shape):
    return "×".join(str(dimension) for dimension in shape)


def render_sdpa_text(result, profile_mode, use_color):
    rows = [
        ("model", result["model"]),
        ("architecture", result["architecture"]),
        ("device", result["device_name"]),
        ("data type", result["data_type"].removeprefix("torch.")),
        ("parameters", format_parameters(result["parameters"])),
        ("target", "SDPA"),
        ("attention", f"{result['attention']} · {result['attention_function']}"),
        ("query", f"{format_shape(result['query_shape'])} · {result['query_dtype'].removeprefix('torch.')}"),
        ("key", f"{format_shape(result['key_shape'])} · {result['key_dtype'].removeprefix('torch.')}"),
        ("value", f"{format_shape(result['value_shape'])} · {result['value_dtype'].removeprefix('torch.')}"),
        ("heads", str(result["heads"])),
        ("attention FLOPs", f"{result['attention_flops']:,}"),
    ]
    if result["mask_shape"] is not None:
        rows.append(("mask", f"{format_shape(result['mask_shape'])} · {result['mask_dtype'].removeprefix('torch.')}"))
    if result["compile"]:
        rows.append(("compile", result["compile"]))
    if result["compile_mode"]:
        rows.append(("compile mode", result["compile_mode"]))
    rows.extend([
        ("timing", f"{result['warmup_steps']} warmup steps · {result['timed_evaluations']} measured evaluations"),
        ("median", f"{result['median_seconds'] * 1000:.2f} ms"),
        ("range", f"{result['minimum_seconds'] * 1000:.2f}–{result['maximum_seconds'] * 1000:.2f} ms"),
        ("performance", f"{result['evaluations_per_second']:.3f} evaluations/second"),
    ])
    if result["peak_memory_bytes"] is not None:
        rows.append(("peak memory", format_bytes(result["peak_memory_bytes"])))
    styles = [["label", None] for _ in rows]
    for index, row in enumerate(rows):
        if row[0] in ("median", "performance"):
            styles[index][1] = "emphasis"
    lines = [style_text("Results", "heading", use_color),
             render_table(rows, styles=styles, use_color=use_color)]
    if "profile" in result:
        lines.extend(("", render_profile(result["profile"], profile_mode, use_color)))
    return "\n".join(lines)


def summarize_profile_buckets(entries):
    buckets = {name: {"time": 0.0, "flops": 0} for name, _ in PROFILE_BUCKETS}
    for entry in entries:
        bucket = buckets[entry["type"]]
        bucket["time"] += entry["time"]
        bucket["flops"] += entry["flops"]
    return buckets


def render_profile(entries, profile_mode, use_color):
    heading = style_text("Profile", "heading", use_color)
    if not entries:
        return f"{heading}\nno operations recorded"
    buckets = summarize_profile_buckets(entries)
    total = sum(bucket["time"] for bucket in buckets.values())
    bucket_rows = []
    bucket_styles = []
    for name, label in PROFILE_BUCKETS:
        seconds = buckets[name]["time"]
        share = seconds / total
        throughput = ""
        if name != "other" and seconds > 0 and buckets[name]["flops"]:
            throughput = f"{buckets[name]['flops'] / seconds / 1e12:.2f} TFLOP/s"
        bucket_rows.append([label, render_share_bar(share), f"{share * 100:.1f}%", f"{seconds * 1000:.2f} ms", throughput])
        bucket_styles.append([name, name, None, None, None])
    lines = [heading, render_table(bucket_rows, alignments=(False, False, True, True, True),
                                   styles=bucket_styles, use_color=use_color)]
    if profile_mode == "full":
        operation_rows = [["type", "operation", "average", "×", "calls", "=", "total"]]
        operation_styles = [["header"] * len(operation_rows[0])]
        for entry in entries:
            average = entry["time"] / entry["calls"]
            operation_rows.append([entry["type"].upper(), entry["name"], f"{average * 1000:.2f} ms", "×",
                                   str(entry["calls"]), "=", f"{entry['time'] * 1000:.2f} ms"])
            operation_styles.append([entry["type"], None, None, None, None, None, None])
        lines.extend(("", style_text("Profile: Full", "heading", use_color),
                      render_table(operation_rows, alignments=(False, False, True, False, True, False, True),
                                   styles=operation_styles, use_color=use_color)))
    return "\n".join(lines)


def create_profile_json(profile_entries):
    buckets = summarize_profile_buckets(profile_entries)
    simple = {}
    for name, _ in PROFILE_BUCKETS:
        seconds = buckets[name]["time"]
        flops = buckets[name]["flops"]
        simple[name] = {"time": seconds, "tflops": flops / seconds / 1e12 if flops and seconds else None}
    return {
        "simple": simple,
        "full": [{"type": entry["type"], "name": entry["name"], "time": entry["time"], "calls": entry["calls"]}
                 for entry in profile_entries],
    }


def render_json(result):
    output = dict(result)
    if "profile" in output:
        output["profile"] = create_profile_json(output["profile"])
    return json.dumps(output, indent=2)


def run_benchmark(arguments, show_progress):
    device = resolve_device(arguments)
    entries, metadata, data_start = read_checkpoint_header(arguments.model)
    state_dict = create_model_state(arguments.model, entries, data_start, device, show_progress)

    model_options = {"load_device": device}
    if arguments.dtype != "auto":
        model_options["dtype"] = DATA_TYPES[arguments.dtype]

    patcher, parameters = load_model(state_dict, metadata, model_options, show_progress)
    del state_dict
    detached = False
    try:
        load_device = patcher.load_device
        architecture = patcher.model.model_config.__class__.__name__
        data_type = str(patcher.model.get_dtype())
        if arguments.profile is not None and load_device.type not in ("cuda", "xpu"):
            raise SystemExit("error: --profile requires a cuda or xpu device")
        if arguments.target == "full":
            configure_tunable(arguments, load_device)
            if arguments.compile is not None:
                compile_backend = None if arguments.compile == "default" else arguments.compile
                patcher.model.diffusion_model = torch.compile(
                    patcher.model.diffusion_model, backend=compile_backend, mode=arguments.compile_mode)

        positive, context_dimension, pooled_dimension = create_conditioning(
            patcher, arguments.context_length, load_device)
        negative = nodes.ConditioningZeroOut().zero_out(positive)[0]
        latent = nodes.EmptyLatentImage().generate(arguments.width, arguments.height, arguments.batch_size)[0]

        if arguments.target == "sdpa":
            shape = extract_sdpa_shape(patcher, arguments, positive, negative, latent)
            del positive, negative, latent
            configure_tunable(arguments, load_device)
            patcher.detach()
            del patcher
            detached = True
            durations, peak_memory, profile_entries = measure_sdpa(shape, arguments, load_device)
            return create_sdpa_result(
                arguments, architecture, data_type, parameters, context_dimension, pooled_dimension,
                shape, durations, peak_memory, load_device, profile_entries)

        durations, peak_memory, profile_entries = measure_sampling(
            patcher, arguments, positive, negative, latent, load_device)
        return create_result(
            arguments, architecture, data_type, parameters, context_dimension, pooled_dimension,
            durations, peak_memory, load_device, profile_entries)
    finally:
        if not detached:
            patcher.detach()


def main():
    arguments = parse_arguments()
    os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
    load_dependencies()
    validate_arguments(arguments)
    show_progress = not arguments.json
    comfy.utils.set_progress_bar_enabled(show_progress)
    if not is_network_source(arguments.model) and not os.path.exists(arguments.model):
        raise SystemExit(f"error: model not found: {arguments.model}")

    result = run_benchmark(arguments, show_progress)
    if arguments.json:
        output = render_json(result)
    elif arguments.target == "sdpa":
        output = render_sdpa_text(result, arguments.profile, color_enabled(arguments))
    else:
        output = render_text(result, arguments.profile, color_enabled(arguments))
    sys.stdout.write(f"{output}\n")


if __name__ == "__main__":
    main()
