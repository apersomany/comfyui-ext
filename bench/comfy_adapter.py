from dataclasses import dataclass

import comfy.model_detection
import comfy.model_management
import comfy.samplers
import comfy.sd
import comfy.utils
import nodes
import torch

from bench.exceptions import BenchmarkError

CONTEXT_DIMENSION_KEYS = (
    "caption_channels",
    "context_dim",
    "cross_attention_dim",
    "text_dim",
    "cap_feat_dim",
    "encoder_hidden_states_dim",
    "c_dim",
)
KEY_PROJECTION_NAMES = ("k_proj", "to_k", "k", "kv_proj", "to_kv", "c_kv", "k_norm")
TEXT_EMBEDDER_NAMES = (
    "cap_embedder",
    "context_embedder",
    "txt_in",
    "caption_projection",
    "text_embedder",
    "y_embedder",
    "context_refiner",
    "encoder_hid_proj",
)
POOLED_EMBEDDER_NAMES = ("vector_in", "y_embedder", "pooled_embedder")
DATA_TYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
FLUX2_DIFFUSERS_ALIASES = {
    "time_guidance_embed.timestep_embedder.linear_1.weight": "time_text_embed.timestep_embedder.linear_1.weight",
    "time_guidance_embed.timestep_embedder.linear_1.bias": "time_text_embed.timestep_embedder.linear_1.bias",
    "time_guidance_embed.timestep_embedder.linear_2.weight": "time_text_embed.timestep_embedder.linear_2.weight",
    "time_guidance_embed.timestep_embedder.linear_2.bias": "time_text_embed.timestep_embedder.linear_2.bias",
    "time_guidance_embed.guidance_embedder.linear_1.weight": "time_text_embed.guidance_embedder.linear_1.weight",
    "time_guidance_embed.guidance_embedder.linear_1.bias": "time_text_embed.guidance_embedder.linear_1.bias",
    "time_guidance_embed.guidance_embedder.linear_2.weight": "time_text_embed.guidance_embedder.linear_2.weight",
    "time_guidance_embed.guidance_embedder.linear_2.bias": "time_text_embed.guidance_embedder.linear_2.bias",
}
FLUX2_DIFFUSERS_MODULATION_KEYS = {
    "double_stream_modulation_img.linear.weight": "double_stream_modulation_img.lin.weight",
    "double_stream_modulation_txt.linear.weight": "double_stream_modulation_txt.lin.weight",
    "single_stream_modulation.linear.weight": "single_stream_modulation.lin.weight",
}


@dataclass(frozen=True)
class Conditioning:
    positive: list
    negative: list
    context_dimension: int
    pooled_dimension: int | None


def resolve_device(configuration):
    if configuration.device is not None:
        try:
            return torch.device(configuration.device)
        except RuntimeError as error:
            raise BenchmarkError(
                f"invalid device {configuration.device!r}: {error}"
            ) from error
    return comfy.model_management.get_torch_device()


def configure_tunable(configuration, device):
    if configuration.tunable_file is None:
        return
    if device.type != "cuda" or torch.version.hip is None:
        raise BenchmarkError("TunableOp requires a ROCm device")
    torch.cuda.tunable.set_filename(configuration.tunable_file)
    torch.cuda.tunable.enable(True)
    torch.cuda.tunable.tuning_enable(True)


def convert_flux2_diffusers_state(state):
    if not all(
        key in state
        for key in (
            "x_embedder.weight",
            "double_stream_modulation_img.linear.weight",
            "double_stream_modulation_txt.linear.weight",
            "single_stream_modulation.linear.weight",
        )
    ):
        return state

    for source, target in FLUX2_DIFFUSERS_ALIASES.items():
        if source in state:
            state[target] = state.pop(source)

    input_embedder_weight = state["x_embedder.weight"]
    state["x_embedder.bias"] = torch.zeros(
        input_embedder_weight.shape[0],
        dtype=input_embedder_weight.dtype,
        device=input_embedder_weight.device,
    )
    converted = comfy.model_detection.convert_diffusers_mmdit(state, "")
    if converted is None:
        raise BenchmarkError("could not convert the Flux2 Diffusers checkpoint")
    converted.pop("img_in.bias", None)
    for source, target in FLUX2_DIFFUSERS_MODULATION_KEYS.items():
        converted[target] = state.pop(source)
    return converted


def load_model(state, metadata, configuration, device, reporter):
    state = convert_flux2_diffusers_state(state)
    parameters = comfy.utils.calculate_parameters(state)
    model_options = {"load_device": device}
    if configuration.data_type != "auto":
        model_options["dtype"] = DATA_TYPES[configuration.data_type]
    task = reporter.add_task("Loading model", total=1)
    patcher = comfy.sd.load_diffusion_model_state_dict(
        state,
        model_options=model_options,
        metadata=metadata,
    )
    reporter.update(task, advance=1)
    reporter.finish(task)
    if patcher is None:
        raise BenchmarkError(
            "ComfyUI could not detect the checkpoint model architecture"
        )
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


def infer_context_dimension(diffusion_model, model_configuration):
    unet_configuration = getattr(model_configuration, "unet_config", None) or {}
    text_layer_count = unet_configuration.get("txtlayers")
    text_dimension = unet_configuration.get("txtdim")
    if (
        isinstance(text_layer_count, int)
        and text_layer_count > 0
        and isinstance(text_dimension, int)
        and text_dimension > 0
    ):
        return text_layer_count * text_dimension
    for key in CONTEXT_DIMENSION_KEYS:
        value = unet_configuration.get(key)
        if isinstance(value, int) and value > 0:
            return value
    for name, module in diffusion_model.named_modules():
        if name.split(".")[-1] in KEY_PROJECTION_NAMES and (
            "cross" in name or "attn2" in name
        ):
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


def infer_pooled_dimension(diffusion_model, model_configuration, model_class_name):
    for name in POOLED_EMBEDDER_NAMES:
        module = getattr(diffusion_model, name, None)
        if module is not None:
            width = first_input_width(module)
            if width:
                return width
    channels = (getattr(model_configuration, "unet_config", None) or {}).get(
        "adm_in_channels"
    )
    if channels:
        embedding_count = 5 if model_class_name == "SDXLRefiner" else 6
        return (
            channels - embedding_count * 256
            if channels > embedding_count * 256
            else channels
        )
    return None


def create_conditioning(patcher, context_length, device):
    model = patcher.model
    data_type = model.get_dtype_inference()
    context_dimension = infer_context_dimension(
        model.diffusion_model, model.model_config
    )
    if context_dimension is None:
        raise BenchmarkError("could not infer the text context width for this model")
    pooled_dimension = infer_pooled_dimension(
        model.diffusion_model,
        model.model_config,
        model.__class__.__name__,
    )
    positive = [
        [
            torch.randn(
                1, context_length, context_dimension, device=device, dtype=data_type
            ),
            {},
        ]
    ]
    if pooled_dimension is not None:
        positive[0][1]["pooled_output"] = torch.randn(
            1,
            pooled_dimension,
            device=device,
            dtype=data_type,
        )
    if model.model_config.__class__.__name__ == "Anima":
        positive[0][1]["t5xxl_ids"] = torch.randint(
            0,
            32128,
            (context_length,),
            device=device,
            dtype=torch.int32,
        )
        positive[0][1]["t5xxl_weights"] = torch.ones(
            context_length,
            device=device,
            dtype=data_type,
        )
    negative = nodes.ConditioningZeroOut().zero_out(positive)[0]
    return Conditioning(positive, negative, context_dimension, pooled_dimension)


def create_latent(configuration):
    return nodes.EmptyLatentImage().generate(
        configuration.width, configuration.height, configuration.batch_size
    )[0]


def run_workflow(patcher, configuration, positive, negative, latent, steps=None):
    return nodes.KSampler().sample(
        model=patcher,
        seed=configuration.seed,
        steps=configuration.steps if steps is None else steps,
        cfg=configuration.classifier_free_guidance,
        sampler_name=configuration.sampler_name,
        scheduler=configuration.scheduler,
        positive=positive,
        negative=negative,
        latent_image=latent,
        denoise=configuration.denoise,
    )


def validate_comfy_choices(configuration):
    if configuration.sampler_name not in comfy.samplers.KSampler.SAMPLERS:
        raise BenchmarkError(f"invalid sampler name {configuration.sampler_name!r}")
    if configuration.scheduler not in comfy.samplers.KSampler.SCHEDULERS:
        raise BenchmarkError(f"invalid scheduler {configuration.scheduler!r}")
