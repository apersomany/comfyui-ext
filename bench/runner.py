import platform
import subprocess
from pathlib import Path

import comfy.model_management
import comfyui_version
import torch

from bench.checkpoint import (
    create_model_state,
    open_checkpoint_source,
    read_checkpoint_header,
)
from bench.comfy_adapter import (
    configure_tunable,
    create_conditioning,
    create_latent,
    load_model,
    resolve_device,
)
from bench.devices import DeviceController
from bench.exceptions import BenchmarkError
from bench.results import DeviceInfo, EnvironmentInfo, ModelInfo, create_result
from bench.targets.full import run_full_target
from bench.targets.sdpa import run_sdpa_target

TARGETS = {
    "full": run_full_target,
    "sdpa": run_sdpa_target,
}

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
COMFYUI_ROOT = REPOSITORY_ROOT / "comfyui"


class ModelLifecycle:
    def __init__(self, patcher):
        self.patcher = patcher
        self.detached = False

    def detach(self):
        if not self.detached:
            self.patcher.detach()
            self.detached = True


def git_revision(path):
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def accelerator_runtime():
    if torch.version.hip is not None:
        return f"ROCm {torch.version.hip}"
    if torch.version.cuda is not None:
        return f"CUDA {torch.version.cuda}"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "XPU"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "MPS"
    return None


def create_environment(device):
    return EnvironmentInfo(
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
        comfyui_version=comfyui_version.__version__,
        comfyui_revision=git_revision(COMFYUI_ROOT),
        accelerator_runtime=accelerator_runtime(),
        device=DeviceInfo(
            type=device.type,
            index=device.index,
            name=comfy.model_management.get_torch_device_name(device),
        ),
    )


def compile_full_model(patcher, configuration):
    if configuration.compile_backend is None:
        return
    compile_backend = (
        None
        if configuration.compile_backend == "default"
        else configuration.compile_backend
    )
    patcher.model.diffusion_model = torch.compile(
        patcher.model.diffusion_model,
        backend=compile_backend,
        mode=configuration.compile_mode,
    )


def run_benchmark(configuration, reporter):
    device = resolve_device(configuration)
    device_controller = DeviceController(device)
    if configuration.profile_mode is not None and device.type not in ("cuda", "xpu"):
        raise BenchmarkError("profiling requires a CUDA or XPU device")

    source = open_checkpoint_source(configuration.model)
    patcher = None
    lifecycle = None
    try:
        checkpoint = read_checkpoint_header(source)
        state = create_model_state(source, checkpoint, device, reporter)
        patcher, parameters = load_model(
            state,
            checkpoint.metadata,
            configuration,
            device,
            reporter,
        )
        del state
        lifecycle = ModelLifecycle(patcher)
        load_device = patcher.load_device
        device_controller = DeviceController(load_device)
        architecture = patcher.model.model_config.__class__.__name__
        data_type = str(patcher.model.get_dtype()).removeprefix("torch.")
        if configuration.profile_mode is not None and load_device.type not in (
            "cuda",
            "xpu",
        ):
            raise BenchmarkError("profiling requires a CUDA or XPU load device")
        if configuration.target == "full":
            configure_tunable(configuration, load_device)
            compile_full_model(patcher, configuration)

        conditioning = create_conditioning(
            patcher, configuration.context_length, load_device
        )
        latent = create_latent(configuration)
        outcome = TARGETS[configuration.target](
            patcher,
            configuration,
            conditioning,
            latent,
            device_controller,
            reporter,
            lifecycle,
        )
        environment = create_environment(load_device)
        model = ModelInfo(
            source=configuration.model,
            architecture=architecture,
            parameters=parameters,
            data_type=data_type,
        )
        return create_result(
            configuration,
            git_revision(REPOSITORY_ROOT),
            environment,
            model,
            conditioning,
            outcome,
        )
    finally:
        source.close()
        if lifecycle is not None:
            lifecycle.detach()
