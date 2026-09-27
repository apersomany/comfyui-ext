import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from bench.exceptions import BenchmarkError


@dataclass(frozen=True)
class BootstrapState:
    comfy_cli_arguments: ModuleType
    attention_options: dict[str, str]

    @property
    def attention_choices(self):
        return ("auto", *self.attention_options)


_STATE = None


def prepare_comfyui():
    global _STATE
    if _STATE is not None:
        return _STATE

    comfyui_path = Path(__file__).resolve().parent.parent / "comfyui"
    sys.path.insert(0, str(comfyui_path))
    import comfy.cli_args

    options = {}
    for action in comfy.cli_args.attn_group._group_actions:
        name = (
            action.dest.removeprefix("use_")
            .removesuffix("_attention")
            .removesuffix("_cross")
        )
        options[name.replace("_", "-")] = action.dest

    _STATE = BootstrapState(comfy.cli_args, options)
    return _STATE


def configure_attention(state, attention):
    if attention == "auto":
        return
    if attention not in state.attention_options:
        choices = ", ".join(state.attention_choices)
        raise BenchmarkError(
            f"invalid attention implementation {attention!r}; choose from {choices}"
        )
    setattr(state.comfy_cli_arguments.args, state.attention_options[attention], True)
    state.comfy_cli_arguments.args.disable_xformers = True
