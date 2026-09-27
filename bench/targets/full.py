from bench.comfy_adapter import run_workflow
from bench.measurement import Measurement, measure_sampling, profile_operation
from bench.profiling import AttentionFlopCounter, create_attention_override
from bench.targets.base import TargetOutcome


def run_full_target(
    patcher,
    configuration,
    conditioning,
    latent,
    device_controller,
    reporter,
    lifecycle,
):
    durations, evaluations, memory = measure_sampling(
        patcher,
        configuration,
        conditioning,
        latent,
        device_controller,
        reporter,
        run_workflow,
    )
    profile_entries = None
    if configuration.profile_mode is not None:
        counter = AttentionFlopCounter()
        transformer_options = patcher.model_options["transformer_options"]
        previous_override = transformer_options.get("optimized_attention_override")
        transformer_options["optimized_attention_override"] = create_attention_override(
            counter
        )
        try:
            profile_entries = profile_operation(
                lambda: run_workflow(
                    patcher,
                    configuration,
                    conditioning.positive,
                    conditioning.negative,
                    latent,
                    steps=1,
                ),
                counter,
                device_controller,
                reporter,
            )
        finally:
            if previous_override is None:
                transformer_options.pop("optimized_attention_override", None)
            else:
                transformer_options["optimized_attention_override"] = previous_override
    measurement = Measurement(durations, evaluations, memory, profile_entries)
    return TargetOutcome(measurement, None, False)
