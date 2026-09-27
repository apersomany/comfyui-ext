import json
import unittest
from io import StringIO
from types import SimpleNamespace

from rich.console import Console

from bench.configuration import BenchmarkConfiguration
from bench.devices import MemoryMeasurement
from bench.measurement import Measurement
from bench.output.console import render_console
from bench.output.json import render_json
from bench.output.markdown import render_markdown
from bench.results import DeviceInfo, EnvironmentInfo, ModelInfo, create_result
from bench.targets.base import TargetOutcome


class ResultTests(unittest.TestCase):
    def test_schema_contains_samples_and_environment(self):
        configuration = BenchmarkConfiguration(
            model="model.safetensors",
            width=512,
            height=512,
            batch_size=1,
            context_length=64,
            steps=3,
            classifier_free_guidance=1.0,
            sampler_name="euler",
            scheduler="normal",
            denoise=1.0,
            seed=0,
            warmup_evaluations=1,
            data_type="fp16",
            device="cpu",
            attention="auto",
            target="full",
            compile_backend=None,
            compile_mode=None,
            tunable_file=None,
            profile_mode=None,
        )
        environment = EnvironmentInfo(
            python_version="3.12.0",
            torch_version="2.0.0",
            comfyui_version="1.0.0",
            comfyui_revision=None,
            accelerator_runtime=None,
            device=DeviceInfo(type="cpu", index=None, name="CPU"),
        )
        model = ModelInfo(
            source=configuration.model,
            architecture="TestModel",
            parameters=1000,
            data_type="float16",
        )
        conditioning = SimpleNamespace(context_dimension=768, pooled_dimension=None)
        measurement = Measurement(
            durations=[0.1, 0.2],
            observed_evaluations=3,
            memory=MemoryMeasurement(None, None, None),
            profile_entries=None,
        )
        result = create_result(
            configuration,
            None,
            environment,
            model,
            conditioning,
            TargetOutcome(measurement, None, False),
        )
        output = json.loads(render_json(result))
        self.assertEqual(output["schema_version"], 1)
        self.assertEqual(output["environment"]["device"]["name"], "CPU")
        self.assertEqual(output["measurements"]["latency"]["samples"], [0.1, 0.2])
        self.assertTrue(output["model"]["synthetic_weights"])

        stream = StringIO()
        render_console(
            result,
            Console(file=stream, force_terminal=False, width=100),
            None,
        )
        self.assertIn("Comfy Bench", stream.getvalue())
        self.assertIn("## Performance", render_markdown(result, None))
