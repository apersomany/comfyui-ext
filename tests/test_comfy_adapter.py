import unittest

import torch

from bench.bootstrap import prepare_comfyui

prepare_comfyui()

from bench.comfy_adapter import (
    convert_flux2_diffusers_state,
    infer_context_dimension,
)


class ComfyAdapterTests(unittest.TestCase):
    def test_converts_biasless_flux2_diffusers_state(self):
        input_embedder_weight = torch.zeros((4, 2))
        timestep_weight = torch.zeros((4, 2))
        state = {
            "x_embedder.weight": input_embedder_weight,
            "double_stream_modulation_img.linear.weight": torch.zeros((24, 4)),
            "double_stream_modulation_txt.linear.weight": torch.zeros((24, 4)),
            "single_stream_modulation.linear.weight": torch.zeros((12, 4)),
            "time_guidance_embed.timestep_embedder.linear_1.weight": timestep_weight,
        }

        converted = convert_flux2_diffusers_state(state)

        self.assertIs(converted["img_in.weight"], input_embedder_weight)
        self.assertIs(converted["time_in.in_layer.weight"], timestep_weight)
        self.assertNotIn("img_in.bias", converted)
        self.assertIn("double_stream_modulation_img.lin.weight", converted)
        self.assertIn("double_stream_modulation_txt.lin.weight", converted)
        self.assertIn("single_stream_modulation.lin.weight", converted)

    def test_leaves_other_state_unchanged(self):
        state = {"weight": torch.zeros(1)}
        self.assertIs(convert_flux2_diffusers_state(state), state)

    def test_infers_layer_packed_context_dimension(self):
        model_configuration = type(
            "ModelConfiguration",
            (),
            {"unet_config": {"txtlayers": 12, "txtdim": 2560}},
        )()

        dimension = infer_context_dimension(torch.nn.Module(), model_configuration)

        self.assertEqual(dimension, 30720)


if __name__ == "__main__":
    unittest.main()
