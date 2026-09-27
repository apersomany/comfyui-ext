import json
import struct
import tempfile
import unittest
from pathlib import Path

import torch

from bench.checkpoint import (
    LocalCheckpointSource,
    create_model_state,
    read_checkpoint_header,
)
from bench.exceptions import BenchmarkError
from bench.progress import ProgressReporter


def write_checkpoint(path, header, data):
    encoded_header = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded_header)) + encoded_header + data)


class CheckpointTests(unittest.TestCase):
    def test_reads_header_and_builds_synthetic_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            header = {
                "weight": {
                    "dtype": "F32",
                    "shape": [2, 2],
                    "data_offsets": [0, 16],
                },
                "bias": {
                    "dtype": "F32",
                    "shape": [2],
                    "data_offsets": [16, 24],
                },
                "__metadata__": {"format": "pt"},
            }
            write_checkpoint(path, header, bytes(24))
            source = LocalCheckpointSource(path)
            checkpoint = read_checkpoint_header(source)
            with ProgressReporter(False) as reporter:
                state = create_model_state(
                    source,
                    checkpoint,
                    torch.device("cpu"),
                    reporter,
                )
            self.assertEqual(checkpoint.metadata, {"format": "pt"})
            self.assertEqual(state["weight"].shape, (2, 2))
            self.assertTrue(torch.equal(state["bias"], torch.zeros(2)))

    def test_rejects_overlapping_tensors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.safetensors"
            header = {
                "first": {
                    "dtype": "U8",
                    "shape": [2],
                    "data_offsets": [0, 2],
                },
                "second": {
                    "dtype": "U8",
                    "shape": [2],
                    "data_offsets": [1, 3],
                },
            }
            write_checkpoint(path, header, bytes(3))
            source = LocalCheckpointSource(path)
            with self.assertRaisesRegex(BenchmarkError, "overlapping data span"):
                read_checkpoint_header(source)
