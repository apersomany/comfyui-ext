import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx
import torch

from bench.exceptions import BenchmarkError

METADATA_TENSOR_NAMES = ("comfy_quant",)
METADATA_READ_GAP = 1 << 20
MAXIMUM_HEADER_LENGTH = 64 << 20

SAFETENSORS_DATA_TYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "C64": torch.complex64,
    "U64": torch.uint64,
    "U32": torch.uint32,
    "U16": torch.uint16,
}


class CheckpointSource(Protocol):
    identity: str

    @property
    def size(self) -> int | None: ...

    def read_range(self, offset: int, length: int) -> bytes: ...

    def close(self) -> None: ...


class LocalCheckpointSource:
    def __init__(self, path):
        self.path = Path(path)
        self.identity = str(self.path)
        try:
            self._size = self.path.stat().st_size
        except OSError as error:
            raise BenchmarkError(f"could not inspect {self.path}: {error}") from error

    @property
    def size(self):
        return self._size

    def read_range(self, offset, length):
        try:
            with self.path.open("rb") as file:
                file.seek(offset)
                data = file.read(length)
        except OSError as error:
            raise BenchmarkError(f"could not read {self.path}: {error}") from error
        if len(data) != length:
            raise BenchmarkError(
                f"short read from {self.path} ({len(data)} of {length} bytes)"
            )
        return data

    def close(self):
        return None


class HttpCheckpointSource:
    def __init__(self, url):
        self.identity = url
        self._size = None
        self.client = httpx.Client(follow_redirects=True, timeout=30.0)

    @property
    def size(self):
        return self._size

    def read_range(self, offset, length):
        expected_prefix = f"bytes {offset}-{offset + length - 1}/"
        try:
            response = self.client.get(
                self.identity,
                headers={"Range": f"bytes={offset}-{offset + length - 1}"},
            )
        except httpx.HTTPError as error:
            raise BenchmarkError(f"could not read {self.identity}: {error}") from error
        content_range = response.headers.get("Content-Range", "")
        if response.status_code != httpx.codes.PARTIAL_CONTENT:
            raise BenchmarkError(
                f"{self.identity} does not support HTTP range requests (status {response.status_code})"
            )
        if not content_range.startswith(expected_prefix):
            raise BenchmarkError(
                f"unexpected Content-Range {content_range!r} from {self.identity}"
            )
        match = re.fullmatch(r"bytes \d+-\d+/(\d+|\*)", content_range)
        if match and match.group(1) != "*":
            self._size = int(match.group(1))
        data = response.content
        if len(data) != length:
            raise BenchmarkError(
                f"short read from {self.identity} ({len(data)} of {length} bytes)"
            )
        return data

    def close(self):
        self.client.close()


@dataclass(frozen=True)
class TensorEntry:
    data_type: str
    shape: tuple[int, ...]
    begin: int
    end: int


@dataclass(frozen=True)
class CheckpointHeader:
    entries: dict[str, TensorEntry]
    metadata: dict[str, str] | None
    data_start: int


def is_network_source(source):
    return source.startswith(("http://", "https://"))


def open_checkpoint_source(source):
    if is_network_source(source):
        return HttpCheckpointSource(source)
    return LocalCheckpointSource(source)


def read_checkpoint_header(source):
    header_length = struct.unpack("<Q", source.read_range(0, 8))[0]
    if header_length <= 0:
        raise BenchmarkError(
            f"invalid safetensors header length {header_length} in {source.identity}"
        )
    if header_length > MAXIMUM_HEADER_LENGTH:
        raise BenchmarkError(
            f"safetensors header length {header_length} in {source.identity} exceeds "
            f"the maximum of {MAXIMUM_HEADER_LENGTH}"
        )
    if source.size is not None and 8 + header_length > source.size:
        raise BenchmarkError(
            f"safetensors header length {header_length} exceeds the file size of {source.identity}"
        )
    try:
        header = json.loads(source.read_range(8, header_length).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BenchmarkError(
            f"invalid safetensors header in {source.identity}: {error}"
        ) from error
    if not isinstance(header, dict):
        raise BenchmarkError(
            f"safetensors header in {source.identity} is not a JSON object"
        )
    metadata = header.get("__metadata__")
    if metadata is not None and (
        not isinstance(metadata, dict)
        or any(not isinstance(value, str) for value in metadata.values())
    ):
        raise BenchmarkError(f"invalid safetensors metadata in {source.identity}")
    data_start = 8 + header_length
    entries = validate_tensor_entries(source, header, data_start)
    return CheckpointHeader(entries, metadata, data_start)


def validate_tensor_entries(source, header, data_start):
    entries = {}
    for name, value in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(value, dict):
            raise BenchmarkError(
                f"invalid safetensors entry for tensor {name} in {source.identity}"
            )
        data_type = value.get("dtype")
        shape = value.get("shape")
        offsets = value.get("data_offsets")
        if data_type not in SAFETENSORS_DATA_TYPES:
            raise BenchmarkError(
                f"unsupported data type {data_type!r} for tensor {name} in {source.identity}"
            )
        if not isinstance(shape, list) or any(
            type(dimension) is not int or dimension < 0 for dimension in shape
        ):
            raise BenchmarkError(
                f"invalid shape for tensor {name} in {source.identity}"
            )
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(offset) is not int for offset in offsets)
        ):
            raise BenchmarkError(
                f"invalid data offsets for tensor {name} in {source.identity}"
            )
        begin, end = offsets
        data_type_size = SAFETENSORS_DATA_TYPES[data_type].itemsize
        if begin < 0 or end < begin or end - begin != math.prod(shape) * data_type_size:
            raise BenchmarkError(
                f"invalid data span for tensor {name} in {source.identity}"
            )
        entries[name] = TensorEntry(data_type, tuple(shape), begin, end)

    spans = sorted((entry.begin, entry.end, name) for name, entry in entries.items())
    previous_end = 0
    for begin, end, name in spans:
        if begin < previous_end:
            raise BenchmarkError(
                f"overlapping data span for tensor {name} in {source.identity}"
            )
        previous_end = end
    if source.size is not None and spans and data_start + spans[-1][1] > source.size:
        raise BenchmarkError(f"tensor data in {source.identity} exceeds the file size")
    return entries


def read_metadata_tensors(source, checkpoint, names, reporter):
    if not names:
        return {}
    spans = sorted(
        (checkpoint.entries[name].begin, checkpoint.entries[name].end, name)
        for name in names
    )
    runs = [[spans[0][0], spans[0][1], [spans[0]]]]
    for begin, end, name in spans[1:]:
        run = runs[-1]
        if begin - run[1] <= METADATA_READ_GAP:
            run[1] = max(run[1], end)
            run[2].append((begin, end, name))
        else:
            runs.append([begin, end, [(begin, end, name)]])

    tensors = {}
    action = (
        "Downloading metadata"
        if isinstance(source, HttpCheckpointSource)
        else "Reading metadata"
    )
    task = reporter.add_task(action, total=len(runs))
    for run_begin, run_end, members in runs:
        blob = source.read_range(checkpoint.data_start + run_begin, run_end - run_begin)
        for begin, end, name in members:
            entry = checkpoint.entries[name]
            buffer = bytearray(blob[begin - run_begin : end - run_begin])
            data_type = SAFETENSORS_DATA_TYPES[entry.data_type]
            tensors[name] = torch.frombuffer(buffer, dtype=data_type).reshape(
                entry.shape
            )
        reporter.update(task, advance=1)
    reporter.finish(task)
    return tensors


def create_synthetic_tensor(name, shape, data_type, device):
    lowered_name = name.lower()
    if data_type == torch.bool:
        return torch.zeros(shape, dtype=data_type, device=device)
    if data_type in (torch.float8_e4m3fn, torch.float8_e5m2):
        return (torch.randn(shape, dtype=torch.float16, device=device) * 0.02).to(
            data_type
        )
    if data_type.is_complex:
        return torch.randn(shape, dtype=data_type, device=device) * 0.02
    if not data_type.is_floating_point:
        if data_type in (torch.uint8, torch.uint16, torch.uint32, torch.uint64):
            return torch.randint(0, 16, shape, dtype=data_type, device=device)
        return torch.randint(-8, 8, shape, dtype=data_type, device=device)
    if name.endswith("bias"):
        return torch.zeros(shape, dtype=data_type, device=device)
    if "norm" in lowered_name:
        return torch.ones(shape, dtype=data_type, device=device)
    if "scale" in lowered_name:
        return torch.full(shape, 0.02, dtype=data_type, device=device)
    return torch.randn(shape, dtype=data_type, device=device) * 0.02


def create_model_state(source, checkpoint, device, reporter):
    metadata_names = [
        name
        for name in checkpoint.entries
        if name.split(".")[-1] in METADATA_TENSOR_NAMES
    ]
    metadata = read_metadata_tensors(source, checkpoint, metadata_names, reporter)
    total = sum(max(1, math.prod(entry.shape)) for entry in checkpoint.entries.values())
    torch.manual_seed(0)
    state = {}
    task = reporter.add_task("Initializing synthetic weights", total=total)
    for name, entry in checkpoint.entries.items():
        if name in metadata:
            state[name] = metadata[name]
        else:
            state[name] = create_synthetic_tensor(
                name,
                entry.shape,
                SAFETENSORS_DATA_TYPES[entry.data_type],
                device,
            )
        reporter.update(task, advance=max(1, state[name].numel()))
    reporter.finish(task)
    return state
