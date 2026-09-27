# Comfy Bench

Comfy Bench benchmarks ComfyUI diffusion models with synthetic weights generated from safetensors metadata. The checkpoint tensor payload is not loaded except for metadata tensors required by ComfyUI.

Local checkpoint paths and HTTP URLs with range request support are accepted.

## Development environment

Enter the CPU shell with:

```console
nix develop
```

Accelerator shells are available through `nix develop .#cuda`, `nix develop .#rocm`, and `nix develop .#xpu`.

The shell creates a backend-specific virtual environment and installs the root benchmark requirements together with the ComfyUI requirements.

## Usage

```console
python bench.py MODEL.safetensors --device cuda:0 --dtype fp16
```

Benchmark the largest observed attention shape independently with:

```console
python bench.py MODEL.safetensors --target sdpa --attention flex --compile inductor
```

Enable operation profiling with `--profile` or `--profile full`.

## Output

Rich terminal output is used by default. JSON and Markdown are available with:

```console
python bench.py MODEL.safetensors --format json --output result.json
python bench.py MODEL.safetensors --format markdown --output result.md
```

JSON output uses a versioned, nested schema and includes raw latency samples, environment information, workload configuration, memory measurements, and optional profile data. The legacy `--json` option remains supported.

## Package layout

`bench/checkpoint.py` handles local and remote safetensors access. `bench/comfy_adapter.py` contains ComfyUI integration. `bench/measurement.py` and `bench/profiling.py` own benchmark instrumentation. Benchmark targets live in `bench/targets`. Result models and output renderers live in `bench/results.py` and `bench/output`.

## Tests

```console
python -m unittest discover -s tests -v
```
