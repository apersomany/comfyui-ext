# Flex Attention SDPA benchmark results

These results were collected with the SDPA target in [`bench.py`](https://github.com/apersomany/comfyui-ext/blob/73abe92ccd647c0c282b80085830ba5fa1882fbc/bench.py).

The attention shapes were extracted from an Anima Turbo sampling pass. The benchmark selected the largest observed attention call and replayed it with synthetic tensors.

The test environment was FP16 on AMD Radeon Graphics with ROCm 7.2 and PyTorch 2.13.0+rocm7.2. The context length was 64 and the batch size was 1.

The `default` mode uses Inductor without an explicit compile mode. The other modes use `max-autotune` and `max-autotune-no-cudagraphs`.

The aggregated measurements are in [`summary.json`](summary.json). The raw JSON outputs are grouped by backend:

- `pytorch-aotriton`
- `flex-triton-3.7.1`
- `flex-triton-3.8.0-local`

The Triton 3.8.0 results use the local Triton patches and are included as experimental comparison data.

## Command arguments

```text
/path/to/anima-turbo.safetensors --device cuda:0 --dtype fp16 --context-length 64 --target sdpa --attention BACKEND --compile inductor --json
```

The size-specific arguments were:

```text
512²:  --width 512  --height 512  --steps 16 --warmup-steps 6
1024²: --width 1024 --height 1024 --steps 8  --warmup-steps 3
2048²: --width 2048 --height 2048 --steps 8  --warmup-steps 3
```

`BACKEND` was `pytorch` or `flex`. The compile mode was selected by adding one of these arguments:

```text
default: omit --compile-mode
max-autotune: --compile-mode max-autotune
max-autotune-no-cudagraphs: --compile-mode max-autotune-no-cudagraphs
```
