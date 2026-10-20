# fast-vla

Making a vision-language-action (VLA) robot policy fast, and measuring what that speed costs in task success.

This project profiles [SmolVLA](https://huggingface.co/HuggingFaceVLA/smolvla_libero) on the [LIBERO](https://libero-project.github.io/) simulation benchmark, then applies inference optimizations one at a time (bf16, `torch.compile`, CUDA graphs, fewer flow-matching steps, prefix KV caching, quantization, a custom Triton kernel). Each change is reported with both its latency speedup and its effect on task success rate.

> Work in progress. The results table below fills in as each milestone lands.

## Results

| Variant | p50 latency (ms) | p95 latency (ms) | Speedup | LIBERO success rate |
| --- | --- | --- | --- | --- |
| Baseline (fp32, eager) | | | 1.0x | |

## Hardware

- NVIDIA GeForce RTX 5070 Laptop GPU (8 GB, Blackwell), Ubuntu 22.04 on WSL2

## Layout

```
src/fastvla/   library code: latency harness, optimizations, eval helpers
scripts/       entry points that produce every number in this README
results/       raw measurements (CSV/JSON) and plots
tests/         correctness checks (e.g. optimized vs reference outputs)
```

## Roadmap

- [ ] M0 Environment: LeRobot + LIBERO running on the GPU
- [ ] M1 Baseline success rate
- [ ] M2 Per-action latency breakdown
- [ ] M3 bf16, torch.compile, CUDA graphs
- [ ] M4 Fewer flow-matching steps, prefix KV caching
- [ ] M5 Quantization, custom Triton kernel
- [ ] M6 Final tradeoff table and write-up
