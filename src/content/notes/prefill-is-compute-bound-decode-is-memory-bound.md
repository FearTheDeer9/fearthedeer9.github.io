---
permalink: /notes/prefill-is-compute-bound-decode-is-memory-bound/
title: "Prefill is compute-bound; decode is memory-bound"
created: 2026-07-13
---

This single asymmetry is the seed from which most of inference optimization grows.

**Prefill** processes the whole prompt at once: parallel matrix–matrix multiplication over all positions, with causal masking to prevent looking into the future. Lots of FLOPs per byte of weights loaded — high arithmetic intensity, so the GPU's compute units are the limiting resource.

**Decode** produces one token at a time: matrix–vector multiplication, where for every single token the entire set of model weights (plus the growing KV cache) must be re-read from HBM to do comparatively few FLOPs. Arithmetic intensity is tiny, sitting far [below the ridge point](kernels-are-memory-bound-below-the-ridge-point.md) — the bottleneck is memory bandwidth, and the tensor cores mostly sit idle waiting on the bus.

Nearly every technique in the serving stack is an answer to one side of this asymmetry:

- [Batching](batching-is-nearly-free-until-the-ridge-point.md) amortizes decode's weight reload across many sequences.
- [Quantization](quantization-loss-is-outlier-driven-not-bit-driven.md) shrinks the bytes that must move per decode step.
- [Speculative decoding](speculative-decoding-is-provably-exact.md) spends decode's idle compute to verify several tokens per weight-load.
- [Chunked prefill](chunked-prefill-trades-ttft-for-itl-stability.md) exists precisely because mixing a compute-bound prefill into a batch of memory-bound decode steps injects latency spikes.

The idle resource differs between the two phases, and [finding the idle resource and spending it](find-the-idle-resource-and-spend-it.md) is one of the recurring moves of the whole field.
