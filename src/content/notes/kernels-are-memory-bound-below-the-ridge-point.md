---
permalink: /notes/kernels-are-memory-bound-below-the-ridge-point/
title: "A kernel is memory-bound below the GPU's ridge point — which is ~300 FLOPs/byte, not 1"
---

The roofline model: every kernel has an *arithmetic intensity* — FLOPs performed per byte moved from HBM. The GPU has a *ridge point* where the two possible bottlenecks cross:

$$\text{AI}_{\text{ridge}} = \frac{\text{Peak FLOP/s}}{\text{Peak bandwidth (bytes/s)}}$$

Below this intensity the kernel is memory-bound (throughput limited by bandwidth); above it, compute-bound (limited by peak FLOPs).

The tempting wrong answer is that the crossover sits at intensity = 1. That would only hold if peak FLOP/s equaled peak bandwidth, which is never true of real hardware. On an H100, ~990 TFLOP/s (BF16) over ~3.35 TB/s gives a ridge of roughly **300 FLOPs per byte** — a kernel must do ~300 floating-point operations for every byte it pulls from memory before compute becomes the constraint.

That number is why [decode sits so deep in the memory-bound regime](prefill-is-compute-bound-decode-is-memory-bound.md): touching every weight byte to do a handful of FLOPs per token is intensity in the single digits, two orders of magnitude below the ridge. It's also why the critical batch size at which [batching stops being free](batching-is-nearly-free-until-the-ridge-point.md) is much larger than intuition suggests: you locate $B_{\text{crit}}$ by writing arithmetic intensity as a function of batch size, $\text{AI}(B)$, and solving $\text{AI}(B) = \text{AI}_{\text{ridge}}$ for the specific chip.

The roofline is the quantitative backbone of [slack arbitrage](find-the-idle-resource-and-spend-it.md): it tells you, per kernel, which resource is saturated and which is idle.
