---
permalink: /notes/find-the-idle-resource-and-spend-it/
title: "Slack arbitrage: find the idle resource and spend it"
created: 2026-07-13
---

When a system is bottlenecked on one resource, every other resource is idle — and idle capacity is free money. The recurring optimization move is to identify which resource the [roofline model](kernels-are-memory-bound-below-the-ridge-point.md) says is saturated, then deliberately spend the *other* one to relieve it.

Because [decode is memory-bound](prefill-is-compute-bound-decode-is-memory-bound.md), the idle resource during generation is almost always compute, which is why the same arbitrage shows up everywhere:

- [FlashAttention](flashattention-keeps-the-attention-matrix-out-of-hbm.md) restructures attention to stop paying the HBM tax — and in the backward pass literally *recomputes* values rather than store and reload them: compute spent, bandwidth saved.
- [Speculative decoding](speculative-decoding-is-provably-exact.md) uses the compute sitting idle during every decode step to verify $\gamma$ draft tokens in one parallel pass.

The general form: this is classical systems thinking — bottleneck analysis — imported into ML by scale. The arbitrage is only visible if you know the exchange rate, which is exactly what the ridge-point number (~300 FLOPs/byte on an H100) gives you.

Sibling threads: [amortization](amortization-pay-once-spread-over-more-work.md), [caching + indirection](caching-plus-indirection-makes-reuse-shareable.md), [verified approximation](verified-approximation.md), [scale-out](scale-out-changes-the-problem-category.md). Hub: [a map of LLM inference optimization](a-map-of-llm-inference-optimization.md).
