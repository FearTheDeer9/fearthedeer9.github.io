---
permalink: /notes/quantization-loss-is-outlier-driven-not-bit-driven/
title: "Quantization loss is outlier-driven, not bit-driven"
---

Quantization speeds up decode for one reason: [decode is memory-bound](prefill-is-compute-bound-decode-is-memory-bound.md), and FP16→INT4 cuts the bytes of weights crossing the bus per step by 4×. The win is bandwidth, not arithmetic speed (unless the hardware natively does low-precision math). Dequantization is a cheap affine reconstruction — store a scale $s$ and zero-point $z$ per group of values, recover $x \approx s\,(q - z)$ — a multiply and an add, not any mantissa gymnastics.

The interesting claim is where the accuracy loss comes from. It is *not* generic "fewer bits, fewer distinguishable values." With 16 levels (INT4), the scale must stretch to cover the block's full range: $s \propto \max(|x|)$. LLM weight and especially activation distributions are outlier-dominated — a few channels with magnitude far above the rest. One shared scale gets dictated by the rare outlier, cramming the ordinary majority into a sliver of the available levels and destroying *their* resolution. The loss concentrates in the non-outlier majority, caused by the outliers.

This reframing explains the actual shape of quantization research: it's about **scale assignment, not round-off**. Granularity ladders (per-tensor → per-channel → per-group) keep one group's outliers from ruining another group's resolution; outlier-aware schemes like LLM.int8() pull outlier channels out to FP16 entirely rather than let them poison a shared scale.

The overhead of fine granularity is sub-linear and nearly free: INT4 at group size 128 stores 64 bytes of data per 2–4 bytes of scale/zero-point — a few percent against a 4× compression.

Quantization sits at the "bounded loss" end of the [approximation spectrum](verified-approximation.md); [speculative decoding](speculative-decoding-is-provably-exact.md) is the provably-free other end.
