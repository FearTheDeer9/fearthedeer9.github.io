---
permalink: /notes/verified-approximation/
title: "Approximation spans a spectrum from bounded-loss to provably-free"
created: 2026-07-13
---

Trading an exact computation for a cheaper one isn't a single move — it's a spectrum, indexed by what happens to correctness:

At one end, **bounded loss you pay for**: [quantization](quantization-loss-is-outlier-driven-not-bit-driven.md) accepts a controlled accuracy cost in exchange for a 4× cut in memory traffic. The engineering effort goes into *shaping* the loss (scale granularity, outlier handling) rather than eliminating it.

At the other end, **approximation you get for free via verification**: [speculative decoding](speculative-decoding-is-provably-exact.md) runs an approximate cheap process (the draft model) but bolts on a rigorous accept/reject correction, so the output distribution is provably identical to the exact computation. The cost moves from accuracy to *variance in speedup* — a badly [calibrated](speculative-decoding-is-provably-exact.md) draft doesn't corrupt outputs, it just stops saving time.

The spectrum's lesson: "approximate" is not a quality verdict, it's a design axis. The question to ask of any lossy-looking technique is *where the error goes* — into the outputs (then bound it), or into the runtime (then it's free lunch, verify and take it). The elegant techniques are usually the ones that found a verification structure letting them sit at the free end.

Sibling threads: [slack arbitrage](find-the-idle-resource-and-spend-it.md), [amortization](amortization-pay-once-spread-over-more-work.md), [caching + indirection](caching-plus-indirection-makes-reuse-shareable.md). Hub: [a map of LLM inference optimization](a-map-of-llm-inference-optimization.md).
