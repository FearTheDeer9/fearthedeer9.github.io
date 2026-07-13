---
permalink: /notes/amortization-pay-once-spread-over-more-work/
title: "Amortization: pay a fixed cost once, spread it over more work"
---

Wherever a large fixed cost must be paid regardless of how much useful work rides along, the move is to attach more work to each payment. In LLM serving the fixed cost is usually one trip of the model weights across the memory bus:

- [Batching](batching-is-nearly-free-until-the-ridge-point.md) is the purest case — weights loaded once per decode step, reused across every sequence in the batch, nearly free until the [ridge point](kernels-are-memory-bound-below-the-ridge-point.md).
- [Speculative decoding](speculative-decoding-is-provably-exact.md) amortizes the target model's expensive forward pass over $\gamma$ verified tokens instead of paying it per token.
- [Prefix sharing](prefix-sharing-is-safe-only-with-refcounts-and-copy-on-write.md) amortizes one physical KV allocation across many logical requests.
- [Microbatching](microbatching-shrinks-the-pipeline-bubble.md) amortizes the pipeline's fixed fill/drain bubble over $M$ work-items.

The boundary of the thread matters as much as its members: amortization has a natural limit wherever the cost *isn't* fixed — [KV-cache traffic scales with B × context](kv-cache-traffic-scales-with-batch-times-context.md) and amortizes across nothing, which is exactly where the caching thread has to take over. And TP/DP don't belong here at all: they aren't making a fixed cost cheaper per unit, they're [adding units when one unit's budget is exceeded](scale-out-changes-the-problem-category.md).

Sibling threads: [slack arbitrage](find-the-idle-resource-and-spend-it.md), [caching + indirection](caching-plus-indirection-makes-reuse-shareable.md), [verified approximation](verified-approximation.md). Hub: [a map of LLM inference optimization](a-map-of-llm-inference-optimization.md).
