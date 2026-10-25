---
permalink: /notes/a-map-of-llm-inference-optimization/
title: "A map of LLM inference optimization"
created: 2026-07-13
---

Inference optimization looks like a zoo of techniques, but it decomposes into a small set of recurring moves applied to one underlying situation. This note is the map; every claim links to a note that argues it.

**The substrate — two facts everything grows from:**

- [Attention is the only inter-token operation in a transformer](attention-is-the-only-inter-token-operation.md), which is why the KV cache exists and is attention-only.
- [Prefill is compute-bound; decode is memory-bound](prefill-is-compute-bound-decode-is-memory-bound.md) — quantified by the roofline: [a kernel is memory-bound below the GPU's ridge point](kernels-are-memory-bound-below-the-ridge-point.md), and decode sits far below it. The bytes that move are governed by [KV-cache traffic scaling with B × context while weight traffic stays fixed](kv-cache-traffic-scales-with-batch-times-context.md).

**Five threads — the recurring moves:**

1. **[Slack arbitrage](find-the-idle-resource-and-spend-it.md)** — find the idle resource and spend it: [FlashAttention](flashattention-keeps-the-attention-matrix-out-of-hbm.md), speculative decoding's use of idle compute.
2. **[Amortization](amortization-pay-once-spread-over-more-work.md)** — pay a fixed cost once, spread it: [batching below the ridge](batching-is-nearly-free-until-the-ridge-point.md), [microbatching](microbatching-shrinks-the-pipeline-bubble.md), spec-decode's one-pass-per-γ.
3. **[Caching + indirection](caching-plus-indirection-makes-reuse-shareable.md)** — never redo identical work, and make the reuse shareable: the KV cache, [PagedAttention](pagedattention-is-virtual-memory-for-the-kv-cache.md), [refcounts + copy-on-write](prefix-sharing-is-safe-only-with-refcounts-and-copy-on-write.md).
4. **[Verified approximation](verified-approximation.md)** — from [quantization's bounded loss](quantization-loss-is-outlier-driven-not-bit-driven.md) to [speculative decoding's provably exact free lunch](speculative-decoding-is-provably-exact.md).
5. **[Scale-out](scale-out-changes-the-problem-category.md)** — when one unit's budget is exceeded, the problem changes category: [TP, PP, DP and their distinct costs](tp-pays-in-all-reduces-pp-pays-in-bubbles-dp-pays-in-memory.md).

**The scheduling layer, where techniques meet traffic:** [continuous batching](continuous-batching-schedules-at-iteration-granularity.md) and [chunked prefill](chunked-prefill-trades-ttft-for-itl-stability.md) — where the throughput-vs-TTFT tension gets settled request by request.

**The lens on top:** [agentic workloads are a workload characterization, not a technique](agentic-workloads-are-a-characterization-not-a-technique.md) — the empirical claim about input shape that decides which threads pay off hardest.

Two meta-observations the map keeps yielding: the most powerful techniques (speculative decoding, PagedAttention) deliberately span multiple threads rather than one; and the whole field is scale forcing AI engineering to import classical hardware-architecture and distributed-systems thinking.
