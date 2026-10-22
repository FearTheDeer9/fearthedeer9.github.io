---
permalink: /notes/kv-cache-traffic-scales-with-batch-times-context/
title: "KV-cache traffic scales with batch × context; weight traffic is fixed per step"
---

Two things must cross the memory bus at every decode step, and they scale differently:

- **Weights**: loaded once per step and reused across the whole batch — a fixed cost, roughly constant in both batch size $B$ and context length.
- **KV cache**: *not* shared across sequences. Each sequence drags its own K,V history, so total cache reads scale with $B \times \text{context length}$, unboundedly.

The tug of war between the two is decided by the operating point: small batch + long context, cache can dominate; large batch + short context, weights dominate; large batch + long context — the regime production serving actually lives in — the cache dominates hard.

This is the economic argument for most of the memory-side stack. It's why [PagedAttention](pagedattention-is-virtual-memory-for-the-kv-cache.md) attacks cache allocation rather than weight storage, why [prefix sharing](prefix-sharing-is-safe-only-with-refcounts-and-copy-on-write.md) is worth engineering machinery for (dedupe the term that grows, not the one that's fixed), and why [agentic workloads](agentic-workloads-are-a-characterization-not-a-technique.md) — long, heavily-shared contexts — concentrate so much value into cache management.

It also sharpens the [batching](batching-is-nearly-free-until-the-ridge-point.md) picture: the "weights amortize across the batch" free lunch is specifically a statement about the fixed term. The $B \times \text{context}$ term amortizes across nothing — it is the part of decode that batching cannot dilute, only multiply.
