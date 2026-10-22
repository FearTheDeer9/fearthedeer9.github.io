---
permalink: /notes/caching-plus-indirection-makes-reuse-shareable/
title: "Caching avoids redoing identical work; indirection makes the reuse shareable"
---

Two ideas that are easy to conflate, layered deliberately:

**Caching** is the decision to never redo identical computation. The KV cache itself is the founding instance: past tokens' K,V vectors are kept so that [attention — the only operation that reads the past](attention-is-the-only-inter-token-operation.md) — never recomputes them at each decode step.

**Indirection** is what turns a private cache into a shared one. [PagedAttention's block table](pagedattention-is-virtual-memory-for-the-kv-cache.md) decouples logical position from physical storage, which is precisely what lets many sequences point at one physical copy of a common prefix — with [refcounts and copy-on-write](prefix-sharing-is-safe-only-with-refcounts-and-copy-on-write.md) keeping divergence safe. Caching without indirection saves work within a sequence; caching *with* indirection saves it across sequences, users, and — in [agentic workloads](agentic-workloads-are-a-characterization-not-a-technique.md) — across the branches of a tree, where block-level granularity means a fork duplicates only what's downstream of the fork point.

The distinction is worth policing because it marks a real boundary: PagedAttention is often misfiled as "avoiding recomputation," but it never touches recomputation — that's the KV cache's job. PagedAttention is pure addressing. Keeping the two in separate buckets is what makes the thread structure visible.

Sibling threads: [slack arbitrage](find-the-idle-resource-and-spend-it.md), [amortization](amortization-pay-once-spread-over-more-work.md), [verified approximation](verified-approximation.md), [scale-out](scale-out-changes-the-problem-category.md). Hub: [a map of LLM inference optimization](a-map-of-llm-inference-optimization.md).
