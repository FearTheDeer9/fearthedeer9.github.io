---
permalink: /notes/pagedattention-is-virtual-memory-for-the-kv-cache/
title: "PagedAttention is virtual memory for the KV cache, not a new attention algorithm"
created: 2026-07-13
---

PagedAttention changes nothing about the attention computation. Every query still attends to every key/value in its history; there is no overlap between pages, no "higher-order attention across pages" (a natural guess I made once — it's a confabulation). What changes is purely storage and addressing, imported almost directly from OS virtual memory.

The problem: pre-PagedAttention systems stored each sequence's KV cache in one *contiguous* GPU allocation, sized up front for the maximum possible length, because generation length is unknown in advance. Result: **internal fragmentation** (a sequence that stops at 50 tokens still holds a 2048-token reservation) and **external fragmentation** (free memory exists but no contiguous chunk fits a new reservation).

The fix: store the cache in small, fixed-size, *non-contiguous* blocks, with a per-sequence **block table** — literally a page table mapping logical position → physical block. Blocks are allocated on demand from a shared pool as the sequence grows. The kernel gathers through the table, so memory *looks* contiguous to the attention math while being physically scattered. Fragmentation of both kinds disappears.

The indirection buys more than waste-reduction: because blocks are addressed by table rather than by position, multiple sequences can point their entries at the *same* physical blocks — near-free [prefix sharing](prefix-sharing-is-safe-only-with-refcounts-and-copy-on-write.md), the seed of vLLM's prefix caching and of the [tree-structured caches agentic workloads need](agentic-workloads-are-a-characterization-not-a-technique.md).

File under [caching + indirection](caching-plus-indirection-makes-reuse-shareable.md): the KV cache itself avoids redoing work; the block table is the indirection that makes the reuse shareable. The economic motivation is that [cache traffic, not weight traffic, is the term that grows](kv-cache-traffic-scales-with-batch-times-context.md).
