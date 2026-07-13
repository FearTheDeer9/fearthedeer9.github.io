---
permalink: /notes/prefix-sharing-is-safe-only-with-refcounts-and-copy-on-write/
title: "Prefix sharing is safe only with reference counts and copy-on-write"
---

[PagedAttention's block table](pagedattention-is-virtual-memory-for-the-kv-cache.md) lets two requests with a common prompt prefix point at the same physical KV blocks. With contiguous storage this would require entire allocations to be identical; with paging it's enough for the shared prefix to fill the initial blocks. But sharing is only trivially safe while everyone *reads*.

The moment request A generates a token that must be appended into a block request B still points at, a write into shared state would silently corrupt B's history. The enforcement mechanism is copy-on-write, straight from OS design: each block carries a **reference count**; a write targeting a block with refcount > 1 first allocates a fresh block, copies the content, and redirects the writer's table entry to the private copy — only then does the write proceed. Refcount of 1 means last owner, write in place. This is `fork()` semantics applied to KV pages instead of process memory — a good cross-domain anchor.

The full recipe: shared prefix + refcount tracking + copy-on-write on divergence.

Two consequences worth keeping:

- Refcounts start as a *safety* mechanism and become an *eviction policy*: a block whose refcount drops to zero (last branch referencing it died) is reclaimable, and systems run LRU-like policies over those to survive memory pressure. In branchy [agentic workloads](agentic-workloads-are-a-characterization-not-a-technique.md) this eviction problem is open and active.
- Copy-on-write at block granularity is what makes *tree-structured* sharing tractable: forking a sequence duplicates only blocks downstream of the fork point, never the whole history.
