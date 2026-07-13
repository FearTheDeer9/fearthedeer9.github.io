---
permalink: /notes/agentic-workloads-are-a-characterization-not-a-technique/
title: "Agentic workloads are a workload characterization, not a technique"
---

Everything else in [the inference optimization map](a-map-of-llm-inference-optimization.md) is a technique — an answer to "how do we compute faster, cheaper, leaner." Agents are not a technique. They're an *empirical claim about the shape of the input distribution*: an agent is a loop (read context → reason → act → observe → repeat), and that loop's byproducts determine which techniques pay off hardest. This is workload characterization, the thing that explains why an inference-optimization company would build for agents specifically.

Three properties, three matches:

**Repeated prefixes → prefix caching.** In one-off chat, shared prefixes are incidental (a system prompt, maybe). In an agent loop, the shared prefix is *most of the context* — tool definitions, instructions, all prior iterations — re-sent every cycle plus one small increment, shared across the agent's own iterations and across users. The shared-to-novel ratio is structurally higher, so [block-table sharing](pagedattention-is-virtual-memory-for-the-kv-cache.md) wins structurally bigger; vLLM's explicit prefix caching exists to harvest this automatically rather than by accident.

**Narrow token distributions → speculative decoding.** An agent emitting the same JSON schemas and function names is a distribution a small draft model can nail *with confidence that is warranted* — precisely the [calibration criterion](speculative-decoding-is-provably-exact.md) that determines acceptance rates. Production setups fine-tune agent-specific drafts for exactly this reason.

**Branching execution → tree-structured KV cache.** Subagents fork from a shared history; failed tool calls retry with different arguments; compaction rewrites history instead of appending. This breaks the linear-sequence assumption under nearly every technique above. The natural structure for "many sequences sharing ancestors, diverging at arbitrary points" is a tree (a radix tree over token sequences, in systems like SGLang), where [copy-on-write triggers at each branch point](prefix-sharing-is-safe-only-with-refcounts-and-copy-on-write.md) and PagedAttention's block granularity is what makes forking affordable — only blocks downstream of the fork ever get copied. Eviction over that tree under memory pressure (refcount-zero reclamation, LRU over leaves) is open, active territory.

The pattern: agents concentrate value into the [caching + indirection](caching-plus-indirection-makes-reuse-shareable.md) and [amortization](amortization-pay-once-spread-over-more-work.md) threads — not because those techniques changed, but because the workload's statistics moved toward their best case.
