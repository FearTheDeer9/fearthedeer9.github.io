---
permalink: /notes/continuous-batching-schedules-at-iteration-granularity/
title: "Continuous batching moves scheduling from batch granularity to iteration granularity"
---

Naive batching forms a fixed cohort: gather 8 requests, launch them together, admit nothing new until all 8 finish. Since decode length is unknown in advance and varies per request, the cohort's duration is dictated by its *slowest member* — every request that finished at step 3 sits holding a GPU slot doing nothing, and the "you up?" ping in the queue waits behind a 500K-token reformatting job. The failure isn't that requests have different resource needs; it's that admission only happens at batch boundaries.

Continuous batching re-decides batch membership **after every single decode step**: sequences that hit EOS are evicted immediately, and their slots are refilled from the queue on the very next iteration. Membership is fluid; nobody waits for a cohort, only for a slot.

This is the site where the memory and kernel techniques actually get *used* — it's the layer deciding, moment to moment, which requests run at all, and it leans on [PagedAttention's](pagedattention-is-virtual-memory-for-the-kv-cache.md) flexible allocation to admit and evict without reshuffling contiguous memory.

It also creates its own problem: a newly admitted request needs its [compute-bound prefill](prefill-is-compute-bound-decode-is-memory-bound.md) run inside an iteration where everyone else is doing short, memory-bound decode steps. Run atomically, a 2000-token prefill is one long kernel launch that stalls every in-flight decode's next token — a synchronous latency spike injected into everyone's inter-token latency. That tension is what [chunked prefill](chunked-prefill-trades-ttft-for-itl-stability.md) resolves.

Structurally, "keep hardware busy by feeding smaller staggered work-items instead of one synchronized cohort" is the same move as [microbatching in pipeline parallelism](microbatching-shrinks-the-pipeline-bubble.md).
