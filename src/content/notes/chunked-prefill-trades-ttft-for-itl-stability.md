---
permalink: /notes/chunked-prefill-trades-ttft-for-itl-stability/
title: "Chunked prefill trades one request's TTFT for everyone's inter-token latency stability"
created: 2026-07-13
---

[Continuous batching](continuous-batching-schedules-at-iteration-granularity.md) lets a new request join a batch that's mid-decode for everyone else. But its prefill is one large, compute-bound matrix–matrix operation, and a batched iteration is one kernel launch: run the prefill atomically and every co-scheduled decode request waits for the whole thing before getting its next token. Under real traffic this makes inter-token latency wildly jittery — great throughput on paper, terrible experience.

Chunked prefill splits the new request's prompt into small pieces (say 256 tokens) and interleaves those chunks with ongoing decode steps across iterations. No single iteration is dominated by one giant prefill; the prompt gets processed gradually, sharing the GPU step by step.

The cost lands on the new request itself, and the mechanism matters: it is **opportunity cost, not overhead**. Even with zero scheduling bookkeeping, the prefill's work is now competing for GPU time-slices with other requests' decode steps — between each of your chunks, the GPU is serving someone else. Time-to-first-token stretches from "however long the compute takes, uninterrupted" to "the compute, plus everyone else's decode steps interleaved in between."

This is the throughput-vs-TTFT tension in its most concrete form — the same abstract trade-off that appears the moment you batch anything at all, here resurfacing as a scheduling-turned-interleaving decision. One request's first token is sacrificed, bounded and knowingly, for the stability of every in-flight stream.
