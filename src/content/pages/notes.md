---
layout: single
title: "Evergreen Notes"
permalink: /notes/
author_profile: false
---

Atomic, densely-linked working notes in the spirit of [Andy Matuschak's evergreen notes](https://notes.andymatuschak.org/Evergreen_notes): each note argues one claim, its title is the claim, and the links are where the structure lives. Unlike blog posts, these are continually revised.

The first series maps LLM inference optimization. Start at the hub:

**[A map of LLM inference optimization](/notes/a-map-of-llm-inference-optimization/)**

## Mechanisms

- [Attention is the only inter-token operation in a transformer](/notes/attention-is-the-only-inter-token-operation/)
- [Prefill is compute-bound; decode is memory-bound](/notes/prefill-is-compute-bound-decode-is-memory-bound/)
- [A kernel is memory-bound below the GPU's ridge point — which is ~300 FLOPs/byte, not 1](/notes/kernels-are-memory-bound-below-the-ridge-point/)
- [Batching is nearly free until the ridge point, then trades latency for throughput ~1:1](/notes/batching-is-nearly-free-until-the-ridge-point/)
- [KV-cache traffic scales with batch × context; weight traffic is fixed per step](/notes/kv-cache-traffic-scales-with-batch-times-context/)
- [FlashAttention keeps the attention matrix out of HBM via fusion, tiling, and online softmax](/notes/flashattention-keeps-the-attention-matrix-out-of-hbm/)
- [PagedAttention is virtual memory for the KV cache, not a new attention algorithm](/notes/pagedattention-is-virtual-memory-for-the-kv-cache/)
- [Prefix sharing is safe only with reference counts and copy-on-write](/notes/prefix-sharing-is-safe-only-with-refcounts-and-copy-on-write/)
- [Continuous batching moves scheduling from batch granularity to iteration granularity](/notes/continuous-batching-schedules-at-iteration-granularity/)
- [Chunked prefill trades one request's TTFT for everyone's inter-token latency stability](/notes/chunked-prefill-trades-ttft-for-itl-stability/)
- [Quantization loss is outlier-driven, not bit-driven](/notes/quantization-loss-is-outlier-driven-not-bit-driven/)
- [Speculative decoding is provably exact; draft calibration, not accuracy, is the lever](/notes/speculative-decoding-is-provably-exact/)
- [TP pays latency in per-layer all-reduces; PP pays in bubbles; DP pays in memory](/notes/tp-pays-in-all-reduces-pp-pays-in-bubbles-dp-pays-in-memory/)
- [Microbatching shrinks the pipeline bubble from (P−1)/P toward (P−1)/M](/notes/microbatching-shrinks-the-pipeline-bubble/)

## Threads

- [Slack arbitrage: find the idle resource and spend it](/notes/find-the-idle-resource-and-spend-it/)
- [Amortization: pay a fixed cost once, spread it over more work](/notes/amortization-pay-once-spread-over-more-work/)
- [Caching avoids redoing identical work; indirection makes the reuse shareable](/notes/caching-plus-indirection-makes-reuse-shareable/)
- [Approximation spans a spectrum from bounded-loss to provably-free](/notes/verified-approximation/)
- [When one unit's budget is exceeded, the problem changes category](/notes/scale-out-changes-the-problem-category/)

## Workload lens

- [Agentic workloads are a workload characterization, not a technique](/notes/agentic-workloads-are-a-characterization-not-a-technique/)
