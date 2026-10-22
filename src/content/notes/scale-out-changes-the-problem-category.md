---
permalink: /notes/scale-out-changes-the-problem-category/
title: "When one unit's budget is exceeded, the problem changes category"
---

Every other optimization thread — [slack arbitrage](find-the-idle-resource-and-spend-it.md), [amortization](amortization-pay-once-spread-over-more-work.md), [caching](caching-plus-indirection-makes-reuse-shareable.md), [approximation](verified-approximation.md) — answers "how do we make existing work more efficient on the hardware we have." Scale-out answers a categorically different question: *this doesn't fit, or doesn't keep up, on one unit at all.* A 70B model at FP16 is ~140GB against 80GB of HBM; no amount of efficiency closes that gap.

[Tensor and pipeline parallelism](tp-pays-in-all-reduces-pp-pays-in-bubbles-dp-pays-in-memory.md) solve *fit* — splitting the model within layers (TP) or between them (PP), each paying its own communication or idleness tax. Data parallelism solves *throughput scale-out* — full replication of the (possibly TP/PP-sharded) unit, batch split across replicas. The two are orthogonal and compose: TP=4 within a node × DP=2 replicas is 8 GPUs answering both questions at once.

The categorization has teeth because it resists a natural mistake: filing all of "parallelism" under amortization. [Microbatching](microbatching-shrinks-the-pipeline-bubble.md) genuinely *is* amortization (spreading bubble waste over more items) — but TP and DP aren't spreading any fixed cost; they're buying capacity. Techniques that live in the same textbook chapter can belong to different threads of the ontology.

This is also the thread where AI engineering most visibly becomes classical distributed-systems engineering — the meta-observation that scale forces the import of an older discipline's tools, undisguised.

Hub: [a map of LLM inference optimization](a-map-of-llm-inference-optimization.md).
