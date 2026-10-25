---
permalink: /notes/tp-pays-in-all-reduces-pp-pays-in-bubbles-dp-pays-in-memory/
title: "TP pays latency in per-layer all-reduces; PP pays in bubbles; DP pays in memory"
created: 2026-07-13
---

A 70B model in FP16 is ~140GB of weights against an H100's 80GB of HBM — some models simply don't fit, which makes parallelism a *feasibility* question before it's a performance one. The three schemes split different things and pay in different currencies:

**Tensor parallelism** splits weight matrices *within* a layer (e.g. the FFN up-projection column-wise across 4 GPUs, each computing a partial output from a full copy of the activations). Partial results must be summed — an **all-reduce**, and not once per forward pass but typically twice per layer (after attention, after FFN), for every layer, at every token step. That's a torrent of small, synchronous communication events on which each layer's forward progress is gated, so TP hurts latency directly (and throughput as a consequence: stalled time is pure overhead). Viable only over NVLink-class interconnect within a node — essentially never across nodes.

**Pipeline parallelism** splits *between* layers — GPU 1 holds layers 1–8, GPU 2 holds 9–16. Communication is tiny (one activation vector per stage boundary) and tolerates slow links, even cross-node. Its defining cost isn't communication at all but the **pipeline bubble**: stage 2 can't start until stage 1 ships, and a naive single-item pipeline idles $\frac{P-1}{P}$ of total GPU-time across $P$ stages. [Microbatching](microbatching-shrinks-the-pipeline-bubble.md) is what makes PP livable.

**Data parallelism** splits nothing about the model — full replication, batch split across replicas. Its cost is memory (every replica holds everything), and it presupposes the model already fits, so it's orthogonal: TP/PP solve *fit*, DP solves *scale-out*. A typical composition: TP=4 within a node × DP=2 replicas of that shard-group. See [when one unit's budget is exceeded, the problem changes category](scale-out-changes-the-problem-category.md).
