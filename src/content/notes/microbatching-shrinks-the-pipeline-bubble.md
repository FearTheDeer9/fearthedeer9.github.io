---
permalink: /notes/microbatching-shrinks-the-pipeline-bubble/
title: "Microbatching shrinks the pipeline bubble from (P−1)/P toward (P−1)/M"
created: 2026-07-13
---

[Pipeline parallelism's defining cost is the bubble](tp-pays-in-all-reduces-pp-pays-in-bubbles-dp-pays-in-memory.md): with one monolithic batch flowing through $P$ stages, each GPU idles while the others work — $\frac{P-1}{P}$ of total GPU-time wasted to fill and drain.

The fix is to split the batch into $M$ microbatches and stagger their entry: as soon as stage 1 finishes microbatch 1 and ships it to stage 2, it starts microbatch 2 instead of waiting. In steady state every stage works simultaneously on a different microbatch; the bubble survives only at the initial fill and final drain, shrinking the wasted fraction to roughly

$$\frac{P-1}{M}$$

which approaches zero as $M \gg P$.

Worth keeping the vocabulary load-bearing: this is *not* "ring style" anything — ring all-reduce is a communication topology for TP-style collectives, a different concept that happens to share a word.

Structurally, microbatching is the same move as [continuous batching](continuous-batching-schedules-at-iteration-granularity.md): keep every unit of hardware busy by feeding a steady stream of smaller work-items instead of one large synchronized cohort — applied to pipeline stages instead of decode iterations. That makes it an [amortization](amortization-pay-once-spread-over-more-work.md) play (spreading the fixed fill/drain cost over more items), which is why it files under a different thread than TP/DP's [capacity scale-out](scale-out-changes-the-problem-category.md) even though all three live in the "parallelism" chapter of every textbook.
