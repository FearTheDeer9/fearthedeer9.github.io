---
permalink: /notes/batching-is-nearly-free-until-the-ridge-point/
title: "Batching is nearly free until the ridge point, then trades latency for throughput ~1:1"
---

During decode, model weights are loaded from HBM once per step and reused across every sequence in the batch — weights are shared, activations are batched, classic GEMM reuse. So in the memory-bound regime, the step time is dominated by that one memory transfer and is essentially independent of batch size $B$: increasing $B$ multiplies useful work while latency stays flat. Throughput scales ~linearly. This is [amortization](amortization-pay-once-spread-over-more-work.md) in its purest form.

The free lunch ends at the [ridge point](kernels-are-memory-bound-below-the-ridge-point.md). As $B$ grows, arithmetic intensity climbs (FLOPs grow with $B$; bytes moved grow much more slowly), and past $B_{\text{crit}}$ the compute units — not the bus — become the constraint. From there, per-step latency climbs roughly linearly with $B$: every extra unit of throughput is paid for in latency at about 1:1.

The correction worth internalizing: the transition is **smooth, not a wall**. Nothing breaks at $B_{\text{crit}}$ and there is no latency cliff — the *marginal cost* of batching changes character, from nearly-free to a straight trade. "Free lunch, then a fair price" is the accurate slogan; "hits a wall" is not.

Two caveats shape where this matters in practice: [KV-cache traffic scales with B × context](kv-cache-traffic-scales-with-batch-times-context.md), so the bytes-moved term is not constant at long context; and forming batches at all raises the scheduling question that [continuous batching](continuous-batching-schedules-at-iteration-granularity.md) answers.
