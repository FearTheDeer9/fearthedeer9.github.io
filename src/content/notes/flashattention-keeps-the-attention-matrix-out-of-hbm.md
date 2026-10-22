---
permalink: /notes/flashattention-keeps-the-attention-matrix-out-of-hbm/
title: "FlashAttention keeps the attention matrix out of HBM via fusion, tiling, and online softmax"
---

Naive attention computes $S = QK^T$ — the full $N \times N$ score matrix — writes it to HBM, reads it back for softmax, writes again, reads again to multiply by $V$. Multiple round-trips of an $N^2$ object to slow memory, to do comparatively few FLOPs: the tensor cores stall not because there's no work but because HBM traffic dwarfs it. Classic [memory-bound](kernels-are-memory-bound-below-the-ridge-point.md) signature.

The forward-pass fix is **fusion + tiling, not recomputation**. Process Q, K, V in blocks small enough for SRAM (~100× faster than HBM), and fuse matmul → mask → softmax → matmul into one kernel, so the $N \times N$ intermediate is never materialized in HBM — it lives and dies in SRAM per block.

The genuinely clever part is what makes tiling legal at all: softmax needs the max and sum over an entire row, but only one block of the row is resident at a time. The **online softmax** maintains a running max and running rescaled sum; whenever a new block reveals a larger max $m_2 > m_1$, the already-accumulated sum and output are multiplied by $e^{m_1 - m_2}$ before the new block's contribution is added. The correction is *incremental, applied at every max update* — not deferred to the end — which is exactly what allows a single streaming pass with O(1) extra memory per row. (The max-subtraction exists in the first place to stop $e^x$ overflowing in fp16.)

**Recomputation** is real but belongs to the *backward* pass: rather than store the huge score matrix for gradients, FlashAttention keeps only Q, K, V and the per-row softmax statistics, and re-derives attention blocks on the fly — deliberately spending idle compute to avoid HBM-sized storage. Two different techniques, one philosophy: minimize HBM traffic even at some compute cost — [find the idle resource and spend it](find-the-idle-resource-and-spend-it.md).

Contrast with [PagedAttention](pagedattention-is-virtual-memory-for-the-kv-cache.md), which attacks a completely different axis of "attention is expensive": memory *allocation* across requests, not memory *traffic* within the op.
