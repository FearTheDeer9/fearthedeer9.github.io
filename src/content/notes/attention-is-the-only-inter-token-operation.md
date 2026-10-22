---
permalink: /notes/attention-is-the-only-inter-token-operation/
title: "Attention is the only inter-token operation in a transformer"
created: 2026-07-13
---

Every layer of a transformer is two blocks, and they divide the labor cleanly: attention performs *inter-token* computation (mixing information across positions), while the MLP/FFN performs only *intra-token* computation — the same weight matrix applied independently to each token's residual stream, with zero mixing between tokens. Delete every attention layer and each token's representation evolves in complete isolation forever.

This division is why the KV cache is an attention-only concept. A future token needs to read the past — and the only operation that reads the past is attention, so the K and V vectors of past tokens are the only thing worth keeping around. There is no "MLP cache" because MLP computation for a past token is never revisited by a future one. The MLP's job is different: it holds the bulk of the model's parametric knowledge and applies per-token nonlinear transformation, blind to context.

Getting this attribution right matters downstream: everything about [prefill/decode asymmetry](prefill-is-compute-bound-decode-is-memory-bound.md) and [KV-cache traffic](kv-cache-traffic-scales-with-batch-times-context.md) is really a statement about attention's memory footprint, and techniques like [FlashAttention](flashattention-keeps-the-attention-matrix-out-of-hbm.md) and [PagedAttention](pagedattention-is-virtual-memory-for-the-kv-cache.md) are attacks on that footprint from two different directions.

Related: [A map of LLM inference optimization](a-map-of-llm-inference-optimization.md)
