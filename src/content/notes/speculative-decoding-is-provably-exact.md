---
permalink: /notes/speculative-decoding-is-provably-exact/
title: "Speculative decoding is provably exact; draft calibration, not accuracy, is the lever"
created: 2026-07-13
---

The mechanism, with the directions right (they're easy to invert): a small **draft** model generates $\gamma$ tokens *sequentially* — cheap, it's a tiny model doing normal memory-bound decode. The large **target** model then verifies all $\gamma$ positions in *one parallel forward pass*, prefill-style. Normally $\gamma$ tokens cost the big model $\gamma$ separate [memory-bound weight reloads](prefill-is-compute-bound-decode-is-memory-bound.md); here that cost is paid once and [amortized](amortization-pay-once-spread-over-more-work.md) across $\gamma$ candidates — using compute that was sitting idle during decode anyway, textbook [slack arbitrage](find-the-idle-resource-and-spend-it.md).

Verification: accept draft token $x$ with probability $\min\!\left(1, \frac{p(x)}{q(x)}\right)$, where $p$ is the target's probability and $q$ the draft's. Guaranteed acceptance when the *target* likes the token at least as much as the draft ($p \ge q$); on rejection, resample from the normalized residual $\max(0, p - q)$, which by construction has mass only where the target preferred what the draft under-weighted. The resulting distribution is **provably identical** to sampling the target alone — a genuine free lunch on quality, unlike [quantization's bounded loss](quantization-loss-is-outlier-driven-not-bit-driven.md). See the [approximation spectrum](verified-approximation.md).

What actually determines the speedup is the expected run of consecutive accepts per round — and the lever on it is **calibration, not accuracy**. A draft that's wrong-but-humble (low $q$ on its own picks) still passes the ratio test easily; a draft that's wrong-and-confident inflates $q$ exactly where the target disagrees, suppressing acceptance across every position and collapsing the expected accept-run toward 1 — at which point you pay for the draft *and* get no amortization: strictly worse than plain decoding.

This is why [narrow agentic token distributions](agentic-workloads-are-a-characterization-not-a-technique.md) are speculative decoding's best case: JSON schemas and repeated tool-call shapes are a distribution a small model can be *confidently right* about, and some production setups fine-tune agent-specific drafts to optimize precisely that calibration.
