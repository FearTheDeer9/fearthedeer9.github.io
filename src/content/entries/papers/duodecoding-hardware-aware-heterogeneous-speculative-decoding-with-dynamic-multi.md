---
title: "DuoDecoding - Hardware-aware Heterogeneous Speculative Decoding with Dynamic Multi-Sequence Drafting"
date: "2026-08-31"
type: "paper-summary"
subtype: null
topic: "Speculative decoding"
source_title: "DuoDecoding - Hardware-aware Heterogeneous Speculative Decoding with Dynamic Multi-Sequence Drafting"
source_author: null
source_url: null
---

They load the target on the GPU and the draft on the CPU, they then simultaneously decode on both, they compare the time for forward pass on each device, define the ratio as c and do $c=\gamma$ . They then modify the forward pass in the following manner:
**Algorithm 2:** Verification Process

**Input:** $n, \mathbf{x}, \tilde{\mathbf{x}}, \mathbf{q}_{\le n}, \hat{\mathbf{q}}_{[s, s/\gamma]}, \hat{\mathbf{x}}_{[s, s/\gamma]}, \mathbf{p}_{\le n}$ from Algorithm 1

```
Function DuoDecVerify:
    ▷ Verify the unverified part in the prefix
    l <- x~.length
    r_1, ..., r_l ~ Uniform(0, 1)
    k <- max({ i | 0 <= i < l, r_{n-i} > p_{n-i} / q_{n-i} })
    if x~.length == 0 or k == 0 then
        ▷ Verify multi-sequence drafts
        k <- -1
        for i <- 1 to s do
            r_i ~ Uniform(0, 1)
            if r_i < p_n / q^_{i,0} then
                k <- i
                x  <- [x_1, x_2, ..., x_n] + x^_{k,:}
                n  <- n + k/γ
                x~ <- x^_{s,:}
                return n, x, x~
            else
                p'_n <- norm(max(p_n - q^_{i,0}, 0))
        ▷ All sequences rejected
        if k == -1 then
            t  ~ p'_n
            x  <- [x_1, x_2, ..., x_n, t]
            n  <- n + 1
            empty x~
    else
        t  ~ norm(max(p_{n-k} - q_{n-k}, 0))
        x  <- [x_1, x_2, ..., x_{n-k}, t]
        n  <- n - k + 1
        empty x~
    return n, x, x~
```
