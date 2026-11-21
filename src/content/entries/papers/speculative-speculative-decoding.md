---
title: "Speculative Speculative Decoding"
date: "2026-08-17"
type: "paper-summary"
subtype: null
topic: "Speculative decoding"
source_title: "Speculative Speculative Decoding"
source_author: null
source_url: null
---

The approach is summarized here: 
```
We introduce speculative speculative decoding (SSD), a unifying framework for methods that aim to parallelize drafting and verification. While in SD the draft model waits for the verification to complete before beginning to speculate the next round, in SSD the draft model predicts what verification outcomes are most likely, and prepares (i.e., pre-speculates) for all of them in parallel to the verification taking place. If any of these prepared outcomes occurs, the draft model can immediately send the pre-speculated tokens to the verifier, thereby avoiding all draft overhead. Like ordinary speculative decoding, SSD is lossless. Unlike ordinary speculative decoding, in SSD the draft model is located on distinct hardware from the target.
```
The speculator predicts possible outcomes for the speculation he proposed. Then, for each of the possible outcomes he creates a separate speculation and stores it in the cache. When the verifier outcome actually returns, the result is compared to the cache and if a cache hit is obtained, that means that a speculation has already been prepared for it ahead of time, and it is immediately returned.
In order to do this efficiently, the following algorithm is proposed: 

**Algorithm 1:** The Speculative Speculative Decoding (SSD) Framework.

```
Function main(prompt, target, primary_draft, backup_draft):
    asynchronously launch speculator(prompt, primary_draft, backup_draft)
    generated_tokens <- verifier(prompt, target)
    return generated_tokens

Function verifier(prompt, target):
    target.prefill(prompt)
    WAIT TO RECEIVE spec_tokens from speculator
    generated_tokens <- []
    while True do
        verify_outcome <- target.verify(spec_tokens)
        generated_tokens.append(verify_outcome.tokens)
        SEND verify_outcome to speculator
        if end_token in verify_outcome then
            return generated_tokens
        end
        WAIT TO RECEIVE spec_tokens from speculator
    end

Function speculator(prompt, primary_draft, backup_draft):
    primary_draft.prefill(prompt)
    spec_tokens <- primary_draft.speculate(prompt)
    while True do
        SEND spec_tokens to verifier for verification
        outcomes <- predict_verify_outcomes(spec_tokens, primary_draft)   // Sec. 4.1
        cache    <- speculate_for_outcomes(outcomes, primary_draft)       // Sec. 4.2
        WAIT TO RECEIVE verify_outcome from verifier
        if end_token in verify_outcome then
            return
        end
        if verify_outcome in cache then
            spec_tokens <- cache[verify_outcome]
        else
            spec_tokens <- fallback_speculate(
                verify_outcome, primary_spec, backup_spec
            )                                                            // Sec. 4.3
        end
    end
```

**Theorem 7.** *Let the primary and backup speculators take time $T_p$ and $T_b$ relative to the verifier. Let $E_{\text{hit}}$ and $E_{\text{miss}}$ denote the expected number of generated tokens from the primary and backup speculators, respectively. The expected speedup of Algorithm 1 relative to autoregressive decoding is then:*
$$
\textit{speedup}_{\text{SSD}} \;=\; \frac{p_{\text{hit}} \cdot E_{\text{hit}} + (1 - p_{\text{hit}}) \cdot E_{\text{miss}}}{p_{\text{hit}} \cdot \max(1, T_p) + (1 - p_{\text{hit}}) \cdot (1 + T_b)}.
$$

**Definition 10.** *We define the **fan-out** $F_k^p := \big|\{v_T := (k', t^*) \in S_T \mid k' = k\}\big|$ at position $k$ to be the number of verification outcomes with $k$ accepted tokens that are included in the speculation cache, given the previous iteration was speculated by the primary speculator. $F_k^b$ is defined analogously for the backup speculator.*

**Definition 11.**[^1] *We say that a speculator has a **r power-law cache hit rate** if the chance of a cache miss with fan-out $F$ is equal to a power-law of $F$ with exponent $r$, for a sequence drafted by this speculator. More specifically, this means $1 - p_{\text{hit},*}(F) = 1/F^r\ \ \forall F \in \mathbb{N}$, for some $r > 0$.*

**Theorem 12.** *(**Saguaro Cache Shape: Geometric Fan-Out**) Consider a draft model with acceptance rate $a_p$ and a $r$ power-law cache hit rate. Then the optimal choice of $F_k^p$ values for $k \in [0, K]$ under the constraint $\sum_{k=0}^{K} F_k^p \le B$, follows a capped geometric series:*
$$
\begin{aligned}
F_k &= F_0 \cdot a_p^{k/(1+r)} \quad \forall k < K, \ \textit{and} \\
F_K &= F_0 \cdot a_p^{K/(1+r)} \cdot (1 - a_p)^{-1/(1+r)},
\end{aligned}
$$
*where $F_0$ can be chosen such that $\sum_{k=0}^{K} F_k^p = B$ (closed form-equation in Appendix A). An equivalent result holds for $F_k^b$.*

### 4.2.1 Algorithm

**Definition 13.** *We define a **sampling scheme** as a function $\sigma \colon \mathbb{R}^V \to \Delta^{V-1}$ from model logits $z_{\text{draft}} \in \mathbb{R}^V$ to a probability distribution $p \in \Delta^{V-1}$.*

To understand how to design a sampling scheme that increases the residual probability mass on the most likely draft tokens, we first note that the probability of token $t$ in the residual distribution is proportional to $\max(p_{\text{target}}(t) - p_{\text{draft}}(t), 0)$. Thus, *decreasing* $p_{\text{draft}}(t)$ *increases* the probability of $t$ in the residual. This is exactly what our cache-aware sampling scheme does.

**Definition 14.** *Given draft logits $z \in \mathbb{R}^V$, we define the **Saguaro sampling scheme** $\sigma_{F,C}(z)$ for fan-out $F$ and downweighting constant $C \in [0, 1]$ as*
$$
\sigma_{F,C}(z) \propto
\begin{cases}
C \cdot \exp(z_t) & \textit{if } t \in \textit{top}_F(z) \\
\exp(z_t) & \textit{otherwise,}
\end{cases}
$$

In practice, $C$ is a hyperparameter that is chosen empirically.

**Theorem 17.** *The optimal fallback strategy for cache misses, given two speculators of varying speeds and quality, is to use the slow and accurate speculator as the backup for batch sizes $b < b^*$, and the fast speculator otherwise. We solve for $b^*$ in Appendix A.*
