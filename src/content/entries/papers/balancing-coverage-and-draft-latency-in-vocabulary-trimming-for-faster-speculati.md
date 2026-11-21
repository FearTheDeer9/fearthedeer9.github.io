---
title: "Balancing Coverage and Draft Latency in Vocabulary Trimming for Faster Speculative Decoding"
date: "2026-08-18"
type: "paper-summary"
subtype: null
topic: "Speculative decoding"
source_title: "Balancing Coverage and Draft Latency in Vocabulary Trimming for Faster Speculative Decoding"
source_author: null
source_url: "https://arxiv.org/html/2603.05210"
---

https://arxiv.org/html/2603.05210#S1.p2.1

The authors note the fundamental tradeoff that exists in draft model design - larger vocabs improve token coverage and agreement with the target model but this increases the draft latency and vice versa. They also note that domain-specific workloads use only a small fraction of the full vocabulary -> They view the vocab selection as a constrained optimization problem that balances token coverage and draft latency and optimize a utility function using a Tree-structured Parzen Estimator. Simultaneous work has dealt with this by adding static, frequency-based statistical methods. The authors criticize this since static reductions are oblivious ot the trade off between coverage and latency. They suggest a reward function that handles the tradeoff natively and optimizes for it while being architecture-aware (necessary for latency)
# Method 
1. Formulate vocabulary selection as constrained optimization
2. Compute token coverage from training data
3. Estimate draft model FLOPs
4. Define a utility function 
5. Optimize via Tree-Structured Parzen Estimator 
## Weaknesses
* Latency calculation only takes FLOPs into account whereas communication is a primary source of latency, specifically in the regime where SpecDec is most efficient

## Observations 
* The method seems robust to training data size (the optimal vocab size doesn't change much) at around 13k.
