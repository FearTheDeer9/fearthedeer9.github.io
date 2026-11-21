---
title: "CoSine - Collaborative Speculative Inference for Efficient LLM Inference Serving"
date: "2026-08-20"
type: "paper-summary"
subtype: null
topic: "Speculative decoding"
source_title: "CoSine - Collaborative Speculative Inference for Efficient LLM Inference Serving"
source_author: null
source_url: null
---

The author note that speculative inference has two key bottlenecks:
1. Disparate Resource Demands - SSMs incurr roughly 1000x lower compute overhead compared to LLMs but the co-location of the models creates resource contention and elevated costs. Further, there is a clash between compute bound LLMs and memory bound SSMs
2. Constrained Drafter Knowledge - the small models are inherently limited with respect to performance and expressivity. CoSine facilitates collaboration among multiple nodes, assigning the most suitable resources to each phase. 
There are two critical scheduling dimensions for optimization:
3. Temporal pipelining to overlap SSM drafting with LLM verification
4. Spatial load balancing that dunamically allocates batches to SSMs according to specialization

# Key Components 
1. Router between disparate components that maximizes exporaltion vs exploitation. the latter is a function of individual confidence + historical accuracy. 
2. Scheduler that is infra aware and tries to balance the batch size (where batch size is an aggregate of num requests * their individual lengths) to fit current compute availability.
