---
title: "Decoding Speculative Decoding"
date: "2026-08-23"
type: "paper-summary"
subtype: null
topic: "Speculative decoding"
source_title: "Decoding Speculative Decoding"
source_author: null
source_url: null
---

Through 350 experiments the authors aim to delineate the factors that effect the performance of speculative decoding and observe that it relies heavily on the latency of the draft model, and that the draft model's language modeling capabilities are not as strongly correlated with performance as one would thing. These insights lead to new design spaces for draft models which are more hardware efficient. 
* One takeaway from these findings is that width>depth in spec-dec and that models need to be optimized for throughput maximization from the start, not accuracy on downstream tasks
