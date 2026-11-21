---
title: "Benchmarking Composable Compression Techniques in Mixture-of-Experts LLMs"
date: "2026-08-26"
type: "paper-summary"
subtype: null
topic: "Inference systems"
source_title: "Benchmarking Composable Compression Techniques in Mixture-of-Experts LLMs"
source_author: null
source_url: null
---

MoE models scale model capacity in an efficient manner through sparse activation. However the large parameter footprint, routing imbalance and long context KV-cache growth make deployment difficult which gave rise to compression techniques of expert pruning, weight quantization and KV cache compression. In this work, the authors present a systematic benchmark which evaluates how all of these techniques compose rather than testing each in isolation. They measure compression quality, workload and architecture robustness, pruning/quantization/KV cache sensitivity and deployment efficiency on commodity hardware. They show that there exists non-trivial interactions among the different methods and that it can't be predicted from standalone techniques, nor does compression rate along reliably predict quality loss. 

Authors Conclusion: "Our takeaway. By extensively benchmarking MoE models across diverse architectures, compression rates, and hardware backends, we find that MoE compression must be evaluated as a deployment pipeline rather than as isolated accuracy-preserving steps. We observe that model quality is dependent on compression axis, architecture, and is non-additive, with expert pruning dominating degradation. Hardware efficiency is often decoupled from memory reduction due to KV-cache and low-bit dequantization overheads. These observations underscore the practical value of MoE-XBench in identifying the compression point that preserves quality while delivering real throughput and memory gains."

The authors craft a benchmark for this and close by noting the following - the effects of pruning are not linear nor are they predictable, they should be done with care, with consideration to the interaction between model <> data <> hardware. They note that practitioners should targer accuracy<>memory<>latency points rather than memory reductions since naive modifications do not always translate to throughput gains. Further, they note that moderate quantization should be the default go to as is seems low risk whereas expert pruning is architecture-specific tuning. KV-cache compressions should be treated as long-context memory reductions not necessarily runtime optimization since the dequantize overhead can slow decode. Finally, to deal with the non-obvious interactions between the various techniques they suggest one validates the composed compression as a complete deployment pipeline on the targeted backend and measure "real" metrics (acc, robustness, architecture reliability, peak memory, prefill throughput, decode throughput)
