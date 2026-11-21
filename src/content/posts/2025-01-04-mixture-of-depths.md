---
layout: single
title: "Mixture-of-Depths: Dynamically allocating compute in transformer-based language models"
date: 2025-01-04
categories:
  - AI
  - paper-summary
tags:
  - model compression
  - Dynamic compute allocation
  - summary
---

Compute is a significant bottleneck for current AI model performance and scale ((cite)), making model compression increasingly important. In this paper, the authors propose a novel method to dynamically allocate compute across layers, reducing total cost while maintaining performance.

 <!-- excerpt-end -->

The proposed Mixture-of-Depths (MoD) method enforces a total compute budget by capping the number of tokens that can participate in self-attention and MLP computations at each layer, allowing the network to determine which tokens are processed using a top-k mechanism. This enables non-uniform FLOP expenditure across both time and model depth dimensions while maintaining predictable total compute cost. Empirically, MoD transformers improve upon vanilla transformers by up to 1.5% on the final log probability training objective for equivalent FLOPs, and can achieve training loss parity with vanilla transformers using only around 50% of the FLOPs per forward step, demonstrating intelligent routing capabilities.

## Implementation

The authors define capacity as the total number of tokens comprising the input to a given computation (in transformers, both MLP and self-attention layers typically process T tokens - the total across sequence and batch). They hypothesize that certain tokens might require less processing than others and that these tokens can be identified through learning. Routing occurs at the block level, where each block assigns scores to tokens across the entire sequence, with the top-k scoring tokens passing through the block computation while others continue unchanged through the residual stream. Score allocation is achieved by learning an additional weight matrix $W_{\theta}^l$ and assigning each token the scalar score $r_i^l = W_{\theta}^Tx_i^l$. The block output is then computed as:

$$
x_i^{l+1} = \begin{cases}
r_i^l f_i(\tilde{X}^l) + x_i^l, & \text{if } r_i^l > P_\beta(R^l) \\
x_i^l, & \text{if } r_i^l < P_\beta(R^l)
\end{cases}
$$

where $P_\beta(R^l)$ is the $\beta$-th percentile of router weights with $\beta = 1-C/S$, C being the user-defined capacity and S the sequence length. The router weights multiply the block's computation output, placing them along the gradient path and enabling learning through backpropagation.

## Results

The authors demonstrated several key findings:

1. MoD transformers with aggressive capacity reduction (12.5% of sequence length) and routing every other block achieved optimal performance
2. Learned routing proved crucial, as stochastic routing performed significantly worse
3. For various FLOP budgets (6e18, 2e19, and 1e20), optimal MoD transformers consistently required more parameters than baseline models but achieved better performance
4. MoD variants achieved comparable or better performance than baselines while requiring fewer FLOPs per forward pass, translating to faster inference
5. The optimal MoD transformer configuration uses approximately the same FLOPs per forward pass as the isoFLOP optimal baseline, allowing for predictable performance optimization
6. The approach showed minimal degradation when switching to autoregressive sampling, with the auxiliary routing prediction task achieving 99% accuracy

This is a good start for an opinion section, but it could be expanded to provide more depth and critical analysis. Here's how I would enhance it:

## My Opinion

This paper introduces a simple, intuitive, and efficient method to substantially decrease FLOPs and increase inference speed while maintaining performance. The simplicity of implementation makes it particularly valuable for practical applications, as it requires minimal modifications to existing transformer architectures and can be readily adapted to various use cases.

The results are compelling, but several interesting questions remain unexplored:

1. Adaptive Capacity:

- While the paper uses fixed capacity per layer, dynamically adjusting capacity based on input complexity could yield further improvements
- The capacity could potentially be learned during training or adjusted based on metrics like prediction entropy
- Different layers might benefit from different capacities, particularly given the known specialization of transformer layers

2. Integration with Other Methods:

- The paper demonstrates compatibility with MoE (Mixture-of-Experts), but exploration with other efficiency techniques like pruning or quantization could be valuable
- The interaction between MoD and attention mechanisms like sparse attention or linear attention remains unexplored

3. Future Directions:

- The authors' suggestion of selective Key/Query token processing is particularly intriguing and could lead to more efficient attention mechanisms
- This could potentially be extended to long-context models, where selective processing of historical context could significantly reduce memory requirements
- The method could be adapted for multi-modal transformers, where different modalities might require different computational resources
- The trade-off between parameter count and per-token computation could be explored more thoroughly

The strength of this work lies in its practical applicability and the clear path it provides for future research. The demonstrated ability to achieve better performance with fewer FLOPs suggests that current transformer architectures may be computationally inefficient, and MoD provides a promising framework for addressing this inefficiency.
