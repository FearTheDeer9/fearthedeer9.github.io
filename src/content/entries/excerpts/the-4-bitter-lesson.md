---
title: "The 4 Bitter Lesson"
date: "2026-10-06"
type: "excerpt"
subtype: "article-summary"
topic: "RL for LLMs"
source_title: null
source_author: null
source_url: null
---

In today's inference scaling, post-training intensive environment RL is a core component of the model training pipeline. In RL there is a constant tog of war between throughput and stability, most techniques that increase throughput cause the sampled and the trained policy to diverge, slowing learning and destabilizing training. Below we will show how one can alleviate some of these issues:
# Improving Gradient Stability 
In order to maintain stability, it's known that gradient and optimizer states are saved in much higher precision than the model weights, so they maintain bf16 precision for the backward pass even though the forward pass used NVFP4. However, this creates a mismatch as the forward pass calculates $y = x \cdot Q(w_{bf16})$ while the backward does $y = x \cdot w_{bf16}$. the backward pass is unaware of the clipping and rounding mistakes made. It is better if we quantize and dequantize the backward pass, i.e we do $y = x \cdot DQ(Q(w_{bf16}))$.![](/attachments/the-4-bitter-lesson-Pasted-image-20261006121748.png)
However, it is worth noting that this method, while leading to faster learning leads to less stable training and may diverge.
# Four over Six for RL Weight & Activation
They selectively utilize max value 4 rather than 6 for some of the layers with the 4 bit format since a different paper showed how this significantly reduces the maximal possible error. Introducing this has engineering overhead as one needs to make sure that training and sampling use the exact same numerical contract. However, the effort seems worth it as it significantly improves stability and reduces the KLD from the previous stage, nearly recovering baseline![](/attachments/the-4-bitter-lesson-Pasted-image-20261006123124.png)
# Selective Layer Precision
Almost no lower precision recipe works uniformly, instead it's common for specific, sensitive layers to use higher precision. They single out the last 15% of layers as well as the "always on" expert (since it's a good tradeoff of "weight" vs "usage" since we use it every single time).
![](/attachments/the-4-bitter-lesson-Pasted-image-20261006123417.png)

# Results 
When combining all of these, they show that gradient spikes are nearly gone while performance is nearly equivalent with significantly less train time:
	![](/attachments/the-4-bitter-lesson-Pasted-image-20261006123550.png)![](/attachments/the-4-bitter-lesson-Pasted-image-20261006123558.png)![](/attachments/the-4-bitter-lesson-Pasted-image-20261006123610.png)
# Conclusion 
The authors wrap this up wonderfully so I'll attach there version verbatim:
```
At this point, we would normally take this chance to mostly connect this exact work to the company and its goals. Instead, let us share a metaphor. In this work, some challenges naturally arise from the individual components but the most critical questions come from the interaction between the pieces. It goes without saying that if making RL with NVFP4 work was as straightforward as applying an NVFP4 pretraining recipe to the trainers and NVFP4 inference on samplers, there would be no need to write or open-source anything.

Most researchers have heard of the [bitter lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html); a less-discussed-but-equally-thoughtful short essay introduced the “[one-step trap](http://incompleteideas.net/IncIdeas/OneStepTrap.html)”. Modeling complex dynamical systems in terms of local predictions is often a trap. In this work, we mostly focused on the interaction between trainers and samplers; in our company, we focus on the dynamics of the interactions between people and models. If we exclusively build models optimized for standalone tasks, they will never learn the non-local long-horizon value that they can bring by understanding people and their goals and interactions with others. The single-step approach is an easier one, but almost never the one worth taking.
```
