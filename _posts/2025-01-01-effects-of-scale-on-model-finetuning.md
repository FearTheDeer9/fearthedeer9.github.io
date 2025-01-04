---
layout: single
title: "Effects of Scale on Model Finetuning"
date: 2024-01-01
categories: paper-summary machine-learning research
tags: [llm, finetuning, scaling-laws]
use_math: true
---

Finetuning is a type of [[knowledge editing for LLM]], it is required and widely adopted to unlock new and robust capabilities for creative tasks, get the most for focused downstream tasks, and align its value with human preferences. Whenever there is more annotated task-specific data that has been accumulated, the more finetuning is a good option. FInetuning performance is affected by pretraining conditions such as LLM model size and pretraining data size as well as finetuning conditions, such as downstream task, finetuning data size and finetuning methods. Intuitively, the pretraining controls the quality of the learned representation and knowledge in pretrained LLMS and the finetuning affects the degree of transfer to the downstream task. This paper explores whether and how LLM finetuning scales with the aforementioned factors.

<!-- excerpt-end -->

# Main Findings

- Proposed multiplicative joint scaling law for LLM finetuning:
  $$\mathbb{f}(L,X,D_f) = A \cdot \frac{1}{X^{\alpha}} \cdot \frac{1}{D_f^{\beta}} + \epsilon$$
  Where {A,E $\alpha, \beta$} are data-specific parameters to be fitted, $D_f$ denotes finetuning data size and X refers to each of the other scaling factors. E is irreducible loss.
- Scaling LLM model benefits LLM finetuning more than scaling pretraining data
  - Intuitively finetuning heavily relies on the knowledge encoded in the LLM, where LLM model size and pretraining data size both matter. Empirically the former has a larger effect, although the difference is scaling is highly task dependent.
- Increasing [[PET]] (parameter efficient tuning) parameters doesn't scale well for LoRA and Prompt, although LoRA shows better training stability
  - The amount of newly added trainable parameters can for a bottleneck on the expressivity of the PET method. Empirically these have marginal effect at best. More parameters for Prompt can make training instable as optimizing larger prompt embedding becomes non-trivial. Carefully optimizing finetuning hyperparameters and initialization can alleviate these to some extent.
- The scaling property for LLM finetuning is highly task- and data-dependent, making the selection of optimal finetuning method for a downstream task non-trivial.
  - The effect of finetuning data size is larger for FMT compared to PET. Lora has a higher scaling factor compared to Prompt. LoRA often perform better than Prompt with more finetuning data while Prompt is better when only a few thousand finetuning examples are available.
- LLM-based finetuning could encourage zero-shot generalization to relevant tasks, and PET performs much better than FMT.
  - With PET the majority of LLM parameters are frozen during finetuning, thus it relies heavily on encoded knowledge in pretrained LLMs when adapting them to downstream tasks. This hypothesis is supported by the fact that the scaling factors for PET are consistently higher for model size and pretrain data size than finetuning data size. It is further supported by the fact that the performance gap between FMT and PET is substantially narrowed with larger LLMs.

# Takeaways

- There is no universal answer for which finetuning method to apply. There should exist a critical point for finetuning data size beyond which one finetunning method performs better than the other but the highly non-linear scaling law causes this to be hard to derive analytically. Instead, we should estimate this empirically. The scaling trends and critical values are highly dependent on the downstream task. All in all, this is not understood well enough to say anything confidently say anything quantitatively but we can rely on the following rule of thumb - When only few thousands of examples are available consider PET, either Prompt or LoRA, when with slightly larger datasets, LoRA would be preferred due to stability and better finetuning data scalability, For million-scale datasets FMT should be best.
- When considering generalization to closely related tasks, PET methods tend to perform better than FMT, particularly when the LLM is large. This is likely due to the fact that with PET most parameters are frozen and the learned knowledge is inherited. This suggests that when generalization capability is a big concern, PET should be considered.

# My Opinion

The paper makes significant practical and methodological contributions to the field of LLM finetuning. For practitioners and architects, it proposes a scaling power law that helps reason about expected performance gains when scaling various training factors. The paper also introduces valuable rules of thumb for selecting appropriate finetuning methods based on specific use cases. Furthermore, its systematic methodology for studying the relationships between different scaling factors provides a valuable framework for future research in this area.

However, most of the paper's findings are empirical and lack theoretical grounding. The high task dependency of their results suggests the existence of important underlying factors that are not currently captured in their models. The strong variation across tasks, while systematically documented, indicates that our understanding of the fundamental principles governing LLM finetuning remains incomplete. This points to the need for further theoretical work to better understand and formalize the relationships observed in their empirical studies.

Despite these limitations, the paper represents an important step forward by providing both practical insights for immediate application and a methodological foundation for future theoretical investigations into LLM finetuning dynamics.

# Setup

- Downstream tasks where machine translation and multilingual summarization. Chosen for their high complexity and rich amount of available finetuning corpora
- The models are decoder-only transformers and trained with the modified [[UL2 Objective]], model sizes ranged from 1-16B and where optimized using [[Adafactor]] for one epoch under a cosine learning decay rate scheduler.
- Evaluation is done on the best checkpoint reached based on [[model perplexity|token level perplexity]].
