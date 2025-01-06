---
layout: single
title: "Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study"
date: 2025-01-05
categories:
  - AI
  - paper-summary
tags:
  - fine tuning
  - model alignment
  - summary
---

RLHF is the most used method to align LLMs with human preferences.
RLHF methods can be roughly categorized as either reward-free or _reward-based_.

 <!-- excerpt-end -->

Reward-Based methods first learn a reward model and then apply actor-critic algorithms such as PPO. Reward-Free methods bypass learning the reward-model and use the feedback directly to modify the underlying model such as DPO.

In this paper the authors try to answer the question of whether DPO or PPO is superior from a performance perspective. They conclude that PPO is able to surpass other alignment methods in all cases and achieve SOTA results. The authors reveal the key factors for the best performances of PPO in fine-tuning LLMs as well as uncovering fundamental limitations of DPO.

## Algorithm backgrounds

### PPO

A reward model $r_\phi \in R$ is first learned from human labeled data. A common practice is to collect a dataset of preference pairs $\mathbb{D}= \{(x,y_w,y_l)\}$, $y_w, y_l$ are responses to x and marked as "win" and "lose" by a human-annotator. The distribution of the preference dataset is assumed to follow a Bradley-Terry model : $$(1) \quad P_\phi(y_w>y_l \mid x) = \frac{exp(r_\phi(x,y_w))}{exp(r_\phi(x,y_w))+exp(r_\phi(x,y_l))}$$$r_\phi$ is trained by minimizing the NLL of 1 given D. After $r_\phi$ is obtained PPO can be executed.

### DPO

Instead of learning a reward model, DPO optimizes the policy over the preference data directly, it does so by deriving a closed-form solution for the RLHF objective which leads to the minimization of$$\mathbb{L}_{DPO}(\pi_\theta) = - \mathbb{E}_{(x,y_w,y_l)\sim D} [log \sigma (\beta(\log \frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}- log \frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}]$$

## DPO Limitations

It is known that PPO can exploit potential failures in the learned reward model to achieve high rewards without meeting the actual human preference. The authors prove a theorem showing that any solution found by PPO also minimizes the DPO objective meaning that DPO also suffers from this. In addition, DPO might discover solutions exploiting out-of-distribution data, posing a risk of deviating excessively from the reference policy. **DPO is prone to generating a biased policy that favors OOD responses, leading to unpredictable behavior.**

### Iterative DPO

The authors mention DPO_Iter (Iterative DPO) as a improvement on vanilla DPO.
Instead of using only the existing preference dataset, it: - Generates new responses using the current model - Uses a learned reward model to label preferences between these newly generated responses - Creates new preference pairs from these labeled responses
Then iteratively: - Trains DPO on these newly created preference pairs - Uses the resulting model to generate new responses - Labels new preferences - Repeats the process
The paper shows this helps address one of DPO's main weaknesses - sensitivity to distribution shift between the model outputs and preference data. By generating new responses with the current model, the preference pairs better match the model's current output distribution. Additionally, Since the preferences are generated from the model's own outputs, the training signal is more relevant to its current capabilities.
However, the paper also notes that even with these improvements, DPO_Iter still doesn't match PPO's performance on more challenging tasks like code generation. While iterative training helps address some of DPO's limitations around distribution shift, it doesn't fully overcome the fundamental limitations in how DPO handles out-of-distribution responses that the paper identifies theoretically.

## Key Factors for PPO

The authors identify three key techniques for RLHF performance of PPO.

1. Advantage normalization - i.e $advantages = \frac{advantages-mean(advantages)}{std(advantages)}$
2. Large batch-size training
3. Updating reference model parameters with exponential moving averages - For each parameter in the reference model, update it as $param_{ref} = \alpha \cdot param_{ref} + (1-\alpha) \cdot param_{policy}$ where $\alpha$ is a decay rate between 0 and 1. This allows the reference policy to change over training rather than remaining completely static, causing the KL term regulation not to be too restrictive. This is especially helpful in complex tasks where the model needs to substantially modify its behavior from the initial pre-trained policy to solve complex programming problems.

# My Opinion

The paper's main contribution lies in giving a comprehensive answer to which RLHF method works best. In doing so, it introduces theoretical insights on the drawbacks of DPO as well as establishing key implementation details to help with optimizing the notoriously tricky performance of PPO.
For simple use-cases it may still be the case that DPO is best since it reduces the optimization overhead that comes with DPO but in complex scenarios it seems as PPO is the best we have at the moment.
