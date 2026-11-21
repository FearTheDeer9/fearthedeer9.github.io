---
title: "BoW -> JEV"
date: "2026-09-29"
type: "excerpt"
subtype: "article-summary"
topic: "Architectures & scaling"
source_title: null
source_author: "Sebastian Raschka"
source_url: "https://magazine.sebastianraschka.com/p/classifier-history-and-jev"
---

https://magazine.sebastianraschka.com/p/classifier-history-and-jev
# JEV 
Raschka claims that the best way to succinctly explain the attention Jev is getting is as follows: Like ChatGPT in 2022 was exciting because it was a general-purpose chat model that could generate all kinds of texts, one of the reasons the tech community is excited about Jev is that it is the ChatGPT moment for classification, where it can cheaply classify all kinds of text inputs without having to fine-tune a custom classifier for each task. There’s not much known about the architecture and exact training algorithm, except that one of the founders said it was trained via “Reinforcement Learning for Calibrated Decisions”
## API 
![](/attachments/bow-jev-Pasted-image-20260929183617.png)
## Architecture 
Speculation currently holds that the JEV architecture is very similar to modern BERT variants but the core reason they show such good classification on many out-of-the-box classification tasks is largely due to their training data. The CEO claims that they use a 100% synthetic pipeline so this is probably where most work went: integrating and creating a high-quality dataset that is both generalizable and informative but enough for scale. Additionally they have not yet disclosed the algorithm details but they hinted that they used a new training method, which they coined as reinforcement learning for calibrated decisions, or RLCD. Rashka goes on to note that in ICLR in 2025, a technique that could be related, named Reinforcement Learning from Calibrated Rewards, or RLCR, was introduced. 
The technique modifies RLVR by asking the model for a numerical confidence score and adjusting $R= c-(q-c)^2$.  For example, an incorrect answer with 90% (0.9) confidence receives a reward of -0.81. At 20% confidence, the reward is -0.04. And a correct answer at 90% confidence earns 0.99. 
	Readers familiar with evaluating calibrated models may notice that the squared-error term is the Brier penalty for the stated probability that the answer is correct.

## Value Proposition 
cheap quick classification tasks are incredibly valuable and numerous. Examples could be:

- screening for prompt injection
- determining reasoning effort or model choice
- selecting skills from a library
- judging based on well-defined criteria
- finding files when building context
  
  
 essentially any domain where you can't build a high-quality fine-tuning dataset but the classification itself is repetitive and you would want high-quality yet cheap alternatives to auto-regressive models is right in Jeff's wheelhouse
