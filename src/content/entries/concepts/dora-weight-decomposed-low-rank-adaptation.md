---
title: "Dora (Weight Decomposed low-rank adaptation)"
date: "2024-02-27"
type: "concept"
subtype: null
topic: "Continual learning & optimization"
source_title: null
source_author: null
source_url: "https://arxiv.org/abs/1602.07868"
---

An extension built on top of LoRa  (Low Rank Adaptation). 
From a high level, DoRA implements two steps:
1. Decomposing a pretrained weight matrix into a magnitude vector **m** and a directional matrix **V**. The decomposition is done by taking the magnitude vector **m** to be the vector-wise norm of the original weight matrix W_0, and the initial directional matrix to be the original weight matrix W_0 where each column is divided by its vector wise norm.
2. Applying LoRA to the directional matrix **V** and training the magnitude vector **m** separately.
The update step is calculated by: **W’ = m (V + Delta_V)/Norm = m (W+ AB)/Norm** where the normalization is based on [Weight Normalization : Reparameterization to Accelerate training of deep neural networks](https://arxiv.org/abs/1602.07868)
Here is an illustration of the process from the original paper:
![](/attachments/dora-weight-decomposed-low-rank-adaptation-image.png)
 The motivation for DRA is based on the authors having analyzed and compared the LoRa and full fine-tuning learning patters and finding out that LoRA either increases or decreases magnitude and direction updates proportionally but lacks the capability to make only subtle directional changes as found in full fine-tuning. They propose decoupling these components, proposing a method which trains the directional component **V** while allowing the magnitude **m** to be trained separately. 
DoRA seems to be much more robust to changes in rank as compared to LoRA, and that it even outperforms LoRA when having half the rank.
## Code Implementation 
![](/attachments/dora-weight-decomposed-low-rank-adaptation-image-2.png)
