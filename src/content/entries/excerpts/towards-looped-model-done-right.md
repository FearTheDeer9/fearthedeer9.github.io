---
title: "Towards Looped Model Done Right"
date: "2026-10-07"
type: "excerpt"
subtype: "article-summary"
topic: "Architectures & scaling"
source_title: null
source_author: null
source_url: null
---

The authors try to formalize and systematize the way we train, architect, and organize loop transformers. To do so they begin by defining the search space. They claim that the framework for looped models should be thought of as three separate choices that are often bundled together. These are:
1. Where recurrence is placed
2. How the input is reintroduced
3. How recurrent states are organized
And they explore each of these questions separately. 
## Notations

Let $\mathbf{x} \in \mathcal{V}^{B \times L}$ denote token indices and $\mathbf{x}_0 = E_\theta(\mathbf{x}) \in \mathbb{R}^{B \times L \times d}$ their embeddings. A general tied iterative model can be written as

$$
\begin{aligned}
\mathbf{e} &= P_\theta(\mathbf{x}_0). \\
\mathbf{z}_0 &= \phi_\theta(\mathbf{e}, \boldsymbol{\xi}). \\
\tilde{\mathbf{z}}_t &= W_\theta(\mathbf{z}_t, \mathbf{e}). \\
\mathbf{z}_{t+1} &= R_\theta(\tilde{\mathbf{z}}_t), \qquad t = 0, \dots, T-1. \\
\mathbf{h} &= C_\theta(\mathbf{z}_T).
\end{aligned}
$$

Here $P_\theta$ is the prelude map [3], $R_\theta$ is the tied iterative body, $C_\theta$ is the coda map. All three are stacks of standard Transformer blocks; the prelude and coda are applied once, whereas the recurrent core is repeatedly applied with shared parameters. $\phi_\theta$ initializes the recurrent state, and $W_\theta$ is the per-step input write. The fixed representation $\mathbf{e}$ is the prelude output, $\mathbf{z}_t$ is the model's recurrent hidden stream, and auxiliary randomness $\boldsymbol{\xi}$ is used only by variants with random state initialization.

## Ouro vs Huginn
Under this notation, an _Ouro_-style topology sets $P_\theta, W_\theta$, and $C_\theta$ to identity maps, initializes $z_0=x_0$, and applies the same stack for iterations.
A _Huginn_-style topology keeps nontrivial prelude and coda maps and writes the fixed representation into the recurrent stream at each iteration![](/attachments/towards-looped-model-done-right-Pasted-image-20261007143747.png)
# Results 
![](/attachments/towards-looped-model-done-right-Pasted-image-20261007143954.png)![](/attachments/towards-looped-model-done-right-Pasted-image-20261007144030.png)![](/attachments/towards-looped-model-done-right-Pasted-image-20261007144140.png)![](/attachments/towards-looped-model-done-right-Pasted-image-20261007144406.png)![](/attachments/towards-looped-model-done-right-Pasted-image-20261007144439.png)![](/attachments/towards-looped-model-done-right-Pasted-image-20261007144454.png)![](/attachments/towards-looped-model-done-right-Pasted-image-20261007144615.png)![](/attachments/towards-looped-model-done-right-Pasted-image-20261007144837.png)

## Transfer to moe
![](/attachments/towards-looped-model-done-right-Pasted-image-20261007145103.png)
![](/attachments/towards-looped-model-done-right-Pasted-image-20261007145019.png)
![](/attachments/towards-looped-model-done-right-Pasted-image-20261007145517.png)
![](/attachments/towards-looped-model-done-right-Pasted-image-20261007145550.png)
# Takeaways

We compare full-stack and middle-loop language models by disentangling three design choices that standard _Ouro_- and _Huginn_-style recipes vary simultaneously: the iteration envelope, input injection, and latent-state organization. Under matched parameter scale, logical depth, and training-token budget, we find that the prelude–loop–coda sandwich architecture improves performance on mathematical reasoning, context extraction, and other reasoning-intensive tasks. Input injection primarily benefits knowledge retrieval and context access, while contributing little to quantitative reasoning. Random state initialization yields mixed results, whereas the shared H/L hierarchy provides no consistent benefit.

Our MoE transfer study evaluates the two complete architectures in a modern sparse-model setting and shows that the advantages of the _Huginn_-style architecture persist. The _Huginn_ MoE achieves more balanced expert utilization and approaches—and on some tasks surpasses—the performance of its feedforward MoE counterpart, despite being trained from scratch under the same data budget and matched memory and FLOP constraints.

More broadly, the architectural design space of looped models remains underexplored and calls for substantially more controlled ablation studies. Meaningful comparisons therefore require carefully designed, apples-to-apples evaluations that match parameter scale, training budget, inference cost, and evaluation protocol. Many important questions remain open, including how loop architectures interact with conditional sparsity and how architecture–system co-design can further advance their performance–efficiency Pareto frontier
