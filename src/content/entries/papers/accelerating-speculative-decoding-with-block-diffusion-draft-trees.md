---
title: "Accelerating Speculative Decoding with Block Diffusion Draft Trees"
date: "2026-08-16"
type: "paper-summary"
subtype: null
topic: "Speculative decoding"
source_title: "Accelerating Speculative Decoding with Block Diffusion Draft Trees"
source_author: null
source_url: null
---

An intuitive idea that I, along with presumably anyone that has thought about what is the next step for spec-dec post block diffusion models, has probably had - replace blocks diffusion, which is a single chain with a tree that maximizes the expected acceptance length given a budget B. It works in the following manner:
1. Run the block diffusion drafter once to obtain per-position distributions for the next L draft positions after b. 
2. Build a draft tree with B nodes from those distributions. The node budget B controls how many candidate tokens the verifier (target model) evaluates.
3. Compile the tree into input tensors for the target model and run one target-model forward pass with tree attention. 
4. Walk the tree, at each step, use the target model’s decoding rule to choose the next token and check whether it matches a child in the tree; accept the matched drafted path, and carry the first unmatched target token (the next bonus token) to the next round


The idea as mentioned is not novel nor very ground-breaking. The key thing here is about the implementation and execution, how efficient can one make this. To this end, the authors here introduced the following innovations:
* The ideal tree-construction objective is

$$
\max_{T:\,|T|\le B,\ T\text{ valid}} \ \mathbb{E}_{Y_{1:L}\sim p(\cdot\mid c,b)}\big[\alpha_T(Y_{1:L})\big],
\tag{5}
$$

however, is infeasible to optimize as it depends on the target-conditioned continuation probabilities, which are unavailable when the tree is constructed. They therefore use a surrogate based on the factorized distribution induced by the block diffusion drafter $Q(\cdot \mid c, b)$, constructed solely from the per-position marginals $\{q_i\}_{i=1}^{L}$. This surrogate does not require full path-conditioned probabilities, which are unavailable from a single block diffusion forward pass.

$$
\max_{T:\,|T|\le B,\ T\text{ valid}} \ \mathbb{E}_{Y_{1:L}\sim Q(\cdot\mid c,b)}\big[\alpha_T(Y_{1:L})\big].
\tag{6}
$$

They show that it is equivalent to greedily choosing the Top-B prefixes where the probability is given by multiplying the full ancestor tree up to said node. Further, they show that this is the optimal tree that can be constructed given the surrogate predictions along. 
* The algorithm they propose works in the following manner. While the budget is not exhausted, do the following: maintain a max heap of prefixes. At each stage, pop the highest prob leaf from said tree and compare the probability of adding the next most likely sibling of the leaf vs probability fo extending the chain with the most likely token from the next level - add the higher prob to the heap and repeat until budget is exhausted.
