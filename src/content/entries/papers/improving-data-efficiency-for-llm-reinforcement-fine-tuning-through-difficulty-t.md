---
title: "Improving Data Efficiency for LLM Reinforcement Fine-tuning through Difficulty-targeted Online Data Selection and Rollout Replay"
date: "2026-10-19"
type: "paper-summary"
subtype: null
topic: "RL for LLMs"
source_title: "Improving Data Efficiency for LLM Reinforcement Fine-tuning through Difficulty-targeted Online Data Selection and Rollout Replay"
source_author: "Yifan Sun, Jingyan Shen, Yibin Wang, Huan Zhang"
source_url: "https://arxiv.org/pdf/2506.05316"
---

- **Authors:** Yifan Sun, Jingyan Shen, Yibin Wang and Huan Zhang 
- **Link/DOI:** chrome-extension://afikdfhpphmiponfgggpopedhllgaeni/viewer.html?file=https://arxiv.org/pdf/2506.05316
- **Date Read:** 2026-10-19
- **Tags:** 

## TL;DR
- 

## 1. Problem Statement
- The most effective approach for fine tuning LLMs is currently through RL. However, RL is extremely resource-intensive and existing work has largely overlooked the data inefficiency of it. In this paper, the authors introduce two techniques to improve the data efficiency of LLM RL fine tuning: 
	- Difficulty targeted online data selection - through adaptive difficulty they prioritize questions that are more likely to yield informative learning signals. 
	- Rollout Replay

## 2. Methodology / Architecture
![](/attachments/improving-data-efficiency-for-llm-reinforcement-fine-tuning-through-difficulty-t-Pasted-image-20261019170352.png)
- **Difficulty Targeted Online Data Selection** (DOTS) - Samples with 0.5 success probability are maximally informative. The authors try to select these for efficient training. TO that end, they execute multiple rollouts for a small reference set and compute groung truth adaptive difficulty for these. Then, the difficulty of the rest of the training set is estimated by comparing them to the reference set
	- The authors explicitly call out the importance of this approach being online, the non-stationarity is crucial given the evolving state of the model. 
![](/attachments/improving-data-efficiency-for-llm-reinforcement-fine-tuning-through-difficulty-t-Pasted-image-20261019171557.png)
	Claude - Here please fill in the details of the adaptive scaling, how the diffficulty is calculated for the subset, then how these are used through embedding to give predicted for the rest of the samples + calibration. Be concise, the equations are nearly sufficient. 
- **Rollout Replay (RR)** - at each training step they generate $\delta B$ rollouts and reuse $1-\delta B$ past rollouts from recent steps. They use a bounded FIFO buffer. Naively reusing past rollouts introduces bias into the policy gradient estimation, they therefore provide a modified GRPO loss which uses importance sampling wrt the original policy under which the rollouts where originally created 
$$
\begin{aligned}
\mathcal{J}_{\text{GRPO-RR}}(\theta) = {} & \mathbb{E}_{q \sim \mathcal{D},\ \color{red}{\{o_i\}_{i=1}^{G} \sim \pi_{\theta_{\text{behavior}}}(\cdot \mid q)}} \\
& \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \Big( \min\big( \tilde{r}_{i,t}(\theta) A_i,\ \text{clip}\big(\tilde{r}_{i,t}(\theta), 1-\epsilon, 1+\epsilon\big) A_i \big) - \beta D_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}}) \Big) \right],
\end{aligned}
$$
The full algorithm:
**Algorithm 1** GRPO with `DOTS` and `RR`

**Require:** Initial policy model $\pi_\theta$, reward model $r_\varphi$, training dataset $\mathcal{D}$, target difficulty $\alpha$, batch size $B$, total steps $T$, reference set size $K$, sampling temperature $\tau$, adaptive difficulty prediction framework DiffPred (§4.1), fresh rollout fraction $\delta \in (0, 1]$, buffer capacity $C$

---

- `1` Initialize replay buffer $\mathcal{R} \leftarrow \emptyset$
- `2` Set $\pi_{\theta_{\text{old}}} \leftarrow \pi_\theta$
- `3` **for** step $= 1, \dots, T$ **do**
	- *// Adaptive difficulty prediction*
	- `4` Sample reference set $\mathcal{D}_{\text{ref}} \subset \mathcal{D}$ uniformly at random, where $|\mathcal{D}_{\text{ref}}| = K$
	- `5` **for each** $q \in \mathcal{D}_{\text{ref}}$ **do**
		- `6` Generate $G$ outputs $\{o_i^q\}_{i=1}^{G} \sim \pi_{\theta_{\text{old}}}(\cdot \mid q)$
		- `7` Compute rewards $r_i^q = r_\varphi(o_i^q)$ for $i \in [G]$ and difficulty score $d_q = \frac{1}{G} \sum_{i=1}^{G} (1 - r_i^q)$
	- `8` **end for**
	- `9` Predict adaptive difficulty $\hat{d}_{q'} = \text{DiffPred}(\mathcal{D}_{\text{ref}}, \{d_q \mid q \in \mathcal{D}_{\text{ref}}\}, q')$ for all $q' \in \mathcal{D} \setminus \mathcal{D}_{\text{ref}}$
	- `10` Sample rollout batch $\mathcal{B}_{\text{rollout}}$ of size $\delta B$ from $\mathcal{D}$ according to:
		$$
		P(q) = \frac{\exp\left(-|\hat{d}_q - \alpha| / \tau\right)}{\sum_{q' \in \mathcal{D}} \exp\left(-|\hat{d}_{q'} - \alpha| / \tau\right)}
		$$
	- `11` **for each** $q \in \mathcal{B}_{\text{rollout}}$ **do**
		- `12` Generate $G$ outputs $\{o_i^q\}_{i=1}^{G} \sim \pi_{\theta_{\text{old}}}(\cdot \mid q)$
		- `13` Compute rewards $r_i^q = r_\varphi(o_i^q)$ for $i \in [G]$ and group average reward $\bar{r}_q = \frac{1}{G} \sum_{i=1}^{G} r_i^q$
		- `14` Obtain advantages $\hat{A}_i^q$ and policy probabilities $\pi_{\theta_{\text{old}}}(o_i^q \mid q)$ for $i \in [G]$
	- `15` **end for**
	- *// Rollout Replay and update*
	- `16` Sample $(1 - \delta)B$ samples from buffer $\mathcal{R}$ to complete batch $\mathcal{B}$ if $|\mathcal{R}| \ge (1 - \delta)B$
	- `17` Update policy $\pi_\theta$ using **modified** GRPO objective $\mathcal{J}_{\text{GRPO-RR}}$ on batch $\mathcal{B}$
	- *// Store informative rollouts in buffer*
	- `18` Add $\left(q,\ \{(o_i^q,\ \hat{A}_i^q,\ \pi_{\theta_{\text{old}}}(o_i^q \mid q))\}_{i=1}^{G}\right)$ to $\mathcal{R}$ for $q \in \mathcal{B}_{\text{rollout}},\ \bar{r}_q \notin \{0, 1\}$
	- `19` Remove the oldest samples from $\mathcal{R}$ until $|\mathcal{R}| \le C$
	- `20` Set $\pi_{\theta_{\text{old}}} \leftarrow \pi_\theta$
- `21` **end for**
## 3. Results & Performance
- 23-62% reduction in fine-tuning time while maintaining the same ultimate level of performance
- They show that the pearson correlation between predicted and actual adaptive difficulty sits above 0.7 for all samples, which is great!
	- 25% more "effective questions" than baseline (effective = strictly within 0 and 1 success. No zero gradient samples)

## 4. Limitations
- Only tested on small models with non-agentic tasks. Experience replay with associated environments is harder 
- How is the replay buffer different from "repeating epochs"?
	- If it's not, can we somehow leverage lessons from this to "batch buidling" in Pre-training where we aim to ahve a representative mixture of samples in each training step, to reduce catastrophic forgetting. 

## 5. Personal Takeaways / Action Items
- DOTS accelerates convergence while RR reduces per-step cost, both of these are effective ways of improving RL efficiency 
- DOTS outperforms external difficulty labels (i.e AI as a judge, methods without access to ground truth/internal representations)
