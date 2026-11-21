---
title: "ProRL"
date: "2025-05-30"
type: "paper-summary"
subtype: null
topic: "RL for LLMs"
source_title: "ProRL"
source_author: "Minjie Liu, Yi Dong"
source_url: "https://arxiv.org/pdf/2505.24864"
---

- **Authors:** Minjie Liu, Yi Dong
- **Link/DOI:** https://arxiv.org/pdf/2505.24864
- **Date Read:** 2025-05-30
- **Tags:** 

## TL;DR
- The authors show that prolonged RL, utilizing a novel training methodology that incorporates KL div control, reference policy resetting and a diverse suite of tasks can uncover novel reasoning strategies that are inaccessible to base models, even under extensive modeling, challenging the prevailing assumption that RL is only about "sharpening" parts of the distribution that correlate with high reward. 

## 1. Problem Statement
- Recent studies argue that RL trained models do not acquire new reasoning capabilities beyond what exists in the base model based on pass@k metrics, and that they merely optimize the sampling efficiency of solutions already embedded in the base model. They posit that these conclustions stem from methodological constraints rather than fundamental limitations of RL approaches themselves. Specifically, they identify two main limitations:
1.  Over reliance on specialized domains, like math, where models are overtrained during pre/post-training, restricting the potential for exploration
2. The premature termination of RL training before models can fully explore and develop new reasoning capabilities based on a limited amount of RL training, typically no more than hundreds of steps. 

## 2. Methodology / Architecture
- The authors address the limitations from the previous section through several key contributions:
1. They introduce ProRL, a recipe designed to enable extended RL training periods that facilitate deeper exploration of reasoning strategies. 
	1. This method demonstrates continued performance improvements after an unprecedented 2k training steps, suggesting that RL training scales effectively with increased compute 
2. They show that RL can indeed discover genuinely new solution pathways entirely absent in base models, when given sufficient training time and applied to novel reasoning tasks. 
	1. They show that their model generates novel insights and performs exceptionally well on tasks with increasingly difficult and out-of-domain tasks
	2. They identify many tasks where the base model fails to produce ANY correct solution and the RL model achieves 100% correct solutions.
	3. They show that the amount of gain from RL on each task is predictable given the base model's performance -- RL expands reasoning boundary most effectively in domains where the base initially struggles 
		1. They introduce a "Creativity Index" which measures the amount of overlap with the pretraining corpus, they show that ProRL leads to higher novelty trajectories. 
* The authors address the key challenges in prolonged RL training, such as entropy collapse and instability, by introducing a KL divergence penalty and periodic resets of the reference policy which ensure stable training across many epochs and continued performance improvement. 
	* **Mitigating Entropy Collapse -** When entropy collapses, the policy prematurely commits to a narrow set of outputs, severely limiting exploration. This is particularly detrimental in methods like GRPO, where the learning signal depends on having a diverse set of sampled outputs to effectively estimate relative advantages. The authors adopt the following remedies for it: 
	1. Decoupled clipping of the PPO objective, with different values for $\epsilon_{high}, \epsilon_{low}$ 
	2. Dynamic sampling - filtering out prompts for which the model consistently succeeds or fails. 
	3. In addition to the above + high temp sampling, they explicitly regularize KLD through a penalty 
	$$
	L_{KL\text{-}RL}(\theta) = L_{GRPO}(\theta) - \beta D_{KL}(\pi_\theta \,\|\, \pi_{ref})
	$$
		1. They observe that as training progresses, the KL term may increasingly dominate the loss, leading to diminishing policy updates. 
		2. For long horizon RL they monitor training progress using a blended validations et derived from the evaluation benchmark. When val performance stagnates or degrades they perform a hard reset of the reference model and optimizer. This restores training stability and facilitates greater policy divergence from the base model. 

## 3. Results & Performance
- They present Nemotron-Research-Reasoning-Qwen-1.5B, a generalist model trained via reinforcement learning on a diverse, verifiable dataset of 136K problems across math, code, STEM, logic puzzles, and instruction following. Leveraging stable reward computation, improved GRPO, and prolonged training, their model achieves strong generalization across domains. It outperforms DeepSeek-R1- Distill-Qwen-1.5B by +15.7% on math, +14.4% on code, +25.9% on STEM, +22.0% on instruction following, and +54.8% on text-based logic puzzles Reasoning Gym1 . It also surpasses domainspecialized baselines in both math (+4.6%) and code (+6.5%), demonstrating the effectiveness of generalist prolonged RL training
![](/attachments/prorl-Pasted-image-20261014115845.png)
* ![](/attachments/prorl-Pasted-image-20261014120009.png)
	This shows that RL has a negative correlation between base model performance and extent of reasoning improvement (*though this seems like it may be a property of the metric/task rather than the model*). Further, we can see that the domains where RL improves the least are the ones where the model responses have the most overlap, suggesting that the base has seen a large amount of similar data during pretraining 
* The authors were further able to show that ProRL trained models are able to generalize to OOD tasks and increasingly harder tasks, indicating that the models learn behaviors and abilities rather than pattern matching, and these translate. 
## 4. Limitations
- Im not sure whether the claims hold up under more scrutiny, some of this feels like metric manipulation while some seems like what you would expect from training for longer. However, the specific method for stability over long horizon seems worthwhile. 
	- Limited task scope
	- Small model

## 5. Personal Takeaways / Action Items
### Train Recipe

**Training Monitoring.** We construct a validation data blend to closely monitor training progress across steps. This validation set includes subsets from our evaluation benchmark, specifically AIME2024, Codeforces, GPQA-diamond, IFEval, and the logic puzzle graph_color from Reasoning Gym. We evaluate model performance using similar sampling parameters as in evaluation settings (other than we use the same context window as in training).

**Reference Model and Optimizer Reset.** Occasionally, we perform a hard reset of the reference model and optimizer, as described in Section 2.3.1, particularly when validation metrics significantly degrade or when improvements plateau. Interestingly, the hard reset not only restores training stability but also provides an opportunity to adjust training hyperparameters and introduce enhancements such as additional training data and reward shaping. Figure 8 presents KL divergence across training runs. The final training recipe comprises several sequential stages, described in the following.

- **Run 1:** We begin training on four tasks from Appendix D. We did not include instruction-following data as it was not available to us initially. In this phase, we limit the response length to 8k where the base model's sequential length is 128k to avoild long sequence rollouts. As shown in Figure 2, model response length first decreases shortly and then keeps increasing along with improved validation scores. Toward the end of this stage, we observe instability and degradation in validation performance.
- **Run 2:** We perform a hard reset of the reference policy and resume training with the same setup as Run 1. Unlike DeepScaleR [3], which proposes increasing the maximum response length, we maintain the maximum response length as 8k because we observe that 8k maximum length is sufficient for the model to learn and improve its validation scores.
- **Run 3:** We incorporate instruction-following data into the training mix and continue training. This stage proceeds until we observe a sudden increase in response length, primarily due to the model repeating answers and failing to terminate with an `<eos>` token.
- **Run 4 and 5:** We introduce reward shaping by penalizing responses that do not terminate correctly. This encourages proper generation behavior, resulting in a modest reduction in response length.
- **Runs 6 and 7:** We increase the rollout count from 16 to 32, performing two hard resets in the process. Interestingly, response length begins to rise again alongside improvements in validation metrics.
- **Run 8:** We extend the context window to 16k tokens and reduce rollout count to 16. Despite the model being trained on an 8k context window for most of the time, it quickly adapts to the extended context window. We observe marginal improvements in hard math tasks like AIME, with more substantial gains coming from other domains.
