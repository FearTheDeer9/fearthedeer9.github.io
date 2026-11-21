---
title: "How Frontier Labs Train LLMs"
date: "2026-09-28"
type: "excerpt"
subtype: "article-summary"
topic: "Pretraining, data & distillation"
source_title: null
source_author: null
source_url: null
---

* In the past 3 years, released technical reports have become most more informative and thorough and reading them has significant alpha
* Through reading them lately one can easily see that labs have converged on a unified pipeline for training frontier LLMs, with only smaller implementation details differences - the main stages are largely consolidated. These are - Acquire Data, Pre-train, Mid-Train, SFT, RL, consolidate (merge experts and more RL), evaluate, safety + alignment training.

```Step back from the stages and the claim from the top of the post holds up: by 2026 there is **one recipe**, and the reports are variations on it. Here is the whole pipeline in one breath — _curate and dedup human-plus-synthetic data and choose the mixture by scaling-law forecasting; pre-train a RoPE/GQA/SwiGLU/RMSNorm MoE on a tokens-per-parameter ladder, deliberately over-trained, in FP8; mid-train on reasoning-dense data and extend context; SFT/cold-start to install behaviors and format; run GRPO-family RL with verifiable rewards, entropy control, and trainability filtering; align with a gated reward stack and an instruction hierarchy; measure with cheap NLL plus a contamination-guarded benchmark zoo; and red-team against a preparedness framework._ If you internalize that sentence, every report in the table below reads like a fill-in-the-blanks.```
# Architecture 
### Settled core 
* RoPE for position embeddings with Yarn stretching for long contexts 
* Some form of KV cache compression (GQA/MHA/MLA)
	* Interleave with Vanilla attention every Nth layer is also experimented 
	* Interleave with Linear/Hybrid attention variants has also recently been explored 
* SwiGLU for FFNs 
* RMSNorm and QK-norm for stabilization. These replace the LayerNorm - together preventing logit blowups but without the mean centering cost
* Sparsification is worthwhile - migration from dense to MoE is largely consensus at this point. The common theme is shared experts along with fine grained experts (small relative to model dimension).
	* aggressive sparsity regimes sit at ~50x 
* Global Batch Aggregation - there are several load balancing losses that are tried but the key insight is that you must balance over a diverse enough population of tokens rather than about loss vs bias 
# Data 
* Extract -> Dedup -> Classify -> Mix 
* This is THE place where models differ
* The pipeline itself is largely consistent - a huge crawl followed by dedup that reduces ~10x and then re-mix using quantitative recipe 
	* dedup uses exact match -> fuzzy match -> semantic similarity 
	* Different domains have different scaling laws -> the quantitative choices for data mixing need to be aware of this and optimize accordingly
# Pre-Training
* Once the data is collected and ready for training, this stage is an enginnering discipline that revolves around three core questions:
  1. Scale - how big is the run and how long are we training it for
  2. Precision - which number format are we using where 
  3. Stability - How do we keep a months-log training run from diverging 
* Chinchilla Scaling Laws are compute optimal, not deployment-optimal. This means that there is benefit for training past the Kaplan laws if models will be served on billions of tokens afterwards. The efficiency gain, though with a smaller slope, is worth it. 
	* Kaplan laws show 20TPP whereas frontier models commonly ship at 500-2000TPP
* The scale of models today means that it's infeasible to "experiment" on the target models anymore. This means that researchers extrapolate from experiments on small proxies. They use three complementary tools for this: 
1.  Scaling Ladders - train a family of models varying across one single axis and fit the loss<>X curbe. 
2. [Maximal update parameterization tricks  to transfer models across scales. ](https://arxiv.or g/abs/2203.03466)
3. Small Scale instability proxies - reproduce the big-run faillure points (loss/attention spikes) by increasing the LR. 
* Optimizer shift for first time in a decade adamW->Muon - orthogonalize the momentum update with a Newton-Schulz iteration and match AdamW's update RMS. 
* Recent work pushes the precision frontier with recent 4-8 bit variants in the open source frontier 
* Stability by removing specific loss-inducing tokens/n-grams, no weight decay on embeddings, constant monitoring etc 
# Post Training I - SFT & Distillation 
### SFT  
* The post training stage is about the "how" rather than the "what". Pre-train->Mid train imbues models with the knowledge. The final stage is about teaching them what kind of behavior we expect, what persona to adapt. The Post Training stage broadly splits into two: SFT and RL. The SFT stage serves as a bootstrap - making the models perform well enough so that "play" in the LR stage will be informative rather than all tasks being too difficult. The RL stage leverages this and nails specific model characteristics in place 
* SFT data is usually obtained by self-generated trajectories on verifiable tasks which are subsequently filtered

### Distillation 
* Usually more efficient dollar-wise than RL for imbuing new capabilities. But unlike RL it cannot "outperform" what it learns from its teacher, there's a "hard ceiling" quality-wise to this approach 
* Three main variants for distillaiton:
  1. Off-Policy - Train on a fixed set the teacher produced. The weakness is that the student only ever sees the teacher trajectories, not its own, so errors compound and persist at inference. 
  2. OPD - the teacher scored student-generated trajectories (reverse rather than forward KL). Fixes the off-policy limitation.
	  1. OPD is stable when the teacher and student are of the same family, when you use stronger external teacher you risk entropy collaspe, accuracy drops and outright divergence. Train intra-family, not just the most performant model available to you
  3. Self Distillation - teacher are student are same lineage but altered in some manner (different checkpoint, teacher has access to privileged information etc)
* MOPD -  The most important distillation development recently is **Multi-Teacher On-Policy Distillation**. Train several _independent, per-domain RL specialists_ (math, code, agentic, instruction-following, chat), then **fuse them into one generalist** by distilling all of them onto the student’s own rollouts — each teacher supplying a dense token-level reverse-KL signal on the prompts in _its_ domain, alongside a verifiable outcome reward. It decouples development and inherits _nearly all_ of each teacher’s peak skill
* distillation is eating the “consolidation” step - two years ago you merged skills by mixing all the RL data together and hoping; the 2026 default is _train specialists separately, then OPD-merge them._ It’s more modular, more stable, and — because the reward is a distributional divergence rather than a scalar — meaningfully harder to reward-hack. The catch (and MAI’s objection) still stands: a student can only be pulled toward teachers that already exist, so OPD **industrializes inheritance** — and only RL still explores past every teacher.

# Post Training II - RL 
* The hottest compnent currently in the AI landscape, this is what makes reasoning and tool use trainable. Its remarkable how standardized yet fragile this is 
* Recent changes from PPO->GRPO which alleviate the need for a value model - which is expensive and awkward when working with long trajectories (though its recently making a comeback for agentic RL - PRMs etc)
* GRPO Variants:
  1. DAPO - the defacto "improved GRPO". **clip-higher** (a looser upper clip so low-probability exploratory tokens can grow — preserving entropy), **dynamic sampling** (drop prompts with pass-rate 0 or 1, whose group advantage is zero), a **token-level loss** (normalize over all tokens, not per-sample, killing a length bias), and **overlong-reward shaping**.
  2. BAPO - Generalizes clip-higher to an adaptive controller for the off-policy regime that async infra creates 
  3. CISPO - clip the IS weight instead of the token update so that rare reflective tokens won't be dropped
* There a genuine split regarding when to keep or drop the KL term. current best practice seems to be to drop it for reasoning but keep it for alignment 
* The rise of **Long Horizon Agentic RL** makes vanilla GRPO with tweaks cumbersome, the following directions emerged:
  1. Sequence level (GSPO) -  rather than token level, IS ratio is calculated on a sequence level for which a reward is given, more table, eliminates the need for routed retrain
  2. Bring back the critic (SAO/PPO) - When a very long agent trajectory is compacted into multiple sub-traces different rollouts of the same prompt have different numbers of trainiable traces and different lengths, the assumptions of GRPO break. A critic can estimate token-level advantages for a single rollout, fits naturally with compaction.
	  1. One rollout per group rather than GRPO - fits better with async RL when a group is only as fast as its slowest member
	  2. To make the value model affordable SAO updated the critic more ofthen than the actor and fine-tunes the value model with frozen attention
	  3. Train only on the tool calls, not intermediate "observation" tokens between successive tool invocations
	  4. Strict double sided token level clipping - Clip both sides of the token ratio to mask divergent gradients 
* The new divergence shows that 2025's claim of "GRPO won, actor-based is dead" was premature, with GRPO/CISPO for short verifiable tasks, GSPO for stable MoE RL and PPO back for long agentic traces. The key takeaway is that trajectory length and structure now drive the choice of RL algorithm 
* It is crucial one monitors the entropy and makes sure it isn't too large or too small. This can be done by clipping, by intervening at the covariance level, or by adding an entropy bonus 
* Verifiable rewards - Have largely won since they're cheap, low-noise and hard to game at the reward level. Alternative of RMs/LLM-as-judge come with their own limitations. Four LLM functions -> four flavors of LLM misalignment 
* The Information Gain one gets from a task is p*(1-p), this is zero at both extremes and maximized at p=0.5. Labs filter prompts to fit into pass-rate bands
* Labs have started introducing difficulty-aware length-penalties -- penalize length in proportion to how easily a prompt is solved, so hard problems can think longer
* **The Stability Tricks** - 
  **top-p mask replay** (reuse the sampling truncation mask at train time so train and inference agree — MAI, DeepSeek-V3.2)
  **MoE router replay** (an MoE picks different experts in the inference vs training engine, an off-policy gap baked into the architecture — fixed by replaying routing, [Ma et al., 2025](https://arxiv.org/abs/2510.11370)
  DeepSeek’s “Keep Routing”)
  **FP32 LM head** to fix train/inference precision mismatch (MiniMax); async-RL **staleness** bounds; and self-distillation as a numerical save-point
* as mentioned above, agentic RL suffers from credit assignment due to the long contexts. As a result Frontier Labs tried to reintroduce PRMs and other agentic systems to reward based on finer-scoped steps (such as term-based, step-based, or token-based) rather than the full trajectory. So far this has mostly been negative due to the fact that PRMs have often been found to be exploitable. 
* it's been found that each stage can create catastrophic forgetting of the knowledge of previous stages and as a result work on continual learning has been done. This hasn't been solved but it has been shown that the reason we forget is due to KL distance. As a result we want to keep the model as close as possible to the previous manifold point on the manifold that maintains previous capabilities while learning the new ones. To do that we labs have identified a mitigation toolbox:
	*  replay. Keep old data and mix it back 
	*  proximity regularization, where we have the KL penalty in RLHF or orthogonal gradient projection 
	* Parameter isolation, which is about freezing the entire model and only modifying specific bits, such as LoRA
	* Model merging, where we average the fine-tuned model back with the base,or sibling checkpoints 
	* distillation-based recovery, where we use the base model for self-distillation of capabilities that may have been lost during the fine-tuning 
*  alignment of helpfulness, honesty, and harmlessness has also become a more pressing challenge with the increased capabilities. The frontier has adopted a mindset of gating safety not averaging it. That is, safety is usually a hard gate. If it passes it's good enough but you don't allow anything to pass if you can't prove it. That is you don't average several safety dimensions and pass the average. You gate on each specific component. Alignment is usually an RL objective now, with a composed reward stack where the trade-off between helpfulness and safety is constantly monitored and measured. We prefer engaging with the material and somehow adding a regularization loss to prevent models from simply refusing instead of engaging in a non-harmful manner 
* evals are now more important than ever. While training, continuous measures like the loss are better than accuracy, whereas when we deploy we want to use the benchmark zoo with specific high-quality benchmarks that we keep 
* Finally we have alignment training and red teaming - Safety has become a pipeline-wide process with a shared vocabulary (preparedness categories, automated red-teaming, dangerous-capability evals), and the reasoning era adds CoT monitoring — useful, but only as long as we don’t train models to hide their thoughts.
* open challenges that we now have:

1. Verification as the ceiling on RL: How do we get high-quality verification that is both informative and scalable and does not suffer from hacking by a smart capable model that is situationally aware?
2. The data dilemma: The high-quality human text is running out and we want to generate more data showcasing specific capabilities and patterns. The way to synthetically generate them is still an open question.
3. The cost and fragility of long-horizon RL algorithms is an open question. How do we scale PRMs? How do we assign responsibility for success or failure
