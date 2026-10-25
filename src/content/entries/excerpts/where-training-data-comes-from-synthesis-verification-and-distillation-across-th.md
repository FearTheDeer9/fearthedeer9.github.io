---
title: "Where Training Data Comes From - Synthesis, Verification and Distillation Across the Training Stack"
date: "2026-10-11"
type: "excerpt"
subtype: "article-summary"
topic: "Pretraining, data & distillation"
source_title: "Where Training Data Comes From"
source_author: "Jiaxin Zhang"
source_url: "https://jxzhangjhu.github.io/blog/2026/where-training-data-comes-from/"
---

- **Authors:** Jiaxin Zhang 
- **Link/DOI:** https://jxzhangjhu.github.io/blog/2026/where-training-data-comes-from/#three-sources-of-supervision
- **Date Read:** 2026-10-11
- **Tags:** 
## TL;DR
Data is the new moat, this can be inferred from the fact that architecture details are shared quite commonly and training data is nothing more than "a diverse mixture of public and licensed sources". The question that one needs to always be able to answer is 
	"At each stage of training, what counts as a single training example, and who guarantees that it is correct?"
		All other steps, filtering, synthesis, mixtures, rubrics, distillation and decontamination is downstream from the answer to the above question.

![](/attachments/where-training-data-comes-from-synthesis-verification-and-distillation-across-th-Pasted-image-20261012175841.png)

## The proper Schema 
Most of the confusion about data stems from the fact that we are using a bad word for it. By giving a single name to six different things we are bound to be confused, and this is easily negated if we instead reframe these as different choices along 4 separate axes -
1. What makes a single data "unit"
2. What is the data source
3. What verifies/certifies a data sample 
4. What is the characteristic failure ![](/attachments/where-training-data-comes-from-synthesis-verification-and-distillation-across-th-Pasted-image-20261011142536.png)
	1. Mental note-to-self: notice that post training is split into 3, and that continual training already has a separate, dedicated step just for this. 
	2. **The unit of data stops being text.** In pre-training you buy tokens. In RLVR you buy _problems with checkers_ — the answer is not the data, the checker is. In agentic RL you buy _environments_, which is to say you buy software. This is why data work migrates from data engineering to software engineering as you go up, and why the headcount profile of a post-training team looks nothing like a pre-training data team.
	3. **The economics invert.** A pre-training corpus is 10⁹–10¹⁰ documents with essentially zero verification cost per document. An RLVR run is 10³–10⁵ prompts, each of which needs a checker that someone wrote and validated. Agentic RL is 10²–10⁴ tasks, each of which may need a Docker image. Per-example cost rises by five or six orders of magnitude while the example count falls by five or six — which is why “just get more data” is good advice at the bottom of the stack and nearly always wrong at the top.
		1. Ankify these numbers 
**Who gives the supervision signal** - Zhang makes this point explicit, and says that the answer to this question is crucial, more than the "real" vs "artificial" debate. It can be a human, a teacher model or the "world". Each of these have characteristics regarding cost, performance ceiling and failure patterns. 
	The current best practice is largely - teachers for the cold start, the world for the climb, humans for taste and for the checkers you can't automate. 
![](/attachments/where-training-data-comes-from-synthesis-verification-and-distillation-across-th-Pasted-image-20261011150141.png)
# How to Synthesize 
Ways of synthesizing new data can be grouped into eight recurring patterns, each with it's own failure pattern and key points of relevance.
![](/attachments/where-training-data-comes-from-synthesis-verification-and-distillation-across-th-Pasted-image-20261011150642.png)
![](/attachments/where-training-data-comes-from-synthesis-verification-and-distillation-across-th-Pasted-image-20261011150911.png)
* The biggest issue industry is currently trying to solve is how to maintain and encourage diversity when using synthetic generation. The current paradigm is conditioning on external context to alter the model between runs + filtering and verifying post generation that diversity is maintained. 
## Verification Ladder 
![](/attachments/where-training-data-comes-from-synthesis-verification-and-distillation-across-th-Pasted-image-20261011152323.png)
The most important things to internalize from this ladder are:
* Rule based verification has false negative as the failure pattern. Model/human based and test suites fail through false-positives.
* Grounding beats sophistication - ground the reward by a component the agent cannot game/rewrite. Whenever the attack surface for the model is large, reward hacking will happen. ![](/attachments/where-training-data-comes-from-synthesis-verification-and-distillation-across-th-Pasted-image-20261011153910.png)

## Pretraining - Filtering is the Name of the Game
* It's common to discuss the "Data wall" pretty soon we will "run out" of high quality human generated text. Then we ask how much does repetition buy? 
	* The canonical answer: up to **4 epochs is nearly free** (an 8.7B model trained 4 epochs on 44B unique tokens ends 0.5% worse in validation loss than one epoch on 178B unique tokens), meaningful gains continue to ~16 epochs, and by 40 epochs repeating is worthless. 
	* The fitted half-lives are the part people forget: repeated tokens decay in value with `R*_D = 15.4` while excess parameters decay with `R*_N = 5.3`, so **when you are data-constrained you should buy epochs before parameters** — the opposite of Chinchilla’s equal split
* Filtering is critical but naive deduplication is not always the best practice, and model based filtering is the current SOTA with "quality" == "shaped like the target task"
* Mode collapse can nearly always be eliminated if one remembers to accumulate rather than replace the original data with the newly generated synthetic data
* The optimal mixture should not be inferred by rank invaraince on toy models, but rather a composition of eval on toy model + the trajectories seen across several toy model scales. Different domains progress differently with more representational power. 

## Mid Training - what it is and how we optimize for it 
Operationally it is everything that happens after you decide pre-training is finished and before you start collecting preference or reward data: up-weighting a domain, extending context, annealing on a high-quality mixture, installing an output format, injecting a proprietary corpus, and preparing the model so that RL has something to amplify.
The unit of data here is a **curated pack**, and the certifier is the weakest of any stage: a proxy eval plus a replay ratio. Nobody checks individual documents. What you are actually deciding is a distribution and a schedule, and the recurring lesson across the literature is that **the same facts in a different format are worth roughly an order of magnitude more.**
* One needs to be highly aware of the outsized impact of the peak LR chosen. It accounts for a third of the "catastrophic forgetting" phenomena and is effected by model size and the distribution shigt. 
	* Likewise, the replay ratio needs to be tuned dependent on the size of the distribution shift 
* Ankify the reference papers regarding canonical CPT LR recipes 

**Knowledge Injection** - A model can perfectly memorize a document and still be unable to answer any question about it. This is because in a single-format corpus the attribute is encoded diffusely across the document's tokens rather than linearly in the hidden state adjacent to the entity's name. Fine tuning cannot fix it becuase where knowledge is stored is a pre-trianing-time property of the corpus. The way to remedy it is through augmentation, specifically diversity ![](/attachments/where-training-data-comes-from-synthesis-verification-and-distillation-across-th-Pasted-image-20261011165249.png)![](/attachments/where-training-data-comes-from-synthesis-verification-and-distillation-across-th-Pasted-image-20261011165804.png)
**Long Context is an unlocked rather than an installed capability**
* 100% Long data is worse than 60% long data even if we're optimizing for long tasks since after SFT we learn that a model specialized on long documents is a porr init for instruction tuning. 
* Train on longer traces than What you expect to need in prod 
* If you mid-train at short context only, assume the long-context is gone https://arxiv.org/abs/2603.17074
**Mid Training as RL Prep** - This is important and long enough for me to paste verbatim
![](/attachments/where-training-data-comes-from-synthesis-verification-and-distillation-across-th-Pasted-image-20261011171403.png)

## SFT Stage - a Readiness Gate 
A sample here is a triplet of a (prompt, response, optional - reasoning trace). The certifier is a design choice, can be a verifier, judge or a human. 
* "How much SFT data do I need" is ill-defined, it cannot be answered if you do not know the base model and the breadth of the eval needed. 
* The unit of SFT data is not an example, it is a _behavior_, and a behavior costs single to double-digit examples on a base model that already has the underlying capability. What it cannot buy is the capability itself — which is the bridge to the distillation section, and the reason the correct mental model for SFT is a readiness gate: it decides whether the model can be _asked_ for what it already knows, and whether RL will have a usable policy to improve.
* For verifiable domains this is the order of "importance" of the different axes:
	* correctness
	* diversity 
	* quality of response 
	* quantity 
![](/attachments/where-training-data-comes-from-synthesis-verification-and-distillation-across-th-Pasted-image-20261011172522.png)
* SFT sharpens more than it broadens, and this matters if RL comes next. If your SFT stage is a cold start for RL, evaluate it at pass@k, not pass@1, because pass@1 is exactly the metric that rewards collapsing the distribution you are about to ask RL to explore.
## RL - The Data is Prompts, Not Answers 
* **The first-order property of a good RL prompt is that it produces reward variance.**
* **Being unsolvable is worse than being absent.** This is the asymmetry to remember.
*  **the amount of prompt data needed is smaller than anyone expected, in a way that should make you suspicious.** One training example reached a 35.7 six-benchmark average against 35.9 for the 1,209-example set containing it.
* **Is your RL teaching or eliciting?** On Qwen2.5-Math-7B, RLVR with _random_ rewards — a coin flip — gains +21.4 points on MATH-500 against +29.1 for ground-truth rewards, which is 73% of the entire supervised gain reproduced by noise. Deliberately _incorrect_ rewards gain +24.1. The mechanism is identified rather than asserted: the clipping term in GRPO asymmetrically favors already-likely tokens, so the update ratchets up whatever the prior already contains, and removing clipping three different ways makes the random-reward gain vanish ([Shao et al., 2025](https://arxiv.org/abs/2506.10947)). On models without the exploitable prior the picture inverts completely: OLMo2-7B gains +15.5 from ground-truth rewards while every spurious alternative is flat or negative (majority vote −6.4, random −8.3, format +0.4).
![](/attachments/where-training-data-comes-from-synthesis-verification-and-distillation-across-th-Pasted-image-20261011175917.png)
* **The practical consequence is where you spend.** If format reward captures most of your gain, you are eliciting, the reward content is nearly fungible, and investment in verifier quality is largely wasted. Elicitation also saturates — models that have already been RL-post-trained show minimal gains under _any_ reward, ground truth included. You get to harvest the prior once. Conversely, every setting where verifier quality produced large durable gains is a setting where the prior could not carry the model: agentic coding with process monitoring, cross-domain science, post-cutoff evaluations. That is where “verification is the asset” is literally rather than rhetorically true.![](/attachments/where-training-data-comes-from-synthesis-verification-and-distillation-across-th-Pasted-image-20261011180116.png)

## Agentic Data is Software
At the top of the stack, a training example is a piece of software. A triplet of a resettable environment, a task specification and a "verifier" of some sort that runs against the environment state and determines if the task was accomplished. 
* **Start by buliding the environment and then mint tasks against it**- this is due to two reasons, the first is that it's easier and more effective. The other is memory cost, each repo requires an image.  One image per repository costs 295 GB for 50k tasks; one image per task costs 6 TB for 2,438.
* A common way of generating tasks is reverse construction, do something, see the behavioral change, describe it post-hoc and now the task is executable by construction. 
* It is crucial when writing verifiers that they are unaware of the solution and that the solving agent is made sure to not be able to recover the verifier. 
* Before you begin rollouts, make sure you run gates to filter incorrect environments. Here are a few common ones:
  ![](/attachments/where-training-data-comes-from-synthesis-verification-and-distillation-across-th-Pasted-image-20261012150027.png)
* When training on trajectories, current best practice is to train only on succesful ones, filtering the failed executions. 
	* Recent work is exploring using failed trajectories as training data for the [verifier](https://arxiv.org/abs/2606.21262)
* Rubrics are the common solution for cases where we don't know how to write a test.  However, current techniques often perform worse than simple binary rewards where the failures usually reside with one of two domains:
## Distillation are a data-acquisition strategy 
Distillation isn't about losses, it's about acquiring supervision. When deciding what suits you, you need to ask yourself:
1. What data can you physically extract from the teacher 
2. How far apart you and the teacher are 
![](/attachments/where-training-data-comes-from-synthesis-verification-and-distillation-across-th-Pasted-image-20261012170629.png)
* The cost of each step down the ladder is measurable and it's small. topk=128 + 30% ground-truth next token loss costs just 0.01 nats compared to the full distribution while being much less memory [expensive](https://arxiv.org/abs/2502.08606)
* **distillation cannot beat supervised learning asymptotically.** The authors’ two-condition test is crisp enough to quote verbatim in a design review: distillation is more efficient if and only if (i) the student’s compute or token budget is below a student-size-dependent threshold, _and_ (ii) a teacher already exists or will be amortized across more than one student. Otherwise, train the student directly. Even weak-to-strong generalization — a student beating its teacher — turns out to be a finite-data artifact: feed the student enough distillation tokens and it drifts back up to the teacher’s cross-entropy.
	* **If your KL to the teacher starts high, a better teacher is a worse teacher**
* The Distillation Ladder - use the first branch that fits:
	* **Step 0. Two pre-flight diagnostics.** Measure initial per-token KL or top-k overlap between candidate teacher and student (calibration above: ~0.04 productive, ~0.19 failing). Measure your student’s pass rate on the target prompts — if it is near zero, on-policy gradients vanish and you need curriculum pacing or an easier warm-up first.
	* **Step 1. Are you compressing, or acquiring capability?** If you are pre-training a smaller model from scratch, apply the two-condition test above literally, and if it fails, just do supervised pre-training. If you are post-training an existing small model, continue.
	* **Step 2. Start off-policy, always**
	* **Step 3. Pay for on-policy when the errors compound.** Switch when the student is above roughly 7B and exploring off the static-trace manifold, when intermediate errors compound across multi-step reasoning, or when off-policy loss plateaus while held-out reward still improves under on-policy rollouts. Then: turn on at least 25% student-generated data ([Agarwal et al., 2023](https://arxiv.org/abs/2306.13649)); pick the divergence by _decoding regime_ rather than by ideology (sampled or open-ended generation wants mode-covering, instruction following and single-correct-answer reasoning want mode-seeking); and use full-vocabulary KL if you can afford it, top-k _with a debias term_ if you are shipping the signal over a network, and sampled-token-only if you add variance reduction.
	* **Step 4. Use the teacher as a verifier, not an author, when any of these hold.** Access is black-box. Teacher and student are from different families. Or you want to _exceed_ the teacher — which is the decisive case, because distillation targets are ceilings and rewards are not.
	* **Step 5. If you must use a cross-family teacher, put it inside the RL gradient instead of beside it.** This is the most useful recent result for the small-model-strong-teacher setting. Rather than matching the teacher’s distribution, use the teacher’s log-probability ratio to _reweight the advantage_ of trajectories the verifier has already blessed, applied **only where the advantage is positive**. The gating is what makes it work: without it, teacher-preferred tokens on failing trajectories receive a _larger_ penalty, pushing the student away from the teacher.
	* **Step 6. Do not add a distillation loss next to an RL loss and hope.** The 1:1 combination was worse than RL alone on two of three students, because distillation converges rapidly and then pins the student in a KL-induced local optimum while RL’s curve is still rising.
* **Takeaway.** Distillation buys behavior reliably and capability unreliably, and the ceiling is the teacher unless something in your loop is grounded in the world. The composite recipe that respects all of that: off-policy traces for the cold start, on-policy correction if the errors compound, the teacher as a _scorer_ rather than an author wherever you want to pass it, and a verifier underneath the whole thing so that the teacher redistributes credit instead of defining the target.
* **Hygiene and Gotchas when distilling**:
	* Style transfers much better than knowledge - the upshot, never accept preference wuns without cross-referencing with benchmark performance. And when there's a disagreement, trust the benchmark 
	* Additional known failures: Entropy collapse, calibration degradation, length and format pathologies, hacking the teacher's ownlikelihood, style without substance, bias and safety inheritance
	*  **filter or rewrite self-identification strings, vendor names, refusal boilerplate, and system-prompt tics out of teacher traces, and add an identity probe to your eval suite.**
## Contamination - eval is a part of the Pipeline 
* **Takeaway.** Four instruments. Report one clearly out-of-distribution eval beside every headline number and read the divergence, not either number. Attach an attempt count to every score: two byte-identical Aya-Vision-8B checkpoints submitted to Chatbot Arena scored 1069 and 1052 with four models ranked between them ([Singh et al., 2025](https://arxiv.org/abs/2504.20879)). If your system retrieves, run every eval search-on and search-off and publish the answer-leakage rate beside the accuracy. And hold back a canary split of vetted items, which is how GSM1k detects leakage through the vendor APIs it needed to score closed models.
# The Data Platform 
Given that data is the new product, a serious lab builds a platform for data rather than maintaining datasets. Here are the components this entails:
1. **A provenance and lineage service.** Every training example needs to carry where it came from, which generator and prompt produced it, which certifier passed it, and which runs consumed it.
2. **A decontamination service, not a decontamination script.** Prompt-side n-gram matching, run against _every_ eval you report, on _every_ corpus you add, including synthetic corpora and RL prompt sets. It belongs in the platform rather than in each project because the published overlap rates are alarming enough that you will not remember to run it manually.
3. **A scorer registry.** Quality classifiers are the highest-leverage knob in pre-training and the most dangerous one in mid-training, because they are length-biased and will silently delete your long-context and agent-trajectory data. Register each scorer with the source distribution it was calibrated on, its length-response curve, and the threshold in use — and always report random selection with source preservation as a baseline, since it beats most published filters.
4. **A verifier registry with adversarial suites attached.** Every checker is a piece of software with a version, a measured precision and recall _on your data_, and a suite of adversarial patterns it must resist. The non-negotiable lesson from the ladder section is that static accuracy does not predict RL outcome, so the artifact you version is the verifier plus its attack suite, and the suite has to be regenerated as policies get stronger.
5. **An oracle channel.** At every checkpoint, sample ~1,000 training prompts, regenerate, and score them with a strong external model. Plot training reward and oracle reward on the same axes. Divergence between those two curves is the most reliable reward-hacking alarm I know of, and unlike a held-out eval it costs one generation pass per checkpoint.
6. **An environment registry.** Container images, seeds, task specifications, rubrics, and a two-sided validation record showing the checker fails on the unfixed instance and passes on the fixed one. This is the part that is unambiguously software engineering, and the part where headcount goes at the top of the stack.
7. **Private, rotating, and freshly collected evals.** Whatever you publish, keep a held-out set that never leaves the building, rotate a fraction of it on a schedule, and maintain a non-regression suite that runs on every checkpoint. The reason is in the [contamination](https://jxzhangjhu.github.io/blog/2026/where-training-data-comes-from/#contamination-your-eval-set-is-part-of-your-pipeline) section, and the incentive analysis behind it is simple: contamination is not only an accident of web crawling, it is what an organization drifts toward when its objective function collapses to a score.
