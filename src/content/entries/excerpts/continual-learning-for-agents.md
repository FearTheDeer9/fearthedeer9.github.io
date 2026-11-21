---
title: "Continual Learning for Agents"
date: "2026-10-18"
type: "excerpt"
subtype: "article-summary"
topic: "Continual learning & optimization"
source_title: "Continual Learning for Next-Generation Agents"
source_author: "Jiaxin Zhang"
source_url: "https://jxzhangjhu.github.io/blog/2026/continual-learning-for-next-generation-agents/"
---

- **Authors:**  Jiaxin Zhang 
- **Link/DOI:** https://jxzhangjhu.github.io/blog/2026/continual-learning-for-next-generation-agents/
- **Date Read:** 2026-10-18
- **Tags:** 

## TL;DR
- LLMs today are already smarter than everyone, they have more knowledge than what we can possibly acquire. To be useful however, one needs to be aware of his environment and adapt to work standards and practices. This is what CL does and this is the next frontier.
- We have several axes we can "touch" in order to imbue the model with additional context and adapt it's behavior accordingly. These are:
	- In-Context Learning - effects the behavior within a session
	- Memory - carry selected experience across sessions 
	- Skills/harness - encode verified procedures 
	- Adapters and weights - compress patterns that transfer broadly 
	- Outer learning mechanism - improve how/which of these updates is chosen 
- The author claims " Continual learning for agents is no longer only “update parameters without catastrophic forgetting.” It is the problem of **routing experience to the right update surface and consolidating it across multiple timescales**—while preserving transfer, privacy, calibration, safety, and rollback."

## What is CL - The stack, and what each layer gives 
The author claims that  "learning" can happen in different levels with them producing a cost gradient. The higher the level, the more powerful it is but it is also more dangerous and more demanding of evidence.  The levels differ in what their scope is, where their updates happen and in their characteristic failures.
![](/attachments/continual-learning-for-agents-Pasted-image-20261018154208.png)
* ***Context** is ideal for tentative beliefs, local tool state, and one-off constraints because it is cheap and reversible. In practice, four things belong in working state: the current plan and its open questions, the constraints the user has asserted, the environment facts discovered this session (which command works, which path exists), and the failures already ruled out. Each is either cheap to rediscover or too unverified to promote.
* **Memory**  crosses the session boundary. It should not be a bag of old messages. A useful agent memory has at least three roles:
	* *episodic*: what happened in a particular run, with the model/harness version and outcome;
	* *semantic*: relatively stable facts and preferences abstracted across episodes;
	* *procedural*: a reusable lesson about how to act.
	* Every durable memory should carry metadata: source, timestamp, scope, confidence, evidence, expiry, and links to contradicting or superseding entries. [ARIA](https://arxiv.org/abs/2507.17131) is a useful concrete pattern: a deployed agent identifies a knowledge gap, asks a human expert a targeted question, writes the guidance into a timestamped repository, compares it with related entries, and requests clarification when rules conflict.
	*  Agent memory needs garbage collection, conflict resolution, and evaluation—not just vector search.
* **Harness and Skills**  Compared with weights, a harness change is cheap, readable, diffable, and easy to roll back. Compared with raw memory, it can encode an executable procedure and its validator. This is the warm path between “remember this” and “retrain the model.”
	* A workable promotion rule from memory to skill has four conditions, and the last one is the one usually missing:
		1. **recurrence:** the situation has appeared often enough that a procedure beats a lookup;
		2. **determinism:** the right response is a sequence of steps, not a judgment that depends on the case;
		3. **checkability:** the procedure can be validated by something other than the model that proposed it;
		4. **non-interference:** running it in situations where it does not apply is harmless, or the skill knows its own preconditions.
* **Adapters and Weights** - Weights are valuable because they compress behavior. A lesson internalized into an adapter or shared model requires no retrieval and can generalize in ways a literal memory cannot.  This surface is appropriate only when a pattern is:
	1. supported by many attributable experiences;
	2. useful beyond one user or project;
	3. stable enough not to expire next week;
	4. legal and private enough to train on;
	5. able to survive replay, held-out transfer, safety, and calibration checks.
	The biological analogy is [**complementary learning systems**](https://doi.org/10.1037/0033-295X.102.3.419): fast episodic storage protects individual experiences while slower, interleaved consolidation extracts structure into semantic memory. [Dorovatas et al. (2026)](https://arxiv.org/abs/2603.01761) makes the agent version explicit: combine rapid in-context adaptation with low-frequency in-weight learning rather than forcing either surface to do both.
* **Updater** - Learning how to learn. Which feedback is trustworthy, who is responsible for the failure, what do we want to test/evaluate the proposed update on, how do we decide between exploration and training? 
	* A calibrated update router can be written as 
	$$
	R(e_t, q_t) \to (s, \sigma, \tau, a, c),
	$$
		where $e_t$ is an experience; $q_t$ summarizes credibility, causal attribution, novelty, recurrence, transferability, privacy, regression risk, and cost; $s$ is the target surface; $\sigma$ is scope; $\tau$ is a time-to-live or review horizon; $a$ is write, quarantine, request evidence, stage, merge, or discard; and $c$ is confidence in that route.
		* Framed this way, the router is a decision policy with an explicit budget rather than a classifier. Each route carries an expected benefit (future tasks improved), an expected cost (compute, latency, review time), and an expected risk (regression, leakage, poisoning). Choosing the smallest reversible write that captures most of the benefit is a bandit-style problem with delayed, partially observed rewards—which is exactly why routing deserves to be learned and measured rather than hard-coded once.
	* 

## The deployment loop - translating experience to durable results 
![](/attachments/continual-learning-for-agents-Pasted-image-20261018160619.png)
### Critical Loop Requirements 
* **Capture the complete experience** the final answe alone is not enough if feedback is delayed. A patch won't be applicable without enough provenance. Include: 
	* task and relevant environment state;
	- full action/tool trajectory;
	- model, prompt, memory, tool, and harness versions;
	- final and intermediate outcomes;
	- explicit feedback and later downstream outcomes;
	- identity, consent, privacy class, and retention policy.
- **Validate the signal and identify the cause** - When a feedback comes, don't trust it blindly. Check provenance, authority, privacy, adversarial risk and independent evidence. Try to reproduce. Then, identify what caused this, is it model knowledge, context management, tool description, environment etc. ![](/attachments/continual-learning-for-agents-Pasted-image-20261018161052.png)
* **Rule of Thumb - route to the highest layer that can solve the problem**
* **Replay, Contrast and Consolidate** - don't make changes based on raw trajectories. Aggregate them and look for recurring patterns. Dedup, search for counterexamples for every proposed change, distill trajectories to facts/procedures and retain links from abstraction to raw traces that produced them for provenance/future inspection
* **Solving the example that triggered a change isn't enough, gate changes on transfer for intra-domain work and non-regression on inter-domain**
* **Stage the release** - when promoting a fix a layer up, save a "quick rollback" option and evaluate against the newly deployed version 
* **Monitor, RollBack and MetaLearn** 
	* A few signals are worth watching continuously rather than at release time: the share of retrieved memories that are never used in the final answer, the age distribution of entries that still fire, the rate at which new writes contradict existing ones, the fraction of promotions later reverted, and the gap between the proxy signal (acceptance, thumbs-up) and any ground-truth outcome you can measure later. Each of those tends to degrade slowly and silently, which is exactly why it needs a dashboard rather than an incident.

## Translating Classic CL to Agentic CL 
* **Replay** - Replay has two jobs. It protects stability by representing old behavior, and it reveals whether a new lesson transfers beyond its trigger. More replay is not always better: stale or irrelevant experiences increase context and compute, violate privacy, and can block adaptation. Selection and coverage are the research problem. 
	* **TODO** We need both to maintain a non-stale, non duplicate list of tasks and have a continual deployment service that runs them 
* **Regularization** - The changes should be targeted, we do not want to change large scale param updates/over-generalize behavioral changes to non-relevant domains 
* **Parameter Isolation** The hard part is to decide the level each new update should be shared in, the "natural" places to share already exist. Permission-Based memories, project specific skills, task vectors & Lora adapters, Sandboxed tools and subagents 
* **Dynamic Architecture** - Add capacity rather than overwrite old ones. In agents this manifests as new skills/tools/memory modules or subagents. At the model level it can be a LoRa adapter/new experts 
* **Distillation** This is about transforming raw trajectories and transcripts into explicit learning. extracting abstractions from raw traces/memories 
* **What Doesn't Transfer** unlike classically assumed, we no longer have clean task boundaries, task IDs, immediate labels, fixed data-generating processes, mutable models or an offline test set independent of deployment. 
	* **TODO** think which of these can we translate these into something continually collected and optimized 
## Current status of the Field 
![](/attachments/continual-learning-for-agents-Pasted-image-20261018165031.png)
* Refer to the table above for reference if we want to adopt something as is, for general insights, the thing to take away is that what differentiates the different systems is to what layer they write their memories and what gates promotion. We will nevertheless extract the lessons from each "grouping" of systems that are concerned with the same layer 
	* **Context and memory** - These systems show that memory can produce real cross-episode adaptation while the base model stays frozen. Memory quality depends on write, revise, retrieve and forget, not retrieval alone. Fast learning remains local and explicit which is why it is easier to inspect and revoke 
	* **Harness Optimization -** Cluster failures, form hypotheses for clusters and ship/iterate/drop recommendation loops 
	* There are early signs for **Weight Updates** and **Updater learning** but these are very Early

## The Rungs of CL
![](/attachments/continual-learning-for-agents-Pasted-image-20261018172640.png)
## Evaluating CL 
* Split tasks into groups, capture average score change on the different groups due to the CL 
	* Accuracy 
	* timeTokens 
	* performance on OOD tasks 
	* cost of learning in memory growth, retrieval latency, human time
## Failure Modes 
Common pitfalls for these systems aren't loud failures, they instead degrade through time in a way that makes it looks like success early on. Make sure to avoid:
* Memory Bloat 
* Preference Laundering (One user’s stylistic preference is written at global scope, and later it is indistinguishable from a verified rule)
* Proxy Capture/Goodharting
* Self-Confirming Evidence - Use different models for proposal/evaluation and verification 
* Poisoning - A one time injection becomes a trusted instruction that isn't scrutinized
* Plasticity Decay - After many update cycles, the system retains old behavior but learns new behavior progressively worse—the loss-of-plasticity failure

# Takeaways 
* Let's try building systems that make each part of this loop "product grade" through iteration. Each component/service should be stand along and allow us to modify a single layer in a methodical manner that maintains credibility, stress tests and makes sure that no regression happens. We should identify which services are per-layer vs generic ahead of time and build them in a mature manner that will scale. 

# Plausible Research Agenda
* **Calibrated Update Router** 
	* **Question:** Can a learner decide where an experience belongs and know when that decision is uncertain?
	* **Success criterion:** the learned router beats every fixed policy on cumulative utility _and_ has a lower false-promotion rate than the most aggressive fixed policy. Beating “always memory” on utility alone is not enough, because the interesting claim is about knowing when not to write.
* **Session-to-weight consolidation**
	* **Question:** How can a “senior” long-session agent teach a fresh-session version of itself?
	* **Success criterion:** a consolidated fresh session matches full-history prompting on future tasks at a fraction of the context cost, without regressing protected capabilities. **Likely obstacle:** the consolidation target is written by the same model whose errors it inherits, so teacher-error propagation needs its own measurement.
* **Trustworthy online agent learning**
	* **Question:** Can an agent learn when feedback is delayed, implicit, selected by its own policy, or adversarial?
	* **Success criterion:** performance degrades gracefully as the poisoned fraction rises, and recovery time after a drift event stays bounded. A method that wins on clean feedback and collapses at 5% poisoning has not solved the deployment problem.
* **Co-evolving memory, harness, and weights**
	* **Question:** When do multiple surfaces cooperate, and when do they hide one another’s regressions?
	* **Success criterion:** the combined system beats the best single surface on held-out families, and each surface’s contribution remains positive when the others are ablated. If removing one surface _improves_ results, the combination was masking a regression rather than compounding gains.
* **Personal-to-fleet continual learning**
	* **Question:** How can local agents share transferable lessons without leaking private or contradictory experience?
	* **Success criterion:** fleet-level learning improves new tenants who contributed nothing, while a deletion request provably removes the contributor’s influence from every derived artifact. The second half is what makes this hard, and what makes naive “just fine-tune on everything” unusable.
	
* **Continual alignment**
	* **Question:** Can safety, honesty, and calibrated abstention remain stable while capability grows?
	* **Success criterion:** after a long update stream, refusal boundaries and calibration are statistically indistinguishable from the pre-update model on a held-out safety suite, while capability has improved. Any method that reports only the capability half of that sentence has left the important question unanswered.
