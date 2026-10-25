---
title: "Environment Scaling for Agentic RL"
date: "2026-10-01"
type: "excerpt"
subtype: "article-summary"
topic: "RL for LLMs"
source_title: null
source_author: null
source_url: null
---

* In order to build an environment in modern RL, one needs five components:
	
	1. The task spec, which includes the description of the task. Commonly, also extra,  hidden information that is hidden from the model being tested and is privileged knowledge for the final judge.
	2. The tool arsenal, their descriptions, and how they work.
	3. The harness, which is about managing the context for the agent executing tool commands, interacting with the environment, parsing, and initiating the agent when new results arrive.
	4. A backend or persistent state, which holds the global state of the environment and tracks progress of verbal artifacts.
	5. A verifier: how do we validate and grade the RL trajectory?
* the pipeline of them building environments using them for training and improving our models is an 8-step pipeline.
1. Identifying source materials from which we can seed new environments. This can be:
    - code repos
    - tool documentation
    - a task set
    - a trajectory of past user interactions
2. We take these source materials and translate them into a tuple of task and environment. This stage has a few common patterns:
    1. Skeleton to scenario: This is when we first create a skeleton of the environment, which can include state variables, tools, etc., and then inject or infer plausible tasks and scenarios to run on this environment, given the initial source and the skeleton we got.
    2. We have an environment-first inversion. That is instead of writing a task and then an environment, we take a working system that we sourced and we break it. We introduce bugs and ask the agent to fix them. Since we already know about this
3.  build and make it executable. We take the artifacts we produced in the previous step and we use an LLM with a repair loop until it works end to end. That is, we generate the code, try to build and run it, observe and monitor any errors, return them to the LLM, and ask it to iterate. At the end of this stage we have an environment that is fully executable 
4. Now that it's executable the next stage is to verify, we check whether the task that we produce is satisfactory. We define satisfactory environments as tasks which we can verify and then, given that we can verify them, it makes no difference to us if they're verifiable yet unsolvable. Once we know it's verifiable we check whether it's solvable as well as non-trivial. That is, two sides of the same coin. 
5.  now that we have filtered the environments to those that are verifiable and solvable, we start collecting trajectories from strong teacher models. If we're preparing a new model we can use the best state-of-the-art models that we currently have to bootstrap. We generate many candidate runs filtered to those that are successful and use them for an initial seed of SFT data. This also serves as the RL bootstrapping signal 
	1. my own note here: I just got an idea for a research proposal. The common issue here is that state-of-the-art models are usually closed models from frontier labs but these typically hide the thinking traces from their models, which have been shown to be quite useful for training. What I'm now thinking is whether we can:
		1. see the initial trajectories from state-of-the-art models
		2. collect successful trajectories
		3. use the leading open-source models to fill in the trajectories between outputted blocks with plausible reasoning steps that led one from the first observation to the latter
6.  now we're ready to train. We start with a short SFT warm-up stage where we train on the previously collected trajectories in order to obtain formatting prowess as well as a baseline of reasoning and execution capabilities to gain some form of signal. Then we move to the RL, where the model trains and gets rewarded based on the verifier
7. A one-shot pipeline leaves signal on the table. After a round of training you know _much more_ than you did at generation time: which skills the agent still fails, which tasks gave a clean learning signal, and which were a waste of compute. The **evolve** step turns that hindsight into the next batch of environmentsThree kinds of diagnostic signal are available, in increasing order of how hard they are to extract:
	- **Inference-level signals** — _what the agent failed at._ Per-skill failure clusters, and failure _modes_ like looping on the same command, exhausting the turn budget, or terminating early. These are cheap and robust, and they drive **coverage repair** (generate more of the under-served skill) and **failure-targeted generation**.
	- **Training-level signals** — _whether a task family is teaching anything._ The fraction of tasks with all-zero or all-one reward, the per-task reward/advantage variance, entropy and KL trajectories. A family that produces no reward variance is producing no gradient (this is the difficulty-vs-trainability point of _Difficulty ≠ trainability_, used here as a control knob), and should be made easier, decomposed into subgoals, or dropped.
	- **Transfer-level signals** — _whether the synthetic version is too easy._ A gap between performance on synthesized development tasks and on a held-out benchmark flags skills whose synthetic rendition is a simplified caricature of the real thing.

*  an important note is that almost the entire pipeline is governed by the choices made regarding sourcing verification and traning, which are independent choices. That is, the generation-to-execution-to-feedback loop, or the GEF loop, is the way to read methods and environment scaling. Each of these GEF stages has ways of being extended and improved so that the environments that it produces would be more beneficial 
	* ![](/attachments/environment-scaling-for-agentic-rl-Pasted-image-20261001164309.png)
	* generation. Here we have:

		- complexity scaling, where we make individual tasks harder
		- diversity scaling, where we cover more environments and domains
		- dynamic scaling, where we adapt the difficulty to the current agent
	- execution. 
		- interactivity, which is giving the agent tools that are genuinely executable and not hallucinated simulators
		- realism, which is how far the gap is between an environment and deployment settings. Can the agent immediately notice that it's being tested 
	- Feedback 
		- Granularity: binary success to partial credit to rubric scores.
		- Automation: Is this plain LLM, deterministic code, or a hybrid?
		- Robustness: How easy is it for the agent to be hacked? Are the environments robust against red teaming efforts from a misaligned model?
-  I know, which I already know, but it's worth reemphasizing that there's a difference between the most informative and the hardest tasks. We want tasks whose probability of success is closest to 0.5 (the pass rate). That means that we need to filter out the easiest ones, never to return or, rather, maybe only to check that no regression happened. The hardest ones, where we're currently unable to get any training signal, should also be filtered out but this time it's only for the current training step. We should come back to these tasks and see whether, after subsequent training, they now become feasible and should be reintroduced
- open challenges include:

1. Robust verifiers, that is, automatic verifiers that give fine accurate descriptions and scores while being hard to hack.
2. Generalization: improvement on the synthesized environments doesn't always scale to production settings.
3. Co-evolving the environments with the agents. As the agent becomes more capable the environments need to get progressively harder. Current systems usually rely on loops where we transition between some form of environment synthesis and further training but a better way of generating environments to fit the current agent is something to explore.
