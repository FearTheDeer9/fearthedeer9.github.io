---
title: "The Ultimate Guide to RL environments"
date: "2026-10-07"
type: "excerpt"
subtype: "course-summary"
topic: "RL for LLMs"
source_title: "The Ultimate Guide to RL Environments"
source_author: null
source_url: "https://huggingface.co/spaces/AdithyaSK/rl-environments-guide"
---

![](/attachments/the-ultimate-guide-to-rl-environments-Pasted-image-20261007183644.png)
Specifically the partition shown and the components and steps on each side of it.
![](/attachments/the-ultimate-guide-to-rl-environments-Pasted-image-20261007185725.png)
# What makes an Environment 
There is currently no standard protocol for what makes an environment, which means that the different frameworks can be partitioned based on the decisions they make along the four following axes:

1. What does the environment include? Is it only the reward function or does it include tools, state management, and the full multi-learn loop? Does it include the training pipeline?
2. Where does it run? Is it a separate HTTP server or is it in-process with the training environment? (The trade-off is whether a network hop is included versus security and isolation).
3. Does it include the trainer? Do we decouple or couple the environment with its own trainer?
4. When does the reward fire? Is it per tool per step, once per episode, or is it external?

# Comparison Dimensions 
## Building the Environment 
Here we deal with: How do we define the base environment? What does the parent class look like? How do I declare tools? What is the return type and what is the entry point to the environment? 
## Communication and Deployment
This is the most fundamental split. Does the environment run as a separate HTTP server or inside the loop? 
 it's worth noting that in both cases most environments don't actually run the servers themselves but rather they delegate it to a sandbox provider like Modal or E2B.
## Tool & Action Model
 all environments expose a list of callable tools, each with its own name, description, and typed parameters. What differs between the environments is where the tool definitions live and how the trainer discovers them, when it's an in-process versus a separate HTTP server  this changes whether the tool definitions are JSON schemas using Pydantic or other frameworks, or end-process Python functions. The trade-off here is ease of definition of end-use versus portability across different frameworks and models
## Reward Architecture 
 this is more of a philosophical difference. The different frameworks can decide whether they want external rewards, server-embedded rewards, post-episode rewards, or environment-embedded rubrics.
  in the external case the training script is what decides it. The environment returns text and a separate reward function compares this raw file to the expected output, returning a reward. The environment is a dumb pipe passing the reward function you chose.

If the rewards are server-embedded it means that every tool's response includes a reward. The trainer calls every reward on what you defined previously.

Post-episode verification means that a verify endpoint is called. The server handles this after a full trajectory. Environment-embedded rubrics mean that the environment exposes a composable rubric object that computes rewards from both actions and observation during each step, and sets the reward for it specifically 
## Episode Control
a rollout is a cycle of the model generating actions, tools getting called, observations going back to the model, and repeating. However we need to decide two things: who runs the cycle and how the cycle ends.
* does the trainer drive the cycle
*  is it the environment itself? 

When does it end?
- a flag
- a dedicated terminate or tool call
- max turn cap
- verifier deciding after each action

Each framework picks their own answers and this affects how we batch rollouts, how we can parallelize work, and what the training data is that we end up with
*  this decision drives whether it's natural for you to plug and play harnesses. The end-driven variant plays more nicely with harnesses leading it, as it's not clear when the trainer runs it. How does the delegation between it and the harness work 
## Tasks & Datasets 
 the frameworks diverge with respect to how strict the schema for a specific task is. The stricter it is, the more you have to work on adapting each environment to the baseline spec but the more portable it is. If you care about one task, it's overhead. If you're running many tasks and using full suites, the specs are what make it a one-time thing. Once you have the adapter you can port and change the kind of rewards and modify the environments easily while keeping the tasks compatible with it, as long as you don't touch their adapters
## Local & Cluster Setup 
 a key distinction to keep in mind is that, in process for Python, the model and the trainer are co-located, which usually means that they need to run on a GPU. For HTTP servers the environment is usually on a CPU, whereas the trainer is on a GPU. If we want to scale to many concurrent environments, we only need extra CPUs rather than extra GPUs since a single model can train based on rollouts from the concurrent CPU environment 
# Conclusion 
The dimensions above can be grouped into - 
* identity
* API surface 
* Deployment 
* Tooling 

It's worth noting that the HTTP vs in-process split is by far the most pivotal, the consequences of this decision propagate to all other decision points due to the constraints they place, start from it when choosing. When facing a practical decision for which framework to adopt, we can consult the table at the bottom of the article:
https://huggingface.co/spaces/AdithyaSK/rl-environments-guide#global-comparison-matrix

**The “environment” boundary is a design choice, not a standard.** Some frameworks (OpenEnv, SkyRL Gym) hand you a thin tool interface and let you own dataset, reward, and trainer wiring. Others (Verifiers, GEM) bundle all four. Neither is wrong, it’s a control vs time-to-running trade.

**HTTP vs in-process is the loudest fork.** Sandboxed execution (code, shell, browser) wants HTTP, you scale env compute independently from training. Pure Python (games, math, text reasoning) wants in-process, zero RPC overhead, no infra to babysit. Pick this axis first, the rest of the table narrows fast.

**Dataset coupling cuts both ways.** Bundled (Verifiers, ORS) means env and dataset come as one unit, can’t swap one without the other. Decoupled (OpenEnv, SkyRL Gym) makes you wire datasets yourself, but any task fits any env.

**Reward timing matters more than reward content.** ORS scores per tool call, NeMo Gym scores post-episode via `/verify`, OpenEnv and Verifiers use composable Rubrics. Per-call is clean for per-step feedback, awkward when reward only makes sense at the end. Post-episode is clean for trajectory-level scoring, no in-episode signal.

**Execution backend dominates per-step latency.** Sandbox creation and tool execution own most of the wall clock regardless of framework. Numbers in [Dimension 10](https://adithyask-rl-environments-guide.hf.space/?__theme=system#dimension-10-scaling-deployment).
