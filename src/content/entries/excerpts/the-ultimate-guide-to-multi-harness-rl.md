---
title: "The Ultimate Guide to Multi-Harness RL"
date: "2026-10-04"
type: "excerpt"
subtype: "article-summary"
topic: "RL for LLMs"
source_title: null
source_author: null
source_url: null
---

I'll be concise here as the point is quite trite - RL in today's AI landscape is a system level problem and the "thing" that we are evaluating and improving is no longer the model individually but rather the coupling of model <> harness (i.e the agent). This means that we need have robust environments that the model interacts with the harness acting as an intermediary. However, the modifications/adaptations the model needs to make are harness-specific and the harness domain is currently a oligopoly. As a result, if we want to train our model for the most generalizable notion of "optimized for real-world, agentic use cases" we need it to easily adapt to new harnesses. This is the subject of this post. Before we dive in, a few preliminaries on RL and RL environments 

# Glossary 
* **Black Box RL** - the harness owns the loop. It starts inside a sandbox, runs its own loop, calls its own tools, compacts its own context, and stops when it decides to. It is an agent program that was never written with training in mind. The trainer is outside that box. All it sees is a sequence of calls arriving at a model endpoint.
* **white-box** environment - the trainer owns the loop. The trainer samples an action, calls `env.step()`, reads the observation, and then samples again. The environment waits to be called. Every token the policy produced is already in the trainer’s hands, because the trainer is the one that sampled it.![](/attachments/the-ultimate-guide-to-multi-harness-rl-Pasted-image-20261004130217.png)
	* It's important to note that for on-policy policy gradient methods the trainer must have the exact tokens sampled and the probabilities given to them. With black-box RL we have neither. As a result only off policy methods are compatible with black-box RL, it's an active topic of discussion if one can recover white-box performance with black-box techniques but worth noting. 
		* To avoid both errors, the sampler must return token ids and per-token log probabilities, and the trainer must keep them. Never re-encode text you decoded. Keep the sampled ids and append the template’s suffix by id concatenation. This works because most chat templates extend token for token when a tool message is appended, and eighteen of the nineteen models tested in TRL’s write-up do so unmodified. It is also why a system that wants the exact tokens has to intercept at the model endpoint rather than at the text boundary.
# Failure Modes 
* When a model trains with a single fixed harness it can overfit to it's actions format, its context structure and its control flow.![](/attachments/the-ultimate-guide-to-multi-harness-rl-Pasted-image-20261004141032.png)
# Current situation 
* Frontier labs have noticed this gap and almost all recent releases come with checkpoints that have been created from models that train across different harnesses - and they have shown that one does not leave "optimized performance" on the table by going this route (i.e the peak performance reached when training across several is equal to that you would get from only training on one bespoke harness. What's still missing is a way to do this with any harness - including ones that you don't control - and plugging it into whichever trainer you already use. 
# The New Proposal 
![](/attachments/the-ultimate-guide-to-multi-harness-rl-Pasted-image-20261004142223.png)
## Components 
* **Capture Proxy** - the harness finds the proxy the way it would find any model provider, through a base URL and an API key. The API key is a capture session id minted for that rollout, which is how one proxy on one port serves every concurrent rollout. A call carrying a key the proxy never registered gets a 401. 
	* The proxy works out which API a request uses from it's path, headers and body shape and routes it accordingly through NVIDIA's polar gateway
	* For each call the proxy asks the engine for the prompt token ids, the sample token ids and the logprob of each sampled token.
		* Remember, one can ONLY train on-policy if BOTH token ids and token logprobs are given
	* Harnesses retry, spawn subagents and compact their context, so a rollout is rarely one clean conversation. The proxy stores each model call as a node and links it to the earlier call whose prompt and completion form the longest exact token prefix of its own prompt. A retry shows up as a sibling that never continued. A subagent has its own system prompt, so its first call extends nothing and starts a new root, and a compacted context does the same.
* **Harbor** -  runs agents against containerized tasks and keeps the task, the harness and the sandbox independent of each other.
	* Returns a score, while the RL extension also adds tokens and probabilities through intercepting them from the inference server. 
* </Add information regarding deployment specific adaptations here/ >
# Training & Results 
* **Reward Shaping**  
	* **Correctness:** 1 for a correct answer and 0 for a wrong one.
	- **Tool efficiency:** a small bonus, at most 0.1, for a correct answer that takes fewer calls. A wrong answer still scores 0, so the model can’t earn the bonus by giving up early.
- Both runs take the same amount of steps but don't see the same amount of data
- The multi harness variant came out as slightly more accurate and significantly more efficient wrt tool calls + token usage
# Conclusion 
* **Multi-harness RL improved both task performance and tool use in our small-model experiments.**
	*  At the final checkpoint, the multi-harness model used **31% fewer tool calls on held-out tasks that both it and the base model solved**, with savings under every harness.
* OpenEnv’s capture proxy makes this practical without rewriting the harnesses or building a separate training loop for each one. With a compatible inference backend, it records the exact prompt and completion tokens, generation log probabilities and loss masks that training needs. Harbor supplies the tasks and sandboxes, and the same TRL trainer learns from the resulting traces. The harness keeps its own tools, context management and execution loop. Changing which harness produces the rollout does not require reimplementing those behaviours inside the trainer.

# Ankify 
* The SWE abstractions one needs to introduce in order to make the RL pipeline model/harness agnostic, enabling this plug and play approach for post training
