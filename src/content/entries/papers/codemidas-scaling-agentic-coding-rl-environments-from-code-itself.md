---
title: "CodeMidas - Scaling Agentic Coding RL Environments from Code Itself"
date: "2026-09-22"
type: "paper-summary"
subtype: null
topic: "RL for LLMs"
source_title: "CodeMidas - Scaling Agentic Coding RL Environments from Code Itself"
source_author: "Xiaomi LLM Core et al. (Tong Yang, Fuli Luo)"
source_url: "https://arxiv.org/pdf/2609.22068"
---

## Provenance
- **Link** — https://arxiv.org/pdf/2609.22068
- **Authors / affiliation** — LLM Core, Xiaomi; Peking University; University of Hong Kong; Renmin University of China. Co-corresponding: Tong Yang, Fuli Luo
- **Venue, date** — arXiv 2609.22068v1, 18 Sep 2026
- **How it reached me** — X
- **Why I clipped it** — Corrections for RL, and RL generally, is a field I want to get further into — specifically how to adapt agents for domain-specific tasks. Part of the curriculum I'm self-imposing. Read alongside [Reinforce-Ada - An Adaptive Sampling Framework under Non-linear RL Objectives](/papers/reinforce-ada-an-adaptive-sampling-framework-under-non-linear-rl-objectives/): that paper extracts more signal from a fixed task set, this one manufactures the task set.

## Claim
<!-- What is the paper asserting? One paragraph, in your words, no jargon borrowed from the abstract. -->
Every existing pipeline for building coding RL environments needs some artifact beyond the code itself — issues, pull requests, commits, existing tests, or a written description — so each one depends on human help of some form and inherits the coverage of whatever happened to get recorded. CodeMidas takes source code as its only task-specific input, which is the natural move given that code is the problem domain being trained on. It deletes an implemented piece of functionality from a real codebase, adapts what remains into a coherent starting point, and asks the solver to re-implement the behavior from a statement describing it. The verifier's trustworthiness comes from re-deriving the same functionality as the original code: tests are grounded in execution of the reference implementation, so expected values are *observed* rather than written down.

## Techniques and evidence
<!-- What techniques and evidence are used? How do they distinguish between true and false beliefs about the model? Which forms of evidence feel stronger vs weaker? Any flaws in the setup? -->

Every stage is agent-driven, and the funnel is steep — 22,575 candidate codebases in, 5,545 tasks out, roughly a quarter surviving.

### The pipeline
1. **Task design and codebase adaptation** (→16,027). An agent finds functionality with public entry points and observable outcomes, traces entry points and shared dependencies to fix the task scope, removes the core implementation, and adjusts what remains into a coherent starting point. The original implementation is kept aside as the reference solution. The statement defines inputs, observable behavior and required public interfaces; internal helpers and algorithms are left to the solver.
2. **Execution-grounded test construction** (→12,746). An agent maps the statement's requirements to inputs and boundary cases, invokes the public entry points on a reference copy, and records what happens. CLI tools get command executions, pure functions input–output cases, stateful APIs sequences of calls exercising ordering and cleanup.
3. **Environment preparation and execution consistency** (→11,930). Cleanup strips anything that could reveal the deleted implementation — compiled outputs, cached copies, files left by the construction agents, and the original tests for the target functionality. Each task is then run in six fresh containers under training runtime settings: two on the starting codebase, which must both fail, and four on the reference solution, which must all pass. Fail-to-pass plus a flakiness screen in one check.
4. **Post-rollout filtering** (→8,173 →5,545). Three passes. *Leakage*: an adversarial agent hunts the solver-visible environment for residual material — compiled artifacts, caches, installed copies of the target project — that would let it recover a solution without doing the work, and a separate review confirms the exploit against the reference and verifier. *Agreement*: a coding agent attempts each task four times, and a reviewing agent compares each implementation against statement, verifier and reference to flag false positives and false negatives; tasks with verifier defects are rejected. *Rollout outcome*: a frontier model makes several attempts and only tasks with both successes and failures are kept.

### The specification problem is not solved
Tests have to reject wrong implementations while still accepting alternative correct ones. Their answer is to let the statement decide what an assertion may pin: reference execution fixes expected values only where the statement specifies behavior, unspecified aspects get only the stated constraints checked (a required exception type, not its message wording), and a review agent then strips assertions resting on exact wording, incidental ordering, or internal structure. A task is rejected if an assertion needs a private symbol with no behavioral substitute.

This is the hard part and it is not solved. What the pipeline is really doing is the reverse problem: guessing the *true* function that was meant to be implemented by looking at one specific attempt at it and the input–output values that attempt produces. The original authors' intent may not be accurately recoverable from that, and nothing in the pipeline can tell the difference between behavior that was intended and behavior that is an accident of one implementation.

### Agents all the way down
The agents that build and validate the environments are the same kind of thing as the agents being trained in them. This is slightly bothersome — it relies on separate agents being able to teach one another. But correct prompting and enough care should be able to identify heuristics that make the loop genuinely automatable, conditional on model intelligence and on where the precision/recall tradeoff is set.

### Setup
MiMo-V2.5 trained with GRPO on the 5,545 tasks, binary execution reward, batch size 32, 32 rollouts per task. The verifier lives outside the solver's environment and is injected only at grading. Five external benchmarks, all reported against the initial policy: SWE-bench Pro +4.1, DeepSWE +11.7 (10.0 → 21.7), ProgramBench +17.0 on Almost Solved (4.5 → 21.5), RepoZero C2Rust +11.3, Terminal-Bench v2.1 +8.5 (63.7 → 72.2). CodeMidas Val, 200 held-out tasks, 35.0 → 44.7. Ablation: 1k → 3k → 5k rises monotonically on all three evaluations, and an unfiltered "vanilla 8k" pool loses to the filtered 3k subset on all three.

*Not audited on this read — the results section was skimmed. See Mine for the baseline that is missing from it.*

## Generalizable vs specific
<!-- What seems model/task-specific? What do you predict will generalize? -->

### General
The insight is that **the produced product is the source of truth**. Relying on what someone naturally creates in the course of their workflow, rather than on a view or artifact built on top of it, gives a much higher chance of identifying the true ontology and abstraction. The secondary artifact adds a layer where effort level, tiredness, oversight and distraction can inject noise and subvert the person from their goal. Read that way, their Table 1 axis is not really about metadata availability — issues, pull requests, commits and documentation are all *descriptions* of the work, written under those conditions, while the code is the work itself.

Also general:
* Deleting the answer from a working artifact and keeping that artifact as the oracle.
* Fail-to-pass across repeated fresh containers as an environment screen — it catches both a broken starting state and flaky execution in one check.
* Adversarial leakage probing as a standard construction step rather than an afterthought.
* Spending agent compute at construction time instead of only at training time.

### Specific
* Everything resting on a deterministic, containerizable executable reference. Code is unusually lucky here — the artifact *runs*, and execution supplies expected values for free. Outside code, something else has to play that role.
* The binary execution reward.
* The statement as arbiter of what an assertion may pin.
* The language and domain mix — Python, TypeScript and Go are 56% of tasks; median reference patch 142 lines; at least two source files touched in 65.9% of tasks.
* One model, MiMo-V2.5.

## Cherry-picking
<!-- Which parts feel most cherry-picked? Which numbers carry the headline, and on which subset? -->

## Limitations and flaws
<!-- Theirs, stated. Yours, found. Keep the two apart. -->
### Theirs

### Mine

## Concrete output
<!-- Pick one: code you ran against the model, a blog post, a newsletter-style summary, an explanation you gave someone. Papers you really like get one. -->

## Anki
<!-- Paste paragraphs verbatim while reading. A later session extracts these into Anki/ as BEGIN_NOTE blocks. Leave them raw. -->
