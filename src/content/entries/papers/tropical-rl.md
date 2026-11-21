---
title: "Tropical RL"
date: "2026-10-08"
type: "paper-summary"
subtype: null
topic: "Reinforcement learning (classical)"
source_title: "Tropical Reinforcement Learning"
source_author: "Arip Asadulaev, Aladin Djuhera, Karim Salta, Holger Boche, Fakhri Karray, Martin Takac"
source_url: "https://arxiv.org/pdf/2610.02478v1"
---

- **Authors:** Arip Asadulaev, Aladin Djuhera, Karim Salta, Holger Boche, Fakhri Karray, Martin Takac
- **Link/DOI:** https://arxiv.org/pdf/2610.02478v1
- **Date Read:** 2026-10-08
- **Tags:** RL, RLVR, LLMs, compositional reasoning, tropical semiring

## TL;DR
- Expected-return RL adds the probabilities of all verified trajectories. This measures how often the current policy succeeds, but it does not retain which solution worked, and a policy update on one sampled action can lower the probability of an unobserved successful alternative.
- Tropical RL changes the aggregation over alternative paths from `sum` to `max` while keeping the sum of log-probabilities along each path. A state value therefore names the most likely verified continuation. Since max-plus is distributive, a best prefix and suffix can be joined at a shared state even if they came from separate, individually failed rollouts.
- **TROPIC** stores valid transitions and verified paths in a persistent graph, composes paths, and trains the policy on selected verified solutions. It beats the strongest on-policy baseline on all four tasks and both models tested, by up to 16.2 percentage points.

## 1. Problem Statement
- RL with verifiable rewards (RLVR) usually credits a trajectory only once the model samples a complete successful episode. In long-horizon sparse-reward tasks, a rollout can contain many useful steps and still receive no credit because it fails near the end.
- Expected return is a sum over successful trajectories. The sum reports the total probability of success, not which concrete computation succeeded. It also acts on normalized policy probabilities: increasing one action can lower another action that was never sampled or shown to be wrong. That can erase a useful fragment needed for a later composition.
- A solution with $d$ fragments, each discovered in separate attempts, may have exponentially small probability of appearing end-to-end. The authors ask how to assemble and reinforce a verified solution from pieces the policy has already produced, without first sampling the whole sequence.
- **Setting:** deterministic, resettable environments with verifiable terminal outcomes and repeated intermediate states. The work studies this setting; it does not establish that arbitrary stochastic environments can be handled the same way.

## 2. Methodology / Architecture
### Tropical value
- In ordinary RL, probabilities along a trajectory multiply, while alternative trajectories sum. In log space, a trajectory score is the sum of its step log-probabilities; expected return's sum over alternatives becomes log-sum-exp.
- Tropical RL replaces only the aggregation over alternatives with a maximum. In the binary-verifier case, the value is the maximum log-probability among verified continuations:
  $$V_{trop}(s)=\max_{\tau\in\mathcal{Y}^{+}(s)}\log P_{\pi}(\tau\mid s).$$
- The authors motivate `max` with two requirements: **certification** (the value must be the score of an actual verified path) and **retention** (adding a newly found path must not lower the value). They prove that these requirements force the maximum as the aggregation operation. Max is also idempotent: finding the same path again does not increase its value just because it was sampled more often.
- The max-plus Bellman update is $Q(s,a)=\lambda\log\pi(a\mid s)+r(s,a)+V(T(s,a))$, with $V(s)=\max_a Q(s,a)$. Here $\lambda$ controls how much the model likelihood contributes to path selection. With $\lambda=1$ and a binary verifier, the value is the log-probability of the most likely verified path.

### TROPIC
- **Build a state graph.** Sample attempts and retain valid transitions even when an episode ultimately fails. A state is an environment configuration plus step count; including the step count makes the graph acyclic and ensures states at different horizons are not merged.
- **Restart from a frontier.** After the initial attempts, choose reachable states that do not yet have a verified completion, favoring shallower under-explored frontiers. Start new attempts there, prepending the best known prefix so each attempt can still be replayed and checked as a complete path.
- **Compose paths.** At shared states, join high-scoring known prefixes with high-scoring suffixes, replay each complete candidate in the environment, and ask the verifier to accept it. This can discover a successful path that no rollout sampled end-to-end. The graph and archive persist across iterations.
- **Train and rehearse.** For each training problem, use the best archived path plus up to $K-1$ paths selected for coverage of steps that have been trained less often. Apply maximum-likelihood (token cross-entropy) updates to those paths. TROPIC uses no critic, reward-weighted advantages, entropy bonus, or reference-policy KL penalty in this update.
- The archive's certified capability is monotone because paths are retained. That guarantee is about memory: shared model parameters can still interfere across states, so the policy itself is not guaranteed never to forget. Coverage-based replay is a heuristic to reduce that risk.

## 3. Results & Performance
- **Setup:** four deterministic, resettable agent tasks (Sokoban, Countdown, FrozenLake, WebShop); Qwen2.5-3B-Instruct and Gemma 4 E4B-it; comparison against PPO, GRPO, DAPO, and SNR-Aware Filtering. Each method uses three training seeds and is evaluated on 512 held-out instances per environment. At evaluation, TROPIC gets no archive, restarts, or composition: the policy must solve each instance in one attempt.
- **Success rate:** TROPIC is best in all eight task/model settings. Against the strongest baseline for each setting:
  - Sokoban: +16.2 points (Qwen: 58.4% vs 42.2%); +6.6 (Gemma: 83.1% vs 76.6%).
  - Countdown: +5.9 (Qwen: 34.8% vs 28.9%); +6.0 (Gemma: 73.6% vs 67.7%).
  - FrozenLake: +9.1 (Qwen: 86.4% vs 77.2%); +7.2 (Gemma: 92.3% vs 85.1%).
  - WebShop: +6.9 (Qwen: 65.7% vs 58.8%); +4.6 (Gemma: 81.4% vs 76.8%).
- **Ablation (Qwen, Sokoban):** full TROPIC reaches 58.4%. Disabling composition drops to 40.8%, close to the strongest baseline at 42.2%; retaining composition but removing frontier restarts gets 51.8%; composing only fragments from already successful rollouts gets 53.3%. This supports the authors' claim that cross-rollout composition is the main source of the gain, and that fragments from failed rollouts help.
- In the same Sokoban ablation, 31.6% of solved training problems were first solved by composition before a complete successful rollout had been sampled. On held-out evaluation the method does not retrieve archived solutions, so the measured gain transfers to the trained policy.
- TROPIC used a comparable or lower number of generated tokens per evaluation episode and trained faster in seven of eight settings; Gemma Countdown was the exception for training time (4.3 h vs 4.5 h for the strongest baseline). These are reported under the paper's standard protocols, not as a matched generated-token or wall-clock budget in every detail.

## 4. Limitations
- The main algorithm assumes deterministic, resettable environments with verifiable outcomes and shared states where prefixes and suffixes can meet. The paper does not establish the same guarantees for stochastic or non-resettable tasks.
- The archive never loses a verified path, but the shared-parameter policy can still forget one after training on another path. Coverage replay mitigates interference heuristically; it does not prove retention in the model weights.
- The experiments cover four tasks and two models. The authors explicitly do not claim systematic generalization to a different distribution of compositions; they leave that evaluation to future work.
- Composition requires graph storage, candidate generation, path replay, and verification. The paper argues replay is cheap relative to token generation in these tasks, but the trade-off may differ in other environments.

## 5. Personal Takeaways / Action Items
- The conceptual move is unusually clean: change the algebra used to aggregate alternatives, rather than add another estimator or reward-shaping term. `max` gives the learner a witness it can replay, while the archive gives it a persistent set of witnesses to compose.
- The distinction between **retaining a capability in memory** and **retaining it in shared model weights** matters. The first follows from the archive; the second remains an open continual-learning problem.
- This seems most useful when tasks have sparse terminal feedback, long correct prefixes, and many attempts that pass through the same intermediate state. The shared-state interface is what makes fragment recombination possible.
- Follow up on how their graph and composition procedure behaves when state identity is approximate, the environment is stochastic, or a verifier can be noisy; those seem like the main boundaries before applying the idea to less structured agent tasks.
