---
title: "Continual LR with Neuroevolution"
date: "2026-10-08"
type: "paper-summary"
subtype: null
topic: "Continual learning & optimization"
source_title: null
source_author: "Eleni Nisioti, Andrea Cossu, Kathrin Korte, Sebastian Risi"
source_url: "https://arxiv.org/pdf/2610.01583"
---

- **Authors:** Eleni Nisioti, Andrea Cossu, Kathrin Korte, Sebastian Risi
- **Link/DOI:** https://arxiv.org/pdf/2610.01583
- **Date Read:** 2026-10-08
- **Tags:** 

## TL;DR
- The authors study whether the neuroevolution optimization paradigm is better suited for continual learning tasks, offering a better plasticity<>stability combination than classical, gradient based methods

## 1. Problem Statement
- Despite the huge success modern RL has shown, it is still static: policies are trained offline with task specific hyperparams and then don't adapt at deployment - failing at even minor task variations and often forgetting core skills already acquired if forced to adapt. 

## 2. Methodology / Architecture
- Prior work has shown that policy gradient methods lose plasticity when the task changes and propose changes to the optimizer/loss/training procedure to restore it. However, it has not been straightforward as symptoms and causes are hard to tell apart. The authors try an alternative approach, that of neuroevolution. NE belongs to evolutionary algorithms, black box optimizers that refine a candidate solution through cycles of selection and mutation. 
### Evolutionary Algorithms Refresher 
Neuroevolution optimises the weights $\theta \in \mathbb{R}^d$ of a policy network $\pi_\theta$ with an evolutionary algorithm (Risi et al., 2025; Stanley et al., 2019). Each generation it evaluates $P$ candidate networks, using the episodic return $R_\tau$ as the fitness of each. A *genetic algorithm* (GA) keeps an archive of $N$ networks. Each generation it perturbs parents drawn uniformly from the archive with Gaussian noise of scale $\sigma$ and keeps the fittest $N$ of archive and offspring (Such et al., 2018). *Evolution strategies* (ES) keep a single network $\theta$. Each generation they evaluate $P$ perturbations $\theta + \sigma\epsilon_i$, $\epsilon_i \sim \mathcal{N}(0, I)$, and move $\theta$ along their fitness-weighted sum, a Monte Carlo estimate of the gradient of the expected return under $\mathcal{N}(\theta, \sigma^2 I)$ (Wierstra et al., 2008; Salimans et al., 2017).

## 3. Results & Performance

### Prior Work 
* Studying evolution in switching environments, Wang & Dai (2019) asked whether a hill climber that alternates between two landscapes can find a local optimum shared by both. They showed that this is possible through neighborhood linking: if the optimum reached on one task lies within a neighborhood of the other, the next switch carries the climber there, and repeated switches chain neighborhoods together until it reaches an optimum shared by both tasks. In gradient-based continual learning, Mirzadeh et al. (2020) linked wider neighborhoods to less forgetting in supervised learning and widened them through the training regime, such as the learning rate and the batch size; here we instead study neighborhood width in RL tasks and focus on how selection under noise increases neighborhood size and affects the stability-plasticity trade-off.
### Evaluations 
Metrics are computed from the return $R_\tau$ (Section 2.1), averaged over several episodes over multiple environment seeds and deterministic actions. 

**Learning accuracy (LA)**, the return on each task at the end of training on it
**Cumulative return (Cum.)** on the current task over training, which also captures how quickly a method recovers after a switch.
**Forgetting (F)**, the return lost on a task between the end of training on it and the end of the run. **stability-plasticity trade-off**  LA $-$ F, as is common in continual learning 
**Zero-shot transfer (ZT)** is the return on the next task at a switch, before any training on it 

**Neighborhood width** The *neighborhood* of a solution is the region of weight space around it in which the policy still solves its task. We say that a policy solves a task when its rescaled return is at least 0.5. To measure the size of the neighborhood we follow Lehman et al. (2018). We take the centroid at the end of a task and perturb the weights of each layer by a random Gaussian direction whose length is a fraction $\epsilon$ of that layer's weight norm, so that the measure does not depend on the scale of the weights (Li et al., 2018). The *neighborhood width* is the radius $\epsilon$ at which half of the perturbed policies no longer solve the task. A wide neighborhood ensures that a method solves the new task immediately if it also contains solutions of the new task. The *shared neighborhood* measures this: we perturb the centroid in the same way at a single radius, $\epsilon = 0.1$, and report the fraction of perturbed policies that solve both the current and the next task.
### Comparison between methods 
- GA is the most plastic method, but the explicit constraint of enforcing high behavioral diversity comes with a cost for stability 
- ES show the best tradeoff between stability and plasticity
- PPO loses plasticity while continual RL variants restore some of the plasticity while at the expense of stability.
#### Analysis 
* ES finds the widest neighborhoods, and the share of a method’s neighborhood that solves both consecutive tasks correlates with its trade-off. 
	* We perturb the weights of a solution and call the region in which the perturbed policies still solve the task its neighborhood. Lehman et al. (2018) showed that ES finds wider neighborhoods than gradient-based optimization, because it maximizes the expected return of perturbed copies of its solution. Here we further propose that neighborhood size matters for continual learning: when the neighborhoods of two consecutive tasks overlap, a solution to the new task lies close to the current one, so at a task switch a method can reach it while still solving the old task, and be both stable and plastic.
* Behavioural diversity makes the population more plastic by raising the chance that some member of the population solves the new task.
* Dormant neurons do not accumulate in NE, and weight drift does not cost it plasticity
	* we show that NE does not exhibit symptoms that accompany plasticity loss in RL: dormant neurons (Sokar et al., 2023) do not accumulate, and weight drift (Muppidi et al., 2024) does not cost it plasticity
* Continual RL variants improve plasticity at the expense of stability
* ES sits in the widest neighborhood, and the shared neighborhood predicts the trade-off
* Novelty search makes populations more plastic
* NE does not show the symptoms of plasticity loss of RL
## 4. Takeaways 
* Neuroevolution does not show the symptoms of plasticity loss reported in continual RL: its networks do not accumulate dormant units, and the GA’s weights drift further than PPO’s without costing it plasticity, which suggests that these symptoms stem from gradient-based optimisation rather than from the non-stationarity
* We propose a mechanism explaining these results: exploring in parameter space biases learning towards wide neighborhoods, which are more likely to overlap with the neighborhood of the next task: across methods, the size of this shared neighborhood predicts the stability-plasticity tradeoff. This builds on evidence that wider neighborhoods reduce forgetting(Mirzadeh et al., 2020) and that ES prefers robust solutions (Lehman et al., 2018). To the best of our knowledge, we are the first to propose parameter-space exploration as a mechanism for balancing stability and plasticity in continual RL
#### Limitations 
* Less sample efficient than RL 
* When finely tuned, has lower ceiling than RL - doesn't climb narrow peaks of the return like gradient based methods
#### Future Work 
* First, the two NE mechanisms we found beneficial for continual learning, parameter-space exploration for stability and population diversity for plasticity, may also benefit RL. For example, injecting noise into the policy’s weights (Plappert et al., 2018) could let PPO follow the gradient of a smoothed return, and population-based methods that reward diversity (Parker-Holder et al., 2020) could keep it plastic.
* Second, NE itself can be improved with mechanisms we left out. For example, ES and GAs can adapt the range of their mutations online (Bck, 1991; Hansen, 2016; Wierstra et al., 2008); raising the mutation rate when performance drops has long been used to track changing environments 
### Questions 
* Whats the difference between genetic algorithms and evolution strategies 
* What is Maclaurin Expansion
* Why is the log likelihood the "natural" metric to optimize in the fist place? Ra
