---
title: "Pedagogical RL - Teaching Models to Teach Temselves from Privileged Information"
date: "2026-10-04"
type: "paper-summary"
subtype: null
topic: "RL for LLMs"
source_title: "Pedagogical RL - Teaching Models to Teach Temselves from Privileged Information"
source_author: null
source_url: null
---

Typical on-policy RL and distillation techniques use privileged information/execution feedback for scoring rollouts but they do not utilize this information in order to aid models to find successful trajectories in the first place. In scenarios where the model is incapable of recovering correct trajectories alone it stalls. This work posits the question - can we leverage the privileged information to actively steer the models towards successful execution. They introduce pedagogical RL, a paradigm teaching models how to teach themselves to generate rollouts that are correct + each step would be plausible and useful for its own learning. 
The ideal solution for this problem is a sampler that would allow us to recover trajectories that are:
1. correct - maximize the tasks reward
2. learnable - Are closest to the current model's distribution
The authors note that several heuristics already exist in the field today but all suffer from obvious limitations:
3. Rejection sampling - sample more, until at least a few successes are collected
	1. Relies on successful trajectories being at least plausible
4. Privileged teacher sampling - imbue the model with access to hidden context, c, and use it to generate a successful trajectory. 
	1. This obviously suffers from the fact that the trajectories may not be reachable without said hidden context
5. Product sampling - sample from a mixture if student and privileged teacher models. 
	1. 
	$$
	q_{\text{mix}}(\tau_t \mid \tau_{<t}, x, c) \propto \pi_\theta(\tau_t \mid \tau_{<t}, x)^{\delta} \, \pi_\theta(\tau_t \mid \tau_{<t}, x, c)^{1-\delta}
	$$
# The method 
1. Define a learnability score $G_\theta(\tau|x)$ that measures how plausible $\tau$ is for the current student and define a pedagogical reward as the product $R(x,c,\tau) G_\theta(\tau|x)$ which assigns a high reward if the continuation is both learnable *and* arrives at a good answer 
	1. The spike aware learnability score $G_{spike}(\tau|x)$ disproportionately penalizes shocking jumps in the trajectory. Define $a_t^{max}= arg max_a \pi_\theta(a|x, \tau_{<t})$ to be the students most likely next token given the prefix and define the surprise gap as $d_t=\log \frac{\pi_\theta(a_t^{max}|x, \tau_{<t})}{\pi_\theta(\tau_t|x, \tau_{<t})}$ Then we have 
	$$
	G_{spike}(\tau|x)= exp [- \frac{\lambda}{\beta}log(\frac{1}{T}\Sigma_{t=1}^T exp(\beta d_t))]
	$$
		Note that as $\beta \rightarrowtail \infty$ the penalty approaches maximum token-level surprise gap and as $\beta \rightarrow 0$ we get the average surprise gap.  This lets us penalize sharp unsupported jumps without ignoring the overall difficulty of the response.
2. Then, they take a pedagogical self-teacher which has access to privileged context c and a student which does not. They train the pedagogical self-teacher using standard GRPO and the reward being 
$$
max_{\theta_T}\mathbb{E}_{\tau \sim \pi \theta_T(\cdot|x,c)}[r_{ped}(x,c,\tau)]
$$
 where $r_{ped}(x,c, \tau)= G_\theta(\tau|x) R(x,c,\tau)$ 
3. Then, one takes these traces and traing the student on the generated trajectories. However, even given this updated reward the teacher may create highly unlikely tokens. To prevent these tokens from dominating the update they down-weight the gradient according to its student likelihood. 
$$
\mathtt{L}_{assim}= \mathbb{E}_{y \sim \hat{\pi_T}} [\frac{1}{\Sigma_{t=1}^Tw_t} \Sigma_{t=1}^T w_t l_t(\theta_s)]
$$
	Where $l_t(\theta_s) = -log \pi \theta_s (\tau_t|x, \tau_{<t})$ and $w_t = \sigma(\kappa(\log \pi \theta_s (\tau_t|x, \tau_{<t} - \gamma)))$ 
# Results 
1. Much higher accuracy and convergence compared to off-policy, OPS and GRPO variations 
2. Faster Convergence 

# Anfiky 
1. The method in pseudocode along with the forms for the spike score, the way it's incorporated into the reward and the subsequent loss on the collected trajectories
