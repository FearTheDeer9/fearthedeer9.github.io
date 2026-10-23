---
title: "Reward Respecting Subtasks for Model-Based RL"
date: "2026-10-06"
type: "paper-summary"
subtype: null
topic: "Reinforcement learning (classical)"
source_title: "Reward Respecting Subtasks for Model-Based RL"
source_author: null
source_url: "https://arxiv.org/pdf/2202.03466"
---

## Provenance
- **Link** — https://arxiv.org/pdf/2202.03466
- **Authors / affiliation** — 
- **Venue, date** — 
- **How it reached me** — 
- **Why I clipped it** — 

## Claim
The authors note that to achieve the ambitious goal of AI RL must be able to reason abstractly in both state and time. They claim that state reasoning has made progress but temporal abstraction lags far behind, largely due to the immense size of possible options. They flag the fact that previous proposals for option discovery do not take how the options will be used in planing into account as a severe limitation. To alleviate this, they propose reward-respecting subtasks, these come with strong contraints on the space of options and thereby provide a partial solution to the problem of option discovery. Further, they can be adapted to work natively with the "standard" algorithms.

## The Challenge of Temporal Abstraction
The issue of abstraction is common in AI in general and RL in particular, the agent sensors work with low-level data such as pixels/tokens while the reward is given based on performance on much higher-level objects such as correct solution/navigation etc. Abstraction in planning in particular relate to how an AI model translates the transition dynamics it has learned into improvements in the policy/value function and corresponding behavior. The authors suggest *options* as the way if formulating temporal abstractions. One poses subsidiary tasks that should theoretically help with solving the problem. Solutions to these tasks produce an option, the consequences of this yield a model. The model is then used for planning (SubTask, Option, Model, Plan == STOMP). In the current work, the authors introduce the notion of reward respecting subtasks. That is, rewards that aren't decoupled from the main task reward as is sometimes the case, and thereby avoiding situations in which solving the subtask is detrimental to solving the main task. 
## Reward Respecting Subtasks 
### GVF 
The first step of the STOMP progression is subtask specification. 
* **GVF -** The authors introduce a generalized value function which has three extensions compared to the "regular" VF. 
	* **Cumulant function $C: \mathbb{S} \rightarrow \mathbb{R}$** generalizes the reward function to an arbitrary quantity to be added up
	* **Stopping Function $\beta: \mathbb{S}\rightarrow[0,1]$**  which gives a probability for stopping at a given time
	* **Stopping Value Function $z: \mathbb{S} \rightarrow \mathbb{R}$** added to the accumulation
To formulate subtasks given a GVF, the functions c,z are considered fixed and the function $\pi, \beta$ can vary. **Reward Respecting Subtasks** are subtasks whose cumulant is identical to the reawrd and the stopping value takes into  the value of the state stopped in. The stopping value is related but not equal to $V_\pi$ since in that case it wouldn't alter behavior in any manner whatsoever. The paper focuses on reward respecting subtasts whose stopping values encourage solutions to stop when a given feature is high. Generally, it is only useful to construct subtasks for attaining a feature i if $w_i$ , its estimated contribution to value on the main task, is sometimes high and sometimes low.
* I glossed over this but the paper assumes linear subtasks of the form $z^i(s)= \boldsymbol{w} \cdot \boldsymbol{x(s)}$ with x some feature function and w a modifiable weight vector.
## Option Learning 
Now that we have specific subtasks, the next stage of the STOMP progression is specifying off-policy learning algorithms to approximate optimal VFs and options. An option here is a policy together with a stopping rule. The authors introduce a general TD error and an update procedure called UpdateWeights&Traces (UWT), so one template can describe learning values, policies, options, and the option models in the next STOMP step.
* **TD error** — $\delta(c,z,v,v',\beta)=c+\beta z+\gamma(1-\beta)v'-v$. Here $c$ is the cumulant (the original reward for these subtasks), $z$ is the stopping value, $v$ and $v'$ are the current and next subtask-value estimates, $\beta$ is the probability of stopping, and $\gamma$ is the discount factor. The update combines the stopping value when the option stops with the next value when it continues.
* **UWT** — a shared weight-and-trace update:
  * $e \leftarrow \rho(e+\nabla)$ — add the current gradient to the eligibility trace, weighted by the importance-sampling ratio $\rho$.
  * $w \leftarrow w+\alpha\delta e$ — update the weights using the TD error and trace.
  * $e \leftarrow \gamma\lambda(1-\beta)e$ — decay the trace according to the discount, trace parameter, and stopping probability.
  In these examples, $\nabla$ is either the gradient of the value estimate with respect to its weights or the policy gradient $\nabla_\theta\log\pi(A_t|S_t,\theta)$. For on-policy learning, $\rho=1$. UWT is a compact way to describe these updates, not a requirement to use this exact procedure in every implementation.
* **Learning each reward-respecting option off-policy** — for each feature-attainment subtask $i$, learn a value estimate $\hat v(x(s),w_i)$ and a policy $\pi_i(\cdot|s,\theta_i)$; $\theta_i$ parameterizes that policy. The subtask still earns the original reward and uses the stopping value $z_i(s)$ to favor ending in states where feature $i$ is high. Its stopping rule is $\beta_i(s)=1$ when $z_i(s)\geq \hat v(x(s),w_i)$: stop when the subtask's stopping value is at least as good as the estimated value of continuing. Otherwise, continue the option.
  * The behavior policy $\mu$ may differ from the option's policy, so each update uses $\rho_t^i=\pi_i(A_t|S_t)/\mu(A_t|S_t)$. UWT updates the value weights and policy weights with the same TD error and this ratio; the policy update uses the policy gradient above.
* **Hallway example** — the agent learns a hallway-attaining option from data generated by a uniform random behavior policy. With linear value functions and a softmax policy, the off-policy actor-critic updates learn the optimal option and value function. The option stops at either the hallway or the goal: it trades off reward earned along the way, the value of the stopping state, and the feature-attainment bonus. The main task itself can also be learned in this setup as another subtask with no stopping, $\beta_0=0$.

## Model Learning 
Moving on to the third stage of the STOMP progression, the authors learn models of the environment's action and option transitions. An option is a policy plus a stopping function; an action is a special case whose policy always selects that action and whose stopping function always stops.
* **Reward model** — $r(s,o)=\mathbb{E}_{\pi_o,\beta_o}[\sum_{t=1}^{K}\gamma^{t-1}R_t \mid S_0=s]$ is the expected reward accumulated at each primitive step until option $o$ stops, following its policy $\pi_o$ and stopping rule $\beta_o$. The learned model $\hat r(x(s),o)$ estimates this from the state feature vector $x(s)$.
* **Transition model** — the paper uses an expectation model, $\hat n(x(s),o) \approx \mathbb{E}_o[\gamma^K x(S_{t+K}) \mid S_t=s]$. Here $K$ is the option duration in primitive steps, and $\mathbb{E}_o$ averages over the environment transitions and stopping outcomes under the option's policy and stopping rule. The model predicts the expected feature vector at the stopping state, discounted by the option duration; it does not represent the full distribution over possible stopping states.
* **Learning the model off-policy** — for every available action or learned option, use the observed one-step rewards and next-state features to update its reward and transition models with UWT. The reward update targets the actual reward and bootstraps through the option's stopping rule; the transition update learns each feature of the expected stopping-state representation. The importance-sampling ratio corrects for the behavior policy differing from the policy of the option being modeled.
* **Hallway example** — the agent learns models for the four actions and the hallway-attaining option from 50,000 steps under a uniform random behavior policy. The state features are one-hot over 72 non-terminal cells; the model is linear, with a weight vector for reward and a matrix for expected next features. The model error falls as learning proceeds, and planning with the models reaches near-optimal performance after about 20,000 steps. More training reduces model error further but gives little additional planning improvement in this environment.
* **Why use an expectation model** — it does not represent the full distribution of possible outcomes, but it is sufficient for planning when the value function is linear in the state features. More general distribution models can represent richer outcomes, but can become unwieldy as the feature space grows. The authors also point out that no model can be complete and accurate because the world is more complex than the agent.
# Planning with Options 
The final step in STOMP uses the action and option models to improve the value function through approximate value iteration.
* **Tabular planning** — with exact models, update a state's value to the best backed-up value across its available options:
  * $V(s) \leftarrow \max_{o\in O(s)}[r(s,o)+\sum_{s'}p(s'|s,o)V(s')]$
  Here $r$ is the option's expected cumulative reward and $p$ is its discounted stopping-state model. When the available options are only primitive actions, this is ordinary value iteration.
* **Approximate planning** — represent values as $\hat v(x,w)$ and update the weights toward the best predicted value:
  * $w \leftarrow w+\alpha[\max_{o\in O(x)}(\hat r(x,o)+\hat v(\hat n(x,o),w))-\hat v(x,w)]\nabla_w\hat v(x,w)$
  The quantity $\hat r(x,o)+\hat v(\hat n(x,o),w)$ is the backed-up value: the reward predicted for taking option $o$ plus the value of the expected discounted features at its stopping state. Computing this for one state-option pair is one planning look-ahead, meaning one option evaluated from one state using its model. With a linear value function, an exact expectation model gives the same backup as summing over all next states; a learned model can still be imperfect. The model call is cheaper than summing over the full state space.
* **What makes an option useful** — an option helps planning if, in at least one state, its model gives a higher backed-up value than the other available options, so the planner selects it. Reward-respecting options keep the original task reward in the subtask; optimistic stopping bonuses favor ending in states where a selected feature and the estimated state value are high.
* **Gridworld result** — the authors applied approximate value iteration using the models learned after 50,000 random steps. Planning with the reward-respecting hallway option reached the optimal policy with fewer look-aheads than using primitive actions alone; the shortest-path option was less useful because it often followed a route with negative reward.
* **Planning assumption** — the method assumes option models are reasonably accurate and stable while the approximate value function still needs improvement. The authors note that this can fit problems where dynamics are known or learnable but accurate values remain difficult to obtain.
# Questions 

* What is a "non-markovian" feature vector, what is the imporance of the fact that it''s "non-markovian" that it's stateful?
* In section 1, why is planning with the shortest path option five insted of four lookaheads
* Remind me what an eligibility trace is and how it's mathemtically implemented 
* Work through the model learning and and option equations, specifically the form of the Sutton thing 
* "The intrinsic rewards are constructed from the successor representation as an approximation to the graph Laplacian of the interconnection topology of the environment"??
* Explain the math of the van Hasselt work which shows the maximization bias in stochastic environments 
* What is the difference in something being planned vs learned? 
## Anki
* The STOMP loop
* GVF components
	* Subtasks and the reward respecting variant 
* Equations for modeling and planning steps
