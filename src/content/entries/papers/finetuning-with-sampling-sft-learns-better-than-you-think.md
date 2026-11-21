---
title: "Finetuning with Sampling - SFT Learns Better Than You Think"
date: "2026-10-05"
type: "paper-summary"
subtype: null
topic: "Pretraining, data & distillation"
source_title: "Finetuning with Sampling - SFT Learns Better Than You Think"
source_author: null
source_url: "https://arxiv.org/pdf/2610.02140"
---

## Provenance
- **Link** — https://arxiv.org/pdf/2610.02140
- **Authors / affiliation** — 
- **Venue, date** — 
- **How it reached me** — 
- **Why I clipped it** — 

## Claim
By utilizing a MCMC sampling algorithm one can progressively transform an off-policy expert trace to be more on-policy given a reference model for fine tuning -> this minimizes KLD directly rather than through a regularizer -> improved capabilities while avoiding the catastrophic forgetting that hinders SFT usually. 
## Techniques and evidence
Here is the pseudocode from the original paper:
**Algorithm 1: Projection Sampling with Autoregressive Models**
- **Input:** base $p$; proposal $\kappa_{\mathcal{C}}$; expert trajectory $\tau$; length $T$
- **Hyperparams:** block size $B$; MCMC steps $N_{\text{MCMC}}$
- **Output:** $(x_0, \dots, x_T) \sim p_{\mathcal{C}}$
---
- `1` **Notation:** Define the unnormalized intermediate target
	$$
	\pi_k(x_{0:kB}) \propto p_{\mathcal{C}}(x_{0:kB}).
	$$
- `2` **for** $k \leftarrow 0$ **to** $\lceil T/B \rceil - 1$ **do**
	- `3` Given prefix $x_{0:kB}$, we wish to sample from $\pi_{k+1}$. Construct initialization $\mathbf{x}^0$ by extending autoregressively with $\kappa_{\mathcal{C}}$:
		$$
		x_t^{(0)} \sim \kappa_{\mathcal{C}}(x_t \mid x_{<t}, \tau), \qquad \text{for } kB+1 \le t \le (k+1)B.
		$$
		Set the current state $\mathbf{x} \leftarrow \mathbf{x}^0$.
	- `4` **for** $n \leftarrow 1$ **to** $N_{\text{MCMC}}$ **do**
		- `5` Sample an index $m \in \{1, \dots, (k+1)B\}$ uniformly.
		- `6` Construct proposal sequence $\mathbf{x}'$ with prefix $x_{0:m-1}$ and resampled completion:
			$$
			x'_t \sim \kappa_{\mathcal{C}}(x_t \mid x_{<t}, \tau), \qquad \text{for } m \le t \le (k+1)B.
			$$
		- `7` Compute acceptance ratio (5):
			$$
			A(\mathbf{x}', \mathbf{x}) \leftarrow \min\left\{1,\ \frac{p(\mathbf{x}')}{p(\mathbf{x})} \cdot \frac{\kappa_{\mathcal{C}}(\mathbf{x} \mid \mathbf{x}', \tau)}{\kappa_{\mathcal{C}}(\mathbf{x}' \mid \mathbf{x}, \tau)}\right\}.
			$$
			Draw $u \sim \mathrm{Uniform}(0, 1)$;
		- `8` **if** $u \le A(\mathbf{x}', \mathbf{x})$ **then accept** and set $\mathbf{x} \leftarrow \mathbf{x}'$
	- `9` **end**
	- `10` Set $x_{0:(k+1)B} \leftarrow \mathbf{x}$ to fix the new prefix sequence for the next stage.
- `11` **end**
- `12` **return** $x_{0:T}$

Let's understand the components one by one:
1. we assume that trajectories can be grouped into "equivalence classes" based on task-specific notion of "correctness". We note this group with $\mathbb{C}$. 
2. We note with $P_\mathbb{C}$ the set of all distributions that are only output trajectories which are equivalent to the expert traces 
3. They prove that $p_c := arg \min _{\pi \in P_C} KL (\pi|p)=. \frac{p(x) \boldsymbol{1}(x \in \mathbb{C})}{\Sigma_{x' \in \mathbb{X'}}p(x')\cdot \boldsymbol{1}(x' \in \mathbb{C})}$ i.e the restriction of the base model onto equivalent trajectories. 
4. They rewrite the Metropolis-Hastings algorithm 
$$
A(\mathbf{x}, \mathbf{x}^i) = \min\left\{1,\ \frac{p(\mathbf{x}) \cdot \kappa(\mathbf{x}^i \mid \mathbf{x}) \cdot \mathbf{1}(\mathbf{x} \in \mathcal{C})}{p(\mathbf{x}^i) \cdot \kappa(\mathbf{x} \mid \mathbf{x}^i) \cdot \mathbf{1}(\mathbf{x}^i \in \mathcal{C})}\right\} = \min\left\{1,\ \frac{p(\mathbf{x}) \cdot \kappa_{\mathcal{C}}(\mathbf{x}^i \mid \mathbf{x})}{p(\mathbf{x}^i) \cdot \kappa_{\mathcal{C}}(\mathbf{x} \mid \mathbf{x}^i)}\right\}. \tag{6}
$$
5. This gives us a convenient recipe for boosting an off-policy dataset: initialize at the expert trajectory, mutate the candidates with some information-preserving proposal distribution $\kappa_\mathbb{C}$ and use (6) to accept or reject candidates. In fact, not only does this process eventually converge to $\pi_\mathbb{C}$ , but it also guarantees a progressive decrease in the policy gap between the data generation policy and the base model.
6. A direct implementation of Metropolis-Hastings for LLMs is infeasible as it requires regenerating full-length token sequences with repeated LLM inference calls. This infeasibility is further compounded by slow mixing, where convergence can require exponentially many MCMC steps due to high dimensional sample spaces or poor choice of proposals and initializations. To help avoid these issues, they adapt the implementation of Metropolis-Hastings for power sampling in Karan and Du [2025], which we refer the reader to for further details. In short, they define restriction distributions of the target projection pC over sequences of increasing length in units of block size B. They use Metropolis-Hastings to sample within each distribution, providing a strong initialization to begin sampling from the next one. T
## Anki
* The pseudocode as well as the computational optimizations needed on top in order to functionalize this for LLMs
