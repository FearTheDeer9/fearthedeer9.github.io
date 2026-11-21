---
title: "Summary of Report"
date: "2025-02-14"
type: "excerpt"
subtype: "course-summary"
topic: "Causal & probabilistic ML"
source_title: null
source_author: null
source_url: null
---

A fundamental limitation of CBO approaches is that they all assume knowledge of the underlying casual graph.
The typical problem formulation in BO is $\min_{x \in \mathbb{A}} f(x)$ where A is the feasible set, x is an input sequence and f is a black box function. 
The two important aspects of the BO algorithm are:
1. Bayesian statistical model: This is a surrogate model for the black-box function and is written as $P(f(\cdot))$, which is a probabilistic estimate for the true function $f(\cdot)$
2. Acquisition function: This is the function that we are optimizing at each step of the BO algorithm and is written as $\alpha(\cdot)$
This work assumes an underlying Gaussian Process. This modeling allows to quantify our current  level of uncertainty regarding the graph and balance the exploration exploitation tradeoff. Further, by having a GP we essentially have a distribution over functions, hence by sampling new datapoints we can update our posterior and then also obtain a closed form formula for the current marginal likelihood of the variable of interest.
In this report the acquisition functions that are used are either Expected Improvement - which aims to select the point with the highest improvement of the global optimum or Entropy Search - which aims to reduce uncertainty about the location of the global optimum by exploring  
regions of the search space that provide the most information.
* Why do we only consider these two distinct options, can we create a composite utility function? What about multivariable MI for the second bit (see page 8, the maximization of uncertainty only assumes one other variable)?

Here we model our surrogate model using a GP, other methods of using BNNs our Neural Processes, which are made to deal with the computational intractability of GPs also exist. This work explores how to extend the GP framework to larger graphs by making use of the underlying causal model rather than just a black-box function optimization.
The cold start problem, of having worthwhile questions at the early stages of exploration is a key theme in BO, here we try to make use of the causal model as well as observational data to solve this 
	Could we perhaps encode "concpets"/"desiderata" etc?
* Can we guarantee that we have identified the direct parents? how do we know that there are no confounders? that there are no intermediate variables between the parents and the nodes that we could improve upon? Could we somehow optimize this? 
	* Could we train a "simple" network that has concepts ingrained in an interpretable way and then generalize it so that it is no longer fully interpretable but it is guaranteed and verifiable that the original concepts still exist in the agent's world model?
* Only consider hard interventions
* In the report we explicitly compare CBO, BO and CEO. CBO consistently performs best being robust both to the size of the graph as well as explicitly optimizing for performance rather than jointly optimizing learning the causal graph. 
	* It was also shown that CBO is sensitive to the causal graph given to it, if the edges of the prior are misspecified both convergence rate and final values found will suffer. However, the report does provide further analysis for this 
		* The CBO algorithm with only the parent edges correctly specified performs  
			competitively with the CBO algorithm when all edges are correctly specified.
* Assume no hidden confounders, what are the underyling assumptions or rather what ar ethe situations in which such an assumption makes sense vs when is it overly optimistic?
	* What do we do if we have a known confounder but it can't be interfered with?
* Shows a closed form for the update in the case of linear dpendence on the parents and an approximation is non-linear
	* How good is this approximation, could other better approximations be made?
* To deal with larger graphs we make use of given observational data. then, to integrate this observational data we rely on prior work that he;ps with estimating the likelihood of a variable being a direct variable of a different variable given some measurements. This process is done by computing
# TODOs
* What are the assumptions needed to justify a GP formalization?
* Read about the doubly robust method of estimating direct parent
* What does it mean for the acquisition function to be robust? Could it be improved then?
* How can we extend this method to deal with larger graphs/non linear and increased noise
* How can we get environments with a realistic way of simulating interventions?
