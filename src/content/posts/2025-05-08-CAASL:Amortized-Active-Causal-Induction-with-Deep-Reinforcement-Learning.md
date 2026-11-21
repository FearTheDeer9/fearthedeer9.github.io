---
date: '2025-05-08'
title: "Paper Review - CAASL: Amortized Active Causal Induction with Deep Reinforcement Learning "
layout: single
tags:
- reinforcemnt learning
- causality
- graph learning
---

CAASL: Amortized Active Causal Induction with Deep Reinforcement Learning
CAASL (Causal Amortized Active Structure Learning) presents a breakthrough approach for selecting informative interventions in causal discovery tasks without requiring access to the data likelihood. This method bridges the gap between traditional causal structure learning and intervention design by using reinforcement learning to train an amortized policy, offering both rapid inference and sample-efficient causal discovery.
Key Innovation
The core innovations of CAASL address fundamental challenges in causal intervention design:

Amortized Intervention Policy: A single transformer-based network that directly proposes interventions without intermediate causal graph inference
Likelihood-Free Approach: Elimination of the need to compute or approximate data likelihoods during intervention design
Reinforcement Learning Framework: Training with SAC to maximize cumulative information gain about causal structure
Symmetry-Preserving Architecture: Neural architecture that encodes key design space symmetries for generalization

This combination enables both speed and sample efficiency that exceeds previous approaches, which typically required either explicit likelihood computations or intermediate causal graph inference steps.
Implementation
Transformer-Based Policy
The policy leverages a transformer architecture with several key features:

Alternating Self-Attention: Applied over both the variable axis and samples axis to ensure permutation equivariance and invariance
History Encoding: Represents past interventional data and outcomes in a format that preserves causal information
Permutation Invariance: The transformer's max pooling over samples ensures the ordering of data doesn't affect intervention selection
Gaussian-Tanh Distribution: Policy outputs are parameterized as a Gaussian-Tanh distribution for both intervention targets and values

Reward Function
The reward function is derived from improvement in causal graph estimation:

AVICI Integration: Uses a pre-trained amortized causal discovery model (AVICI) that directly estimates graph posteriors
Adjacency Matrix Accuracy: Measures improvement in correctly identified edges rather than log-likelihood
Telescoping Reward Structure: Rewards represent incremental improvement at each step
Mathematical Formulation:
R(ht, It, ht−1, {A, θ}) = E[∑i,j I[Â(i,j) = A(i,j)]] - R(ht−1, It−1, ht−2, {A, θ})


Training Process
The policy is trained through reinforcement learning in simulated environments:

Simulator-Based Training: Environments sample causal graphs from a prior where ground truth is known
REDQ/SAC Algorithm: Uses an off-policy RL algorithm that improves sample efficiency
Hidden-Parameter MDP: Formulates the problem as a HiP-MDP where hidden parameters are the unknown causal graph
Q-Function Networks: Multiple Q-function approximators with transformer-based history encoding

Results
CAASL demonstrated exceptional performance in comparisons:

Synthetic Environment Performance:

Superior returns and structural hamming distance compared to random interventions and observational data
Competitive with or outperforming likelihood-based methods like DiffCBED and SS Finite
Converges to optimal intervention strategies in ~20 iterations


Single-Cell Gene Expression Performance:

Successfully applies to complex SERGIO simulator with differential equation mechanics
Handles significant technical noise and data missingness (~74% dropout rate)
Outperforms random intervention strategies despite biological complexity


Generalization Capabilities:

Robust to distribution shifts in graph priors (Erdős–Rényi to Scale-Free)
Handles changes in noise distributions and mechanism parameters
Zero-shot generalization to higher dimensional problems (up to 3x training dimensionality)
Adapts to different intervention types not seen during training



Theoretical Foundations
The method's success is grounded in several key theoretical insights:

Connection to Bayesian Experimental Design: CAASL's reward function is related to a lower bound on Expected Information Gain (EIG)
Barber-Agakov Bound: The theoretical foundation lies in the bound:
EIG(A; πϕ) ≥ E[log q(A | ht)] + const.

From Log-Likelihood to Accuracy: Replacing log-likelihood with adjacency matrix accuracy provides a more direct and effective reward signal
Bernoulli Representation of Causal Graphs: The posterior treats each potential edge as an independent Bernoulli random variable
Feature Integration: The mathematical conditioning of the transformer through alternating attention ensures both variable and sample symmetries are preserved

Key Lessons

Amortization Matters: Training a single policy network that generalizes across causal graphs dramatically reduces computational cost compared to per-instance optimization
Likelihood-Free Approach: Eliminating the need for likelihood computation enables application to complex domains where likelihoods are intractable
Symmetry Encoding: Designing the network architecture to respect fundamental symmetries in the causal discovery problem enables generalization
Reinforcement Learning for Design: RL provides an effective framework for sequential decision-making under uncertainty, particularly valuable for intervention design
Simulator-Based Training: Using simulated environments where ground truth is known enables effective policy learning that transfers to real-world scenarios
Adjacency Accuracy as Reward: Directly rewarding correct graph structure identification provides a more effective signal than complex information-theoretic quantities

Points to Remember

Likelihood-Free vs. Likelihood-Based: Traditional intervention design requires computing p(data|causal model), which is intractable in many real-world settings. CAASL eliminates this requirement, making it applicable to complex biological systems.
Bernoulli Graph Representation: Representing the causal graph posterior as independent Bernoulli random variables for each edge provides a probabilistic framework that quantifies uncertainty rather than just making point estimates.
Simulation for Training: While we can't simulate environments we don't know the structure of, we can train on many simulated environments where we do know the structure, and the learned strategies transfer to new, unknown environments.
Adjacency Matrix Accuracy: Using the number of correctly identified edges as a reward is a practical simplification that still maximizes information gain about the graph structure while being more numerically stable and interpretable.
Two Model Roles: The system uses separate models for graph inference (AVICI) and intervention selection (CAASL), with the policy learning to propose interventions that help the inference model discover the true graph structure.