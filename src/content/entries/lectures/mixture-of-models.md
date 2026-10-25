---
title: "Mixture of Models"
date: "2025-02-05"
type: "lecture"
subtype: null
topic: "Causal & probabilistic ML"
source_title: null
source_author: null
source_url: null
---

LATENT VARIABLE MODELS (LVMs) & MIXTURE MODELS

- LVMs introduce hidden variables that aren't directly observed
- Two key advantages:
    1. Fewer parameters than direct correlation modeling
    2. Hidden variables create useful compressed representations
- Mixture models are the simplest form of LVMs
- Most common: Gaussian Mixture Models (GMMs)

CLUSTERING APPROACHES

- Soft Clustering (like GMMs)
    - Assigns probabilities of belonging to each cluster
    - Uses responsibilities (rik) to weight contributions
- Hard Clustering (like K-means)
    - Makes definitive 0/1 assignments to clusters
    - Simpler but less nuanced

THE EM ALGORITHM

- Alternates between two steps:
    - E-step: Estimates hidden variables given current parameters
    - M-step: Updates parameters to maximize likelihood
- Guaranteed to never decrease log-likelihood
- Can be used for both MLE and MAP estimation

MODEL SELECTION & COMPLEXITY

- Probabilistic models (GMMs) naturally penalize complexity through likelihood
- Non-probabilistic methods (K-means) lack this property
    - Reconstruction error always decreases with more clusters
    - Cross-validation doesn't work well
- This is a key advantage of probabilistic approaches

ESTIMATION APPROACHES

- Maximum Likelihood Estimation (MLE)
    - Only considers observed data
    - Can overfit
- Maximum A Posteriori (MAP)
    - Incorporates prior beliefs
    - Better regularization properties
    - Generally preferred when possible

PRACTICAL CONSIDERATIONS

- Initialization matters for both EM and K-means
- Multiple random restarts often needed
- Missing data can be handled naturally in the EM framework
- Online variants exist for large-scale applications

The big picture takeaway is that probabilistic approaches like GMMs, while more complex, offer significant advantages over simpler methods like K-means, particularly in terms of model selection and interpretation of results.
