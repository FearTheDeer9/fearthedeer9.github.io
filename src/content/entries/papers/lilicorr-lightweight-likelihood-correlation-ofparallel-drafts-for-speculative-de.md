---
title: "LiLiCorr - Lightweight Likelihood Correlation ofParallel Drafts for Speculative Decoding"
date: "2026-08-24"
type: "paper-summary"
subtype: null
topic: "Speculative decoding"
source_title: "LiLiCorr - Lightweight Likelihood Correlation ofParallel Drafts for Speculative Decoding"
source_author: null
source_url: null
---

The authors extend DFLash with a - you guessed it - causality-injection head that is meant to "fix" the inherent limitation of diffusion models, the fact that they sample from the per-position marginals rather than the joint. This is done by using a lightweight likelihood based model that correlates the per-position marginal distribution that the drafter produces. It keeps the topK tokens from each position and for each is produces an in and an out vector. These vectors are used to link candidates at consecutive position where high cosine similarity "up weights" and vice versa, capturing the block's joint structure without materializing the exponential weight of the joint distribution. the network produces the 2K vectors in one pass and then computes the scores in parallel as batched matrix multiplication, leaving a cheap, greedy sequential walk as the final step. 
The authors maintain three separate contributions:
1. A novel likelihood based formulation of the marginals problem - cast the recovery of a coherent sequence from per slot marginals as a learned scoring model over candidate paths. 
2. Correlation implemented as an efficient parallel matrix operations - unlike pror works which run the correlation network once per slot. 
3. SOTA results
