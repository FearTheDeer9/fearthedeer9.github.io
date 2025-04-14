---
date: '2025-04-14'
layout: single
tags:
- natural language processing
- representation learning
- concept representation
- hierarchical concepts
- geometric representations
title: Preliminaries
---

In this work the authors attempt to extend the linear representation hypothesis to cases where the concepts no longer have natural binary contrasts and hence do not admit a straightforward notion of *direction* (i.e {male, female} vs. {is_animal}). They extend the theory for both binary concepts, categorical concepts and hierarchical representations. This is done by replacing the notion of a representation as a *direction* to representation as a *vector*. 
The authors summarize there main contributions as:
1. Showing that categorical concepts are represented as polytopes where each vertex is the vector representation of one of the elements of the concept. 
2. Semantic hierarchy between concepts is encoded geometrically as orthogonality between representations
3. Validating these results on Gemma and LLaMA-3 using the WordNet hierarchy
## Preliminaries 
### Concepts 
The authors formalize a concept as a latent variable W that is caused by the context X and causes the output Y. 
Further, they introduce the notion of *causally separable* concepts. Concepts W and Z are causally separable if the potential outcome $Y(W=w, Z=z)$ is well defined for all w, z. That is, two variables are causally separable if they can be freely manipulated. 
### The Causal inner product 
Following the results of Park et al. the authors unify the two distinct representation spaces dealt with, the context embedding and the token unembeddings, by performing an invertible affine transformation on them.  The  Euclidean inner product in the transformed spaces is the causal inner product, an inner product in which causally separable concepts are orthogonal, and the Riesz isomorphism between the embedding and unembedding spaces is the vector transpose operation. 

# Representation of concepts 
### Categorical concepts 
#### *Definition 1* 

<!-- excerpt-end -->

A vector $\bar{\ell}_W$ is a linear representation of a binary concept $W$ if
$\mathbb{P}(W = 1 \mid \ell + \alpha\bar{\ell}_W) > \mathbb{P}(W = 1 \mid \ell)$, and (3.1)
$\mathbb{P}(Z \mid \ell + \alpha\bar{\ell}_W) = \mathbb{P}(Z \mid \ell)$, (3.2)
for all contexts $\ell$, all $\alpha > 0$, and all concept variables $Z$ that are either subordinate to or causally separable with $W$. Here, if $W$ is a binary feature for an attribute $w$, then $W = 1$ denotes $W = \text{is\_w}$
#### *Definition 2* 
A value $z$ is *subordinate* to a value $w$ (denoted by $z \prec w$) if $\mathcal{U}(z) \subseteq \mathcal{U}(w)$. We say a categorical concept $Z \in_R \{z_0,\ldots,z_{n-1}\}$ is subordinate to a categorical concept $W \in_R \{w_0,\ldots,w_{m-1}\}$ if there exists a value $w_Z$ of $W$ such that each value $z_i$ of $Z$ is subordinate to $w_Z$.

Once they had these definitions in hand the authors set out to enhance the definition of a representation and transform it from a *direction* to a *vector* by imbuing it with the notion of magnitude, to do so they rely on the 
#### Theorem of Magnitudes of linear representation 
*Suppose there exists a linear representation (normalized direction) $\bar{\ell}_W$ of a binary feature $W$ for an attribute $w$. Then, there is a constant $b_w > 0$ and a choice of unembedding space origin $\bar{\gamma}_0^w$ in  such that
$\bar{\ell}_W^{\top}g(y) = b_w \quad \text{if } y \in \mathcal{U}(w)$ 
$\bar{\ell}_W^{\top}g(y) = 0 \quad \text{if } y \notin \mathcal{U}(w)$ (4.1)

*Further, if there are $d$ causally separable attributes $\{w_0,\ldots,w_{d-1}\}$ with linear representations, we can choose a canonical origin $\bar{\gamma}_0$ in as

$\bar{\gamma}_0 = \sum_{i=0}^{d-1} \bar{\gamma}_0^{w_i}.$ (4.2)

The theorem says that, if a (perfect) linear representation of the animal feature exists, then every token having the animal attribute has the same dot product with the representation vector; i.e., “cat” is exactly as much animal as “dog” is. If this weren’t true, then increasing the probability that the output is about an animal would also increase the relative probability that the output is about a dog rather than a cat.

Using this, the authors introduce the notion of a vector representation of an attribute:
#### Vector representation 
We say that a binary feature $W$ for an attribute $w$ has a *vector representation* $\bar{\ell}_w \in \mathbb{R}^d$ if $\bar{\ell}_w$ satisfies Definition 3 and $\|\bar{\ell}_w\|_2 = b_w$ in Theorem 4. If the vector representation of a binary feature is not unique, we say $\bar{\ell}_w$ is the vector representation that maximizes $b_w$.

* Now, using this they define the *Polytope representation* of a categorical concept $W={w_0,...w_{k-1}}$ as the convex hull of the vector representations of the elements of the concept
### Hierarchical Concepts 
Next, the authors turn to the question of how hierarchical relationships between concepts are encoded in the representation space. Their core intuition is that manipulating the “animal” concept should not affect relative probabilities of the “mammal” and “bird” concepts, so we might expect the representations of animal and mammal ⇒ bird to be orthogonal. They formalize this by connecting the vector and semantic concepts in the theorem of Hierarchical Orthogonality 
####  (Hierarchical Orthogonality)
*Suppose there exist the vector representations for all the following binary features. Then, we have that*
*(a)* $\bar{\ell}_w \perp \bar{\ell}_z - \bar{\ell}_w$ *for* $z \prec w$;

*(b)* $\bar{\ell}_w \perp \bar{\ell}_{z_1} - \bar{\ell}_{z_0}$ *for* $Z \in_R \{z_0,z_1\}$ *subordinate to* $W \in_R \{\text{not\_w},\text{is\_w}\}$;

*(c)* $\bar{\ell}_{w_1} - \bar{\ell}_{w_0} \perp \bar{\ell}_{z_1} - \bar{\ell}_{z_0}$ *for* $Z \in_R \{z_0,z_1\}$ *subordinate to* $W \in_R \{w_0,w_1\}$; *and*

*(d)* $\bar{\ell}_{w_1} - \bar{\ell}_{w_0} \perp \bar{\ell}_{w_2} - \bar{\ell}_{w_1}$ *for* $w_2 \prec w_1 \prec w_0$.

## Results and Validation
The authors validate their theoretical framework through several empirical analyses on the Gemma and LLaMA-3 models using WordNet's hierarchical structure:

### Existence of Vector Representations
Using Linear Discriminant Analysis (LDA), they estimate vector representations ℓ̄w for each WordNet concept by finding projections that:
1. Maximize separation between words with and without the attribute
2. Minimize variance of projections within each class

They validate these representations by showing that:
- Test words with attribute w project to values near bw
- Words without the attribute project near 0
- This structure disappears when embeddings are randomly shuffled

### Validation of Hierarchical Structure
They demonstrate the hierarchical orthogonality through multiple analyses:
1. Cosine similarity between vector representations reflects WordNet's hierarchical structure
2. Child-parent vectors are orthogonal to parent vectors
3. Sibling concepts form simplices in subspaces orthogonal to their parent
4. The orthogonality persists even when controlling for high-dimensional effects

## My Explanation - Intuitive Understanding of the Concepts

### The Core Problem
Language models seem to organize concepts linearly, but how do they represent concepts like "is_animal" that don't have natural opposites? Previous work focused on contrasts like male↔female, but many concepts don't work this way.

### Key Mathematical Ideas and Their Motivation
## **From Directions to Vectors** 
Previous work just had the notion of directions. Now by adding the concept of magnitude we transformed the representation into vectors which allows us to say "how much" of a concept is present, compose concepts while measuring their presence independently
## The Whitening Transformation and Dual Spaces
The mathematical foundation of this work relies on understanding and aligning two fundamental spaces:
1. The embedding space Λ ≃ ℝᵈ (contexts)
2. The unembedding space Γ ≃ ℝᵈ (tokens)

These spaces are naturally dual to each other through their interaction in the softmax layer:
P(y|x) = exp(λ(x)ᵀγ(y)) / ∑y'∈Vocab exp(λ(x)ᵀγ(y'))

**The Riesz Isomorphism and Duality**
The Riesz isomorphism provides a canonical way to identify a Hilbert space with its dual. In our context:
- Each embedding acts as a functional on unembeddings (and vice versa) via the inner product
- We want this duality to reflect semantic relationships
- The goal is to make this relationship as simple as possible: semantic similarity should correspond directly to geometric similarity

**The Whitening Transformation**
We achieve this through a specific transformation:
1. On unembeddings: g(y) ← A(γ(y) - γ̄₀)
2. On embeddings: ℓ(x) ← A⁻ᵀλ(x)

Where A = Cov(γ)⁻¹/² is computed via eigendecomposition of the unembedding covariance matrix.

This transformation has several crucial properties:
1. **Preservation of Model Behavior**:
   - The softmax probabilities remain unchanged because the transformation appears in both numerator and denominator
   - This means we're not changing the model, just our way of analyzing it

2. **Statistical Properties**:
   - Makes the covariance matrix identity
   - Decorrelates the dimensions
   - Normalizes the variance in each direction

3. **Geometric-Semantic Alignment**:
   - Causally separable concepts become orthogonal
   - Magnitudes become meaningful measures of concept strength
   - Inner products directly reflect semantic relationships

4. **Dual Space Alignment**:
   - The Riesz isomorphism becomes simply the vector transpose operation
   - Embeddings and unembeddings can be treated as living in the same space
   - Geometric operations (like addition and inner products) have consistent semantic interpretations

**Mathematical Foundation**
The transformation works because:
1. The covariance structure of the unembedding vectors contains information about semantic relationships
2. Whitening removes spurious correlations while preserving essential semantic structure
3. The resulting space has a natural geometric interpretation where:
   - Orthogonality ⟺ Causal separability
   - Magnitude ⟺ Concept strength
   - Inner product ⟺ Semantic similarity

This creates a unified space where:
- Semantic relationships are reflected in geometric ones
- Vector operations correspond to meaningful semantic operations
- The hierarchical structure of concepts manifests as orthogonal subspaces
- Both discrete categorization and continuous interpolation are naturally represented

This transformation reveals the inherent structure of how language models organize semantic knowledge, making it accessible to geometric analysis while preserving the model's fundamental behavior.

## **Polytope Representations**
The authors were able to show that categorical concepts form geometrical shapes in latent space, this gives a natural way to represent multiple related options with each vertex acting as a "pure" instance of a specific sub-category. This captures both discrete categories and continuous interpolation.
### **Hierarchical Orthogonality**
Finally, the paper demonstrates that different levels of hierarchy live in orthogonal subspaces. This means that we can modify one level without affecting others, i.e one can make something "more animal" without changing whether it's a mammal or bird. Resulting in the direct sum decomposition of semantic space

### Theoretical Contribution
The paper shows that language models organize semantic knowledge in a remarkably clean geometric structure:
1. Features have consistent magnitudes
2. Categories form regular geometric shapes
3. Hierarchical relationships manifest as orthogonality
4. The whole structure supports both discrete categorization and continuous interpolation

This provides a mathematical framework for understanding how language models represent and manipulate meaning at different levels of abstraction.