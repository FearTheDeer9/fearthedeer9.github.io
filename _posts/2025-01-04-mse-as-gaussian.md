---
layout: single
title: "The Mathematics and Philosophy Behind MSE"
date: 2024-01-04
categories:
  - math
  - short-form
  - writing
tags:
  - knowledge
  - learning
  - meditation
---

When we perform linear regression, we're making two fundamental modeling choices that are worth examining separately:

1. Assuming our data follows a linear relationship
2. Using mean squared error (MSE) as our loss function
<!-- excerpt-end -->

Let's understand why we make these choices and what they imply about our view of the world.

## The Linearity Assumption

First, we assume there exists some linear relationship between our variables:

$y = Xw$

This is a strong assumption about the structure of our data. We're saying that our target variable $y$ can be expressed as a linear combination of our features $X$, weighted by some parameters $w$. This assumption isn't justified by the probabilistic framework - it's a modeling choice we make upfront.

## Justifying Mean Squared Error

Given that we've assumed linearity, why do we use mean squared error to find the best parameters? This is where probability theory provides insight.

Let's assume our observations deviate from the true linear relationship due to Gaussian noise:

$y = Xw + \epsilon$, where $\epsilon \sim \mathcal{N}(0, \sigma^2)$

This means for any observation:

$P(y_i \mid X,w) = \mathcal{N}(y_i \mid Xw, \sigma^2)$

Taking the negative log likelihood:

$-\log L(w) \propto \sum(y_i - Xw)^2$

This is exactly the MSE loss function! So while the Gaussian likelihood doesn't justify our linear model, it does justify using MSE as our loss function once we've decided to use a linear model.

## The Role of the Prior

If we add a uniform prior over weights $P(w) \propto \text{constant}$, our posterior becomes:

$P(w \mid X,y) \propto P(y\mid X,w)P(w) \propto P(y\mid X,w)$

The uniform prior drops out because it's constant, meaning that maximizing the posterior (MAP) is equivalent to maximizing the likelihood. This explains why standard linear regression can be interpreted as both maximum likelihood and MAP estimation.

## Philosophical Implications

This separation of assumptions helps us think more clearly about what we're really doing in linear regression:

1. We first make a strong assumption about the structure of reality (linearity)
2. We then make an assumption about how measurements deviate from this structure (Gaussian noise)
3. Finally, we express uncertainty about the parameters of our linear model (uniform prior)

The Gaussian noise assumption is often justified by the Central Limit Theorem - when many small, independent factors affect our measurements, their sum tends toward a Gaussian distribution. But the linearity assumption is harder to justify theoretically - it's usually chosen for simplicity and often works well enough in practice.

## Conclusion

Understanding these distinctions helps us be more thoughtful practitioners. When linear regression fails, we should ask:

- Is our linearity assumption wrong?
- Is our Gaussian noise assumption wrong?
- Is our uniform prior inappropriate?

Each of these represents a different type of model failure requiring different solutions. The mathematics of linear regression isn't just about finding lines through points - it's about making explicit our assumptions about the structure of reality and the nature of measurement uncertainty.
