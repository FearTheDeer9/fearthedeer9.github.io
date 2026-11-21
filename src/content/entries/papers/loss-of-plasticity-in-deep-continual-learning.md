---
title: "Loss of Plasticity in Deep Continual Learning"
date: "2026-10-07"
type: "paper-summary"
subtype: null
topic: "Continual learning & optimization"
source_title: "Loss of Plasticity in Deep Continual Learning"
source_author: "Rich Sutton"
source_url: null
---

- **Authors:** Rich Sutton
- **Link/DOI:** 
- **Date Read:** 2026-10-07
- **Tags:** 

## TL;DR
- Continual Learning in "regular" Deep learning is broken 

## 1. Problem Statement
- Artificial neural networks, deep-learning methods and the backpropagation algorithm form the foundation of modern machine learning and artificial intelligence. These methods are almost always used in two phases, one in which the weights of the network are updated and one in which the weights are held constant while the network is used or evaluated. This contrasts with natural learning and many applications, which require continual learning. It has been unclear whether or not deep learning methods work in continual learning settings. Here we show that they do not—that standard deep-learning methods gradually lose plasticity in continual learning settings until they learn no better than a shallow network.
## 3. Results & Performance
- They show that continual learning is "broken" for all "regular" algorithm in regular DL. After a few rounds the continual learning model severely lags the one who is trained on the same data from scratch. This gap is only alleviated if one explicitly constrains layer activation norms to be small + re-initializes neurons after every couple of rounds. This points at a structural gap in modern continual learning methods - one I want to think further about and gain a mental model for.
