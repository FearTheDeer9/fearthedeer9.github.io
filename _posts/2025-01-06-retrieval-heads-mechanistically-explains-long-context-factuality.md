---
layout: single
title: "Retrieval Head Mechanistically Explains Long-Context Factuality"
date: 2025-01-06
categories:
  - AI
  - paper-summary
tags:
  - mechanistic interpretability
  - model compression
  - summary
---

In this paper the researcher are able to show that there exists a special kind of attention heads which are responsible for retrieval of information from long context. They outline a few intriguing properties that these heads posess:

 <!-- excerpt-end -->

1. Universal - all models explored contain a set of retrieval heads
2. Sparse - on average less than 5% of attention heads are retrieval
3. Intrinsic - Retrieval heads already exist in models which are pretrained with short context, when extending to long context the same set of heads generalizes to information retrieval.
4. Dynamically activated - Specific heads attend to the required information no matter how the context changed, others are activated for different contexts.
5. Causal - Pruning the retriebal heads leads to failure in retrieving relevant information and results in hallucination. It severely damages performance on CoT tasks while barely impacting tasks where the models directly generates the answer using its intrinsic knowledge.

## Retrieval head detection

In order to detect the retrieval heads, the authors introduce a _retrieval score_ measuring the frequency of a head's copy-pasts behavior during autoregressive decoding.
The retrieval score is defined as such - denote the current token being generated as _w_, the context as x, the query as q, the corresponding answer as k and the attention scores of a head as $a \in R^{|x|}$. Then attention head h copies ans pastes a token from the needle to the output sentence if the two following criteria are satisfied:

1. $w \in k$ i.e _w_ is a token within the needle sentence.
2. $x_j=w, j = argmax(a), j \in i_q$ i.e the input token that receives the most attention probability by this head is a token within the needle and is the same token as the currently generated token.
   Let $g_h$ be the set containing all tokens copy and pasted by a given head $h$, then define:$$Retrieval\_score\_h = \frac{|g_h \cap k|}{|k|}$$

### Detection algorithm

The authors calculate the retrieval score for all attention heads under a diverse set of input contexts. They use the following methodology - they create a tuple of a query, needle and context and make sure that the query and needle are semantically irrelevant from the context, and that the needle cannot be answered with the model's current knowledge. Then, for each tuple they insert the query uniformly between the context start or end in various context length, noting the average retrieval score. The authors classify a head as a retrieval head if it passes the threshold of 0.1, i.e it performs copy-paste 10% of the time.

## Implications

This paper's findings have several important implications for the future of language model architecture and optimization. The discovery that only about 5% of attention heads are responsible for retrieval suggests a promising direction for KV cache compression - by potentially pruning cache entries for non-retrieval heads while maintaining those critical for information retrieval, we might dramatically reduce memory requirements without sacrificing the model's ability to access information from its context. However, the paper also demonstrates why attempts to optimize attention through mechanisms like sliding windows or local attention patterns have struggled with tasks requiring precise information retrieval - retrieval heads need to be able to form attention patterns between any query and any key in the sequence. This necessity for full attention, at least in retrieval heads, suggests a fundamental constraint on certain types of efficiency optimizations. Furthermore, the clear functional differentiation between retrieval and non-retrieval heads, along with their varying impacts on different types of tasks, strengthens the emerging understanding that attention heads specialize in distinct functionalities, with some being critical for specific capabilities while others might be more dispensable depending on the task at hand.

# My opinion

This is a very cool paper!
It introduces a new type of attention-head, identifies fascinating features that these heads possess and, at least to my interpretation, does a good job of backing up its claims.
I think that the fact that these heads are so pivotal and have such direct effect on the output of the model further strengthens my hunch that different heads account for different functionalities, and that perhaps some heads account for several less frequent/important functionalities - which allows the possibility of them being ablated while maintaining performance, at least until it comes to more complex, [generative tasks](/2025/01/04/mixture-of-depths.html).
