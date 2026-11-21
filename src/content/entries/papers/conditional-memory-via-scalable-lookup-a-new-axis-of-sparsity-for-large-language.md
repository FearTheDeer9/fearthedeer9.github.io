---
title: "Conditional Memory via Scalable Lookup - A New Axis of Sparsity for Large Language Models"
date: "2026-09-06"
type: "paper-summary"
subtype: null
topic: "Architectures & scaling"
source_title: "Conditional Memory via Scalable Lookup - A New Axis of Sparsity for Large Language Models"
source_author: null
source_url: null
---

* N-gram, the model introduced by DeepSeek in this paper, is a way to produce a trade-off between neural computation and static memory by imbuing the model with a module that modernizes classic n-gram embeddings for lookup. 
* A visual diagram of the n-gram module and its various components: how it builds and retrieves the known n-grams by utilizing the context as well as the hash table 
	* ![](/attachments/conditional-memory-via-scalable-lookup-a-new-axis-of-sparsity-for-large-language-Pasted-image-20260906161916.png)
*   the fact that, due to the Zipfian distribution of engrams, the multi-level cache hierarchy is particularly appealing 
*  the scaling laws when we keep iso flop and iso param, as well as when we're in the infinite memory regime and the log-linear validation loss performance there 
	* ![](/attachments/conditional-memory-via-scalable-lookup-a-new-axis-of-sparsity-for-large-language-Pasted-image-20260906163007.png)![](/attachments/conditional-memory-via-scalable-lookup-a-new-axis-of-sparsity-for-large-language-Pasted-image-20260906163019.png)
* Equation for CKA 
	* Intuition for what Gram Matrices and HSIC are
# Questions 
* Explain the XOR hash, how it's implemented and how it solves combinatorial explosion of possible N-Grams
*  what is the expansion rate of 4 in the experimental setup in 4.1 why is the learning rate scaled by 5x with no weight decay for the Engram model? Isn't this a confounder 
*  remind me what yarn is and why one needs it for long contexts 
*  explain the passage on why they changed from base model Isoflav to base model accuracy for the comparison in long context and explain the answers that were observed there 
*  what are the smoothing techniques commonly applied to Engram 
* PatchScope, what is it?
# Thoughts 
*  how well does super BPE or Scone fit in with speculative decoding where 100% coverage isn't as important compared to overall tokens per second*
