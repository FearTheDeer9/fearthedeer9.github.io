---
title: "Speculative Probing - LLM Monitoring at Speculative-Decoding Cost"
date: "2026-09-01"
type: "paper-summary"
subtype: null
topic: "Speculative decoding"
source_title: "Speculative Probing - LLM Monitoring at Speculative-Decoding Cost"
source_author: null
source_url: "https://aclanthology.org/2021.emnlp-main.243/"
---

They repurpose the speculative decoding head, be that a MTP or EAGLE head to work as a sequence classifier (for prompt safety for instance). They achieve this by performing [Soft Prompt Tuning ](https://aclanthology.org/2021.emnlp-main.243/) on the draft heads. 
## Pipeline 
Given a sequence of n tokens with precomputed base model hidden states and token IDs, one can constuct the following:
1. Single step probing - replace the hidden state of the next predicted token (What's usually concatenated for draft heads) with a learned soft prompt. These are fed into the draft model and subsequently routed to the classification head.
	* They use a single layer weight vector for classification. i.e $out = \sigma ( w^T \cdot a)$
2. Recursive probing - Same thing, but recursively chain multiple soft prompts.
