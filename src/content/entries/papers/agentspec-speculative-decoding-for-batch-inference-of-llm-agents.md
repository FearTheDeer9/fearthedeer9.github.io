---
title: "AgentSpec Speculative Decoding for Batch Inference of LLM Agents"
date: "2026-08-27"
type: "paper-summary"
subtype: null
topic: "Speculative decoding"
source_title: "AgentSpec Speculative Decoding for Batch Inference of LLM Agents"
source_author: null
source_url: null
---

The authors begin by noting the inherent friction that exists in modern speculative decoding - the algorithms are designed to provide speed ups at low batch sizes but modern AI service platforms all operate under high-load. The authors attempt to adapt speculative decoding to the modern agentic AI era, and they do this through two main mechanisms:
1. Lowering rejection rates - AgentSpec constraing speculation to semantically coherent segments of the agent workflow. By avoiding speculation across heterogeneous execution stages, agentspec reduces the generation and wasted verification of irrelevant tokens.
2. Dynamic Token Budges - introduces a redundancy metric to guide the allocation of available token budgets across requests. 

# Relevancy 
* "A key distinction between LLM agent applications and standard LLM workloads is that a single user query often spans multiple requests associated with different semantic blocks (e.g., reasoning, tool execution, and result interpretation). While generation patterns are highly consistent within the same semantic block, transitions across blocks or queries introduce substantial semantic shifts"
	* MOD??

# Questions 
* "To address this, AgentSpec adopts a design similar to XGrammar (Dong et al., 2025) by maintaining a cached token-to-string mapping M, enabling efficient online conversion of generated tokens without invoking the tokenizer. Based on this mapping, AgentSpec maintains a lightweight string-based pushdown automaton (PDA) P on the server, which is incrementally updated to track the current semantic block. For each request ri , newly generated tokens are converted using M and fed into the corresponding PDA P(ai , qi) to identify the active semantic block b k i . AgentSpec maintains a separate structure-isolated cache for each semantic block and retrieves speculative drafts exclusively from the matched block, avoiding semantically irrelevant contexts and substantially reducing speculative rejections. As shown in Table 1, AgentSpec achieves a rejection rate a" 
	* Explain the issue, and the mechanism
* How are the calculation overhead of 4.1 and 4.2 not prohibitive of actual speed ups
* What can we take from here for ourselves? 
* How is batch size defined at sizes that exceed hardware capacity? 
* What model do they use?  How do they generate candidates? other than the redundancy metric and semantic blocks they didn't really give a lot of detail...
