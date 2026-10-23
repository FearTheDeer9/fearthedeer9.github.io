---
title: "ThunderAgent"
date: "2026-09-15"
type: "paper-summary"
subtype: null
topic: "Inference systems"
source_title: "ThunderAgent"
source_author: null
source_url: "https://arxiv.org/pdf/2602.13692"
---

## Provenance
- **Link** — https://arxiv.org/pdf/2602.13692
- **Authors / affiliation** — 
- **Venue, date** — 
- **How it reached me** — 
- **Why I clipped it** — 

## Claim
<!-- What is the paper asserting? One paragraph, in your words, no jargon borrowed from the abstract. -->
Current LLM usage patterns are no longer a model-level but rather, agentic inference is a composition of inference engines with tool orchestrators. The existing optimizations are component-specific and don't have the "entire scope", without said global view system-level optimization of KV cache utilization (clean up as well as hit rates), tool lifecycles and cross-memory node imbalances (from tail latencies) are impossible. This paper presents an abstraction of agentic workflows as *agentic programs* which is a fist-class scheduling unit that persists across model innovations and tool executions and by exposing metadata for the workflow allows optimizations from state pausing, migration and tool resource management.
## Techniques and evidence
<!-- What techniques and evidence are used? How do they distinguish between true and false beliefs about the model? Which forms of evidence feel stronger vs weaker? Any flaws in the setup? -->
The authors show that current decomposed pipelines neglect optimized memory usage, tool lifecycle management and KV Cache optimizations.
### Abstraction of Programs 
A program is represented as a tuple of $<ID,c, \mathcal{T}, \mathcal{L}, \tau, s>$  where ID represents the unique global identifier. c denotes the number of tokens in the context, corresponding to the KV cache memory footprint during active execution. T tracks the set of tool environments used by the program, enabling garbage collection when no program requires them further. L, τ , and s denote the node placement, execution phase, and scheduling status, respectively, facilitating program-level KV cache thrashing reduction and cross-node transferring.
#### Cost Model 
Their cost model uses the Space-Time product (STP) as the primary metric, which is defined as the integral of the memory footprint (They use token count as a proxy) over time. They further decomose total cost as the sum of decode, prefill, recomputation, unused capacity and idle caching while external tools are processing - only the first two are "effective compute". 
## Generalizable vs specific
<!-- What seems model/task-specific? What do you predict will generalize? -->
### general
* abstractions of sessions as programs with associated cost, memory requirements, idle time and status
* System awareness across nodes 
	* Global waiting queue for instance 
* Hook based garbage collection 
#### Specific 
* Form of cost model
	* Subsequent optimization of eviction strategy 
* Assumption that all sessions have binary done vs active status 
## Cherry-picking
<!-- Which parts feel most cherry-picked? Which numbers carry the headline, and on which subset? -->

## Limitations and flaws
<!-- Theirs, stated. Yours, found. Keep the two apart. -->
### Theirs 
### Mine
**4.3.1** The way they derived the "optimal" evicition strategy assumes all sessions are equally likely to resume idle time simultaneously, this is what begets the optimized solution being least sum of squared context lengths 

## Concrete output
<!-- Pick one: code you ran against the model, a blog post, a newsletter-style summary, an explanation you gave someone. Papers you really like get one. -->

## Anki
<!-- Paste paragraphs verbatim while reading. A later session extracts these into Anki/ as BEGIN_NOTE blocks. Leave them raw. -->
