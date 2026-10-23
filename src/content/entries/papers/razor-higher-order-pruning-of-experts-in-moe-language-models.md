---
title: "Razor & Higher Order Pruning of Experts in MoE language Models"
date: "2026-10-05"
type: "paper-summary"
subtype: null
topic: "Inference systems"
source_title: "Razor & Higher Order Pruning of Experts in MoE language Models"
source_author: null
source_url: "https://arxiv.org/pdf/2609.18916"
---

## Provenance
- **Link** — https://arxiv.org/pdf/2609.18916, https://arxiv.org/pdf/2609.30465
- **Authors / affiliation** — 
- **Venue, date** — 
- **How it reached me** — 
- **Why I clipped it** — 

## Claim
Both of these papers introduce more principled ways to prune experts in MoE models to reduce memory/compute constraints while maintaining performance on a specific test-set/domain
## Techniques and evidence
<!-- What techniques and evidence are used? How do they distinguish between true and false beliefs about the model? Which forms of evidence feel stronger vs weaker? Any flaws in the setup? -->
Both papers note that by inspecting $c(x)= \Sigma_{i \in S(x) }w_i(x)f_i(x)$ , the routed output of each layer we can identify the consensus. Which experts are more/less important on the dataset on which we'd like to maintain performance. They both note that in order to go from consensus residual to exact deletion effect, one needs to take into account the higher order effects that result from the deletion beyond just the magnitude of the removed component. They differ in the way they go about this. 
**Razor** - 
**HOPE** - 
## Generalizable vs specific
<!-- What seems model/task-specific? What do you predict will generalize? -->

## Limitations and flaws
<!-- Theirs, stated. Yours, found. Keep the two apart. -->
Both papers note the same issue but use severe relaxations to make the problem tractable, relaxtions that lose a significant amount of information. A principled approach would approximate the full counterfactual - how would removing this subset of experts effect the ENTIRE delta. i.e we need to consider all of:
1. The entire subset we remove rather than each expert in isolation 
2. The experts that replace them (we now that models are highly redundant
We clearly see how each of these papers handles only one of the two components and a relaxed version of it at that. 
## Anki
<!-- Paste paragraphs verbatim while reading. A later session extracts these into Anki/ as BEGIN_NOTE blocks. Leave them raw. -->
* The two methods, how they decide which experts are removed along with a consolidation card that compares and contrasts them.
