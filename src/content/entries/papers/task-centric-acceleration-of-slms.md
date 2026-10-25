---
title: "Task-Centric Acceleration of SLMs"
date: "2026-09-08"
type: "paper-summary"
subtype: null
topic: "Inference systems"
source_title: "Task-Centric Acceleration of SLMs"
source_author: null
source_url: "https://arxiv.org/pdf/2602.24174"
---

## Provenance
- **Link** — https://arxiv.org/pdf/2602.24174
- **Authors / affiliation** — 
- **Venue, date** — 
- **How it reached me** — 
- **Why I clipped it** — Part of the  SuperTokenizers for Draft Models Research 

## Claim
By considering the N-gram statistics of output distributions on a representative dataset, we can add a finetuning step that enhances the base vocabulary with popular N-Grams in a way that reduces latency when serving it.
## Techniques and evidence
<!-- What techniques and evidence are used? How do they distinguish between true and false beliefs about the model? Which forms of evidence feel stronger vs weaker? Any flaws in the setup? -->
### Tokenization
They generate outputs over a train set that is pre-collected
* Since they only want to improve decode-time latency rather than prefill they only consider outputs, not inputs. under the assumption that the stats under each regime may differ. 
Then, they define a *merge reward* by its exprected contribution to seq length reduction by 
$$
f_{merge}(t^n) \eqcirc frq(t^n)\cdot (n-1)
$$
. Doing this naively could bias greedy decoding towards collapsing multiple plausible continuations into a single dominant path - surpressing alternatives whose prefix collides with the new token's suffix 
* For instance collapsing the bigram ["in","the"] can make tokens like "therapy" or "theorem" less likely to appear. 
To mitigate, they propose the Prefix Collision Score (PCS) defined as 
$$
\mathrm{PCS}(t^n)
= P\left(
\bigcup_{t' \in \mathcal{V}'}
\operatorname{str}(t_n)
\text{ is a prefix of }
\bigl(\operatorname{str}(t')\bigr)
\right)
$$
  and only collapse into a super-token if the merge reward is high enough AND the PCS doesn't pass a certain, predefined threshold. 
### Fine Tuning 
They initialize each new "mega-token" as the mean of its constituent token embeddings $e_{new}= \frac{1}{n}\sum_{j=1}^{n}e_{t_j}$ 

## Generalizable vs specific
<!-- What seems model/task-specific? What do you predict will generalize? -->
* The technique of adding N-grams "straight" into the model seems general, there's nothing architecture/data specific about it. 
## Cherry-picking
<!-- Which parts feel most cherry-picked? Which numbers carry the headline, and on which subset? -->
* Choosing datasets with extremely structured output data, classification style tasks where the benefit is obvious due to low entropy and abundance of recurring results
## Limitations and flaws
<!-- Theirs, stated. Yours, found. Keep the two apart. -->
The threshold for PCS is flaky, not to mention that it doesn't take into account the magnitude of the merge reward. 
## Concrete output
<!-- Pick one: code you ran against the model, a blog post, a newsletter-style summary, an explanation you gave someone. Papers you really like get one. -->

## Anki
<!-- Paste paragraphs verbatim while reading. A later session extracts these into Anki/ as BEGIN_NOTE blocks. Leave them raw. -->
* We find additional motivation in the uniform information distribution (UID) hypothesis which posits that efficient linguistic representations tend to distribute information evenly across tokens, leading to approximately uniform per-token surprisal along sequences
