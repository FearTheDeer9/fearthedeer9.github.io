---
title: "Breaking the Token Ceiling - Distilling Smaller, Stronger Byte Models"
date: "2026-09-16"
type: "paper-summary"
subtype: null
topic: "Pretraining, data & distillation"
source_title: "Breaking the Token Ceiling - Distilling Smaller, Stronger Byte Models"
source_author: null
source_url: "https://arxiv.org/pdf/2609.12303v1"
---

## Provenance
- **Link** — https://arxiv.org/pdf/2609.12303v1
- **Authors / affiliation** — 
- **Venue, date** — 
- **How it reached me** — 
- **Why I clipped it** — 

## Claim
<!-- What is the paper asserting? One paragraph, in your words, no jargon borrowed from the abstract. -->
The authors introduce scaling laws for distillation models where they vary tokenization scheme (Tokens/Bytes/Bytes w/ \<eot\>) as well as the training objective (Distillation vs CE). Their core contribution is a one step mapping of tokens -> bytes which reduces the vocabulary size, which allows better asymptotic behavior, improved data and storage efficiency. 

## Techniques and evidence
<!-- What techniques and evidence are used? How do they distinguish between true and false beliefs about the model? Which forms of evidence feel stronger vs weaker? Any flaws in the setup? -->
#### Marginalize-It Logits

For each token used in teacher inference, we have a $|V|$-sized logit tensor denoting the next-token distribution, where $V$ is the vocabulary of the BPE tokenizer. We also have the decoded byte sequence of every BPE token in $V$.

To compute the first byte distribution of each BPE token, we marginalize over the full vocabulary $V$ based on the first byte. For each subsequent byte, we restrict the vocabulary to the tokens whose prefixes match the ground-truth bytes and compute the byte distribution conditioned on this subset. We refer to this as the **Marginalize-It** method.

While this approach always yields an exact distribution for the first byte, the distributions over subsequent bytes are only approximate. Figure 2 (column 1) shows the Marginalize-It logit conversion process step by step.

To predict the distribution $B_3$, the continuations of token `is` (whose probability is $0.125$) are silently dropped and only the tokens `isu` and `isk` are carried forward. While obtaining the next-token distribution of token `is` can fix this problem, it requires running additional teacher inference over the sequence:

`[<bos>, T, iram, is]`

The number of additional inferences required to obtain exact conversions can get intractable for long sequences. To avoid that problem, in the Marginalize-It method remaining probability is redistributed among tokens `isu` and `isk`:

$$
P(\text{isu}) = \frac{0.5}{0.5 + 0.125} = 0.8,
\qquad
P(\text{isk}) = \frac{0.125}{0.5 + 0.125} = 0.2.
$$

During training, the byte sequence used is:

`[<bos>, T, i, r, a, m, i, s, u, <eos>]`

We provide additional details in Appendix F.2. Marginalize-It serves as the main baseline throughout the study.

#### End-Of-Token Logits

The **End-Of-Token** method is essentially the Marginalize-It method with two key changes.

First, while forming the byte logits after teacher inference, we append an `<eot>` token to mark the end of each BPE token, as shown in Figure 2 (column 2).

Second, we expand the byte vocabulary by one to store the `<eot>` probabilities and then apply marginalization over byte positions to form the byte logits.

During training, we append `<eot>` after every BPE token in the pretraining dataset:

`[<bos>, <eot>, T, <eot>, i, r, a, m, <eot>, i, s, u, <eot>, <eos>, <eot>]`

This resolves the missing-probability problem of Marginalize-It. To compute the $B_3$ distribution, unlike Marginalize-It, End-Of-Token considers all tokens with prefix `is`.

Since the total probability of tokens starting with prefix `is` is

$$
0.5 + 0.125 + 0.125 = 0.75,
$$

we obtain:

$$
P(\text{isu}\langle\text{eot}\rangle)
= \frac{0.5}{0.75}
= 0.667,
$$

$$
P(\text{isk}\langle\text{eot}\rangle)
= \frac{0.125}{0.75}
= 0.167,
$$

$$
P(\text{is}\langle\text{eot}\rangle)
= \frac{0.125}{0.75}
= 0.167.
$$

In this way, the probability mass of the possible continuations of `is` is absorbed into the `<eot>` token.

This approach is exact and does not require additional inferences over the teacher. Inserting `<eot>` tokens also has an unintended but remarkable consequence: the $\approx 30.94\%$ additional compute it spends per unit of data (one `<eot>` every approximately 4.5 bytes) turns out to further improve asymptotic performance relative to the Marginalize-It method.
## Generalizable vs specific
<!-- What seems model/task-specific? What do you predict will generalize? -->
The goal of reduced vocab size seems generic, and reverting back to characters seems like we rediscovered ancient babel. However, we already know that tokenizers help in some way, we didn't start off with BPE - i want to understand why this is, is this small models? New compute? new data algorithms or due to distillation rather than from scratch? 
## Cherry-picking
<!-- Which parts feel most cherry-picked? Which numbers carry the headline, and on which subset? -->
* BPB Seems like it'd be "easier" for smaller vocabularies for smaller models in some sense, not sure why though (perhaps you don't consider alternative synonyms after the first byte).
* They  stop training token versions. BPP is surpassed by the byte versions but for the accuracy of downstream tasks they still perform better and we don't see how that generalizes. We're only left with extrapolation 
## Limitations and flaws
<!-- Theirs, stated. Yours, found. Keep the two apart. -->
* Lower BPB but also lower downstream accuracy seemingly across the board
## Concrete output
<!-- Pick one: code you ran against the model, a blog post, a newsletter-style summary, an explanation you gave someone. Papers you really like get one. -->

## Anki
<!-- Paste paragraphs verbatim while reading. A later session extracts these into Anki/ as BEGIN_NOTE blocks. Leave them raw. -->
