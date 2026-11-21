---
title: "AdaptiVocab - Enhancing LLM efficiency in focused domains through lightweight vocabulary adaptation"
date: "2026-09-09"
type: "paper-summary"
subtype: null
topic: "Inference systems"
source_title: "AdaptiVocab - Enhancing LLM efficiency in focused domains through lightweight vocabulary adaptation"
source_author: "Itai Nakash, Roi Reichart"
source_url: "https://arxiv.org/pdf/2503.19693"
---

## Provenance
- **Link** — https://arxiv.org/pdf/2503.19693
- **Authors / affiliation** — Technion, Itai Nakash, Roi Reichart 
- **Venue, date** — Aug 2025
- **How it reached me** — 
- **Why I clipped it** — SuperTokenizers for Draft Models

## Claim
<!-- What is the paper asserting? One paragraph, in your words, no jargon borrowed from the abstract. -->
When focusing on narrow domains, one can achieve significant speedups by removing rare tokens and replacing them with common N-Grams 
## Techniques and evidence
<!-- What techniques and evidence are used? How do they distinguish between true and false beliefs about the model? Which forms of evidence feel stronger vs weaker? Any flaws in the setup? -->
1. Identify highest scoring N-grams
2. Identify lowest scoring tokens
3. Replace set from 1 with set from 2.
4. Apply tokenizer to new text -
	1. Tokenize text with original tokenizer 
	2. Decompose tokens that were removed from the original vocab into smaller chunks 
	3. Merge existing tokens into the new n-tokens where possible
		1. This is done greedily, otherwise it is too computationally expensive to be worthwhile 
5. Exponential embedding initialization 
	1. 
## Generalizable vs specific
<!-- What seems model/task-specific? What do you predict will generalize? -->

## Cherry-picking
<!-- Which parts feel most cherry-picked? Which numbers carry the headline, and on which subset? -->

## Limitations and flaws
<!-- Theirs, stated. Yours, found. Keep the two apart. -->
Why is this a one<>one replacement of rare and new 
## Concrete output
<!-- Pick one: code you ran against the model, a blog post, a newsletter-style summary, an explanation you gave someone. Papers you really like get one. -->

## Anki
<!-- Paste paragraphs verbatim while reading. A later session extracts these into Anki/ as BEGIN_NOTE blocks. Leave them raw. -->
