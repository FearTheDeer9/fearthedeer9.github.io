---
title: "Generalization Dynamics of LM Pre-Training"
date: "2026-10-03"
type: "paper-summary"
subtype: null
topic: "Pretraining, data & distillation"
source_title: "Generalization Dynamics of LM Pre-Training"
source_author: null
source_url: "https://arxiv.org/pdf/2609.33150"
---

## Provenance
- **Link** — https://arxiv.org/pdf/2609.33150
- **Authors / affiliation** — 
- **Venue, date** — 
- **How it reached me** — 
- **Why I clipped it** — 

## Claim
The authors claim that wide spread belief which states that LMs stably mature from pattern-matching parrots to generalizable intelligence during pre-training is ill-founded. Instead, they show that LMs hop between modes of parrot and intelligence computations. Morover they show it's locally stable and cannot be fixed by checkpoint averaging -> not the result of standard optimization dynamics. They suggest thinking of the phenomena as one of capacity-bounds - generalizable circuits must compete with shallow ones learned early in training and the data in each pre-training window decides which circuits ultimately wins. They leverage these finding to suggest:
1. How one can select intermediate checkpoint with stronger generalization performance
2. Select pre-training data that controls and stabilizes generalization dynamics. 
## Techniques and evidence
The authors craft adversarial eval suites that are specifically crafted to differentiate parrot-like heuristics from true reasoning. 
![](/attachments/generalization-dynamics-of-lm-pre-training-Pasted-image-20261005145105.png)
And show that all of these show fluctuating performance rather than a monotonic function, for example:
![](/attachments/generalization-dynamics-of-lm-pre-training-Pasted-image-20261005145206.png)
## Generalizable vs specific
<!-- What seems model/task-specific? What do you predict will generalize? -->
* The authors show that "standard" evals show consistent performance gain 
## Limitations and flaws
* This is empirical rather than gears-level, They show that this occurs but are unable to explain why it happens and how one would prevent this from happening in the first place. The "solution" of choosing checkpoints that perform well on held out evals is ad hoc. Moreover, it works for "specific" evals at each time (i.e the model hops between per-domain parrot vs reasoner, it isn't a universal propensity which means that one would need to define which domain is more important for "true" reasoning, a patch rather than a principled solution). 
* 
## Anki
<!-- Paste paragraphs verbatim while reading. A later session extracts these into Anki/ as BEGIN_NOTE blocks. Leave them raw. -->
