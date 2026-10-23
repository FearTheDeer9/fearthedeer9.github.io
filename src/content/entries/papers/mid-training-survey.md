---
title: "Mid Training Survey"
date: "2026-09-30"
type: "paper-summary"
subtype: null
topic: "Pretraining, data & distillation"
source_title: "Mid Training Survey"
source_author: "Peking University"
source_url: "https://arxiv.org/pdf/2510.23081"
---

## Provenance
- **Link** — https://arxiv.org/pdf/2510.23081
- **Authors / affiliation** — Peking University
- **Venue, date** — Nov 25
- **How it reached me** — 
- **Why I clipped it** — 

## Claim
<!-- What is the paper asserting? One paragraph, in your words, no jargon borrowed from the abstract. -->
Mid-training is an emerging field creating a bridge between pre-training (which is about skill and knowledge creation) and post-training (which is about honing the behavior and interaction patterns). Mid-training sits in the middle of this, creating a bridge between enhancing capabilities and honing the behavior profile, setting up the subsequent post-training. 
### Techniques and evidence
<!-- What techniques and evidence are used? How do they distinguish between true and false beliefs about the model? Which forms of evidence feel stronger vs weaker? Any flaws in the setup? -->
because this is a survey it's less about technique and evidence and more about creating a taxonomy and an ontology of where this field is at. The authors claim three distinct contributions:

1. Defining more rigorously what we mean when we say "mid-training".
2. Providing a best practice for how to systematize this in the field, given what we know so far. The framework for optimizing mid-training consists of five steps:
    1. Data curation
    2. Architecture optimization
    3. Training algorithms
    4. Evaluations
    5. The scaling laws for optimizer K rates
3. They know the concrete optimization implementation that mainstream models have adopted.
### Generalizable vs specific
<!-- What seems model/task-specific? What do you predict will generalize? -->
### Data Curation 
the first step in the creation of data is collecting it. The data collection phase is similar to what you do in pre-training, where we have web-crawled corpora and digitized books. However more strict filtering is then applied in order to create higher-quality data. In addition some amount of reformatting and transitioning is applied to the data to modify it to fit in with QA pairs, so as to prepare the seed of capabilities we want present for post-training. This approach is to create the bridge between pre-training and post-training on one hand and to remain on data and in domain to reduce distributional shift weaknesses. It is to imbue the skills we would subsequently need, already present for post-training techniques like RL, to actually work. After data is collected we need to synthesize it. This step is mostly about, again, improving the quality of the data collected and enhancing the information density due to the data scarcity of agentic reasoning tasks. A high-quality demonstration of specific assistant behavior: we rely heavily on synthetic data augmentation techniques. These fall into three broad types:

- distillation, which is about leveraging more powerful LLMs to generate data directly for the smaller models
- extraction, which is about leveraging the LLM to transform web content into high-quality QA pairs. That is, we take an information-dense passage, phrase a question that can be answered from it, and provide the answer as an example for an LLM.
- evolution, which utilizes cyclical iterations to enhance problems and the solutions. This is mostly for verifiable domains like math and computing.

Now that we've collected and synthesized the data, we're left with selecting and decontaminating it. This is about ensuring that the quality of the data that we created is high. data selection can be split into two different types:

- Rating-based and targeted sampling. Targeted sampling is about oversampling and undersampling domains based on the representation we want them to have in the subsequent mixture, whereas writer-based is about using classifier models to rate the quality and only filtering those that pass a certain level.
- Decontamination is about removing data that could be part of public benchmarks, that we'd want to release as sensitive data, or nonsense data.

The current best practices are using n-gram-based filtering but this is brittle yet low-cost. More compute-heavy alternatives like classifier models or embedding models that rely on cosine similarity are being explored but are less well adopted right now. Finally we now have our data and it's about how we mix it. That is the final step here. The mixture used in mid-training is a combination of high-quality general corpora filtered from the pre-training data, as well as the novel datasets created in order to tailor specific objectives. As a result the data in this stage mostly falls into QA data, instruction data, and long context data. These data formats are strategically selected to support specific model capabilities that this stage is about creating, which are knowledge mastery, enhanced reasoning abilities, and adoption of a JT agentic behavior pattern

### Training Techniques 
 the main thing here is mostly about introducing a warmup-stable-decay type of optimizer that is data-mixture-aware. That is, the data mixture is aware of the phases of the optimizer and the most high-quality information-dense data is usually reserved for the final decay step, where the optimization and convergence to the local minima are done 

### Model Architecture Optimization
 not quite sure what they're getting at here. They introduce rope as well as yarn in order to highlight the fact that better positional embeddings that can handle long contexts that were not seen earlier in pre-training are needed now. They introduce rope and yarn for that. The benefits are obviously known but not sure why it fits into mid-training specifically. Maybe the introduction of mid-training goes hand in hand with the long context revolution, kind of 
### Decay Scaling Laws 
 here they don't provide anything specific. They just mention that further research is required to better understand how model size, checkpoints, training token count, and the data ratios influence the decay scaling law (and subsequently how one should introduce decay, at which stage, and at which slope) 
### Objective Driven Implementations 
Objective-driven implementations constitute the methodological cornerstone of mid-training, where capability enhancement objectives can be systematically categorized into the following interconnected domains: general capabilities, core cognitive abilities, task execution capabilities, and extension capabilities. 
Empirical analyses substantiate that retaining rigorously curated subsets of pre-training data is paramount, optimized composition and proportional balancing of which demonstrably elevate downstream task performance and generalization robustness.  the mid-training phase can also be considered as improving areas where pre-trained models are weaker due to natural data distribution biases. Probe experiments can be conducted to identify these gaps and then to address them Systematic enhancement of reasoning capabilities is one of the central focuses during mid-training, as shown in Table 4. It not only targets the direct improvement of core cognitive dimensions—such as logical deduction, causal inference, and multi-step problem solving, but also lays a robust foundation for subsequent RL development.
### Long Context extension
 all models use a multi-stage extension where context length progressively increases throughout the mid-training phase and the increased context length corresponds to smaller batch sizes.  even during the stages where context length becomes larger, models converge on maintaining 70-75% short context mixed with 25-30% long context, which is the empirical sweet spot 
## Cherry-picking
<!-- Which parts feel most cherry-picked? Which numbers carry the headline, and on which subset? -->

## Limitations and flaws
<!-- Theirs, stated. Yours, found. Keep the two apart. -->

## Anki
<!-- Paste paragraphs verbatim while reading. A later session extracts these into Anki/ as BEGIN_NOTE blocks. Leave them raw. -->
