---
title: "Neural Scaling and the Quanta Hypothesis"
date: "2026-02-09"
type: "excerpt"
subtype: "article-summary"
topic: "Architectures & scaling"
source_title: "On Neural Scaling and the Quanta Hypothesis"
source_author: "Eric J. Michaud"
source_url: "https://ericjmichaud.com/quanta/"
---

## Context
Research article by Eric J. Michaud exploring a theoretical framework for understanding how neural networks change as they scale up in size, training data, and compute.

## Source
[On Neural Scaling and the Quanta Hypothesis](https://ericjmichaud.com/quanta/#big-picture)

## Core Insight

Neural networks learn discrete **quanta** — fundamental units representing distinct algorithms or pieces of knowledge. Two key assumptions:

1. **Discreteness**: Each quantum is either fully learned or not; networks cannot partially learn mechanisms
2. **Power-law distribution**: The frequencies at which different quanta are needed follow a power law, with some knowledge common and other knowledge increasingly rare

## Why It Matters

This framework reconciles two seemingly contradictory observations:
- **Smooth scaling laws**: Aggregate loss decreases as a smooth, predictable power law
- **Sharp emergent abilities**: Language models exhibit discrete capability jumps at specific scales

The resolution: mean loss decreases smoothly because it averages over many discrete phase transitions occurring at different scales.

## Key Findings

- **Multitask sparse parity experiments**: Synthetic tasks with power-law-distributed subtasks demonstrate networks learning subtasks in frequency order, matching predicted scaling behavior
- **Per-token analysis (Pythia models)**: Loss curves reveal diverse patterns — some discrete, some smooth — suggesting most tokens benefit from multiple quanta ("polygenic" vs "monogenic")
- **Diminishing returns at scale**: Practical impact diminishes as scaling reaches increasingly niche knowledge
- **Interpretability connection**: Sparse autoencoder "features" may represent individual quanta or feature manifolds

## Limitations

- Empirical scaling exponents don't always match theoretical predictions
- Many per-token curves show smoother behavior than strict discreteness predicts
- Seed-dependent variations suggest partial progress, complicating the discreteness assumption

## Implications for AI Development

- Scaling provides predictable aggregate improvement but unpredictable per-capability jumps
- Understanding quanta could inform more efficient training (targeting specific capabilities)
- Mechanistic interpretability tools need to scale with model size to extract increasingly sparse mechanisms
- The framework suggests a natural ordering of what models learn: frequent patterns first, rare ones later

## Related
- Neural scaling laws (Kaplan et al., Hoffmann et al.)
- Emergent abilities debate (Wei et al. vs Schaeffer et al.)
- Mechanistic interpretability and sparse autoencoders
- Phase transitions in learning
