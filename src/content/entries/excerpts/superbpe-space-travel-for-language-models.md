---
title: "SuperBPE - Space Travel for Language Models"
date: "2026-09-07"
type: "excerpt"
subtype: "quoted"
topic: "Pretraining, data & distillation"
source_title: "SuperBPE - Space Travel for Language Models"
source_author: null
source_url: null
---

* "BPE is a tokenization algorithm that greedily learns a subword vocabulary given training data.2 The algorithm takes a sample of text and a target vocabulary size T as input.3 The first step of BPE is pretokenization, which splits the text into chunks that limit the extent of tokenization; merges cannot bridge these chunks, so the final learned tokens are parts of these chunks. Canonically, pretokenization in BPE consists of splitting on whitespace so that common word sequences do not become a single token. This made sense given the historical context of Sennrich et al. (2016), which aimed to improve word-level tokenization by segmenting words into morphologically meaningful subwords. After pretokenization, the iterative learning algorithm begins. Training text is first split into bytes; the starting vocabulary is the set of all bytes. Then, the frequencies of all pairs of neighboring tokens are recorded, and the most frequent pair is merged into a single, new token at every position in the text where it occurs. The newly merged token is added to the vocabulary. For instance, if the merge is (t, he), then all instances of the token sequence [t, he] will be replaced with the, which is added to the vocabulary. The token pair frequencies are then updated, and the next most frequent pair is again merged into a new token. This continues until the vocabulary reaches the target size T."
* SuperBPE introduces a simple intervention in the pretokenization step, separating tokenizer training into two discrete phases, wherein the tokenizer (1) first learns subwords (by using pretokenization to prevent merges across whitespace) and then (2) learns superwords (by lifting this restriction). Stage 1 is equivalent to regular BPE training and continues up to a certain vocabulary size t, which we call the transition point (t < T). In stage 2, tokenizer training resumes from the vocabulary learned thus far, but this time whitespace pretokenization is skipped. As a result, token pairs that bridge whitespace are considered, enabling superwords to be added to the vocabulary. Intuitively, we intend for our tokenizer to first learn base units of semantic meaning, then combine these units into common sequences for a much more efficient vocabulary. Note that t = T corresponds to BPE, and t = 0 corresponds to a naive revision of BPE that foregoes whitespace pretokenization at any point in training.
# Questions 
* Can the bits for bytes metric be extended for a reward for draft tokenizers.
*
