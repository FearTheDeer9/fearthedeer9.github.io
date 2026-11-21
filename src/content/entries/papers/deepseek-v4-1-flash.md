---
title: "DeepSeek V4.1-Flash"
date: "2026-09-15"
type: "paper-summary"
subtype: null
topic: "Inference systems"
source_title: "DeepSeek V4.1-Flash"
source_author: "DeepSeek"
source_url: "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf"
---

## Provenance
- **Link** — https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf
- **Authors / affiliation** — DeepSeek
- **Venue, date** — 
- **How it reached me** — 
- **Why I clipped it** — 

## Claim
<!-- What is the paper asserting? One paragraph, in your words, no jargon borrowed from the abstract. -->
Agentic workloads have made llm workflows increasingly input-heavy, this means that one should rethink the requirements for such a system - due to ever longer KV cache the memory requirements (over HBM) become increasingly demanding - the authors introduce several innovations that reduce the bytes per param as well as the number of params (through sparsity) for improved performance along the cost<>latency<>quality frontier.
## Techniques and evidence
<!-- What techniques and evidence are used? How do they distinguish between true and false beliefs about the model? Which forms of evidence feel stronger vs weaker? Any flaws in the setup? -->
### CSA V2
They extend compressed sparse attention by sharing global KV entries across layers and reusing the indices that select which entries to attend to. Full layers compute new KV entries and selections; Reindex layers share the KV but make a new selection; Reuse layers share both. Every layer still computes its own query and local SWA KV. In the decoder, a hierarchical indexer scans the full context once per query to build a candidate pool of up to 16,384 positions; later Reindex layers select their top 512 entries from this pool. This reduces both cache storage and repeated indexing work. Together with FP4 caching, the global KV footprint falls to 890 bytes per token, roughly 1/4 of V4-Flash's.
### Engram tables
They add 196B parameters of conditional memory to separate memorization from computation. Short token sequences (2-, 3- and 4-grams) are hashed into learned embedding tables, with context-aware gating controlling how the retrieved memory contributes to the model. The tables are split across two modules and stored in FP8. Since lookup addresses depend only on the input tokens, embeddings can be prefetched from host memory while the model computes, adding memory capacity without running all those parameters for every token. They retain the original Engram design but remove its short causal convolution and use Sinkhorn-balanced embedding updates.
### SWA with bounded replay
Reconstructing missing SWA KV exactly across $L$ layers requires replaying $L \times n_{win}$ tokens, because each layer's local window depends on hidden states that themselves depend on earlier windows. They instead replay only the last $n_{win}=128$ tokens, truncating local attention at the replay boundary and accepting approximate states. On an encoder SWA cache miss, they reuse the cached global KV and replay the prefix to rebuild only its SWA state, processing new tokens alongside it. For the decoder, they run the last 128 prompt tokens' encoder outputs through its layers to initialize SWA for decoding, allowing most prompt tokens to skip the decoder. This removes SWA from persistent storage; with global KV compression, that cache shrinks to roughly 1/8 of V4-Flash's. They report negligible quality loss from the approximation.

### CED
They introduce a CausalEncoder-Decoder architecture to alleviate prefill bottlenecks. By designating the lower half of the layers to be encoders that generate the KV cache, the upper half of the layers directly share the aforementioned cache. The bottom L/2 layers are the causal encoder while the upper half's KV entries are derived from the hidden state of $H_{L/2}$ enabling the upper-layer's global KC cache to be produced with minimal computational cost.
### Synthetic data pipeline 
They create a large scale automated pipeline for data synthesis and environment construction - this is claimed to be the key benefit for quality (as there are no algorithmic innovations in the post-training stage). 
They adopt a holistic approach for data generation, that is systematic and standardized. They filter model-generated content that has limited information gain as they assume that it is indicative of implicit duplication. They also add more domain experts for specific evaluation dimensions as well as model-in-the-loop data iteration approaches. 
### Post Training Pipeline 
They emphasize the fact that they use a relatively vanilla workflow of SFT -> RL -> OPD and instead focused on optimize the pipeline of synthesizing diverse verifiable training task with reference solutions and reward signal -> construct and scale interactive agent environments with trajectories collected and evaluated at low cost -> apply rigorous filtering, dedup and difficulty calibration for data quality and curriculum balance. The tasks and environments are obtained from continual collection of data from production environments (employees and partners) with special attention given to trajectories where negative feedback/frustration was recognized -> opportunity for learning experience.
For controlling reasoning efforts they use a scalar $b \in \{1,100\}$ and sample $M_b$ responses at each effort layer. The responses that share the same x,b are considered  "equivalent" and are scored together for GRPO (responses between different levels are not compared). 
Extra care and engineering effort was poured into making the async RL pipeline efficient and optimized, overcoming the inherent complexity associated with it. They use regularizers and loss functions that are meant to reduce KLD between generation and target model due to the async nature. They support token level checkpoint switching to allow in-flight adaptation to new checkpoints, and they also hold over 40 teacher models and dynamically adapt which of these is used to supply signal based on capabilities, training recipe and dataset mixture. 
#### Gotchas 
* Several separate agents for task construction, env set up (dependencies, permissions etc), separated individual agents attempting the task, grader agent that analyzes trajectories and sees where things failed and why. 
* Rollout execution decoupled into agent sandbox and worker container. The sandbox handles the sandbox and tools while the worker standardizes trajectories, schemas and interactions across envs. 
* 
### Minor details 
* Separate biases for tex vs image modalities
* Single Pass mHC computation, engrams, DSpark, FP4 quantization 
## Generalizable vs specific
<!-- What seems model/task-specific? What do you predict will generalize? -->

## Cherry-picking
<!-- Which parts feel most cherry-picked? Which numbers carry the headline, and on which subset? -->

## Limitations and flaws
<!-- Theirs, stated. Yours, found. Keep the two apart. -->

## Concrete output
<!-- Pick one: code you ran against the model, a blog post, a newsletter-style summary, an explanation you gave someone. Papers you really like get one. -->

## Anki
- Sparsity Ratio used
- Architectural differences of the CED architecture amd how it effects throughput
- Bytes per token for DeepSeek, compare this to other architectures 
- Ablations on each component here: 
	  ![](/attachments/deepseek-v4-1-flash-Pasted-image-20260915124259.png)
* 

# Questions
* Where do different (2x) number of params for prefill vs decode come from
* Explain the SWA bounded replay thing, what is meant by what's needed for exact reconstruction, what optimization do they make and how does it help
* How/Why did they make make the move back to encode<>decode steps. IS this done elsewhere? What are the tradeoffs? What are the intuitions for the difference between these options
* explain "To align the ViT more closely with LLM design principles, we replace the patch embedding layer’s convolution with a linear projection to ensure compatibility with the Muon optimizer"
* What is main vs indexer K
* Not sure I followed the async workflow for dispatching. Explain what they settled on and what were prblems with the alternatives
