---
date: '2025-04-14'
layout: single
title: "Paper Review - A Primer on the Inner Workings of Transformer-Based Language Models"
tags:
- transformers
- language models
- attention mechanisms
- NLP
---

This work summarizes the current techniques used to interpret the inner workings of transformer language models. As such, summarizing a summary is a challenge, but we will give it a try nonetheless.
The paper starts by introducing the transformer architecture and  notation. The authors adopt the *residual stream* perspective regarding interpretability. In this view each input embedding gets updated via vector additions from the attention and feed-forward blocks, producing *residual stream states*. The final layer residual stream state is then projected into the vocabulary space via the unembedding matrix and normalized via softmax to obtain a distribution over our sampled output token. 
## Transformer Architecture 
### Transformer Layer 
#### Layer Norm 
A common operation used to stabilize the training process, today commonly employed before the attention block due to empirical success. The layer norm operation given representation z is $((\frac{z-\mu(z)}{\sigma(z)}) \cdotp \gamma + \beta)$  where $\mu$ and $\sigma$ calculate the mean and std, and $\gamma \in \mathbb{R^d}$ and $\beta \in \mathbb{R}^d$ refer to learned element-wise transformation and bias respectively. 
The authors discuss how layer norm can be interpreted by visualizing the mean subtraction as a projection onto a hyperplane defined by the normal vector $[1,1,1,1....,1] \in \mathbb{R}^d$ and the following scaling as a mapping of the resulting representation to a hypersphere (look into this further).
#### Attention Block 
The attention block is composed of multiple *attention head*. At a decoding step i, each attnetion head reads from residual streams across previous position, decides which position to attend to, gathers information from those and finally writes it into the current residual stream. 
Using tensor rearrangement operations one can simplify the analysis of residual stream contributions. 

<!-- excerpt-end -->

$$ Attn ^{l,h}(X_{\leq i}^{l-1}) = \Sigma_{j \leq i} a_{i,j}^{l,h}x_j^{l-1}W_V^{l,h}W_O^{l,h}== \Sigma_{j \leq i} a_{i,j}^{l,h}x_j^{l-1}W_{OV}^{l,h}$$
$W_V^{l,h}, W_O^{l,h} \in \mathbb{R}^{d_h \times d}$ are learnable weight matrices. The attention weights for every key given the current query are obtained as: $$a_i^{l,h} = softmax ( \frac{x_i^{l-1}W_Q^{l,h} (X_{\leq i}^{l-1}W_K^{l,h})^T}{\sqrt{d_h}}) = softmax (\frac{x_i^{l-1}X_{QK}^hX_{\leq i}^{l-1^T}}{\sqrt{d_h}})$$
These two decomposition enable a view of QK and OV circuits as units responsible for reading from and writing to the residual stream. The attention block output is the sum of individual attention heads, which is subsequently added back into the residual stream. $$Attn^l(X_i^{l-1})= \Sigma_h=1^H Attn^{l,h}(X_{\leq i }^{l-1}))$$$$x_i^{mid,l} = x_i^{l-1} + Attn^l(X_i^{l-1})$$
#### Feedforward Network Block 
A feedforward block is composed of two learnable weight matrices $W_{in}^l \in \mathbb{R}^{d \times d_{ffn}}$ and $W_{out}^l \in \mathbb{R}^{d_{ffn} \times d}$. $W_{in}$ reads from the residual stream state $x_i^{mid,l}$ and its result is passed through an element wise non linear activation function g, producing the neuron activations. These get transformed by $W_{out}$ to produce the output FFN$(x_i^{mid})$ which is then added back to the residual stream. Geva et al introduced the notino of regarding the computation performed by the FFN block as *key-value memory retrieval*, with key $w_{in}^l$ stored in columns of $W_{in}$ acting as pattern detectors over the input sequence and values $w_{out}^l$, rows of $W_{out}$ being upweighted by each neuron activation. Using these perspective the computation done in the FFN block can be re-written as: $$FFN^l(x_i^{mid,l}) = \sum_{u=1}^{d_{ffn}} g_u(x_i^{mid,l}W_{in_u}^l)w_{out_u}^l = \sum_{u=1}^{d_{ffn}}n_u^lw_{out_u}^l$$
with $n^l \in \mathbb{R}^{d_{ffn}}$ being the vector of neuron activations. The element-wise non-linearity inside FFNs creates a *privileged basis* which encourages features to align with basis directions, making basis dimensions more likely to be independently meaningful, and therefore better suitable for interpretability analysis. 
#### Prediction head and transformer Decompositions 
**Prediction as a sum of component outputs** - The residual stream view shows that every model component interacts with it through addition. Thus, the unnormalized scores (logits) are obtained via a linear projection of the summed component outputs. Due to the properties of linear transformation, we can rearrange the tradition forward pass formulation so that each model component contributes directly to the output logits. This decomposition plays an important role when localizing components responsible for a prediction since it allows us to measure the direct contribution of every component to the logits of the predicted token.
$$f(\mathbf{x}) = \mathbf{x}_n^L\mathbf{W}_U \\
= \left(\sum_{l=1}^L\sum_{h=1}^H \text{Attn}^{l,h}(\mathbf{X}_{\leq n}^{l-1}) + \sum_{l=1}^L \text{FFN}^l(\mathbf{x}_n^{\text{mid},l}) + \mathbf{x}_n\right)\mathbf{W}_U \\
= \sum_{l=1}^L\sum_{h=1}^H \text{Attn}^{l,h}(\mathbf{X}_{\leq n}^{l-1})\mathbf{W}_U + \sum_{l=1}^L \text{FFN}^l(\mathbf{x}_n^{\text{mid},l})\mathbf{W}_U + \mathbf{x}_n\mathbf{W}_U. \quad (10)$$
**Prediction as an ensemble of shallow networks forward passes** - Residual networks work as ensembles of shallow networks where each subnetwork defines a path in the computation graph. For a simplified version of this analysis we can consider a two layer attention only transformer, where each attention head is composed just by an OV matrix. We can decompose the forward pass in this case as:
$$f(\mathbf{x}) = \mathbf{x}\mathbf{W}_U + \mathbf{x}\mathbf{W}_{OV}^1\mathbf{W}_U + \mathbf{x}\mathbf{W}_{OV}^1\mathbf{W}_{OV}^2\mathbf{W}_U + \mathbf{x}\mathbf{W}_{OV}^2\mathbf{W}_U . \quad $$
The path traversing a single OV matrix is named full OV circuits while the path involving both attention heads is referred to as virtual attention heads doing V-Composition, since the sequential writing and reading of the two heads is seen as OV matrices composing together. It has been proposed to measure the amount of composition as $\frac{||W_{OV}^1W_{OV}^2||_F}{||W_{OV}^1||_F||W_{OV}^2||_F}$.

## Behavior localization 
### Input attribution 
Commonly used to localize model behavior by estimating the contribution of input elements in defining model predictions. 
**Gradient-based input attribution** involves a first-order taylor expansion of a transformer at a point x. The resulting gradient $\nabla f_w(x) \in \mathbb{R}^{n \times d}$ captures the sensitivity of the model to each element in the input when predicting token w. These scores are usually aggregated at a token level to obtain a more intuitive overview of the influence of individual tokens. This is commonly done by taking the $L^p$ norem of the gradient vector wrt the i-th input embedding. By taking the dot product between the gradient vector and the input embedding $\nabla_{x_i}f_w)x \cdot x_i$ this sensitivity can be converted to an importance estimate. These methods are known to exhibit gradient saturation and shattering issues which prompted introduction of methods such as *integrated gradients*, *SmoothGrad* and Layer-wise Relevance Propagation. 

**Perturbation-based input attribution** - Another popular family of approaches estimates input importance by adding noise or ablating input elements and measuring the resulting impact on model predictions. 

**Context mixing for input attribution** - This method includes the use of the norm of value-weighted vectors,  output-value-weighted vectors or the use of vectors' distances to estimate contributions. A common strategy among these approaches involves aggregating intermediate per-layer attributions reflecting *context mixing* patterns using techniques such as attention rollout resulting in input attribution scores. 

**Contrastive Input Attribution** - An important limitation of input attribution methods is that attributed output tokens belong to a large vocabulary space. In this context attribution scores are likely to misinterpret several overlapping factors such as grammatical correctness and semantic appropriateness driving the model prediction. To address this recent work proposes a contrastive formulation fo such methods producing counterfactual explanations for why the model predicts token w instead of an alternative token o.

### Model Component Importance 
Early studies on the importance of transformer LMS highlighted a high degree of sparsity in model capabilities. These results motivated a new line of research studying how various model components in an LM contribute to its wide array of capabilities. 

**Logit attribution**
- Leverages model decomposition to measure direct contributions to output token prediction
- Uses direct logit attribution (DLA) to evaluate component effects:
  - Measures how component output affects logit score when projected by unembedding matrix
  - Can analyze attribution at multiple levels:
    - Individual neurons within FFNs
    - Attention head paths
    - Complete model components

Types of Attribution Analysis:
- Direct Logit Attribution (DLA):
  - For component c computing fc(x), measures $A_{f_w(x) \leftarrow c}^{DLA} = f^c(x)W_{U[:,w]}$ 
  - Neuron DLA: Measures  $A_{f_w(x) \leftarrow n_u^l}^{DLA} = n_u^l w_{out_u}^{l-1}W_{U[:,w]}$ 
  - Attention path DLA: Measures  $A_{f_w(x) \leftarrow^h c}^{DLA} = a_{n,j}^{l,h}w_{out_u}^{l-1}W_{OV}^{l,h}W_{U[:,w]}$  for attention paths

- Logit Difference (LD):
  - Extends DLA to measure relative preference between tokens
  - Computes difference in logits between target and alternative tokens
  - Helps understand component's role in selecting specific outputs
  - Similar to contrastive attribution framework
  
**Causal Interventions**
- Views model as causal graph with nodes as computations and edges as activations 
- Uses activation patching to intervene by replacing component values during forward pass
- Common intervention approaches:
  - Resample: Uses value from different input example
  - Mean: Uses average of values from multiple examples
  - Zero: Sets activation to null vector
  - Noise: Adds random perturbation to activation
- Faces challenges with ecological validity when interventions produce unnatural activations
- Can be applied in noising setup (patch during clean run) or denoising setup (patch during corrupted run)

**Circuit Analysis**
- Aims to identify subgraphs of components working together for specific tasks
- Uses edge and path patching to study component interactions:
  - Edge patching examines direct connections between components
  - Path patching analyzes multiple connected edges
- Main limitations:
  1. Requires careful design of task-specific input templates
  2. Needs human inspection to isolate important subgraphs
  3. Can produce second-order effects in downstream components
Methods to Address Limitations:
- ACDC algorithm automates circuit identification but is computationally expensive
- Alternative gradient-based methods:
  - Edge Attribution Patching (EAP) approximates patching effects
  - EAP-IG combines with Integrated Gradients for improved faithfulness
  - AtP* provides more robust approximation while maintaining efficiency
- Context mixing approaches can find information flow routes without patching

Here's a draft summary of sections 4.1 & 4.2 in a style consistent with your summary:

Here's an expanded version of the summary with those additional details:

## Information decoding

### Probing
Probes are supervised models trained to predict input properties from representations, aiming to assess information encoded in them. Given intermediate representation $f^l(x)$, a probe p maps to some input features z (like part-of-speech tags or semantic information).

Key limitations:
- Correlational rather than causal - high probe accuracy shows information is encoded but not that it's actually used by the model
- Tension between probe capability and task learning - powerful probes may solve tasks independently rather than reveal encoded information

Methods to address probe limitations:
- Using control tasks with randomized datasets as baselines
- Measuring information gain after applying control functions
- Evaluating both probe quality and "effort" required

Probe interventions:
- Linear erasure: Remove concept information by projecting away from probe direction
  - Used to mitigate bias or study impact of removing linguistic features
  - Can validate influence of probed properties on model predictions
- Steering generation: Modify outputs by manipulating probe-identified directions
  - Adding/subtracting probe direction can change features like sentiment
  - Activation addition method modifies residual stream: $x^{l'} \leftarrow x^l - \alpha u$ 

### Linear Representation Hypothesis and Sparse Autoencoders
Features are hypothesized to be encoded as linear subspaces of representation space. First shown in Word2Vec where word relationships could be captured by vector arithmetic (e.g., "Madrid" - "Spain" + "France" ≈ "Paris").

Sparse Autoencoders (SAEs):
- Train to reconstruct internal representations while promoting sparsity
- Loss combines reconstruction and sparsity terms:
  $L(z) = ||z - SAE(z)||^2_2 + \alpha ||h(z)||_1$ 
- Performance metrics:
  - L₀ norm of feature activations (average number of active features)
  - Loss recovered (% of original cross-entropy preserved)
  - Feature density histogram (distribution of feature activation frequencies)
  - Manual or automated interpretability assessment via DLA and maximally activating examples

The $L_1$ sparsity penalty in standard SAEs introduces a significant limitation known as "shrinkage". Since this penalty directly penalizes the magnitude of feature activations, the SAE is incentivized to reconstruct representations using the smallest possible activation values that still achieve reasonable reconstruction. This creates several complications: important features might be underrepresented due to artificially reduced magnitudes, it becomes difficult to distinguish between genuinely small activations and those that have been artificially shrunk, and the overall quality of feature interpretability is diminished as magnitude information no longer reliably indicates feature importance.

SAE variants addressing shrinkage:
- Gated SAEs: Separate feature detection (binary gate) from magnitude estimation
- TopK SAEs: Replace L1 penalty with keeping k largest features
- JumpReLU SAEs: Use threshold-based activation with L0 penalty

All variants move sparsity from loss function to architectural constraints, allowing better magnitude preservation while maintaining sparsity. This enables clearer interpretation of feature importance through activation magnitudes.
### Decoding in Vocabulary Space
The model interacts with the vocabulary in two ways: through the embedding matrix WE for input tokens and through the unembedding matrix WU for output space interaction. Due to this interpretable nature, analyzing model representations via vocabulary tokens provides valuable insights.

**Decoding intermediate representations**
- Logit lens: Projects intermediate residual stream states via $W_U$
- Can be interpreted as model prediction if skipping later layers
- Some models may fail to produce plausible predictions, leading to development of translators:
  - Linear or affine transformations applied before unembedding projection
  - Can be trained on attention head outputs (attention lens)
  - WU can be viewed as weights of a probe with vocabulary tokens as classes

**Decoding model weights**
- Uses SVD decomposition of model weight matrices: $W = U \Sigma V^T$
- When multiplying input z by W: $zW = \Sigma_i (zu_i \sigma_i) v_i^T$
  - $u_i$acts as key compared to input via dot product
  - $sigma_i$ weights the contribution
  - $v_i^T$ represents transformation direction
- Project top right singular vectors onto vocabulary space via $W_U$
- Reveals tokens matrix primarily interacts with
- Can analyze low-rank approximations by using subset of singular vectors
- Projection of weight matrices to backward pass (backward lens) can reveal how new information is stored

**Logit spectroscopy**
- Extension of logit lens examining finer structure of internal representations
- Splits right singular matrix of $W_U$ into N bands associated with different singular values
- Can project representations onto or suppress from these subspaces
- Enables analysis of different aspects of token prediction across model layers

## Discovered Inner Behaviors 
### Attention Block Behaviors

#### Attention Heads with Interpretable Attention Patterns
**Positional heads**
- Attend to specific positions relative to processed token
- Include heads attending to token itself, previous token, or next position
- Previous token heads crucial for induction mechanism and name concatenation

**Subword joiner heads**
- First found in machine translation encoders
- Attend exclusively to previous tokens that are subwords of same word as current token

**Syntactic heads**
- Attend to tokens with specific syntactic roles relative to processed token
- Specialize in dependency relations like obj, nsubj, advmod, and amod
- Appear suddenly during training and crucial for linguistic ability development

**Duplicate token heads**
- Attend to previous occurrences of same token in context
- Hypothesized to copy position information of previous occurrences
#### Attention Heads with Interpretable QK and OV Circuits
**Copying heads**
- OV matrices exhibit copying behavior
- Detected by number of positive real eigenvalues of $W_EW_{OV}W_U$
- Positive eigenvalues indicate increase in logits of same tokens

**Induction heads**
- Two-component mechanism involving previous token head (PTH) and induction head
- PTH copies information from first token A to residual stream of B
- Induction head at last position attends to B, its OV circuit increases B's logit score
- Can perform fuzzy matching for semantically related tokens
- Important for in-context learning
- Formation affected by diversity of context tokens

**Copy suppression heads**
- Reduce logit score of attended token if it appears in context and is confidently predicted
- Improves model calibration by avoiding naive copying
- OV circuit can suppress almost all vocabulary tokens when attended to
- Connected to self-repair mechanism as ablation deactivates suppression behavior

**Successor heads**
- Given ordinal sequence element input (e.g., "one", "Monday")
- First FFN creates 'numerical structure' representation 
- 'Effective OV circuit' $FFN^1(W_E)W_{OV}W_V$ increases logits of next sequence elements
	- Badly worded intuition: We identify an input token that commonly is an element in an ordinal sequence. Adopting the view of FFNs as key-value pairs, we can confidently assume that the weights in our model learn to deduce from the existence of this token (from this key) that the following token in the sequence is more likely (which is the value of the ffn). Then, the information that originates in a previous FFN block in the model  is added to the residual stream. It is eventually embedded by an attention head of a future layer, translated by the OV circuit and then unembedded into the space of the current token, upsampling the probability of the next item of the list.
- Found across multiple models (Pythia, GPT2, Llama 2)
Based on our previous discussions and matching your style, here's a draft for sections 5.2-5.4:

### Feedforward Network Block
The dimensions in the FFN activation space (neurons) following the non-linearity are more likely to be independently meaningful, making them prime candidates for interpretability analysis.

**Neuron behavior** can be analyzed through two complementary lenses that align with the key-value perspective:
- Input behavior (keys): How neurons respond to specific inputs
  - Position-specific firing ranges
  - Task/skill correlation 
  - Concept detection (languages, domains)
  - Linguistic feature detection
- Output behavior (values): How activations affect model predictions
  - Knowledge neurons for factual recall
  - Linguistic correctness neurons
  - Language-specific neurons for multilingual generation
  - Token frequency neurons adjusting prediction probabilities
  - Suppression neurons preventing improbable continuations

**Polysemantic neurons and superposition**
Early layer neurons typically exhibit polysemanticity, functioning as n-gram detectors. This may stem from superposition - encoding more features than available dimensions. The model appears to perform a two-stage tokenization process:
- Early layers: Map tokens to semantic meanings (detokenization)
- Later layers: Aggregate context back to token predictions (re-tokenization)

**Universality of neurons**
About 1-5% of neurons show consistent behavior across different model initializations, exhibiting higher monosemanticity:
- Alphabet neurons
- Previous token neurons
- Attention control neurons
- Entropy neurons modulating prediction uncertainty

### Residual Stream
The residual stream serves as the main communication channel, with components writing to linear subspaces that can be read by downstream components or directly influence predictions.

Key characteristics:
- Exponential growth in norm across layers (including output matrices WO, Wout)
- Input matrices (WQ, WK, WV, Win) maintain constant norms
- Memory management through:
  - Attention heads with negative eigenvalues
  - FFN neurons with opposing input/output directions
  
**Rogue dimensions**
Certain dimensions exhibit unusually large magnitudes, leading to:
- Anisotropic representations (vectors pointing in similar directions)
- Strong correlation with token training frequency
- Magnitude increase with model size
- Critical role in enabling no-op attention updates
- Impact on model quantization and downstream performance

### Emergent Multi-component Behaviors

**Circuit stability under fine-tuning**
Fine-tuning preserves circuit functionality while improving component efficiency:
- Basic operations (induction heads, name-movers) remain unchanged
- Components become better at detecting and encoding task-relevant features
- Benefits come from enhanced pattern recognition rather than architectural changes

**Grokking and circuit types**
Models transition from dense memorizing circuits to sparse generalizing ones:
- Memorizing circuits: Low initial cost but inefficient scaling
  - Direct input-output mappings
  - Poor compression efficiency
- Generalizing circuits: Higher fixed cost but better scaling
  - Implement actual algorithmic operations
  - More efficient for handling large datasets
  - Higher initial cost reflects needed computational machinery
Here's a draft summary of section 5.4.1 that matches the style and detail level of your existing summary:

### Factuality and hallucinations in model predictions

**Intrinsic views on hallucinatory behavior**
- Detection approaches analyze model's internal representations
- Methods include:
  - Probing classifiers trained on middle/last layer representations to predict output truthfulness
  - Finding and manipulating "truthfulness directions" with causal influence on outputs
  - Analyzing eigenvalues of responses' covariance matrix for semantic consistency
  - Using logit lens scores of predicted attributes in higher layers to indicate correctness

**Recall of factual associations**
- Studies factual recall using subject-relation-attribute tuples (e.g., "LeBron James plays basketball")
- Circuit components specialize by function:
  - Early-middle FFNs: Add subject information to residual stream
  - Early attention heads: Pass relation information to last token
  - Later attention heads: Extract appropriate attribute from subject representation
- Division of responsibilities appears in both attention-based and attention-less architectures

**Recall vs Grounding**
- Models engage differently with memorized vs contextual facts
- Given conflicting information like "The capital of Poland is London", two mechanisms compete:
  - In-context heads favor context answer (London)
  - Memory heads promote memorized answer (Warsaw)
- FFNs show higher contributions for ungrounded (memorized) versus grounded answers
- Early layer FFNs particularly important for memorized information retrieval
- Low-rank approximations of FFN matrices can improve performance by removing components encoding wrong but semantically plausible answers
## Input Attribution Tools:
- Captum: Core library providing gradient and perturbation-based attribution methods for PyTorch models
- Inseq: Focused on generative LMs, supports advanced contrastive context attribution
- SHAP: Popular toolkit for perturbation-based input attribution and model-agnostic explanations
- LIT: Framework-agnostic tool with visual interface for debugging LLM prompts

Component Analysis Tools:
- TransformerLens: Core toolkit for mechanistic interpretability of language models, provides hooks for interventions
- NNsight: Enables extraction of internal information from remote LMs through delayed execution
- Pyvene: Supports complex intervention schemes and serialization of intervention patterns

SAE Development Tools:
- SAELens: Supports advanced visualization of SAE features
- dictionary-learning: Built on NNsight, addresses SAE weaknesses
- Belrose: Specialized for Top-K SAE training

Visualization Tools:
- BERTViz/exBERT: Visualize attention weights and activations
- LM-Debugger: Inspects representation updates through logit attribution
- CircuitsVis: Provides reusable components for visualizing Transformer internals
- Neuronpedia: Open repository for visualizing SAE features with gamified annotation interface

Specialized Tools:
- RASP/Tracr: Language and compiler for creating interpretable transformer behaviors
- Pyreft: Toolkit for fine-tuning and sharing trainable interventions
- ViT Prisma: Focused on mechanistic interpretability of vision models
- MAIA: Multimodal model to automate common interpretability workflows