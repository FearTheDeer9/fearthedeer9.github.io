---
date: '2025-04-14'
layout: single
tags:
- neural rendering
- neural graphics primitives
- multiresolution hash encoding
- real-time rendering
- neural representations
title: Implementation
---

Müller et al. introduced a versatile input encoding for neural networks that dramatically accelerates the training and inference of neural graphics primitives. By combining a multiresolution hash table structure with small MLPs, they achieved training speeds several orders of magnitude faster than previous approaches while maintaining high quality across diverse graphics applications. This approach enables real-time rendering and training of neural representations that previously required hours to converge.

## Implementation

### Multiresolution Hash Encoding

The core innovation of this paper is a multiresolution hash encoding that maps spatial coordinates to feature vectors through a hierarchy of hash tables:

1. **Multiresolution Structure**: The method uses L=16 resolution levels with a geometric progression between the coarsest resolution Nmin and finest resolution Nmax:

   $$N_l = \lfloor N_{min} \cdot b^l \rfloor$$

   where b is determined by:

<!-- excerpt-end -->


   $$b = \exp\left(\frac{\ln N_{max} - \ln N_{min}}{L-1}\right)$$

2. **Hash Function**: For coordinate x at level l, the paper uses a spatial hash function:

   $$h(\mathbf{x}) = \left(\bigoplus_{i=1}^{d} x_i\pi_i\right) \mod T$$

   where ⊕ is bitwise XOR, πi are prime numbers, and T is the hash table size.

3. **Interpolation**: The feature vectors at each corner are d-linearly interpolated:

   $$\mathbf{w}_l = \mathbf{x}_l - \lfloor\mathbf{x}_l\rfloor$$

   where xl = x·Nl and the interpolation weights are determined by the relative position within the voxel.

4. **Feature Combination**: The interpolated feature vectors from all L levels are concatenated with auxiliary inputs ξ to produce the encoded input:

   $$\mathbf{y} = \text{concat}(f_1, f_2, ..., f_L, \xi) \in \mathbb{R}^{LF+E}$$

### Neural Network Architecture

The encoding is paired with small MLPs that vary by application:

- For Gigapixel images, SDFs, and Neural Radiance Caching: 2 hidden layers with 64 neurons each
- For NeRF: 1-hidden-layer density MLP followed by a 2-hidden-layer color MLP, both with 64 neurons

Training uses Adam with carefully tuned learning rates and L2 regularization on network weights (but not hash table entries).

## Results

The approach demonstrated remarkable results across multiple tasks:

1. **Training Speed**:
   - Gigapixel image: From 36.9 hours (ACORN) → 2.5 minutes (884× speedup)
   - NeRF: From hours → 5-15 seconds for comparable quality
   - Achieved convergence in seconds instead of hours across all applications

2. **Performance**:
   - Real-time rendering at 60 FPS at HD resolution
   - Neural radiance caching at 133 FPS at 1920×1080 resolution
   - SDF training and visualization with interactive feedback

3. **Quality Comparison**:
   - Comparable or better quality than specialized approaches (mip-NeRF, NSVF)
   - State-of-the-art performance on various scenes with significantly less training time
   - Demonstrated 20-60× improvement from the hash encoding itself vs. implementation optimizations

4. **Memory Efficiency**:
   - Used a fraction of the parameters of dense grid approaches
   - T=2^19 hash table size offered optimal quality-performance tradeoff
   - Automatic adaptation to important details without explicit structure modifications

## Opinion

This paper represents a breakthrough in neural graphics primitives by addressing fundamental efficiency challenges through an elegant combination of hash tables and multiresolution encoding:

1. **Conceptual Strengths**:
   - The implicit hash collision resolution through gradient dominance is remarkably elegant
   - The multiresolution approach ensures both global context and fine detail
   - The O(1) lookups with predictable memory patterns enable extraordinary GPU performance

2. **Practical Impact**:
   - Transforms previously offline techniques into interactive tools
   - Enables rapid iteration for researchers and practitioners
   - Makes neural representations viable for real-time applications

3. **Future Directions**:
   - The microstructure artifact in SDFs suggests the need for improved smoothness constraints
   - Hash function optimization could potentially yield further improvements
   - Extending to generative models presents interesting research challenges
   - Applications to heterogeneous volumetric fields (clouds, smoke) show promise

The discussion revealed several nuanced aspects of the approach:

- The role of hash collisions in creating adaptive behavior without explicit structure modifications
- The tradeoff between memory efficiency and reconstruction quality
- The importance of d-linear interpolation for continuity
- The challenges in achieving higher-order smoothness for applications like SDFs

## Key Lessons from Our Discussion

• **Parametric Encodings** store trainable features outside the neural network itself, allowing smaller MLPs to achieve high-quality results while significantly accelerating training

• **Hash Table Design** ensures uniform distribution of features across the table through prime number multiplication and XOR operations, maximizing effective capacity

• **Resolution Hierarchy** spans from coarse (collision-free) to fine (detail-rich) levels, with each resolution contributing different strengths to the overall representation

• **Implicit Collision Resolution** allows the network to automatically prioritize important regions through gradient competition, eliminating need for complex data structure updates

• **O(1) Lookups** with predictable memory patterns enable optimal GPU utilization, avoiding the branching and pointer-chasing operations that slow down tree-based approaches

• **Small MLPs** with just 2 hidden layers and 64 neurons are sufficient when paired with the multiresolution hash encoding, dramatically reducing computation

• **Smoothness Considerations** are important for applications requiring continuous derivatives, achievable through smoothstep functions at some cost to reconstruction quality

• **Hash Collisions** cause competing gradients to average, with important features (those with larger gradients) naturally dominating less important ones

• **Microstructure Artifacts** are particularly visible in SDFs because normal calculation amplifies small discontinuities in the function through differentiation

• **Two Key Parameters** (T and Nmax) control the memory-quality tradeoff, with F=2 and L=16 found to be optimal across applications

• **Orders of Magnitude Speedup** results from both the compact encoding (allowing smaller networks) and GPU-friendly implementation with minimal control flow divergence

## Test Your Understanding

1. **Q: Why does the multiresolution hash encoding use a hash function with prime number multiplication and XOR operations rather than simple modulo of coordinates?**
   
   A: The prime-XOR hash function ensures uniform distribution of entries throughout the hash table, maximizing effective capacity. A simple modulo would create clustering of collisions for nearby points or regular structures, leaving many table entries unused. The hash function's purpose is to make efficient use of the limited table size by spreading features evenly across all entries.

2. **Q: How does the encoding achieve adaptivity without explicit data structure modifications during training?**
   
   A: When hash collisions occur, the gradients of colliding points average during backpropagation. Points with larger gradients (more important for the loss function) naturally dominate these averages, causing the hash table entries to prioritize important features. This creates implicit adaptivity where the system automatically focuses computational resources on regions with significant detail.

3. **Q: Why are SDFs more susceptible to the grainy artifacts caused by hash collisions than other applications?**
   
   A: SDFs require computing surface normals as the gradient (derivative) of the distance field. Taking derivatives amplifies small discontinuities in the original function. When visualized through shading models, these normal discontinuities become immediately visible as microstructure on surfaces. Other applications either directly predict color values or have integration steps that smooth out small discontinuities.

4. **Q: What is the total number of trainable parameters in the encoding, and how many parameters are updated for each sample?**
   
   A: The total number of trainable parameters is O(T) = L × T × F (all hash table entries). However, for each input sample, only 2^d × L × F parameters are updated, where d is the dimensionality (typically 2 or 3). This selective update is key to the encoding's efficiency – only a tiny fraction of parameters receive gradients for each sample.

5. **Q: How does the paper address the tradeoff between continuity and higher-order smoothness in the encoding?**
   
   A: The basic d-linear interpolation ensures C⁰ continuity (no jumps in function values) but allows discontinuities in derivatives. For applications requiring higher-order smoothness (C¹ continuity), the authors propose applying a smoothstep function S₁(x) = x²(3-2x) to the interpolation weights, forcing derivatives to zero at cell boundaries. This ensures smooth transitions but tends to reduce overall reconstruction quality.

6. **Q: What are the key differences between this approach and tree-based parametric encodings like NGLOD?**
   
   A: While both approaches are parametric encodings with multiresolution representations, the key differences are: (1) Hash tables use O(1) lookups while trees require traversal; (2) Hash encoding doesn't require structural updates during training; (3) The hash encoding is task-agnostic while tree structures are often task-specific; (4) Hash tables have predictable memory access patterns that are more GPU-friendly than pointer-chasing in trees.

7. **Q: Why is the hash encoding challenging to use in a generative setting?**
   
   A: Generative models typically expect features to be organized in a regular spatial pattern where nearby features correspond to nearby spatial locations. The hash encoding breaks this spatial correspondence, making it difficult for generator networks to produce coherent feature vectors. The generator would need to somehow understand the hash function mapping to produce features that make sense when accessed through that mapping.

8. **Q: What causes the performance cliff observed around T = 2¹⁹ in the experiments?**
   
   A: The performance cliff occurs when the hash table size exceeds the L2 cache size of the GPU (6MB on the authors' RTX 3090). When 2 · T · F > 6 · 2²⁰ bytes, the tables no longer fit entirely in cache, causing significantly more expensive memory accesses and slowing down computation. This hardware limitation creates a practical upper bound on efficient table sizes.