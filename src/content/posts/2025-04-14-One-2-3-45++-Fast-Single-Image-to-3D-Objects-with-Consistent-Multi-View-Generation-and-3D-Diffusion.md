---
date: '2025-04-14'
title: "Paper Review - One-2-3-45++: Fast Single Image to 3D Objects with Consistent Multi-View Generation and 3D Diffusion"
layout: single
tags:
- single image to 3D
- 3D generation
- multi-view consistency
- diffusion models
---

One-2-3-45++ presents a breakthrough approach for transforming a single image into a high-quality 3D textured mesh in approximately one minute. This method bridges the gap between image-based and 3D modeling by combining the power of 2D diffusion models with 3D native diffusion, offering both rapid generation and high fidelity to input images.

## Key Innovation

The core innovations of One-2-3-45++ address two fundamental challenges in image-to-3D conversion:

1. **Consistent Multi-View Generation**: A novel approach to generate multiple coherent views of an object from a single image
2. **3D Diffusion with Multi-View Conditioning**: A two-stage 3D diffusion process guided by multi-view images
3. **Texture Refinement**: A lightweight optimization technique to enhance texture quality
4. **End-to-End Pipeline**: Integration of these components into a system that produces high-quality 3D meshes in under one minute

<!-- excerpt-end -->


This combination enables both speed and quality that exceeds previous state-of-the-art methods, which typically required either extensive optimization time (tens of minutes to hours) or produced lower-quality results.

## Implementation

### Consistent Multi-View Generation

The method generates coherent multi-view images through an innovative approach:

- **Multi-View Tiling**: Six views are arranged in a 3×2 grid layout and generated simultaneously in a single diffusion process
- **Camera Pose Configuration**: Uses fixed absolute elevation angles (30° and -20°) combined with relative azimuth angles (starting at 30° and incrementing by 60°)
- **Network Architecture**:
    - Local conditioning using reference attention
    - Global conditioning using CLIP image embeddings
    - Linear noise schedule for fine-tuning

This approach enables views to interact during diffusion, producing consistent multi-view images that serve as a foundation for 3D reconstruction.

### 3D Diffusion with Multi-View Conditioning

The 3D generation process uses native 3D diffusion with multi-view guidance:

- **3D Volume Representation**:
    
    - SDF volume: Captures signed distance from grid cells to the nearest surface
    - Color volume: Stores color information for the surface
- **Two-Stage Diffusion**:
    
    - Stage 1: Low-resolution (64³) full occupancy volume to capture overall shape (dense convolution)
    - Stage 2: High-resolution (128³) sparse volume for fine-grained details (sparse convolution)
- **Multi-View Conditioning**:
    
    - 2D patch features are extracted from multi-view images using DINOv2
    - Features are projected into 3D space based on known camera poses
    - Aggregation of 2D features through a shared MLP and max pooling
    - 3D feature volumes are concatenated with UNet features during diffusion
- **Mathematical Conditioning**:
    
    - The diffusion process is conditioned by modifying the neural network's prediction
    - Conditioning happens through feature concatenation at multiple resolution levels
    - The process is defined as: `Lx0 = Eϵ∼N(0,I),t∼U(0,1) ||f(xt, t, c) - x0||²`

### Texture Refinement

To enhance texture detail beyond what the 3D diffusion can produce:

- Fixed geometry from the diffusion process
- Optimization of a color field represented by TensoRF
- Multi-view images guide the texture optimization using L2 loss
- Surface normal serves as the viewing direction when baking the color

## Results

One-2-3-45++ demonstrated exceptional performance in comparisons:

1. **Image-to-3D Performance**:
    
    - Superior F-Score (93.6%) and CLIP similarity (81.0%) compared to other methods
    - Overwhelmingly preferred in user studies (87.6% preference rate)
    - Generates results in ~60 seconds compared to minutes or hours for other methods
2. **Text-to-3D Performance**:
    
    - Higher CLIP similarity (26.8%) than optimization-based and feed-forward methods
    - 84.1% user preference rate compared to other approaches
    - Orders of magnitude faster than optimization-based approaches
3. **Qualitative Results**:
    
    - High-fidelity meshes that accurately reflect the input images
    - Diverse results across a wide range of object categories
    - Consistent texturing and geometry across viewpoints

## Theoretical Foundations

The method's success is grounded in several key theoretical insights:

1. **Consistency Through Joint Generation**: Generating multi-view images simultaneously allows for inter-view communication and consistency.
    
2. **Absolute vs. Relative Camera Poses**: Using absolute elevation but relative azimuth addresses the lack of canonical orientation while avoiding elevation estimation errors.
    
3. **3D-Native Diffusion**: Directly learning a mapping from multi-view images to 3D volumes leverages both 2D priors and 3D data efficiently.
    
4. **Dense vs. Sparse Representation**: Using dense convolutions for rough shape and sparse convolutions for details balances computational efficiency and quality.
    
5. **Feature Integration**: The mathematical foundation of conditioning diffusion models through feature concatenation allows for effective guidance from 2D to 3D.
    

## Key Lessons

- **Consistency Matters**: Generating consistent multi-view images is crucial for high-quality 3D reconstruction.
    
- **Two-Stage Diffusion**: Separating coarse and fine detail generation allows computational resources to be focused where needed.
    
- **Hybrid Approach**: Combining the strengths of 2D diffusion models (rich priors) with 3D native models (spatial understanding) creates a powerful synergy.
    
- **Sparse Computation**: For high-resolution 3D, focusing computation on occupied regions is essential for efficiency.
    
- **End-to-End Integration**: A carefully designed pipeline can maintain quality while dramatically reducing generation time.
    
- **Camera Pose Configuration**: The hybrid approach of absolute elevation and relative azimuth angles elegantly resolves orientation ambiguity.
    

# Test Your Understanding

## Q1: How does One-2-3-45++ fundamentally differ from previous image-to-3D methods like Zero123 + One-2-3-45 in terms of multi-view generation?

**A1**: Previous methods like Zero123 generate each view independently, modeling the conditional marginal distribution for each view in isolation. This leads to inconsistencies across views since there's no communication between the different view generation processes. One-2-3-45++ fundamentally changes this by simultaneously generating all six views arranged in a 3×2 grid layout as a single image. By tiling the views together in one diffusion process, the model can attend to all views during generation, ensuring cross-view consistency. This consistency is crucial for downstream 3D reconstruction, as inconsistent views lead to artifacts and incorrect geometry. Additionally, One-2-3-45++ uses a combination of absolute elevation angles and relative azimuth angles to define camera poses, which resolves orientation ambiguity without requiring estimation of the input image's elevation.

## Q2: Explain how the camera pose configuration in One-2-3-45++ works and why the authors chose to use absolute elevation angles but relative azimuth angles.

**A2**: The camera pose configuration in One-2-3-45++ uses fixed absolute elevation angles (30° and -20°) combined with relative azimuth angles (starting at 30° and incrementing by 60° for each subsequent pose). This hybrid approach addresses two challenges: First, using fully absolute camera poses would be problematic because 3D shapes in training data lack aligned canonical poses, creating ambiguity for the generative model. Second, using fully relative poses (like Zero123) would require downstream applications to first infer the elevation angle of the input image to deduce camera poses for the multi-view images, introducing potential errors. The hybrid approach works because most objects have a consistent up/down orientation due to gravity, even without a standardized front/back orientation. This means absolute elevation can be determined relative to the horizontal plane (with positive angles looking down and negative angles looking up), while azimuth needs to be relative to the input view since there's no universal "front" for many objects. This elegant solution resolves orientation ambiguity without requiring additional elevation estimation steps.

## Q3: Describe the two-stage diffusion process in One-2-3-45++ and explain why it's more efficient than a single-stage approach.

**A3**: One-2-3-45++ employs a two-stage diffusion process: Stage 1 generates a low-resolution (64³) full occupancy volume to capture the overall shape using dense convolution, while Stage 2 generates a high-resolution (128³) sparse volume focused on fine-grained SDF values and color within the occupied area using sparse convolution. This approach is more efficient than a single-stage high-resolution diffusion for several reasons: (1) Computational efficiency: High-resolution 3D grids require substantial memory and computation; using full resolution everywhere would be prohibitively expensive, (2) Focused detail: Most of the interesting detail in 3D shapes is concentrated near surfaces, making full-volume high-resolution computation wasteful, (3) Hierarchical refinement: The coarse-to-fine approach naturally aligns with how shapes are perceived and constructed, (4) Memory optimization: By using sparse convolution in the second stage, computation is only performed on the ~5% of voxels near surfaces instead of the entire volume. The first stage identifies the approximate location and shape of the object, and the second stage refines it with high-resolution details only where needed. This coarse-to-fine strategy achieves both efficiency and quality by allocating computational resources intelligently.

## Q4: How does the conditioning mechanism work in the 3D diffusion process of One-2-3-45++, and how does it differ from methods like DreamFusion?

**A4**: In One-2-3-45++, the 3D diffusion process is conditioned on multi-view images through a feature integration mechanism that works in the opposite direction from methods like DreamFusion. While DreamFusion projects 3D representations to 2D and uses 2D priors to guide optimization, One-2-3-45++ lifts 2D information to 3D to guide the diffusion process. Mathematically, the conditional diffusion is expressed as: `Lx0 = Eϵ∼N(0,I),t∼U(0,1) ||f(xt, t, c) - x0||²` where c is the multi-view conditioning.

The conditioning works by: (1) Extracting 2D patch features from each multi-view image using DINOv2, (2) Projecting each 3D voxel onto the multi-view images using known camera poses, (3) Gathering corresponding 2D patch features from views where the voxel is visible, (4) Aggregating these features through a shared-weight MLP and max pooling, (5) Creating 3D conditional feature volumes at multiple resolutions, and (6) Concatenating these with UNet feature maps during diffusion.

This approach doesn't directly modify voxel values but instead influences the neural network's prediction of what the clean data should be. By integrating multi-view information directly into the 3D features, the model can generate geometry and appearance that are consistent with the 2D views. This differs fundamentally from DreamFusion's approach, which optimizes a 3D representation by rendering it to 2D and computing loss/gradients in the 2D domain.

## Q5: What is sparse convolution and why is it crucial for the second stage of diffusion in One-2-3-45++?

**A5**: Sparse convolution is a specialized convolution operation that only processes "active" voxels (typically those near surfaces) rather than the entire 3D volume. It's implemented using specialized data structures (like hash tables, coordinate lists, or octrees) that efficiently store and process only the occupied positions. When performing a convolution on an active voxel, the algorithm identifies which neighboring positions within the kernel's range contain active voxels, applies the convolution only to those positions, and handles missing neighbors through masking, zero-padding, or dynamic normalization.

Sparse convolution is crucial for the second stage of diffusion in One-2-3-45++ for several reasons: (1) Computational efficiency: A dense 128³ volume requires processing 2.1M voxels, but with sparse convolution focusing on the ~5% near surfaces, only ~100K voxels need processing, (2) Memory efficiency: Storing only active voxels dramatically reduces memory requirements, enabling higher resolution, (3) Detail preservation: The high resolution enabled by sparse computation allows for capturing fine geometric and texture details crucial for visual quality, (4) Adaptive allocation: Computational resources are automatically focused on complex regions where detail matters most.

Without sparse convolution, the second stage would either need to operate at a much lower resolution (sacrificing detail) or would require prohibitively expensive computation and memory resources. The two-stage approach with sparse convolution in the second stage represents an elegant solution that achieves both high quality and reasonable computational requirements.

## Q6: Describe the texture refinement process in One-2-3-45++ and explain why it's beneficial even after the 3D diffusion generates colored meshes.

**A6**: The texture refinement process in One-2-3-45++ enhances the visual quality of the generated mesh through a lightweight optimization step. After extracting a mesh from the diffusion-generated volumes, the method: (1) Keeps the geometry of the mesh fixed, (2) Optimizes a color field represented by a TensoRF (Tensorial Radiance Field), (3) Renders the mesh with the current color field through rasterization, (4) Computes L2 loss between the rendered result and the multi-view images, (5) Updates the color field based on the gradient, and (6) Finally bakes the optimized color field onto the mesh using surface normals as viewing directions.

This refinement is beneficial even after 3D diffusion for several reasons: First, the multi-view images typically have higher resolution than the 3D color volume from diffusion (128³), allowing the optimization to capture finer texture details. Second, the diffusion process might introduce slight inconsistencies or blurring in colors due to the volume representation and discretization artifacts. Third, the optimization can correct view-dependent effects and improve shading consistency across the surface. Fourth, by fixing the geometry and only optimizing color, the process is computationally efficient while still significantly improving visual quality.

The combination of 3D diffusion for robust geometry and coarse coloring, followed by targeted texture refinement, enables One-2-3-45++ to achieve high visual fidelity while maintaining its speed advantage over methods that require lengthy optimization.

## Q7: How does One-2-3-45++ balance the trade-off between generation speed and fidelity to the input image? Compare its approach to optimization-based methods like SyncDreamer or DreamGaussian.

**A7**: One-2-3-45++ achieves an exceptional balance between speed and fidelity through a carefully designed pipeline that combines the best aspects of both feed-forward and optimization-based approaches:

For speed, One-2-3-45++ employs: (1) Feed-forward generation of consistent multi-view images in a single pass, (2) Efficient two-stage 3D diffusion using pre-trained networks, (3) Sparse computation for high-resolution details, and (4) Lightweight, targeted texture optimization focused only on appearance.

For fidelity, it incorporates: (1) Multi-view consistency through simultaneous generation, (2) 3D-native diffusion trained on extensive 3D data, (3) Both local and global conditioning mechanisms, and (4) Final texture refinement supervised by the generated views.

In contrast, optimization-based methods like SyncDreamer or DreamGaussian: (1) Optimize a 3D representation from scratch for each input, (2) Require many iterations (typically thousands) of rendering and gradient updates, (3) Often struggle with the "multi-face" or Janus problem, and (4) Can produce over-saturated colors or artifacts from their representation.

The fundamental difference is that One-2-3-45++ leverages pre-trained knowledge and explicit conditioning rather than starting from random noise and optimizing from scratch. This allows it to achieve comparable or better fidelity (93.6% F-Score vs. 84.8% for SyncDreamer) in a fraction of the time (60 seconds vs. 6 minutes for SyncDreamer or 2 minutes for DreamGaussian). The approach demonstrates that with the right architecture and training, feed-forward methods with minimal optimization can match or exceed the quality of methods requiring extensive per-shape optimization.