---
date: '2025-04-14'
layout: single
title: "Paper Review - LION: Latent Point Diffusion Models for 3D Shape Generation"
categories:
  - paper-summary
tags:
- 3d shape generation
- variational autoencoders
- denoising diffusion models
- point clouds
- generative models
---

## Overview
This paper introduces LION (Latent Point Diffusion Model), a novel approach for 3D shape generation that combines variational autoencoders (VAEs) with denoising diffusion models (DDMs) in latent space. The authors aim to create a 3D generative model that satisfies three key requirements for digital artists: high-quality shape generation, flexibility for manipulation, and the ability to output smooth meshes. LION outperforms previous state-of-the-art methods on various benchmarks and enables multiple applications such as multimodal shape generation, voxel-guided synthesis, and shape interpolation.

## Architecture

LION employs a hierarchical framework with two main components:

1. **Hierarchical VAE Structure**:
   - **Global Shape Latent (z₀)**: A vector representation that captures overall shape information
   - **Point-structured Latent (h₀)**: A point cloud structure with 3D coordinates and additional features that represents local details
   - The point latent h₀ is conditioned on the global shape latent z₀, creating a hierarchical relationship

2. **Latent Diffusion Models**:
   - One diffusion model trained on the global shape latent z₀
   - A second diffusion model trained on the point-structured latent h₀, conditioned on z₀
   - Both models operate entirely in latent space rather than directly on point clouds

<!-- excerpt-end -->


The model uses Point-Voxel CNNs (PVCNNs) with adaptive Group Normalization for processing point clouds. The adaptive Group Normalization serves as the conditioning mechanism, allowing the global shape latent to influence the point latent processing.

## Training Process

LION is trained in two separate stages:

1. **First Stage**: Train the VAE with a modified ELBO objective
   - Optimize encoder and decoder parameters using both reconstruction loss and KL divergence terms
   - Regularize both latent spaces toward standard Gaussian distributions

2. **Second Stage**: Freeze the VAE and train the diffusion models in latent space
   - Train the global shape latent diffusion model
   - Train the conditional point latent diffusion model
   - Use a mixed score parametrization where the models predict corrections to an analytic Gaussian score

This separation of training stages improves stability and performance, allowing the diffusion models to focus on modeling the remaining mismatch between the VAE's latent distributions and perfect Gaussians.

## Key Innovations

1. **Latent Space Diffusion**: By applying diffusion in latent space rather than directly on point clouds, LION achieves better generation quality and more flexibility
   
2. **Hierarchical Structure**: The separation of global shape and local details enables targeted editing and natural disentanglement

3. **VAE Framework**: The encoder-decoder architecture allows encoding of arbitrary inputs (like voxelized shapes) into the latent space, enabling various applications without retraining

4. **Mesh Reconstruction**: LION integrates with Shape As Points (SAP) to generate smooth meshes from the generated point clouds

## Applications

LION enables several practical applications:

1. **Multimodal Generation**: Using the diffuse-denoise technique, LION can generate multiple variations of a shape by:
   - Encoding a shape into latent codes
   - Diffusing these codes for a small number of steps
   - Running the reverse generation process to create variations with different details

2. **Voxel-guided Synthesis**: By fine-tuning the encoders, LION can take voxelized inputs and generate detailed outputs that respect the voxel constraints

3. **Shape Denoising**: Similar to voxel-guided synthesis, LION can denoise noisy point clouds in a multimodal fashion

4. **Shape Interpolation**: LION enables smooth interpolation between different shapes by interpolating in latent space

5. **Unconditional Generation**: LION can generate diverse and high-quality shapes from multiple classes without conditioning

## Results

The paper demonstrates that LION achieves state-of-the-art performance on ShapeNet benchmarks, outperforming previous point cloud generative models including other diffusion-based approaches like PVD and DPM. Key results include:

1. Better quantitative metrics (1-NNA with both Chamfer distance and Earth Mover's distance)
2. High-quality generation even when trained jointly over multiple classes without conditioning
3. Successful generation with small datasets
4. Effective voxel-guided synthesis and shape denoising
5. Fast generation using DDIM sampling (under one second per shape)

## Technical Implementation Details

1. **Mixed Score Parametrization**: The diffusion models use a hybrid approach combining an analytic Gaussian score with a learned correction
2. **Accelerated Sampling**: DDIM sampling enables generation in under one second instead of ~27 seconds
3. **PVCNNs**: Efficiently combine point-based and voxel-based processing for better feature extraction
4. **Adaptive Group Normalization**: Enables conditioning of point processing on the global shape latent

## Advantages Over Traditional DDMs

Traditional DDMs applied directly to point clouds struggle with the three criteria for digital artists because:

1. They lack encoders that can map arbitrary inputs into the generative process
2. They don't naturally separate global structure and local details
3. They generate only point clouds without a direct path to mesh output

LION's two-layer approach (VAE + latent diffusion) addresses these limitations by:

1. Providing encoders that can map various inputs to the latent space
2. Creating a hierarchical structure that separates shape and details
3. Enabling integration with mesh reconstruction methods

## Limitations and Future Work

While LION advances 3D shape generation significantly, the authors note several limitations:

1. The model currently focuses only on single object generation
2. It doesn't directly generate textured shapes
3. Generation could be further accelerated

Future directions mentioned include incorporating image-based training through differentiable rendering, extending to full 3D scene synthesis, and integrating texture generation.