---
date: '2025-04-14'
layout: single
title: "Paper Review - pixelNeRF: Neural Radiance Fields from One or Few Images"
tags:
- 3D rendering
- neural radiance fields
- novel view synthesis
- computer vision
- machine learning

---

pixelNeRF presents a learning framework that enables predicting Neural Radiance Fields (NeRF) from just one or a few images in a feed-forward manner. This approach overcomes key limitations of the original NeRF, which requires many calibrated views and significant per-scene optimization time, by introducing an architecture that conditions a neural radiance field on image features in a fully convolutional manner.

## Key Innovation

The core insight of pixelNeRF is that by conditioning a neural radiance field on image features, the network can learn scene priors across multiple scenes, enabling it to:


<!-- excerpt-end -->

1. Perform novel view synthesis with very few input images (as few as one)
2. Achieve feed-forward inference without per-scene optimization
3. Generalize to unseen object categories and complex scenes

This architecture leverages the volume rendering approach of NeRF, allowing the model to be trained directly from multi-view images without explicit 3D supervision.

## Implementation

### Neural Radiance Fields Foundation

pixelNeRF builds upon NeRF, which represents a 3D scene as a continuous volumetric radiance field of color and density. For a 3D point x and viewing direction d, NeRF returns:

$$f(x, d) = (σ, c)$$

Where σ is density and c is RGB color. The volumetric field is rendered into 2D images via:

$$\hat{C}(r) = \int_{t_n}^{t_f} T(t)σ(t)c(t)dt$$

Where $T(t) = \exp(-\int_{t_n}^t σ(s)ds)$ handles occlusion along ray r from near bound $t_n$ to far bound $t_f$.

### Image-Conditioned Architecture

pixelNeRF introduces a fully convolutional image conditioning framework:

1. **Feature Extraction**: A CNN encoder E extracts a feature volume W = E(I) from input image I
    
2. **Point Projection**: For any query point x in 3D space, project it onto the image plane to get image coordinates π(x)
    
3. **Feature Sampling**: Retrieve the corresponding image feature via bilinear interpolation: W(π(x))
    
4. **Conditional NeRF**: Feed these features, along with position and viewing direction, to the NeRF network:
    
    $$f(γ(x), d; W(π(x))) = (σ, c)$$
    
    where γ(·) is positional encoding with exponentially increasing frequencies
    

### Multi-View Integration

For multiple input views, pixelNeRF:

1. Processes each view independently in its own coordinate system
2. Extracts intermediate feature vectors for each view: $V^{(i)} = f_1(γ(x^{(i)}), d^{(i)}; W^{(i)}(π(x^{(i)})))$
3. Aggregates these features using average pooling: $ψ(V^{(1)}, ..., V^{(n)})$
4. Feeds the aggregated features through the final network layers: $(σ, c) = f_2(ψ(V^{(1)}, ..., V^{(n)}))$

### Residual Feature Integration

To effectively incorporate image features, pixelNeRF:

- Uses a ResNet architecture for the NeRF network
- Adds transformed image features as residuals at the beginning of each ResNet block
- Trains separate linear layers to transform features appropriately for each block

## Results

pixelNeRF demonstrated impressive performance across several experimental scenarios:

1. **Single-Category View Synthesis**:
    
    - Outperformed state-of-the-art methods on ShapeNet chairs and cars
    - Achieved 23.72 PSNR for chairs (vs. 22.89 for SRN) in the single-view case
    - Showed even greater improvements in the two-view setting
2. **Category-Agnostic Reconstruction**:
    
    - Trained on 13 ShapeNet categories with a single model
    - Significantly outperformed baselines (26.80 PSNR vs. 23.28 for SRN)
    - Preserved fine details like chair textures, tabletop patterns, and thin structures
3. **Unseen Categories**:
    
    - Maintained strong performance on 10 categories not seen during training
    - Demonstrated true generalization capabilities beyond semantic recognition
4. **Multi-Object Scenes**:
    
    - Successfully reconstructed scenes with multiple objects
    - Leveraged view-space formulation to handle complex arrangements
5. **Real Images**:
    
    - Applied to DTU dataset, producing plausible novel views of real scenes
    - Showed sim-to-real transfer on Stanford Cars dataset without fine-tuning

## Theoretical Foundations

pixelNeRF's success stems from several key theoretical insights:

1. **View Space vs. Canonical Space**:
    
    - Operating in the coordinate system of the input camera rather than a canonical object-centric space
    - Enables generalization to unseen categories and multiple-object scenes
    - Removes the need for category-specific knowledge
2. **Local vs. Global Features**:
    
    - Using spatially-aligned local features preserves fine details
    - Maintains the spatial relationship between image pixels and 3D points
    - Allows reconstruction of complex patterns and textures
3. **Volume Rendering Physics**:
    
    - Leverages the physically-based volume rendering approach from NeRF
    - Enables proper handling of occlusion through transmittance calculation
    - Allows training from 2D supervision only

## My Opinion

The elegance of pixelNeRF lies in its reconciliation of two seemingly contradictory goals:

1. **Conceptual Strengths**:
    
    - Combines the photorealistic rendering capabilities of NeRF with the ability to generalize across scenes
    - Provides a pathway from single images to 3D representations without explicit 3D supervision
    - The view-space formulation elegantly sidesteps limitations of canonical space approaches
2. **Future Directions**:
    
    - Improving rendering speed remains an important challenge
    - Methods to extract meshes from the volumetric representation would increase utility
    - Extending to more complex real-world scenarios and larger scenes
3. **Open Questions**:
    
    - The optimal balance between learned priors and geometric consistency
    - How to handle scenes with incomplete or ambiguous visual information
    - Whether similar conditioning approaches could be effective for other neural representations

pixelNeRF represents a significant advancement in neural scene representations by enabling few-shot novel view synthesis without optimization, bridging the gap between 2D and 3D vision.

## Key Lessons

- **Feed-Forward vs. Optimization**: pixelNeRF enables feed-forward inference without per-scene optimization, unlike the original NeRF
    
- **View Space vs. Canonical Space**: Operating in view space allows generalization to unseen categories and complex scenes, while canonical space simplifies learning but restricts applications
    
- **Local Image Features**: Pixel-aligned local features preserve spatial details that would be lost in global representations
    
- **Volume Rendering**: The physically-based volume rendering approach enables training from 2D images without 3D supervision
    
- **Multi-View Fusion**: The architecture enables incorporation of multiple views through feature aggregation at an intermediate stage
    
- **Scene Priors**: Training across multiple scenes allows the network to learn generalizable priors about 3D structure
    
- **Projection Mechanism**: The explicit projection of 3D points to image coordinates creates a direct mapping between 3D space and image features
    
- **Residual Feature Integration**: Adding image features as residuals at multiple network depths enables effective integration of image information
    

# Test Your Understanding

## Q1: What fundamental limitation of Neural Radiance Fields does pixelNeRF address?

**A1**: pixelNeRF addresses NeRF's inability to generalize across scenes, which requires optimizing a separate network for each scene, demanding many calibrated views (often dozens) and significant per-scene optimization time (hours). pixelNeRF enables feed-forward novel view synthesis from as few as one image without per-scene optimization by conditioning the neural radiance field on image features and learning scene priors across multiple scenes.

## Q2: Explain the difference between view space and canonical space for 3D reconstruction.

**A2**: View space (used by pixelNeRF) performs reconstruction in the coordinate system of the input camera, while canonical space (used by most previous methods) aligns objects to a standardized orientation. View space offers better generalization to unseen categories and multiple-object scenes since it doesn't require knowing what the object is or its "standard" orientation. Canonical space simplifies learning by reducing orientation variance but limits applications to known categories and single objects with well-defined orientations.

## Q3: How does pixelNeRF integrate image features into the neural radiance field?

**A3**: pixelNeRF integrates image features through a multi-step process: First, a CNN extracts a feature volume from input images. For any 3D query point, it projects this point onto the image plane to determine which image features correspond to it. These features are sampled using bilinear interpolation. Then, these features are added as residuals at the beginning of each ResNet block in the NeRF network, using separate learned linear transformations for each block. This allows image information to influence computation at multiple network depths.

## Q4: How does pixelNeRF handle occlusion and what is the role of the transmittance term T(t)?

**A4**: pixelNeRF handles occlusion through volume rendering, using the same approach as the original NeRF. The transmittance term T(t) represents the probability that a ray travels from its origin to distance t without hitting any particles. It's calculated as T(t) = exp(-∫ₜₙᵗ σ(s)ds), where σ is density. In the discrete implementation, pixel color is approximated as a weighted sum where each point's contribution is scaled by both its density and the transmittance (probability of reaching that point). This naturally handles occlusion by reducing contributions from points behind dense regions.

## Q5: How does pixelNeRF enable reconstruction from multiple views?

**A5**: pixelNeRF handles multiple views through a specialized architecture that: (1) Processes each input view independently in its respective camera coordinate system, extracting feature vectors through the first part of the network; (2) Aggregates these intermediate feature vectors across views using average pooling; (3) Processes the combined features through the final layers of the network to predict density and color. This design allows incorporating information from multiple viewpoints while handling arbitrary numbers of input views and maintaining viewpoint-specific information.

## Q6: Why does pixelNeRF not require 3D supervision during training?

**A6**: pixelNeRF doesn't require 3D supervision because it leverages multi-view consistency and the physics of volume rendering. During training, it's given some views of a scene and asked to predict how the scene looks from other viewpoints. The loss compares rendered images to ground truth images. When rendered pixels match ground truth across multiple views, the underlying 3D geometry must be correct. The volume rendering equation naturally encourages accurate density predictions that produce the correct images when integrated. This self-supervised approach allows training solely from multi-view images without explicit 3D ground truth.

## Q7: How does pixelNeRF's feature extraction differ from methods that use global image features?

**A7**: pixelNeRF uses spatially-aligned local features rather than global image features. It maintains a feature grid where each spatial location corresponds to a region in the input image. When querying a 3D point, it projects that point onto the image to retrieve exactly the features corresponding to that location. This preserves fine spatial details and creates an explicit mapping between 3D space and image features. In contrast, global feature methods encode the entire image into a single vector, losing spatial correspondence and often producing results that resemble "retrieval" rather than detailed reconstruction.