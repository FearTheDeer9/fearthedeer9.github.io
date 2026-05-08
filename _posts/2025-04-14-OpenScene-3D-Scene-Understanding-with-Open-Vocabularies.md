---
date: '2025-04-14'
layout: single
title: "Paper Review - OpenScene: 3D Scene Understanding with Open Vocabularies"
tags:
- 3D scene understanding
- computer vision
- semantic segmentation
- zero-shot learning
- open-vocabulary querying
---

OpenScene presents a breakthrough approach to 3D scene understanding that eliminates reliance on labeled 3D data and enables open-vocabulary querying. By co-embedding 3D points with text and image pixels in the CLIP feature space, OpenScene can perform zero-shot 3D semantic segmentation and novel tasks like querying scenes for materials, affordances, and activities.

## Core Innovations

### Open-Vocabulary 3D Scene Understanding

Traditional 3D scene understanding relies on supervised learning with fixed label sets. OpenScene introduces:

- **Zero-shot learning**: No labeled 3D data required
- **Open-vocabulary querying**: Ability to use arbitrary text to query 3D scenes
- **Co-embedding**: 3D points, image pixels, and text exist in the same semantic feature space
- **Extended capabilities**: Beyond object categorization to materials, affordances, activities, and room types

### Co-Embedding with CLIP Feature Space

The key insight is aligning 3D point features with CLIP's rich semantic space:

<!-- excerpt-end -->


$$\text{similarity} = \cos(f^{3D}, t) = \frac{f^{3D} \cdot t}{||f^{3D}|| \cdot ||t||}$$

Where:

- $f^{3D}$ is the feature vector for a 3D point
- $t$ is the text embedding from CLIP's text encoder
- Cosine similarity measures semantic relevance in shared feature space

## Technical Approach

### Image Feature Fusion

For each 3D point, features from multiple camera views are aggregated:

1. **Camera projection**: Each 3D point $p$ is projected to pixel $u$ using camera matrices: $$\tilde{u} = I_i \cdot E_i \cdot \tilde{p}$$ where $I_i$ is the intrinsic matrix and $E_i$ is the extrinsic matrix
    
2. **Occlusion testing**: Ensures only visible points are considered by comparing:
    
    - The distance from camera to 3D point
    - The depth value stored at the projected pixel
3. **Feature aggregation**: Point feature is computed by averaging features from K views: $$f^{2D} = \phi(f_1, ..., f_K) = \frac{1}{K}\sum_{i=1}^{K} f_i$$
    

### 3D Distillation

A 3D convolutional network is trained to produce features directly from 3D geometry:

$$F^{3D} = \mathcal{E}^{3D}(P), \quad \mathcal{E}^{3D}: \mathbb{R}^{M\times 3} \mapsto \mathbb{R}^{M\times C}$$

The network is trained with a cosine similarity loss to match 2D fused features:

$$\mathcal{L} = 1 - \cos(F^{2D}, F^{3D})$$

This allows inference without 2D images while maintaining CLIP-aligned features.

### 2D-3D Ensemble

The final approach combines strengths of both 2D and 3D features:

1. Compute similarity to text prompts for both feature types: $$s^{2D}_n = \cos(f^{2D}, t_n), \quad s^{3D}_n = \cos(f^{3D}, t_n)$$
    
2. For each 3D point, determine which feature gives highest confidence: $$s^{2D} = \max_n(s^{2D}_n), \quad s^{3D} = \max_n(s^{3D}_n)$$
    
3. Select the feature with higher maximum similarity
    

## Mathematical Properties

### Feature Space Alignment

The method leverages transitive property of embedding spaces:

- CLIP aligns text and images
- Image features are projected to 3D points
- 3D network learns to reproduce these features

This creates a three-way alignment between 3D points, images, and text.

### Zero-Shot Transfer

Since features are embedded in CLIP space, the model can recognize:

- Concepts never seen during training
- New datasets without retraining
- Arbitrary queries expressible in natural language

### Spatial Consistency

The 3D convolutional backbone ensures:

- Local geometric patterns are captured
- Feature consistency across viewpoints
- Robustness to occlusion and viewpoint changes

## Key Insights from Results

### Performance with Increasing Classes

- **Supervised methods**: Performance drops sharply with more classes (64.5% → 18.4% mAcc)
- **OpenScene**: Degrades gracefully (59.2% → 23.1% mAcc)
- On rare classes, OpenScene outperforms supervised methods significantly

### 2D-3D Feature Selection Patterns

- ~70% of points use 3D features (better for common geometric structures)
- 2D features preferred for:
    - Small objects
    - Objects with ambiguous geometry
    - Rare classes with limited examples

### Novel Applications

The open-vocabulary nature enables unprecedented applications:

- Object search with text queries ("find a teddy bear")
- Material and property recognition ("wooden," "soft")
- Affordance identification ("places to sit")
- Activity site localization ("where to cook")
- Room type detection ("kitchen," "bedroom")

## Points to Remember

• **Co-embedding vs. Classification**: OpenScene produces features that live in CLIP's semantic space rather than directly predicting class probabilities, enabling open-vocabulary queries.

• **Camera Intrinsics and Extrinsics**: Precise 3D-to-2D projection using calibrated camera parameters ensures accurate feature fusion.

• **Occlusion Handling**: Depth testing prevents incorrect feature association for points not visible from a particular viewpoint.

• **Feature Selection**: The 2D-3D ensemble dynamically selects the most confident feature modality for each point and concept.

• **No Ground Truth Dependency**: Unlike previous approaches, OpenScene never requires 3D labeled data, even for initial training.

• **Transfer Learning Approach**: Previous methods like Rozenberszki used CLIP only for pretraining, while OpenScene maintains CLIP alignment throughout the pipeline.

• **Model Generality**: The same model works for any number of classes without retraining, unlike supervised methods that require retraining for new label sets.

• **Domain Transfer**: Models trained on one dataset can be applied to new datasets with different environments and label sets.

• **Focal Length and Principal Point**: Camera intrinsic parameters calibrated through known patterns determine the exact projection from 3D to 2D.

• **Complementary Features**: 2D features excel at recognizing visual details and texture, while 3D features better capture geometry and structure.

# Test Your Understanding

## Fundamental Concepts

**Q1: What is the key difference between OpenScene and traditional 3D scene understanding approaches?**

A1: Traditional approaches require labeled 3D datasets and can only recognize predefined categories they were trained on. OpenScene maps 3D points into CLIP's feature space, enabling zero-shot recognition of arbitrary concepts without any labeled 3D data. This allows open-vocabulary querying beyond just object categories.

**Q2: Explain what "co-embedding" means in the context of OpenScene.**

A2: Co-embedding means mapping 3D points, image pixels, and text into the same shared high-dimensional feature space (CLIP space) where semantically similar items are located close to each other. This allows cross-modal understanding - a 3D point on a couch, an image pixel showing a couch, and the text "couch" would all have similar vector representations, enabling text-based queries of 3D scenes.

**Q3: How does OpenScene enable zero-shot capabilities without labeled 3D data?**

A3: OpenScene leverages pre-trained 2D vision-language models (OpenSeg/LSeg) by projecting 3D points onto 2D images and collecting corresponding pixel features. It then trains a 3D network to predict these CLIP-aligned features using only geometry as input. By maintaining alignment with CLIP's semantic space, it inherits all of CLIP's knowledge about visual concepts and language, enabling arbitrary text queries without needing 3D labels.

## Technical Details

**Q4: Walk through how a 3D point gets associated with features from 2D images.**

A4: First, the 3D point is projected to pixel coordinates using camera matrices (u = I·E·p). Then, occlusion testing is performed by comparing the distance from camera to point against the depth value at that pixel. If the point is visible, the feature vector from the pixel (extracted using OpenSeg/LSeg) is associated with the 3D point. This process repeats for multiple camera views, and features are aggregated using average pooling to get a final fused 2D feature for the 3D point.

**Q5: What is the loss function used for 3D distillation and why is this choice appropriate?**

A5: The loss function is L = 1 - cos(F²ᴰ, F³ᴰ), measuring the cosine dissimilarity between 2D fused features and 3D predicted features. This is appropriate because: 1) CLIP features are normalized and compared using cosine similarity, 2) it focuses on feature direction rather than magnitude, 3) it directly aligns with how features will be compared at inference time, and 4) it's invariant to scaling, helping with optimization stability.

**Q6: How does the 2D-3D ensemble method decide which feature to use for each point?**

A6: For each 3D point, the system computes the similarity of both its 2D and 3D features to all text prompts in the query set. It takes the maximum similarity score for each feature type (s²ᴰ = max(cos(f²ᴰ, tₙ)) and s³ᴰ = max(cos(f³ᴰ, tₙ))). It then selects the feature (2D or 3D) that has the higher maximum similarity score, effectively choosing the feature that more confidently matches some semantic concept.

## Cameras and Projection

**Q7: What is the principal point in camera intrinsics, and how is it determined?**

A7: The principal point (cx, cy) is the location where the optical axis intersects the image plane, essentially where "straight ahead" from the camera maps on the image. In an ideal camera, this would be exactly the center of the image. It's determined through camera calibration - a process where multiple images of a known pattern (like a checkerboard) are analyzed to solve an optimization problem that finds the intrinsic parameters that best explain the observed projections.

**Q8: Why is occlusion testing important when associating 2D pixels with 3D points?**

A8: Occlusion testing prevents incorrectly associating 3D points with pixels that can't actually see them. When a 3D point is projected to a 2D image, there might be other objects between the camera and that point. By comparing the projected distance with the stored depth value at that pixel, we can determine if the point is actually visible. Without this test, features would be incorrectly assigned to occluded points, introducing noise in the feature fusion process.

## Comparison and Analysis

**Q9: How does OpenScene differ from Rozenberszki's approach in how they use CLIP features?**

A9: Rozenberszki uses CLIP only for pretraining a 3D encoder, where the classification weights are CLIP text embeddings. After pretraining, this CLIP connection is discarded, and the system is fine-tuned with 3D ground truth annotations, limiting it to predefined categories. OpenScene maintains direct alignment with CLIP's feature space throughout the entire pipeline, producing features that can be directly compared with any text embedding, preserving CLIP's open-vocabulary capabilities.

**Q10: Why does OpenScene perform better than fully-supervised methods on rare classes?**

A10: OpenScene performs better on rare classes because: 1) It leverages knowledge from CLIP's large-scale vision-language pretraining, 2) It doesn't rely on 3D labeled examples which are scarce for rare classes, 3) CLIP's feature space already contains knowledge about thousands of concepts, 4) Supervised methods struggle with class imbalance where rare classes have few training examples, and 5) The ensemble approach particularly favors 2D features for rare classes, which better capture visual details.

## Applications and Implications

**Q11: Describe three novel applications enabled by OpenScene beyond traditional 3D semantic segmentation.**

A11: 1) Open-vocabulary 3D object search - finding specific objects like "teddy bear" in a 3D scene database, 2) Material and physical property recognition - identifying surfaces that are "wooden," "metal," "soft," or "comfy," 3) Affordance and activity site identification - locating places in a scene where someone could "sit," "cook," "sleep," or "work," 4) Room type classification without explicit labels - identifying "kitchen," "bedroom," "bathroom" areas, 5) Abstract concept queries - finding "cluttered" or "open" spaces.

**Q12: How might OpenScene be extended to handle temporal data like videos?**

A12: OpenScene could be extended to handle videos by: 1) Incorporating temporal consistency constraints between frames, 2) Adding recurrent or attention mechanisms to track features over time, 3) Using video-based CLIP variants that understand motion and temporal concepts, 4) Leveraging the improved multi-view coverage that video provides, and 5) Enabling temporal queries like "person walking" or "door opening" by mapping spatiotemporal patterns to CLIP space.

**Q13: What are the main limitations of OpenScene mentioned in the paper?**

A13: The main limitations include: 1) Performance gap compared to SOTA supervised methods on standard benchmarks with common classes, 2) Limited success with earlier fusion approaches between 2D and 3D, 3) Lack of quantitative evaluation for novel open-vocabulary tasks due to absence of ground truth, 4) Dependency on pre-trained 2D models like OpenSeg and LSeg, and 5) Need for better methods to take advantage of pixel features when images are present at test time.

**Q14: How does the performance of OpenScene change as the number of classes increases, and why?**

A14: As the number of classes increases from K=21 to K=160, OpenScene's performance degrades much more gracefully than supervised methods (59.2% → 23.1% vs 64.5% → 18.4% mAcc). For K=21, supervised methods outperform OpenScene, but for K≥40, OpenScene begins to outperform them. This occurs because supervised methods struggle with limited examples for rare classes, while OpenScene leverages CLIP's broad semantic knowledge that already contains these concepts, performing more consistently across both common and rare classes.

**Q15: In the future, how might you design experiments to quantify the success of open-vocabulary queries where ground truth is not available?**

A15: Potential approaches include: 1) Creating small-scale datasets with manual annotations for novel tasks like material recognition or affordance detection, 2) Designing user studies where humans evaluate the quality of results for subjective queries, 3) Developing proxy metrics that correlate with performance but don't require exhaustive labeling, 4) Using cross-modal validation where results are compared with other modalities like text descriptions, 5) Creating synthetic environments with known ground truth for novel properties.