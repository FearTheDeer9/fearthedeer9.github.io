---
title: "Gradient Surgery for Multi-Task Learning"
date: "2026-08-16"
type: "paper-summary"
subtype: null
topic: "Continual learning & optimization"
source_title: "Gradient Surgery for Multi-Task Learning"
source_author: null
source_url: null
---

The main contribution here - a model-agnostic form of gradient surgery that project a task's gradient onto the normal plane of the gradient of any other task that has a conflicting gradient in multi-task setting. The authors show that in multi-task learning enbironments the gradients from the different tasks can often be 1) conflicting, 2)with high positive curvature and 3) have large differences in gradient magnitudes. This triad nullifies the optimizer's ability of reaching the joint valley and optimizing both. The authors propose that when two gradients are conflicting, one should alter them by projecting each onto the normal plane of the other, preventing intergering components. This is referred to *projecting conflicting gradients* or PCGrad. 
# PCGrad
## Definitions
**Definition 1.** *We define $\phi_{ij}$ as the angle between two task gradients $\mathbf{g}_i$ and $\mathbf{g}_j$. We define the gradients as **conflicting** when $\cos\phi_{ij} < 0$.*

**Definition 2.** *We define the **gradient magnitude similarity** between two gradients $\mathbf{g}_i$ and $\mathbf{g}_j$ as*
$$
\Phi(\mathbf{g}_i, \mathbf{g}_j) = \frac{2\|\mathbf{g}_i\|_2 \|\mathbf{g}_j\|_2}{\|\mathbf{g}_i\|_2^2 + \|\mathbf{g}_j\|_2^2}.
$$

When the magnitude of two gradients is the same, this value is equal to 1. As the gradient magnitudes become increasingly different, this value goes to zero.

**Definition 3.** *We define **multi-task curvature** as*
$$
\mathbf{H}(\mathcal{L}; \theta, \theta') = \int_0^1 \nabla\mathcal{L}(\theta)^T \nabla^2\mathcal{L}\big(\theta + a(\theta' - \theta)\big) \nabla\mathcal{L}(\theta)\, da,
$$
*which is the averaged curvature of $\mathcal{L}$ between $\theta$ and $\theta'$ in the direction of the multi-task gradient $\nabla\mathcal{L}(\theta)$.

## The Algorithm
---

**Require:** Model parameters $\theta$, task minibatch $\mathcal{B} = \{\mathcal{T}_k\}$

1: $\mathbf{g}_k \leftarrow \nabla_\theta \mathcal{L}_k(\theta)\ \ \forall k$
2: $\mathbf{g}_k^{\text{PC}} \leftarrow \mathbf{g}_k\ \ \forall k$
3: **for** $\mathcal{T}_i \in \mathcal{B}$ **do**
4: &nbsp;&nbsp;&nbsp;&nbsp;**for** $\mathcal{T}_j \overset{\text{uniformly}}{\sim} \mathcal{B} \setminus \mathcal{T}_i$ in random order **do**
5: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**if** $\mathbf{g}_i^{\text{PC}} \cdot \mathbf{g}_j < 0$ **then**
6: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*// Subtract the projection of $\mathbf{g}_i^{\text{PC}}$ onto $\mathbf{g}_j$*
7: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Set $\mathbf{g}_i^{\text{PC}} = \mathbf{g}_i^{\text{PC}} - \dfrac{\mathbf{g}_i^{\text{PC}} \cdot \mathbf{g}_j}{\|\mathbf{g}_j\|^2} \mathbf{g}_j$
8: **return** update $\Delta\theta = \mathbf{g}^{\text{PC}} = \sum_i \mathbf{g}_i^{\text{PC}}$

---
