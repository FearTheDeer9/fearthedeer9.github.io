---
title: "Effect of condition number of hessian matrix on convergence"
date: "2024-02-27"
type: "concept"
subtype: null
topic: "Continual learning & optimization"
source_title: null
source_author: null
source_url: null
---

The classic zig-zag picture is misleading since it suggests that the slow convergence is due to overshooting. But for ill-conditioned problems, convergence is still slow even on a purely quadratic function with perfect "ridge-line" line search, where you step to the exact ridge-line of the mountain so that there is no overshooting at all.
The correct intuition is that, since the gradient points in the steepest direction, gradient descent cannot effectively explore a direction in parameter space until it has already eliminated all other directions in parameter space that are steeper. So even in the best case scenario of perfect line search, it will take one iteration to eliminate the steepest direction, then the next iteration to eliminate the second steepest direction, then the next iteration to eliminate the third steepest direction, and so forth.
Imagine climbing up a mountain using the method of steepest ascent. You will not walk straight towards the peak. Rather, you will climb up the steepest face of the mountain until you reach a ridge, then follow the ridge to the top, making a dog-leg path.
![](/attachments/effect-of-condition-number-of-hessian-matrix-on-convergence-ScNMT.png) (image credit to Mountains to Sound Greenway Trust)
Were it possible to climb an N -dimensional mountain, the path of steepest ascent would first climb to the top of the (N − 1)-dimensional ridge perpendicular to the steepest direction, then to the top of the (N − 2)-dimensional ridge perpendicular to the first two steepest directions, and so on. The more the eigenvalues of the Hessian are separated, the more the path will resemble the set of N edges on a box, traversing from one corner of the box to the opposite corner. This is illustrated below for ill-conditioned functions 
 in increasing numbers of dimensions.
When climbing to a ridge, only minor progress will be made in the direction of future, less steep, ridges. This minor progress is what the classic convergence bounds for gradient descent rely on. The rate of this minor progress will be proportional to the ratio of steepnesses: the greater the difference in steepness, the less progress will be made on the future ridge during the process of climbing the current ridge. The eigenvalues of the Hessian characterize these different steepnesses. The condition number of the Hessian, being the ratio of the largest eigenvalue to the smallest, is the ratio of the steepest ridge's steepness to the shallowest ridge's steepness, which is why it enters into the convergence bounds.
2D
![](/attachments/effect-of-condition-number-of-hessian-matrix-on-convergence-CSx2a.png)
3D
![](/attachments/effect-of-condition-number-of-hessian-matrix-on-convergence-cOPZW.png)
![](/attachments/effect-of-condition-number-of-hessian-matrix-on-convergence-7Q1oE.png)
ND
![](/attachments/effect-of-condition-number-of-hessian-matrix-on-convergence-S0HzM.png)
This is discussed in greater detail in Section 3.1 of my Ph.D. thesis here:
