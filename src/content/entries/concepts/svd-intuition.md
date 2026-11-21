---
title: "SVD intuition"
date: "2024-02-27"
type: "concept"
subtype: null
topic: "Math & CS foundations"
source_title: null
source_author: null
source_url: null
---

![](/attachments/svd-intuition-Pasted-image-20240227193111.png)This is in essence what SVD is, inspecting the last equation closely, one sees that what happens is that x is projected onto each one of the eigenvectors of $\Sigma$ , and then the resulting projection is used to scaled the vector that the eigenvector is mapped to by A. 
## The Power Method
![](/attachments/svd-intuition-Pasted-image-20240227194939.png)hence, by applying $A^TA$ to $v_i$ one gets $\sigma_i^2v_i$. From this follows the theorem leading to the power method:
"Let x be a random unit vector and define $B=A^TA$ then with high probability, $lim_{s\rightarrow \inf}B^sx$ is in the span of the first singular vector $v_1$.  If we normalize $B^sx$ to a unit vector at each s then furthermore the limit is $v_1$." 
* The proof is immediate from the fact that $Bx=B\Sigma_{i}a_iv_i = \Sigma_{i}\sigma_i^2a_iv_i$ and so by applying B infinity times and normalizing we get that the ratio of the coefficients of the result is $(\frac{\sigma_1}{\sigma_i})^{2s}\rightarrow \inf$ 
* In this way we can find v1 by applying B to a random unit vector as long as the ration between $B^sx \and B^{s+1}x$ is very small  (i.e it is no longer being rotated since it is almost entirely in the span of v1). 
* Afterwards, to complete the SVD algorithm all one needs to do is define $A^1=A-u_iv_i^T\sigma_i$ (i.e subtracting the rank-1 component of A corresponding to v1) and applying the same procedure.
	* Note that for the convergence of SVD using the power method, one assumes that the ratio of consecutive eigenvalues is not too close to 1, if that is the case the convergence may take a long time, as a result the power method is not the industry method to compute SVD, but rather a technique using something called housefelder reflections
