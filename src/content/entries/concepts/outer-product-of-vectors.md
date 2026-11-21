---
title: "Outer Product of Vectors"
date: "2024-02-27"
type: "concept"
subtype: null
topic: "Math & CS foundations"
source_title: "Tensorphobia and the Outer Product"
source_author: "Jeremy Kun"
source_url: "https://jeremykun.com/2016/03/28/tensorphobia-outer-product/"
---

* The basic confusion over the meaning of an outer product is that the type don’t match, you start off with two vectors which can commonly be thought of as objects and you finish with a matrix that is a mapping. The reason for why this is the correct way to turn two vectors into a matrix is unclear, leaving me unsatisfied.
* The first thing we must do to build intuition is to recognize that the canonical way to create a linear mapping from a single vector is to define $v \rightarrow  <v \cdot  \_ >$   . That is we take the inner product of v with w for any input vector w. 
* The next thing to remember is that a vector space V and its dual-vector space V* are isomorphic
* At this point, we have a connection between single vectors and the dual space. We now want to create a similar connection between $V\times V \rightarrow Hom(V,V)$  . This has two problems
	* The dimension of $V \times V$ is 2n while the dimension of $Hom(V,V)$ is $n^2$ 
	* The only way to embed $V \times V$ in $Hom(V,V)$  is by mapping v,w to the first two rows of the matrix with zeros elsewhere
 However, once we remember our map from V to V* we can think of defining the map:
 $\beta_{v,w}(x)=\phi_v(x)w = <x \cdot v > w$ that is, we project x onto v and then scale w by the result. And by remembering that V and V* are isomorphic we have a map $V\times V = V \times V* \rightarrow Hom(V,V)$ 
 * In our previous point we defined B: $V\times V \rightarrow Hom(V,V), B(v,w)=\beta_{v,w}$ we want to show that B is an isomorphism of vector spaces. 
 * ![](/attachments/outer-product-of-vectors-Pasted-image-20240227112411.png)
 * ![](/attachments/outer-product-of-vectors-Pasted-image-20240227112350.png)
