---
title: "Value function Iteration methods"
date: "2024-10-20"
type: "lecture"
subtype: null
topic: "Reinforcement learning (classical)"
source_title: null
source_author: null
source_url: null
---

We systematically apply Bellman's equation over and over again to obtain better estimates for the value function iteratively. Each iteration provides successively better approximations to the solution. A typical stopping condition for iterative policy evaluation is to measure by how little the largest change in value function was between two iteration steps and set a cut off threshold. 
![](/attachments/value-function-iteration-methods-Pasted-image-20241020115116.png)

# Value Iteration 
The Bellman Optimality Equation (Henceforth, BOE) $V^{\pi^*}(s) = \max_{a \in A}\Sigma_{s'}P_{ss'}^a(R_{ss'}^a + \gamma V^{\pi^*}(s'))$ is a so called fixed point equation.
As a result, we can start with some arbitrary value function V(...) and apply the right hand side of the BOE to said V(...) to get a new value function V'(...). Thus, the right hand side of the BOE defines an operator L on the space of value functions. The Bellman optimality equation states that the optimal value function is a fixed point $V^{\pi^*} = K(V^{\pi^*})$. A simple process to find a fixed point of K is as follows, Begin with any value function, apply K to it again and again until $K(V_n)=V_n$. The process is guarenteed to converge to a unique limit since K is a contraction $||K(V_1)-K(V_2)||_{\infty} \leq \gamma ||V_1-V_2||_{\infty}$ 
![](/attachments/value-function-iteration-methods-Pasted-image-20241020122023.png)
# Policy Iteration 
![](/attachments/value-function-iteration-methods-Pasted-image-20241020121334.png)
