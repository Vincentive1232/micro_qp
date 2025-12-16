# micro_qp — Math Induction

## 1. Problem Setup

The standard convex QP that can be solved by `micro_qp` can be written as:
$$
\begin{aligned}
\min_{x \in \mathbb{R}^n} \quad & \tfrac{1}{2}x^\top Hx + f^\top x \\
\text{s.t.} \quad & l \le Ax \le u
\end{aligned}
$$
where:
$$
\begin{aligned}
H &\succeq 0 &&: \text{Symmetric Positive Definite/Semi Positive Definite} \\
f &\in \mathbb{R}^n &&: \text{Linear term} \\
A &\in \mathbb{R}^{m \times n} &&: \text{Constraints matrix} \\
l,u &\in \mathbb{R}^m &&: \text{Lower and upper limit（could be } \pm \infty \text{）}
\end{aligned}
$$
---
## 2. ADMM Formulation

### Introducing an auxiliary variable 
Let
$$
z = Ax
$$
Then the original constraints $l \le Ax \le u$ is equivalent to:
$$
z \in [l,u], \quad Ax - z = 0
$$
Therefore, the original problem is rewritten into the classical ADMM form as following, which means that the problem can only contain equal constraints.
$$
\begin{aligned}
\min_{x,z} \; & \tfrac{1}{2}x^\top Hx + f^\top x + I_{[l,u]}(z) \\
\quad \text{s.t. } & Ax - z = 0
\end{aligned}
$$
where $I_{[l,u]}(z)$ is **Indicator Funtion**：  
$$
I_{[l,u]}(z)=
\begin{cases}
0, & z \in [l,u] \\
+\infty, & \text{otherwise}
\end{cases}
$$

---

### Formulate Augmented Lagrangian Function 

Introduce a dual variable $y$ for $Ax-z=0$, and add a punishment:  

$$
\mathcal{L}_\rho(x,z,y) =
\tfrac{1}{2}x^\top Hx + f^\top x + I_{[l,u]}(z)
+ y^\top (Ax - z) + \tfrac{\rho}{2}\|Ax - z\|^2
$$

---

## 3. ADMM 3-Steps Iteration
Idea of ADMM: Alternatively minimizing $\mathcal{L}_\rho$ w.r.t $x$ and $z$. Then update $y$. We recursively do these 3 steps until convergence.

---

### (1) x-update  

Fix $(z^k, y^k)$, then we get:
$$
x^{k+1} = \arg\min_x \;\tfrac{1}{2}x^\top Hx + f^\top x + y^{k\top}(Ax-z^k)+\tfrac{\rho}{2}\|Ax-z^k\|^2
$$

Expand the norm:
$$
\phi(x) = \tfrac{1}{2}x^\top Hx + f^\top x + y^{k\top}Ax - y^{k\top}z^k + \tfrac{\rho}{2}\|Ax\|^2 - \rho z^{k\top}Ax + \tfrac{\rho}{2}\|z^k\|^2
$$

Ignore term $(-y^{k\top}z^k + \tfrac{\rho}{2}\|z^k\|^2)$ since it is irrelevant with $x$, we get:
$$
\phi(x) = \tfrac{1}{2}x^\top Hx + f^\top x + y^{k\top}Ax + \tfrac{\rho}{2}\|Ax\|^2 - \rho z^{k\top}Ax
$$

Since we want to formulate the last 3 terms into $\|Ax - (z^k - y^k)\|^2$, consider:
$$
\|Ax - (z^k - \tfrac{y^k}{\rho})\|^2 = \|Ax\|^2 - 2(z^k - y^k/\rho)^\top Ax + \|z^k - y^k/\rho \|^2
$$

Multiply by a factor $\tfrac{\rho}{2}$, we get:
$$
\tfrac{\rho}{2}\|Ax\|^2 - \rho z^{k\top} Ax + y^{k\top} Ax + consts.
$$
This exactly matches the last three terms in $\phi(x)$ with some constants irrelevant with $x$. Then we substitude it back to $\phi(x)$ and we can get:
$$
x^{k+1} = \arg\min_x \;\tfrac{1}{2}x^\top Hx + f^\top x + \tfrac{\rho}{2}\|Ax-(z^k-y^k/\rho)\|^2
$$

Compute the gradient w.r.t $x$ and finally we arrive at a linear function:
$$
(H + \rho A^\top A + \sigma I)x^{k+1} = -f + \rho A^\top(z^k - y^k/\rho)
$$

Then solve $x^{k+1}$ by using **Cholesky Decomposition**.

---

### (2) z-update  

Fix $(x^{k+1}, y^k)$：  
$$
z^{k+1} = \arg\min_z \; I_{[l,u]}(z) + \tfrac{\rho}{2}\|z-(Ax^{k+1}+y^k)\|^2
$$

This equivalent to following **Projection**：  
$$
z^{k+1} = \Pi_{[l,u]}(Ax^{k+1}+y^k)
$$

Element-wise:
$$
z^{k+1}_i = \min(\max(Ax^{k+1}_i+y^k_i,\,l_i),\,u_i)
$$

---

### (3) y-update  

Update dual variable：  
$$
y^{k+1} = y^k + (Ax^{k+1} - z^{k+1})
$$

## 4. References
[S. Boyd et al., Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers, 2011.](https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf)

[J. Nocedal, S. Wright, Numerical Optimization, 2006.](https://www.math.uci.edu/~qnie/Publications/NumericalOptimization.pdf)

[Y. Wang, S. Boyd, Fast Model Predictive Control Using Online Optimization, IEEE T-CST, 2010.](https://ieeexplore.ieee.org/document/5153127)
