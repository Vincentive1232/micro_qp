# micro_qp — A Lightweight QP Solver for Embedded Systems

`micro_qp` is a light-weight quadratic programming solver based on **ADMM (Alternating Direction Method of Multipliers)** and **IPM(Interior Point Method)**. It is capable for solving convex QP with linear constraints on platforms with tight computational budget, e.g. STM32, RP2040 etc.


## 1. Problem Description and Solution Induction
The standard convex QP problem is described as:

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

Further derivation is written [here](https://github.com/Vincentive1232/micro_qp/blob/main/Math_Induction.md).


## 2. Algorithm-Flow
`micro_qp` is based on ADMM 3-steps iteration. The algorithm is described in detail [here](https://github.com/Vincentive1232/micro_qp/blob/main/Algorithm_Description.md).


## 3. How to use `micro_qp`
### STEP 0: Add `micro_qp` to your project

**Option A: Local path (recommended for development)**

Download `micro_qp`and place the folder in the same level of the `src` folder of the project.

```toml
# Cargo.toml
[dependencies]
micro_qp = { path = "../micro_qp" }
```

**Option B: From Git**
```toml
[dependencies]
micro_qp = { git = "https://your.git.host/yourname/micro_qp.git", rev = "xxxx" }
```
**Please make sure that `std` features has already been disabled.**


### STEP 1: Define problem dimensions and import types
- `N` = number of decision variables.
- `M` = number of constraints.

```rust
use micro_qp::types::{MatMN, VecN};
use micro_qp::admm::{Solver, AdmmSettings};

const N: usize = 2; // number of variables
const M: usize = 2; // number of constraints
```

### STEP 2: Build the QP matrices (H, f, A, l, u)
- `H` is an `NxN` symmetric positive semidefinite matrix.
- `f` is a length `N` Vector.
- `A` is `MxN`, and `l`, `u` are length `M` vectors for bounds.
- For box constraints, use `A = I` and set `l[i], u[i]`.
```rust
fn build_qp() -> (MatMN<N,N>, VecN<N>, MatMN<M,N>, VecN<M>, VecN<M>) {
    let mut H = MatMN::<N,N>::zero();
    H.set(0,0, 1.0); 
    H.set(1,1, 1.0);

    let mut f = VecN::<N>::zero(); // default linear term = 0

    let mut A = MatMN::<M,N>::zero();
    A.set(0,0, 1.0);
    A.set(1,1, 1.0);

    let mut l = VecN::<M>::zero();
    let mut u = VecN::<M>::zero();
    l.data = [-10.0, -10.0];
    u.data = [ 10.0,  10.0];

    (H, f, A, l, u)
}
```

### STEP 3: Create solver and prepare (factorization)
- `prepare($H, &A)` forms the system matrix $P = H + \rho A^{\top}A + \sigma I$.
- It then performs a Cholesky decomposition, which will be reused each iteration for efficiency.
```rust
let (H, f, A, l, u) = build_qp();

let mut solver = Solver::<N,M>::new();
solver.settings = AdmmSettings {
    rho: 0.1,
    eps_pri: 1e-5,
    eps_dual: 1e-5,
    max_iter: 200,
    sigma: 1e-9,
    mu: 10.0, tau_inc: 2.0, tau_dec: 2.0,
    rho_min: 1e-6, rho_max: 1e6,
    adapt_interval: 25,
};  // specified by the user

assert!(solver.prepare(&H, &A));
```

### STEP 4: Solve the problem
Call `solve(f, l, u)` with your QP data.
It returns `(x, iters)` where `x` is the solution vector, and `iters` is the number of ADMM iterations used.
```rust
let (x, iters) = solver.solve(&f, &l, &u);
```

### STEP 5: Warm start for sequential problem
When solving a sequence of similar problem (in control problem or MPC), use the previous `(x, z, y) as the starting point. This usually reduces the number of iterations drastically.
```rust
solver.warm_start(&solver.x, &solver.z, &solver.y);
let (_x2, _it2) = solver.solve(&f, &l, &u);
```

**If `H` or `A` changes, call `prepare(&H, &A)` again. If only `f,l,u` change, you can directly call `solve`.**


## 4. Troubleshooting & performance tips

- **Cholesky fails**: Increase `sigma`(regularization term), check if `H` is symmetric and PSD, or rescale problem data.

- **Slow convergence**:
    - Adjust `rho`, or use adaptive `rho` (`adapt_interval > 0`).

    - Normalize problem data to avoid large scale differences.

- Use warm starting.

- Need faster convergence: Consider over-relaxation (e.g. replace `Ax` by $\alpha Ax + (1-\alpha)z$, with $\alpha \approx 1.6$).

## 5. Limitations (What micro_qp currently can't colve)
- **Non-convex** QP (i.e., $H \not\succeq 0$) -> no global optimality guarantees.
- **Nonlinear / conic constraints** beyond 
$l \leq Ax \leq u$(unless extended with proper proximal operators).
- **Mixed-integer** constraints (binary/integer variables).
- Large-scale problems requiring high-accuracy in few iterations (**interior-point method** may be preferable).


## 6. References
[S. Boyd et al., Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers, 2011.](https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf)

[J. Nocedal, S. Wright, Numerical Optimization, 2006.](https://www.math.uci.edu/~qnie/Publications/NumericalOptimization.pdf)

[Y. Wang, S. Boyd, Fast Model Predictive Control Using Online Optimization, IEEE T-CST, 2010.](https://ieeexplore.ieee.org/document/5153127)