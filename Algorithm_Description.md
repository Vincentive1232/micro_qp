# micro_qp — ADMM Algorithm Flow

## 1. Pre-factorization
- Form the system matrix via `form_p`: 

$$
P = H + \rho A^\top A + \sigma I
$$

- Perform in-place Cholesky factorization `cholesky_in_place`.
- This factor $P = LL^\top$ is reused across iterations unless $\rho$ changes.


## 2. ADMM iterations
Repeat for $k = 0, 1, ..., (max\_iter-1)$:
### (a) x-update (solve linear system)
- Build **RHS**: $b = -f + \rho A^{\top}(z^k - y^k)$
```rust
for i in 0..N { self.rhs.data[i] = -f.data[i]; }
for i in 0..M { self.tmp_m.data[i] = self.z.data[i] - self.y.data[i]; } // z - y
at_mul_v(&self.a, &self.tmp_m, &mut self.tmp_n);                         // A^T (z - y)
for i in 0..N { self.rhs.data[i] += rho * self.tmp_n.data[i]; }          // -f + ρ A^T(…)
chol_solve(&self.p, &self.rhs, &mut self.x);                             // (H+ρA^TA+σI)x=b
```

### (b) z-update (projection)
- Compute $Ax$, then project $Ax + y$ onto $[l, u]$:
```rust
a_mul_x(&self.a, &self.x, &mut self.ax); // ax = A x
for i in 0..M {
    let mut zi = self.ax.data[i] + self.y.data[i];
    if zi < l.data[i] { zi = l.data[i]; }
    if zi > u.data[i] { zi = u.data[i]; }
    self.z.data[i] = zi; // z = Π_[l,u](Ax + y)
}
```

### (c) y-update (scaled dual)
- Compute $y^{k+1} = y^k + (Ax^{k+1} - z^{k+1})$
```rust
for i in 0..M {
    self.y.data[i] += self.ax.data[i] - self.z.data[i]; // y += Ax - z
}
```

### (d) Residuals & stopping
- Primal residual: $r^{pri} = Ax - z$.
- Dual residual: $r^{dual} = \rho A^{\top}(z^{k-1} - z^k)$
```rust
// r_pri = Ax - z
for i in 0..M { self.tmp_m.data[i] = self.ax.data[i] - self.z.data[i]; }
let r_pri = norm_inf(&self.tmp_m);

// r_dual = ρ A^T (z_prev - z)
for i in 0..M { self.tmp_m.data[i] = self.z_prev.data[i] - self.z.data[i]; }
at_mul_v(&self.a, &self.tmp_m, &mut self.tmp_n);
for i in 0..N { self.tmp_n.data[i] *= rho; }
let r_dual = norm_inf(&self.tmp_n);

if r_pri <= self.settings.eps_pri && r_dual <= self.settings.eps_dual {
    self.settings.rho = rho;
    return (self.x, k+1);
}
```

### (e) Adaptive $\rho$ (QSQP-style) + refactor
- if $\|r^{rpi}\| > \mu\|r^{dual}\|$ ----> increase $\rho$.
- if $\|r^{dual}\| < \mu\|r^{pri}\|$ ----> decrease $\rho$.
- Scale the dual: $y$ <---- $(\rho/\rho_{new})y$
- Recompute Cholesky for new $P$.
```rust
if self.settings.adapt_interval > 0 && k > 0 && (k % self.settings.adapt_interval == 0) {
    let mut new_rho = rho;
    if r_pri > self.settings.mu * r_dual {
        new_rho = (rho * self.settings.tau_inc).clamp(self.settings.rho_min, self.settings.rho_max);
    } else if r_dual > self.settings.mu * r_pri {
        new_rho = (rho * self.settings.tau_dec).clamp(self.settings.rho_min, self.settings.rho_max);
    }

    if (new_rho - rho).abs() > 0.0 {
        let scale = rho / new_rho;              // rescale dual (scaled form)
        for i in 0..M { self.y.data[i] *= scale; }
        rho = new_rho;
        self.refactor_with_rho(rho);            // rebuild P and refactorize
    }
}
```

### (f) $I_{[l,u]}(z)$ implementation in the code
For people who is curious about how $I_{[l,u]}(z)$ is implemented, $I_{[l,u]}(z)$ **won't be written explicitly**，since its value could only be 0 or $\infty$.

ADMM doesn't need the function but its **Proximal Operator**, which is the projection of set $[l,u]$.

In practice，it is **z-update element-wise clamp** in `solve()`:

```rust
// z-update
a_mul_x(&self.a, &self.x, &mut self.ax);
for i in 0..M {
    let mut zi = self.ax.data[i] + self.y.data[i];
    if zi < l.data[i] { zi = l.data[i]; }
    if zi > u.data[i] { zi = u.data[i]; }
    self.z.data[i] = zi;
}
```

## 3. Complexity
Per-iteration cost is dominated by triangular solves from the cached Cholesky of $P$. When $\rho$ is unchanged, no refactorization is needed.


## 4. Practical Notes
- Regularization $\sigma I$: ensures $P = H + \rho A^{\top}A + \sigma I$ is positive definite; increase $\sigma$ if Cholesky fails.
- Scaling: pre-scale $H, A$ to similar magnitudes to improve convergence.
- Warm start: keep $(x, z, y)$ from the previous call (e.g., across control steps) to reduce iterations.
- Stopping tolerances: Users can adjust settings to tune $\epsilon_{pri}, \epsilon_{dual}$ to balance speed and accuracy.

