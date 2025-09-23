/* Interior Point Method */
use crate::types::{VecN, MatMN};
use crate::math::{a_mul_x, at_mul_v, cholesky_in_place, chol_solve};


#[derive(Debug, Clone, Copy)]
pub struct IpmSettings {
    pub tol_grad: f32,
    pub max_newton: usize,
    pub mu_init: f32,
    pub mu_decay: f32,
    pub mu_min: f32,
    pub sigma_reg: f32,
    pub ls_beta: f32,
    pub ls_c: f32,
}

impl IpmSettings {
    pub fn new() -> Self {
        Self {
            tol_grad: 1e-4,
            max_newton: 50,
            mu_init: 1.0,
            mu_decay: 0.5,
            mu_min: 1e-6,
            sigma_reg: 1e-8,
            ls_beta: 0.5,
            ls_c: 1e-4,
        }
    }
}

pub struct IpmSolver<const N: usize, const M: usize, const P: usize> {
    // P = 2*M
    pub x: VecN<N>,
    pub h: MatMN<N, N>,
    pub a: MatMN<M, N>,

    // extended G (2M x N), h_bar (2M)
    pub g: MatMN<P, N>,
    pub h_bar: VecN<P>,

    // temp vector
    pub grad: VecN<N>,
    pub rhs: VecN<N>,

    pub settings: IpmSettings,
}

impl<const N: usize, const M: usize, const P: usize> IpmSolver<N, M, P> {
    pub fn new() -> Self {
        Self {
            x: VecN::zero(),
            h: MatMN::zero(),
            a: MatMN::zero(),
            g: MatMN::zero(),
            h_bar: VecN::zero(),
            grad: VecN::zero(),
            rhs: VecN::zero(),
            settings: IpmSettings::new(),
        }
    }

    /// Set/Update H、A and decomposite P = H + rho A^T A + sigma I
    pub fn prepare(&mut self, h: &MatMN<N, N>, a: &MatMN<M, N>) -> bool {
        self.h = *h;
        self.a = *a;

        // G = [ A; -A ]
        for i in 0..M {
            for j in 0..N {
                self.g.set(i, j, self.a.get(i, j));
                self.g.set(i+M, j, -self.a.get(i, j));
            }
        }
        true
    }


    pub fn warm_start(&mut self, x: &VecN<N>) {
        self.x = *x;
    }

    pub fn reset(&mut self) {
        self.x = VecN::zero();
    }

    pub fn solve(&mut self, f: &VecN<N>, l: &VecN<M>, u: &VecN<M>) -> (VecN<N>, usize) {
        // h_bar = [ u; -l ]
        for i in 0..M {
            self.h_bar.data[i] = u.data[i];
            self.h_bar.data[i+M] = -l.data[i];
        }

        // Initialization
        for i in 0..N {
            self.x.data[i] = 0.0;
        }

        let mut mu = self.settings.mu_init;
        let mut iters: usize = 0;

        while mu > self.settings.mu_min && iters < self.settings.max_newton {
            // s = h_bar - Gx
            let mut s = VecN::<P>::zero();
            a_mul_x(&self.g, &self.x, &mut s);
            for i in 0..P {
                s.data[i] = self.h_bar.data[i] - s.data[i];
            }

            // grad = Hx + f + mu * G^T (1/s)
            a_mul_x(&self.h, &self.x, &mut self.grad);
            for i in 0..N {
                self.grad.data[i] += f.data[i];
            }
            let mut inv_s = VecN::<P>::zero();
            for i in 0..P {
                inv_s.data[i] = mu / s.data[i];
            }
            let mut tmp = VecN::<N>::zero();
            at_mul_v(&self.g, &inv_s, &mut tmp);
            for i in 0..N {
                self.grad.data[i] += tmp.data[i];
            }

            // Hessian approx: H + mu * G^T diag(1/s^2) G + sigma I
            let mut hess = self.h;
            for i in 0..N {
                let v = hess.get(i, i);
                hess.set(i, i, v + self.settings.sigma_reg);
            }
            let mut ws = VecN::<P>::zero();
            for i in 0..P {
                ws.data[i] = mu / (s.data[i] * s.data[i]);
            }

            // accumulate G^T * diag(ws) * G
            for i in 0..N {
                for j in 0..N {
                    let mut acc = 0.0;
                    for k in 0..P {
                        acc += self.g.get(k, i) * ws.data[k] * self.g.get(k, j);
                    }
                    let v = hess.get(i, i);
                    hess.set(i, i, v + acc);
                }
            }

            // Solve Newton Direction: hess * dx = -grad
            for i in 0..N {
                self.rhs.data[i] = -self.grad.data[i];
            }
            let mut hess_chol = hess;
            if !cholesky_in_place(&mut hess_chol) {
                break;
            }
            let mut dx = VecN::<N>::zero();
            chol_solve(&hess_chol, &self.rhs, &mut dx);

            // line search
            let mut alpha = 1.0;
            while alpha > 1e-6 {
                let mut x_new = self.x;
                for i in 0..N {
                    x_new.data[i] += alpha * dx.data[i];
                }
                let mut s_new = VecN::<P>::zero();
                a_mul_x(&self.g, &x_new, &mut s_new);
                let mut feasible = true;
                for i in 0..P {
                    s_new.data[i] = self.h_bar.data[i] - s_new.data[i];
                    if s_new.data[i] <= 0.0 { feasible = false; break; }
                }
                if feasible {
                    self.x = x_new;
                    break;
                }
                alpha *= self.settings.ls_beta;
            }
            mu *= self.settings.mu_decay;
            iters += 1;
        }
        
        (self.x, iters)
    }
}