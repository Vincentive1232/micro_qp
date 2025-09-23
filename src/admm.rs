/* Alternating Direction Method of Multipliers */

use crate::types::{VecN, MatMN};
use crate::math::{a_mul_x, at_mul_v, form_p, cholesky_in_place, chol_solve, norm_inf};


#[derive(Debug, Clone, Copy)]
pub struct AdmmSettings {
    pub rho: f32,
    pub eps_pri: f32,
    pub eps_dual: f32,
    pub max_iter: usize,
    pub sigma: f32,

    pub mu: f32,
    pub tau_inc: f32,
    pub tau_dec: f32,
    pub rho_min: f32,
    pub rho_max: f32,
    pub adapt_interval: usize,
}

#[derive(Debug)]
pub struct AdmmSolver<const N: usize, const M: usize> {
    pub h: MatMN<N, N>,
    pub a: MatMN<M, N>,
    pub p: MatMN<N, N>,

    pub x: VecN<N>,
    pub z: VecN<M>,
    pub y: VecN<M>,

    ax: VecN<M>, 
    rhs: VecN<N>, 
    tmp_m: VecN<M>, 
    tmp_n: VecN<N>, 
    z_prev: VecN<M>,
    pub settings: AdmmSettings,
}

impl<const N: usize, const M: usize> AdmmSolver<N, M> {
    pub fn new() -> Self {
        Self {
            h: MatMN::zero(), a: MatMN::zero(), p: MatMN::zero(),
            x: VecN::zero(), z: VecN::zero(), y: VecN::zero(),
            ax: VecN::zero(), rhs: VecN::zero(), tmp_m: VecN::zero(), tmp_n: VecN::zero(), z_prev: VecN::zero(),
            settings: AdmmSettings { 
                rho: 0.01, eps_pri: 1e-6, eps_dual: 1e-6, max_iter: 300, sigma: 1e-6, 
                mu: 10.0, tau_inc: 2.0, tau_dec: 2.0, rho_min: 1e-6, rho_max: 1e6, adapt_interval: 25 }
        }
    }

    /// Set/Update H、A and decomposite P = H + rho A^T A + sigma I
    pub fn prepare(&mut self, h: &MatMN<N, N>, a: &MatMN<M, N>) -> bool {
        self.h = *h;
        self.a = *a;

        form_p(&self.h, &self.a, self.settings.rho, self.settings.sigma, &mut self.p);
        cholesky_in_place(&mut self.p)
    }

    /// re-decomposite when rho is changing
    fn refactor_with_rho(&mut self, new_rho: f32) {
        form_p(&self.h, &self.a, new_rho, self.settings.sigma, &mut self.p);
        let _ = cholesky_in_place(&mut self.p);
    }

    pub fn warm_start(&mut self, x: &VecN<N>, z: &VecN<M>, y: &VecN<M>) {
        self.x = *x;
        self.z = *z;
        self.y = *y;
    }

    pub fn reset(&mut self) {
        self.x = VecN::zero();
        self.z = VecN::zero();
        self.y = VecN::zero();
    }

    pub fn solve(&mut self, f: &VecN<N>, l: &VecN<M>, u: &VecN<M>) -> (VecN<N>, usize) {
        let mut rho = self.settings.rho;

        for k in 0..self.settings.max_iter {
            self.z_prev = self.z;

            // x-update
            for i in 0..N { self.rhs.data[i] = -f.data[i]; }
            for i in 0..M { self.tmp_m.data[i] = self.z.data[i] - self.y.data[i]; }
            at_mul_v(&self.a, &self.tmp_m, &mut self.tmp_n);
            for i in 0..N { self.rhs.data[i] += rho * self.tmp_n.data[i]; }
            chol_solve(&self.p, &self.rhs, &mut self.x);

            // z-update
            a_mul_x(&self.a, &self.x, &mut self.ax);
            for i in 0..M {
                let mut zi = self.ax.data[i] + self.y.data[i];
                if zi < l.data[i] { zi = l.data[i]; }
                if zi > u.data[i] { zi = u.data[i]; }
                self.z.data[i] = zi;
            }

            // y-update
            for i in 0..M {
                self.y.data[i] += self.ax.data[i] - self.z.data[i];
            }

            // residual
            for i in 0..M {
                self.tmp_m.data[i] = self.ax.data[i] - self.z.data[i];
            }
            let r_pri = norm_inf(&self.tmp_m);

            for i in 0..M {
                self.tmp_m.data[i] = self.z_prev.data[i] - self.z.data[i];
            }
            at_mul_v(&self.a, &self.tmp_m, &mut self.tmp_n);
            for i in 0..N {self.tmp_n.data[i] *= rho; }
            let r_dual = norm_inf(&self.tmp_n);

            if r_pri <= self.settings.eps_pri && r_dual <= self.settings.eps_dual {
                self.settings.rho = rho;
                return (self.x, k+1)
            }

            // adaptive rho
            if self.settings.adapt_interval > 0 && k > 0 && (k % self.settings.adapt_interval == 0) {
                let mut new_rho = rho;
                if r_pri > self.settings.mu * r_dual {
                    new_rho = (rho * self.settings.tau_inc).clamp(self.settings.rho_min, self.settings.rho_max);
                }
                else if r_dual > self.settings.mu * r_pri {
                    new_rho = (rho * self.settings.tau_dec).clamp(self.settings.rho_min, self.settings.rho_max);
                }

                if (new_rho - rho).abs() > 0.0 {
                    let scale = rho / new_rho;
                    for i in 0..M { self.y.data[i] *= scale; }
                    rho = new_rho;
                    self.refactor_with_rho(rho);
                }
            }
        }

        self.settings.rho = rho;
        (self.x, self.settings.max_iter)
    }
}