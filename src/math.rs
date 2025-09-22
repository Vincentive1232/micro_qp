use crate::types::{VecN, MatMN};
use libm::sqrtf;

/// y = A * x  (A: MxN, x: N, y: M)
pub fn a_mul_x<const M: usize, const N: usize>(a: &MatMN<M,N>, x: &VecN<N>, y: &mut VecN<M>) {
    for i in 0..M {
        let mut acc = 0.0;
        for j in 0..N {
            acc += a.get(i, j) * x.data[j]; 
        }
        y.data[i] = acc;
    }
}

/// y = A^T * v  (A: MxN, v: M, y: N)
pub fn at_mul_v<const M: usize, const N: usize>(a: &MatMN<M, N>, v: &VecN<M>, y: &mut VecN<N>) {
    for j in 0..N {
        let mut acc = 0.0;
        for i in 0..M { acc += a.get(i, j) * v.data[i]; }
        y.data[j] = acc;
    }
}


/// C = A^T A 
pub fn ata<const M: usize, const N: usize> (a: &MatMN<M, N>, c: &mut MatMN<N, N>) {
    for r in 0..N {
        for cc in r..N {
            let mut acc = 0.0;
            for i in 0..M { acc += a.get(i, r)*a.get(i, cc);}
            c.set(r, cc, acc);
            c.set(cc, r, acc);
        }
    }
}

/// P += alpha * I
pub fn add_diag<const N: usize>(p: &mut MatMN<N, N>, alpha: f32) {
    for i in 0..N {
        let _ii = MatMN::<N, N>::idx(i, i);
        let mut v = p.get(i, i);
        v += alpha;
        p.set(i, i, v);
    }
}

/// P = H + rho * (A^T A) + sigma * I
pub fn form_p<const M: usize, const N: usize>(
    h: &MatMN<N, N>, a: &MatMN<M, N>, rho: f32, sigma: f32, p: &mut MatMN<N, N>
) {
    *p = *h;
    let mut c = MatMN::<N, N>::zero();
    ata(a, &mut c);
    for i in 0..N {
        for j in 0..N {
            let v = p.get(i, j) + rho * c.get(i, j);
            p.set(i, j, v);
        }
    }
    add_diag(p, sigma);
}

/// Cholesky decomposition (lower triangle)
pub fn cholesky_in_place<const N: usize>(p: &mut MatMN<N, N>) -> bool {
    for i in 0..N {
        for j in 0..=i {
            let mut sum = p.get(i, j);
            for k in 0..j {
                sum -= p.get(i, k) * p.get(j, k);
            }
            if i==j {
                if sum <= 0.0 { return false; }
                p.set(i, j, sqrtf(sum));
            }
            else {
                p.set(i, j, sum / p.get(j, j));
            }
        }
        for j in (i+1)..N { p.set(i, j, 0.0);}
    }
    true
}

/// Solve L L^T x = b
pub fn chol_solve<const N: usize>(l: &MatMN<N, N>, b: &VecN<N>, x: &mut VecN<N>) {
    let mut y = VecN::<N>::zero();

    // forward
    for i in 0..N {
        let mut sum = b.data[i];
        for j in 0..i {
            sum -= l.get(i, j) * y.data[j];
        }
        y.data[i] = sum / l.get(i, i);
    }

    // backward
    for i in (0..N).rev() {
        let mut sum = y.data[i];
        for j in (i+1)..N {
            sum -= l.get(j, i) * x.data[j];
        }
        x.data[i] = sum / l.get(i, i);
    }
}

/// ||v||_inf
pub fn norm_inf<const N: usize>(v: &VecN<N>) -> f32 {
    let mut m = 0.0;
    for i in 0..N { let a = v.data[i].abs(); if a > m { m = a; }}
    m
}