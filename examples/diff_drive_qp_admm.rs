/*
This is an example of using micro_qp admm
to solve a differential drive vehicle control problem
*/
#![allow(non_snake_case)]

use micro_qp::types::{MatMN, VecN};
use micro_qp::admm::{AdmmSolver, AdmmSettings};

const N: usize = 2; // decision variables: [ur, ul]
const M: usize = 2; // constraints: box constraints

fn build_problem(
    r: f32,  // wheel radius [m]
    l: f32,  // wheel base [m]
    lambda: f32,
    v0: f32, w0: f32,
    ur_min: f32, ur_max: f32,
    ul_min: f32, ul_max: f32,
) -> (MatMN<N,N>, VecN<N>, MatMN<M,N>, VecN<M>, VecN<M>) {
    // a, b
    let a = [ r / l, -r / l ];
    let b = [ r / 2.0, r / 2.0 ];

    // H = 2(aa^T + λ bb^T)
    let mut H = MatMN::<N,N>::zero();
    let aa = [
        [ a[0]*a[0], a[0]*a[1] ],
        [ a[1]*a[0], a[1]*a[1] ],
    ];
    let bb = [
        [ b[0]*b[0], b[0]*b[1] ],
        [ b[1]*b[0], b[1]*b[1] ],
    ];
    for i in 0..N {
        for j in 0..N {
            H.set(i,j, 2.0 * ( aa[i][j] + lambda * bb[i][j] ));
        }
    }

    // f = -2 ( w0 a + λ v0 b )
    let mut f = VecN::<N>::zero();
    for i in 0..N {
        f.data[i] = -2.0 * ( w0 * a[i] + lambda * v0 * b[i] );
    }

    // A = I, box constraints
    let mut A = MatMN::<M,N>::zero();
    A.set(0,0, 1.0);
    A.set(1,1, 1.0);

    let mut l = VecN::<M>::zero();
    let mut u = VecN::<M>::zero();
    l.data = [ur_min, ul_min];
    u.data = [ur_max, ul_max];

    (H, f, A, l, u)
}

fn main() {
    let r = 0.05;     // 5 cm
    let L = 0.20;     // 20 cm
    let lambda = 1.0; // v and w have the same weights
    let v0 = 0.6;     // desired linear velocity [m/s]
    let w0 = 1.2;     // desired angular velocity [rad/s]

    // ===== define lower and upper limits =====
    let ur_min = -20.0;
    let ur_max =  20.0;
    let ul_min = -20.0;
    let ul_max =  20.0;

    // ===== construct QP =====
    let (H, f, A, l, u) = build_problem(
        r, L, lambda, v0, w0, ur_min, ur_max, ul_min, ul_max
    );

    // ===== construct and configure the solver =====
    let mut solver = AdmmSolver::<N,M>::new();
    solver.settings = AdmmSettings {
        rho: 0.01,
        eps_pri: 1e-7,
        eps_dual: 1e-7,
        max_iter: 300,
        sigma: 1e-9,
        mu: 10.0,
        tau_inc: 2.0,
        tau_dec: 2.0,
        rho_min: 1e-6,
        rho_max: 1e6,
        adapt_interval: 25,
    };

    assert!(solver.prepare(&H, &A));

    // ===== solve =====
    let (x, iters) = solver.solve(&f, &l, &u);
    let ur = x.data[0];
    let ul = x.data[1];

    println!("Solved in {iters} iterations.");
    println!("ur = {ur:.6}, ul = {ul:.6}");

    // check whether the results fullfill the box constraints
    assert!(ur >= ur_min - 1e-4 && ur <= ur_max + 1e-4);
    assert!(ul >= ul_min - 1e-4 && ul <= ul_max + 1e-4);

    let v = 0.5*r*(ur + ul);
    let w = (r/L)*(ur - ul);
    println!("Resulting v = {v:.6} m/s, w = {w:.6} rad/s");
}
