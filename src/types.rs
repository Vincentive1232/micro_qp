#[derive(Debug, Clone, Copy)]
pub struct VecN<const N: usize> {
    pub data: [f32; N],
}

impl<const N: usize> VecN<N> {
    #[inline]
    pub fn zero() -> Self {
        Self { data: [0.0; N] }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MatMN<const M: usize, const N: usize> {
    pub data: [[f32; N]; M]
}


impl <const M: usize, const N: usize> MatMN<M, N> {
    #[inline]
    pub fn zero() -> Self {
        Self { data: [[0.0; N]; M] }
    }

    #[inline]
    pub const fn idx(r: usize, c: usize) -> usize {
        r * N + c
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f32 {
        self.data[r][c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f32) {
        self.data[r][c] = v;
    }
}
