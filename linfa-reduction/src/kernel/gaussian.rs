use num_traits::NumCast;
use ndarray::{Array2, ArrayView1};
use sprs::CsMat;
use std::iter::Sum;

use crate::Float;
use crate::kernel::{IntoKernel, dense_from_fn, sparse_from_fn};

fn kernel<A: Float>(a: ArrayView1<A>, b: ArrayView1<A>, eps: A) -> A {
    let distance = a.iter().zip(b.iter()).map(|(x,y)| (*x-*y)*(*x-*y))
        .sum::<A>();

    (-distance / eps).exp()
}

pub struct GaussianKernel<A> {
    pub data: Array2<A>
}

impl<A: Float + Sum<A>> GaussianKernel<A> {
    pub fn new(dataset: &Array2<A>, eps: f32) -> Self {
        let eps = NumCast::from(eps).unwrap();

        GaussianKernel {
            data: dense_from_fn(dataset, |a,b| kernel(a,b,eps))
        }
    }
}

impl<A: Float> IntoKernel<A> for GaussianKernel<A> {
    type IntoKer = Array2<A>;

    fn into_kernel(self) -> Self::IntoKer {
        self.data
    }
}

pub struct SparseGaussianKernel<A> {
    similarity: CsMat<A>
}

impl<A: Float> SparseGaussianKernel<A> {
    pub fn new(dataset: &Array2<A>, k: usize, eps: f32) -> Self {
        let eps = NumCast::from(eps).unwrap();
        
        SparseGaussianKernel { 
            similarity: sparse_from_fn(dataset, k, |a,b| kernel(a,b,eps))
        }
    }
}

impl<A: Float> IntoKernel<A> for SparseGaussianKernel<A> {
    type IntoKer = CsMat<A>;

    fn into_kernel(self) -> Self::IntoKer {
        self.similarity
    }
}

