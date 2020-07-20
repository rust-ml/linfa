use num_traits::NumCast;
use ndarray::{Array2, ArrayView1};
use sprs::CsMat;
use std::iter::Sum;

use crate::Float;
use crate::kernel::{IntoKernel, dense_from_fn, sparse_from_fn};

fn kernel<A: Float>(a: ArrayView1<A>, b: ArrayView1<A>, c: A, d: A) -> A {
    (a.dot(&b) + c).powf(d)
}

pub struct PolynomialKernel<A> {
    pub data: Array2<A>
}

impl<A: Float + Sum<A>> PolynomialKernel<A> {
    pub fn new(dataset: &Array2<A>, c: f32, d: f32) -> Self {
        let c = NumCast::from(c).unwrap();
        let d = NumCast::from(d).unwrap();

        PolynomialKernel {
            data: dense_from_fn(dataset, |a,b| kernel(a,b,c,d))
        }
    }
}

impl<A: Float> IntoKernel<A> for PolynomialKernel<A> {
    type IntoKer = Array2<A>;

    fn into_kernel(self) -> Self::IntoKer {
        self.data
    }
}

pub struct SparsePolynomialKernel<A> {
    similarity: CsMat<A>
}

impl<A: Float> SparsePolynomialKernel<A> {
    pub fn new(dataset: &Array2<A>, k: usize, c: A, d: A) -> Self {
        let c = NumCast::from(c).unwrap();
        let d = NumCast::from(d).unwrap();

        SparsePolynomialKernel { 
            similarity: sparse_from_fn(dataset, k, |a,b| kernel(a,b,c,d))
        }
    }
}

impl<A: Float> IntoKernel<A> for SparsePolynomialKernel<A> {
    type IntoKer = CsMat<A>;

    fn into_kernel(self) -> Self::IntoKer {
        self.similarity
    }
}

