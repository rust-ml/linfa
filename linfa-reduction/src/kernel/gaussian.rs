use num_traits::NumCast;
use ndarray::{Array2, Axis, NdFloat};
use std::iter::Sum;

use crate::Float;
use crate::kernel::IntoKernel;

pub struct GaussianKernel<A> {
    pub data: Array2<A>
}

impl<A: NdFloat + Sum<A>> GaussianKernel<A> {
    pub fn new(dataset: &Array2<A>, eps: f32) -> Self {
        let eps = NumCast::from(eps).unwrap();

        let n_observations = dataset.len_of(Axis(0));
        let mut similarity = Array2::eye(n_observations);

        for i in 0..n_observations {
            for j in 0..n_observations {
                let a = dataset.row(i);
                let b = dataset.row(j);

                let distance = a.iter().zip(b.iter()).map(|(x,y)| (*x-*y)*(*x-*y))
                    .sum::<A>();

                similarity[(i,j)] = (-distance / eps).exp();
            }
        }

        GaussianKernel {
            data: similarity
        }
    }
}

impl<A: Float> IntoKernel<A> for GaussianKernel<A> {
    type IntoKer = Array2<A>;

    fn into_kernel(self) -> Self::IntoKer {
        self.data
    }
}
