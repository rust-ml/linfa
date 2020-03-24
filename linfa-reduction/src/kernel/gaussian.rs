use ndarray::{Array2, Axis};
use crate::kernel::IntoKernel;

pub struct GaussianKernel {
    data: Array2<f64>
}

impl IntoKernel<f64> for GaussianKernel {
    type IntoKer = Array2<f64>;

    fn into_kernel(self) -> Self::IntoKer {
        self.data
    }
}

impl GaussianKernel {
    pub fn new(dataset: &Array2<f64>, eps: f64) -> Self {
        let n_observations = dataset.len_of(Axis(0));
        let mut similarity = Array2::eye(n_observations);

        for i in 0..n_observations {
            for j in 0..n_observations {
                let a = dataset.row(i);
                let b = dataset.row(j);

                let distance = a.iter().zip(b.iter()).map(|(x,y)| (x-y).powf(2.0))
                    .sum::<f64>();

                similarity[(i,j)] = (-distance / eps).exp();
            }
        }

        GaussianKernel {
            data: similarity
        }
    }
}
