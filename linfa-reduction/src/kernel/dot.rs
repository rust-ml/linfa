use ndarray::{Array1, ArrayView2, Array2};
use crate::Float;
use crate::kernel::{Kernel, IntoKernel};

pub struct DotKernel<A> {
    data: Array2<A>
}

impl<A> DotKernel<A> {
    pub fn new(data: Array2<A>) -> DotKernel<A> {
        DotKernel { data }
    }
}

impl<A: Float> Kernel<A> for DotKernel<A> {
    fn mul_similarity(&self, rhs: &ArrayView2<A>) -> Array2<A> {
        if self.data.ncols() > self.data.nrows() {
            self.data.dot(&self.data.t().dot(rhs))
        } else {
            self.data.t().dot(&self.data.dot(rhs))
        }
    }

    fn size(&self) -> usize {
        usize::min(self.data.ncols(), self.data.nrows())
    }

    fn sum(&self) -> Array1<A> {
        self.mul_similarity(&Array2::ones((self.size(), 1)).view())
            .into_shape(self.size()).unwrap()
    }
}

impl<A: Float> IntoKernel<A> for DotKernel<A> {
    type IntoKer = DotKernel<A>;

    fn into_kernel(self) -> Self::IntoKer {
        self
    }
}
