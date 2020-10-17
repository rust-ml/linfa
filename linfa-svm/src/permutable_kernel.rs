use crate::Float;
use linfa_kernel::Kernel as LinfaKernel;
use ndarray::{Array1, ArrayView2};

pub type Kernel<'a, A> = LinfaKernel<ArrayView2<'a, A>>;

pub trait Permutable<'a, A: Float> {
    fn swap_indices(&mut self, i: usize, j: usize);
    fn distances(&self, idx: usize, length: usize) -> Vec<A>;
    fn self_distance(&self, idx: usize) -> A;
    fn inner(&self) -> &'a Kernel<'a, A>;
}

/// Kernel matrix with permutable columns
///
/// This struct wraps a kernel matrix with access indices. The working set can shrink during the
/// optimization and it is therefore necessary to reorder entries.
pub struct PermutableKernel<'a, A: Float> {
    kernel: &'a Kernel<'a, A>,
    kernel_diag: Array1<A>,
    kernel_indices: Vec<usize>,
    targets: Vec<bool>,
}

impl<'a, A: Float> PermutableKernel<'a, A> {
    pub fn new(kernel: &'a Kernel<'a, A>, targets: Vec<bool>) -> PermutableKernel<'a, A> {
        let kernel_diag = kernel.diagonal();
        let kernel_indices = (0..kernel.size()).collect::<Vec<_>>();

        PermutableKernel {
            kernel,
            kernel_diag,
            kernel_indices,
            targets,
        }
    }
}

impl<'a, A: Float> Permutable<'a, A> for PermutableKernel<'a, A> {
    /// Swap two indices
    fn swap_indices(&mut self, i: usize, j: usize) {
        self.kernel_indices.swap(i, j);
    }

    /// Return distances from node `idx` to all other nodes
    fn distances(&self, idx: usize, length: usize) -> Vec<A> {
        let idx = self.kernel_indices[idx];

        let kernel = self.kernel.column(idx);
        let target_i = self.targets[idx];

        // reorder entries
        (0..length)
            .map(|j| {
                let val = kernel[self.kernel_indices[j]];
                let target_j = self.targets[self.kernel_indices[j]];

                if target_j != target_i {
                    -val
                } else {
                    val
                }
            })
            .collect()
    }

    /// Return internal kernel
    fn inner(&self) -> &'a Kernel<'a, A> {
        self.kernel
    }

    /// Return distance to itself
    fn self_distance(&self, idx: usize) -> A {
        let idx = self.kernel_indices[idx];

        self.kernel_diag[idx]
    }
}

pub struct PermutableKernelOneClass<'a, A: Float> {
    kernel: &'a Kernel<'a, A>,
    kernel_diag: Array1<A>,
    kernel_indices: Vec<usize>,
}

impl<'a, A: Float> PermutableKernelOneClass<'a, A> {
    pub fn new(kernel: &'a Kernel<'a, A>) -> PermutableKernelOneClass<'a, A> {
        let kernel_diag = kernel.diagonal();
        let kernel_indices = (0..kernel.size()).collect::<Vec<_>>();

        PermutableKernelOneClass {
            kernel,
            kernel_diag,
            kernel_indices,
        }
    }
}

impl<'a, A: Float> Permutable<'a, A> for PermutableKernelOneClass<'a, A> {
    /// Swap two indices
    fn swap_indices(&mut self, i: usize, j: usize) {
        self.kernel_indices.swap(i, j);
    }

    /// Return distances from node `idx` to all other nodes
    fn distances(&self, idx: usize, length: usize) -> Vec<A> {
        let idx = self.kernel_indices[idx];

        let kernel = self.kernel.column(idx);

        // reorder entries
        (0..length)
            .map(|j| kernel[self.kernel_indices[j]])
            .collect()
    }

    /// Return internal kernel
    fn inner(&self) -> &'a Kernel<'a, A> {
        self.kernel
    }

    /// Return distance to itself
    fn self_distance(&self, idx: usize) -> A {
        let idx = self.kernel_indices[idx];

        self.kernel_diag[idx]
    }
}

pub struct PermutableKernelRegression<'a, A: Float> {
    kernel: &'a Kernel<'a, A>,
    kernel_diag: Array1<A>,
    kernel_indices: Vec<usize>,
    signs: Vec<bool>,
}

impl<'a, A: Float> PermutableKernelRegression<'a, A> {
    pub fn new(kernel: &'a Kernel<'a, A>) -> PermutableKernelRegression<'a, A> {
        let kernel_diag = kernel.diagonal();
        let kernel_indices = (0..2 * kernel.size())
            .map(|x| {
                if x < kernel.size() {
                    x
                } else {
                    x - kernel.size()
                }
            })
            .collect::<Vec<_>>();
        let signs = (0..kernel.size() * 2)
            .map(|x| x < kernel.size())
            .collect::<Vec<_>>();

        PermutableKernelRegression {
            kernel,
            kernel_diag,
            kernel_indices,
            signs,
        }
    }
}

impl<'a, A: Float> Permutable<'a, A> for PermutableKernelRegression<'a, A> {
    /// Swap two indices
    fn swap_indices(&mut self, i: usize, j: usize) {
        self.kernel_indices.swap(i, j);
        self.signs.swap(i, j);
    }

    /// Return distances from node `idx` to all other nodes
    fn distances(&self, idx: usize, length: usize) -> Vec<A> {
        let kernel = self.kernel.column(self.kernel_indices[idx]);

        // reorder entries
        let sign_i = self.signs[idx];
        (0..length)
            .map(|j| {
                let val = kernel[self.kernel_indices[j]];
                let sign_j = self.signs[j];

                if sign_i != sign_j {
                    -val
                } else {
                    val
                }
            })
            .collect()
    }

    /// Return internal kernel
    fn inner(&self) -> &'a Kernel<'a, A> {
        self.kernel
    }

    /// Return distance to itself
    fn self_distance(&self, idx: usize) -> A {
        let idx = self.kernel_indices[idx];

        self.kernel_diag[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::{Permutable, PermutableKernel};
    use linfa_kernel::{Kernel, KernelInner};
    use ndarray::array;

    #[test]
    fn test_permutable_kernel() {
        let dist = array![[1.0, 0.3, 0.1], [0.3, 1.0, 0.5], [0.1, 0.5, 1.0]];
        let targets = vec![true, true, true];
        let dist = Kernel {
            inner: KernelInner::Dense(dist.clone()),
            fnc: Box::new(|_, _| 0.0),
            dataset: dist.view(),
            linear: false,
        };

        let mut kernel = PermutableKernel::new(&dist, targets);

        assert_eq!(kernel.distances(0, 3), &[1.0, 0.3, 0.1]);
        assert_eq!(kernel.distances(1, 3), &[0.3, 1.0, 0.5]);
        assert_eq!(kernel.distances(2, 3), &[0.1, 0.5, 1.0]);

        // swap first two nodes
        kernel.swap_indices(0, 1);

        assert_eq!(kernel.distances(0, 3), &[1.0, 0.3, 0.5]);
        assert_eq!(kernel.distances(1, 3), &[0.3, 1.0, 0.1]);
        assert_eq!(kernel.distances(2, 3), &[0.5, 0.1, 1.0]);

        // swap second and third node
        kernel.swap_indices(1, 2);

        assert_eq!(kernel.distances(0, 3), &[1.0, 0.5, 0.3]);
        assert_eq!(kernel.distances(1, 3), &[0.5, 1.0, 0.1]);
        assert_eq!(kernel.distances(2, 3), &[0.3, 0.1, 1.0]);
    }
}
