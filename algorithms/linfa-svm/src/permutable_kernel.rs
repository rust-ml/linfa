use crate::Float;
use linfa_kernel::Kernel;
use ndarray::Array1;

pub trait Permutable<F: Float> {
    fn swap_indices(&mut self, i: usize, j: usize);
    fn distances(&self, idx: usize, length: usize) -> Vec<F>;
    fn self_distance(&self, idx: usize) -> F;
    fn inner(&self) -> &Kernel<F>;
    fn to_inner(self) -> Kernel<F>;
}

/// KernelView matrix with permutable columns
///
/// This struct wraps a kernel matrix with access indices. The working set can shrink during the
/// optimization and it is therefore necessary to reorder entries.
pub struct PermutableKernel<F: Float> {
    kernel: Kernel<F>,
    kernel_diag: Array1<F>,
    kernel_indices: Vec<usize>,
    targets: Vec<bool>,
}

impl<F: Float> PermutableKernel<F> {
    pub fn new(kernel: Kernel<F>, targets: Vec<bool>) -> PermutableKernel<F> {
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

impl<F: Float> Permutable<F> for PermutableKernel<F> {
    /// Swap two indices
    fn swap_indices(&mut self, i: usize, j: usize) {
        self.kernel_indices.swap(i, j);
    }

    /// Return distances from node `idx` to all other nodes
    fn distances(&self, idx: usize, length: usize) -> Vec<F> {
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
    fn inner(&self) -> &Kernel<F> {
        &self.kernel
    }

    /// Return internal kernel
    fn to_inner(self) -> Kernel<F> {
        self.kernel
    }

    /// Return distance to itself
    fn self_distance(&self, idx: usize) -> F {
        let idx = self.kernel_indices[idx];

        self.kernel_diag[idx]
    }
}

pub struct PermutableKernelOneClass<F: Float> {
    kernel: Kernel<F>,
    kernel_diag: Array1<F>,
    kernel_indices: Vec<usize>,
}

impl<F: Float> PermutableKernelOneClass<F> {
    pub fn new(kernel: Kernel<F>) -> PermutableKernelOneClass<F> {
        let kernel_diag = kernel.diagonal();
        let kernel_indices = (0..kernel.size()).collect::<Vec<_>>();

        PermutableKernelOneClass {
            kernel,
            kernel_diag,
            kernel_indices,
        }
    }
}

impl<F: Float> Permutable<F> for PermutableKernelOneClass<F> {
    /// Swap two indices
    fn swap_indices(&mut self, i: usize, j: usize) {
        self.kernel_indices.swap(i, j);
    }

    /// Return distances from node `idx` to all other nodes
    fn distances(&self, idx: usize, length: usize) -> Vec<F> {
        let idx = self.kernel_indices[idx];

        let kernel = self.kernel.column(idx);

        // reorder entries
        (0..length)
            .map(|j| kernel[self.kernel_indices[j]])
            .collect()
    }

    /// Return internal kernel
    fn inner(&self) -> &Kernel<F> {
        &self.kernel
    }

    /// Return internal kernel
    fn to_inner(self) -> Kernel<F> {
        self.kernel
    }

    /// Return distance to itself
    fn self_distance(&self, idx: usize) -> F {
        let idx = self.kernel_indices[idx];

        self.kernel_diag[idx]
    }
}

pub struct PermutableKernelRegression<F: Float> {
    kernel: Kernel<F>,
    kernel_diag: Array1<F>,
    kernel_indices: Vec<usize>,
    signs: Vec<bool>,
}

impl<'a, F: Float> PermutableKernelRegression<F> {
    pub fn new(kernel: Kernel<F>) -> PermutableKernelRegression<F> {
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

impl<'a, F: Float> Permutable<F> for PermutableKernelRegression<F> {
    /// Swap two indices
    fn swap_indices(&mut self, i: usize, j: usize) {
        self.kernel_indices.swap(i, j);
        self.signs.swap(i, j);
    }

    /// Return distances from node `idx` to all other nodes
    fn distances(&self, idx: usize, length: usize) -> Vec<F> {
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
    fn inner(&self) -> &Kernel<F> {
        &self.kernel
    }

    /// Return internal kernel
    fn to_inner(self) -> Kernel<F> {
        self.kernel
    }

    /// Return distance to itself
    fn self_distance(&self, idx: usize) -> F {
        let idx = self.kernel_indices[idx];

        self.kernel_diag[idx]
    }
}

#[cfg(test)]
mod tests {
    use super::{Permutable, PermutableKernel};
    use approx::assert_abs_diff_eq;
    use linfa_kernel::{Kernel, KernelInner, KernelMethod};
    use ndarray::array;

    #[test]
    fn test_permutable_kernel() {
        let dist = array![[1.0, 0.3, 0.1], [0.3, 1.0, 0.5], [0.1, 0.5, 1.0]];
        let targets = vec![true, true, true];
        let dist = Kernel {
            inner: KernelInner::Dense(dist),
            method: KernelMethod::Linear,
        };

        let mut kernel = PermutableKernel::new(dist, targets);

        assert_abs_diff_eq!(*kernel.distances(0, 3), [1.0, 0.3, 0.1]);
        assert_abs_diff_eq!(*kernel.distances(1, 3), [0.3, 1.0, 0.5]);
        assert_abs_diff_eq!(*kernel.distances(2, 3), [0.1, 0.5, 1.0]);

        // swap first two nodes
        kernel.swap_indices(0, 1);

        assert_abs_diff_eq!(*kernel.distances(0, 3), [1.0, 0.3, 0.5]);
        assert_abs_diff_eq!(*kernel.distances(1, 3), [0.3, 1.0, 0.1]);
        assert_abs_diff_eq!(*kernel.distances(2, 3), [0.5, 0.1, 1.0]);

        // swap second and third node
        kernel.swap_indices(1, 2);

        assert_abs_diff_eq!(*kernel.distances(0, 3), [1.0, 0.5, 0.3]);
        assert_abs_diff_eq!(*kernel.distances(1, 3), [0.5, 1.0, 0.1]);
        assert_abs_diff_eq!(*kernel.distances(2, 3), [0.3, 0.1, 1.0]);
    }
}
