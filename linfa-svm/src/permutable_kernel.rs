use crate::Float;
use linfa_kernel::Kernel;
use ndarray::Array1;

pub trait Permutable<'a, A: Float> {
    fn swap_indices(&mut self, i: usize, j: usize);
    fn distances(&self, idx: usize, length: usize) -> Vec<A>;
    fn self_distance(&self, idx: usize) -> A;
    fn inner(&self) -> &'a Kernel<A>;
}

/// Kernel matrix with permutable columns
///
/// This struct wraps a kernel matrix with access indices. The working set can shrink during the
/// optimization and it is therefore necessary to reorder entries.
pub struct PermutableKernel<'a, A: Float> {
    kernel: &'a Kernel<A>,
    kernel_diag: Array1<A>,
    kernel_indices: Vec<usize>,
    targets: Vec<bool>,
}

impl<'a, A: Float> PermutableKernel<'a, A> {
    pub fn new(kernel: &'a Kernel<A>, targets: Vec<bool>) -> PermutableKernel<'a, A> {
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
            .into_iter()
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
    fn inner(&self) -> &'a Kernel<A> {
        self.kernel
    }

    /// Return distance to itself
    fn self_distance(&self, idx: usize) -> A {
        let idx = self.kernel_indices[idx];

        self.kernel_diag[idx]
    }
}

pub struct PermutableKernelOneClass<'a, A: Float> {
    kernel: &'a Kernel<A>,
    kernel_diag: Array1<A>,
    kernel_indices: Vec<usize>,
}

impl<'a, A: Float> PermutableKernelOneClass<'a, A> {
    pub fn new(kernel: &'a Kernel<A>) -> PermutableKernelOneClass<'a, A> {
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
            .into_iter()
            .map(|j| kernel[self.kernel_indices[j]])
            .collect()
    }

    /// Return internal kernel
    fn inner(&self) -> &'a Kernel<A> {
        self.kernel
    }

    /// Return distance to itself
    fn self_distance(&self, idx: usize) -> A {
        let idx = self.kernel_indices[idx];

        self.kernel_diag[idx]
    }
}

pub struct PermutableKernelRegression<'a, A: Float> {
    kernel: &'a Kernel<A>,
    kernel_diag: Array1<A>,
    kernel_indices: Vec<usize>,
    signs: Vec<A>,
}

impl<'a, A: Float> PermutableKernelRegression<'a, A> {
    pub fn new(kernel: &'a Kernel<A>) -> PermutableKernelRegression<'a, A> {
        let kernel_diag = kernel.diagonal();
        let kernel_indices = (0..2*kernel.size()).map(|x| if x < kernel.size() { x } else { x - kernel.size() }).collect::<Vec<_>>();
        let signs = (0..kernel.size()*2).map(|x| if x < kernel.size() { A::one() } else { -A::one() }).collect::<Vec<_>>();

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
        let idx = self.kernel_indices[idx];

        let kernel = self.kernel.column(idx);

        // reorder entries
        let sign_i = self.signs[idx];
        (0..length)
            .into_iter()
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
    fn inner(&self) -> &'a Kernel<A> {
        self.kernel
    }

    /// Return distance to itself
    fn self_distance(&self, idx: usize) -> A {
        let idx = self.kernel_indices[idx];

        self.kernel_diag[idx]
    }
}
