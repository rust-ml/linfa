use crate::Float;
use linfa_kernel::Kernel;
use ndarray::Array1;

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

    /// Swap two indices
    pub fn swap_indices(&mut self, i: usize, j: usize) {
        self.kernel_indices.swap(i, j);
    }

    /// Return distances from node `idx` to all other nodes
    pub fn distances(&self, idx: usize, length: usize) -> Vec<A> {
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
    pub fn inner(&self) -> &'a Kernel<A> {
        self.kernel
    }

    /// Return distance to itself
    pub fn self_distance(&self, idx: usize) -> A {
        let idx = self.kernel_indices[idx];

        self.kernel_diag[idx]
    }
}
