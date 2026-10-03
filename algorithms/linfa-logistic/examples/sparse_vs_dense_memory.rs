//! Memory companion to `benches/sparse_vs_dense.rs`: for the same problems, reports the size of
//! the feature matrix and the peak heap allocated while fitting, stored as CSR and densified.

use linfa::prelude::*;
use linfa_logistic::LogisticRegression;
use ndarray::{Array1, Ix1};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256Plus;
use sprs::{CsMat, TriMat};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// System allocator that tracks the bytes currently allocated and their peak.
struct CountingAllocator;

static CURRENT: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            let now = CURRENT.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        CURRENT.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Runs `f` and returns the peak heap it allocated on top of what was allocated before.
fn peak_extra_heap(f: impl FnOnce()) -> usize {
    let baseline = CURRENT.load(Ordering::Relaxed);
    PEAK.store(baseline, Ordering::Relaxed);
    f();
    PEAK.load(Ordering::Relaxed) - baseline
}

/// Same generator as `benches/sparse_vs_dense.rs`.
fn make_problem(n_samples: usize, n_features: usize, density: f64) -> (CsMat<f64>, Array1<bool>) {
    let mut rng = Xoshiro256Plus::seed_from_u64(42);
    let weights: Vec<f64> = (0..n_features).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let nnz_per_row = ((density * n_features as f64).round() as usize).max(1);
    let mut triplets = TriMat::new((n_samples, n_features));
    let mut labels = Array1::from_elem(n_samples, false);
    for (row, label) in labels.iter_mut().enumerate() {
        let mut z = 0.0;
        for _ in 0..nnz_per_row {
            let col = rng.gen_range(0..n_features);
            triplets.add_triplet(row, col, 1.0);
            z += weights[col];
        }
        *label = rng.gen::<f64>() < 1.0 / (1.0 + (-z).exp());
    }
    (triplets.to_csr(), labels)
}

fn mb(bytes: usize) -> f64 {
    bytes as f64 / 1e6
}

fn main() {
    let params = LogisticRegression::default().max_iterations(50);

    println!("| Features | Density | Samples | Dense matrix | Dense fit peak | CSR matrix | CSR fit peak |");
    println!("|---|---|---|---|---|---|---|");
    for n_features in [1_000, 5_000] {
        for density in [0.01, 0.1] {
            for n_samples in [1_000, 10_000, 20_000] {
                let (x, y) = make_problem(n_samples, n_features, density);

                let csr_matrix = std::mem::size_of_val(x.data())
                    + std::mem::size_of_val(x.indices())
                    + std::mem::size_of_val(x.indptr().raw_storage());
                let sparse = DatasetBase::new(x, y);
                let csr_fit = peak_extra_heap(|| {
                    params.fit(&sparse).unwrap();
                });

                let dense_matrix = n_samples * n_features * std::mem::size_of::<f64>();
                let dense: Dataset<f64, bool, Ix1> =
                    Dataset::new(sparse.records.to_dense(), sparse.targets.clone());
                drop(sparse);
                let dense_fit = peak_extra_heap(|| {
                    params.fit(&dense).unwrap();
                });

                println!(
                    "| {n_features} | {}% | {n_samples} | {:.1} MB | {:.2} MB | {:.1} MB | {:.2} MB |",
                    density * 100.0,
                    mb(dense_matrix),
                    mb(dense_fit),
                    mb(csr_matrix),
                    mb(csr_fit),
                );
            }
        }
    }
}
