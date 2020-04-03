use sprs::{CsMat, CsMatBase};
use hnsw::{Searcher, HNSW};
use space::{Neighbor, MetricPoint};
use ndarray::{Array2, Axis, ArrayView1};

use num_traits::NumCast;

use crate::Float;
use crate::kernel::IntoKernel;

struct Euclidean<'a, A>(ArrayView1<'a, A>);

impl<A: Float> MetricPoint for Euclidean<'_, A> {
    fn distance(&self, rhs: &Self) -> u32 {
        let val = self.0
            .iter()
            .zip(rhs.0.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<A>()
            .sqrt();

        space::f32_metric(val.to_f32().unwrap())
    }
}

pub struct SparseGaussianKernel<A> {
    data: CsMat<A>
}

impl<A: Float> SparseGaussianKernel<A> {
    pub fn new(dataset: &Array2<A>, k: usize) -> Self {
        let data = find_k_nearest_neighbours(dataset, k);
        
        SparseGaussianKernel { data }
    }
}

impl<A: Float> IntoKernel<A> for SparseGaussianKernel<A> {
    type IntoKer = CsMat<A>;

    fn into_kernel(self) -> Self::IntoKer {
        self.data
    }
}

pub fn find_k_nearest_neighbours<A: Float>(dataset: &Array2<A>, k: usize) -> CsMat<A> {
    let n_points = dataset.len_of(Axis(1));

    // ensure that the number of neighbours is at least one and less than the total number of
    // points
    assert!(k < n_points);
    assert!(k > 0);

    let mut searcher = Searcher::default();
    let mut hnsw: HNSW<Euclidean<A>> = HNSW::new();

    // insert all columns as data points into HNSW graph
    for feature in dataset.gencolumns().into_iter() {
        hnsw.insert(Euclidean(feature), &mut searcher);
    }

    // allocate buffer for k neighbours (plus the points itself)
    let mut neighbours = vec![Neighbor::invalid(); k + 1];

    // allocate buffer to initialize the sparse matrix later on
    //  * data: we have exact #points * k positive entries
    //  * indptr: has structure [0,k,2k,...,#points*k]
    //  * indices: filled with the nearest indices
    let data = vec![NumCast::from(1.0).unwrap(); n_points * k];
    let indptr = (0..n_points+1).map(|x| x * k).collect::<Vec<_>>();
    let mut indices = Vec::with_capacity(n_points * k);

    // find neighbours for each data point
    for feature in dataset.gencolumns().into_iter() {
        hnsw.nearest(&Euclidean(feature), 3 * k, &mut searcher, &mut neighbours);

        // sort by indices
        neighbours.sort_unstable();

        // push each index into the indices array
        for n in &neighbours {
            indices.push(n.index);
        }
    }

    // create CSR matrix from data, indptr and indices
    CsMatBase::new((n_points, n_points), indptr, indices, data)
}
