use sprs::{CsMat, CsMatBase};
use hnsw::{Searcher, HNSW, Params};
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
    pub fn new(dataset: &Array2<A>, k: usize, eps: f32) -> Self {
        let mut data = find_k_nearest_neighbours(dataset, k);
        
        for (i, mut vec) in data.outer_iterator_mut().enumerate() {
            for (j, mut val) in vec.iter_mut() {
                let a = dataset.row(i);
                let b = dataset.row(j);

                let distance = a.iter().zip(b.iter()).map(|(x,y)| (*x-*y)*(*x-*y))
                    .sum::<A>();

                *val = (-distance / NumCast::from(eps).unwrap()).exp();
            }
        }
        
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
    let n_points = dataset.len_of(Axis(0));

    // ensure that the number of neighbours is at least one and less than the total number of
    // points
    assert!(k < n_points);
    assert!(k > 0);

    let params = Params::new()
        .ef_construction(1000);

    let mut searcher = Searcher::default();
    let mut hnsw: HNSW<Euclidean<A>> = HNSW::new_params(params);

    // insert all rows as data points into HNSW graph
    for feature in dataset.genrows().into_iter() {
        hnsw.insert(Euclidean(feature), &mut searcher);
    }

    // allocate buffer for k neighbours (plus the points itself)
    let mut neighbours = vec![Neighbor::invalid(); k + 1];

    // allocate buffer to initialize the sparse matrix later on
    //  * data: we have exact #points * k positive entries
    //  * indptr: has structure [0,k,2k,...,#points*k]
    //  * indices: filled with the nearest indices
    let mut data = Vec::with_capacity(n_points * (k+1));
    let mut indptr = Vec::with_capacity(n_points + 1);
    //let indptr = (0..n_points+1).map(|x| x * (k+1)).collect::<Vec<_>>();
    let mut indices = Vec::with_capacity(n_points * (k+1));
    indptr.push(0);

    // find neighbours for each data point
    let mut added = 0;
    for (m, feature) in dataset.genrows().into_iter().enumerate() {
        hnsw.nearest(&Euclidean(feature), 3 * k, &mut searcher, &mut neighbours);

        //dbg!(&neighbours);

        // sort by indices
        neighbours.sort_unstable();

        indices.push(m);
        data.push(NumCast::from(1.0).unwrap());
        added += 1;

        // push each index into the indices array
        for n in &neighbours {
            if m != n.index {
                indices.push(n.index);
                data.push(NumCast::from(1.0).unwrap());
                added += 1;
            }
        }

        indptr.push(added);
    }


    // create CSR matrix from data, indptr and indices
    let mat = CsMatBase::new((n_points, n_points), indptr, indices, data);
    let mut mat = &mat + &mat.transpose_view();
    //dbg!(mat.to_dense());

    let val: A = NumCast::from(1.0).unwrap();
    for i in 0..(n_points) {
        mat.set(i, i, val);
    }

    mat
}
