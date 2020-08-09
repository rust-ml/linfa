use sprs::{CsMat, CsMatBase};
use hnsw::{Searcher, HNSW, Params};
use space::{Neighbor, MetricPoint};
use ndarray::{Array2, Axis, ArrayView1, NdFloat};

/// Implementation of euclidean distance for ndarray
struct Euclidean<'a, A>(ArrayView1<'a, A>);

impl<A: NdFloat + std::iter::Sum + Default> MetricPoint for Euclidean<'_, A> {
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

/// Create sparse adjacency matrix from dense dataset
pub fn adjacency_matrix<A: NdFloat + std::iter::Sum + Default>(dataset: &Array2<A>, k: usize) -> CsMat<A> {
    let n_points = dataset.len_of(Axis(0));

    // ensure that the number of neighbours is at least one and less than the total number of
    // points
    assert!(k < n_points);
    assert!(k > 0);

    let params = Params::new()
        .ef_construction(k);

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
        data.push(A::one());
        added += 1;

        // push each index into the indices array
        for n in &neighbours {
            if m != n.index {
                indices.push(n.index);
                data.push(A::one());
                added += 1;
            }
        }

        indptr.push(added);
    }


    // create CSR matrix from data, indptr and indices
    let mat = CsMatBase::new((n_points, n_points), indptr, indices, data);
    let mut mat = &mat + &mat.transpose_view();

    // ensure that all values are one
    let val: A = A::one();
    mat.map_inplace(|_| val);

    mat
}
