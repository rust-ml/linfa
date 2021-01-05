use hnsw::{Params, Searcher, HNSW};
use linfa::Float;
use ndarray::{ArrayBase, ArrayView1, Axis, Data, Ix2};
use space::{MetricPoint, Neighbor};
use sprs::{CsMat, CsMatBase};

/// Implementation of euclidean distance for ndarray
struct Euclidean<'a, F>(ArrayView1<'a, F>);

impl<F: Float> MetricPoint for Euclidean<'_, F> {
    fn distance(&self, rhs: &Self) -> u32 {
        let val = self
            .0
            .iter()
            .zip(rhs.0.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<F>()
            .sqrt();
        space::f32_metric(val.to_f32().unwrap())
    }
}

/// Create sparse adjacency matrix from dense dataset
pub fn adjacency_matrix<F: Float, D: Data<Elem = F>>(
    dataset: &ArrayBase<D, Ix2>,
    k: usize,
) -> CsMat<F> {
    let n_points = dataset.len_of(Axis(0));

    // ensure that the number of neighbours is at least one and less than the total number of
    // points
    assert!(k < n_points);
    assert!(k > 0);

    let params = Params::new().ef_construction(k);

    let mut searcher = Searcher::default();
    let mut hnsw: HNSW<Euclidean<F>> = HNSW::new_params(params);

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
    let mut data = Vec::with_capacity(n_points * (k + 1));
    let mut indptr = Vec::with_capacity(n_points + 1);
    //let indptr = (0..n_points+1).map(|x| x * (k+1)).collect::<Vec<_>>();
    let mut indices = Vec::with_capacity(n_points * (k + 1));
    indptr.push(0);

    // find neighbours for each data point
    let mut added = 0;
    for (m, feature) in dataset.genrows().into_iter().enumerate() {
        hnsw.nearest(&Euclidean(feature), 3 * k, &mut searcher, &mut neighbours);

        //dbg!(&neighbours);

        // sort by indices
        neighbours.sort_unstable();

        indices.push(m);
        data.push(F::one());
        added += 1;

        // push each index into the indices array
        for n in &neighbours {
            if m != n.index {
                indices.push(n.index);
                data.push(F::one());
                added += 1;
            }
        }

        indptr.push(added);
    }

    // create CSR matrix from data, indptr and indices
    let mat = CsMatBase::new((n_points, n_points), indptr, indices, data);
    let mut mat = &mat + &mat.transpose_view();

    // ensure that all values are one
    let val: F = F::one();
    mat.map_inplace(|_| val);

    mat
}

#[cfg(test)]
mod tests {

    use super::*;
    use ndarray::{Array2, ArrayView1};

    #[test]
    fn euclidean_distance_test() {
        let p1 = Euclidean {
            0: ArrayView1::from_shape(2, &[0., 0.]).unwrap(),
        };
        let p2 = Euclidean {
            0: ArrayView1::from_shape(2, &[1., 1.]).unwrap(),
        };

        assert_eq!(p1.distance(&p2), 2_f32.sqrt().to_bits());

        let p2 = Euclidean {
            0: ArrayView1::from_shape(2, &[4., 3.]).unwrap(),
        };

        assert_eq!(p1.distance(&p2), 5_f32.to_bits());

        let p2 = Euclidean {
            0: ArrayView1::from_shape(2, &[0., 0.]).unwrap(),
        };

        assert_eq!(p1.distance(&p2), 0);
    }

    #[test]
    #[allow(clippy::if_same_then_else)]
    fn adjacency_matrix_test() {
        // pts 0 & 1    pts 2 & 3    pts 4 & 5     pts 6 & 7
        // |0.| |0.1| _ |1.| |1.1| _ |2.| |2.1| _  |3.| |3.1|
        // |0.| |0.1|   |1.| |1.1|   |2.| |2.1|    |3.| |3.1|
        let input_mat = vec![
            0., 0., 0.1, 0.1, 1., 1., 1.1, 1.1, 2., 2., 2.1, 2.1, 3., 3., 3.1, 3.1,
        ];
        let input_arr = Array2::from_shape_vec((8, 2), input_mat).unwrap();
        // Elements in the input come in pairs of 2 nearby elements with consecutive indices
        // I expect a matrix with 16 non-zero elements placed in the diagonal and connecting
        // consecutive elements in pairs of two
        let adj_mat = adjacency_matrix(&input_arr, 1);
        assert_eq!(adj_mat.nnz(), 16);

        for i in 0..8 {
            for j in 0..8 {
                // 8 diagonal elements
                if i == j {
                    assert_eq!(*adj_mat.get(i, j).unwrap() as usize, 1);
                // (0,1), (2,3), (4,5), (6,7) -> 4 elements
                } else if i % 2 == 0 && j == i + 1 {
                    assert_eq!(*adj_mat.get(i, j).unwrap() as usize, 1);
                // (1,0), (3,2), (5,4), (7,6) -> 4 elements
                } else if j % 2 == 0 && j == i - 1 {
                    assert_eq!(*adj_mat.get(i, j).unwrap() as usize, 1);
                // all other 48 elements
                } else {
                    // Since this is the first test we check that all these elements
                    // are `None`, even if it follows from `adj_mat.nnz() = 16`
                    assert_eq!(adj_mat.get(i, j), None);
                }
            }
        }

        // Elements in the input come in triples of 3 nearby elements with consecutive indices
        // I expect a matrix with 26 non-zero elements placed in the diagonal and connecting
        // consecutive elements in triples
        let adj_mat = adjacency_matrix(&input_arr, 2);
        assert_eq!(adj_mat.nnz(), 26);

        // diagonal -> 8 non-zeros
        for i in 0..8 {
            assert_eq!(*adj_mat.get(i, i).unwrap() as usize, 1);
        }

        // central input elements have neighbours in the previous and next input elements
        // -> 12 non zeros
        for i in 1..7 {
            assert_eq!(*adj_mat.get(i, i + 1).unwrap() as usize, 1);
            assert_eq!(*adj_mat.get(i, i - 1).unwrap() as usize, 1);
        }

        // first and last elements have neighbours respectively in
        // the next and previous two elements
        // -> 4 non-zeros
        assert_eq!(*adj_mat.get(0, 1).unwrap() as usize, 1);
        assert_eq!(*adj_mat.get(0, 2).unwrap() as usize, 1);
        assert_eq!(*adj_mat.get(7, 6).unwrap() as usize, 1);
        assert_eq!(*adj_mat.get(7, 5).unwrap() as usize, 1);

        // it follows then that the third and third-to-last elements
        // have also neighbours respectively in the first and last elements
        // -> 2 non-zeros -> 26 total
        assert_eq!(*adj_mat.get(0, 2).unwrap() as usize, 1);
        assert_eq!(*adj_mat.get(7, 5).unwrap() as usize, 1);

        // it follows then that all other elements are `None`
    }

    #[test]
    #[should_panic]
    fn sparse_panics_on_0_neighbours() {
        let input_mat = [
            [[0., 0.], [0.1, 0.1]],
            [[1., 1.], [1.1, 1.1]],
            [[2., 2.], [2.1, 2.1]],
            [[3., 3.], [3.1, 3.1]],
        ]
        .concat()
        .concat();
        let input_arr = Array2::from_shape_vec((8, 2), input_mat).unwrap();
        let _ = adjacency_matrix(&input_arr, 0);
    }

    #[test]
    #[should_panic]
    fn sparse_panics_on_n_neighbours() {
        let input_mat = [
            [[0., 0.], [0.1, 0.1]],
            [[1., 1.], [1.1, 1.1]],
            [[2., 2.], [2.1, 2.1]],
            [[3., 3.], [3.1, 3.1]],
        ]
        .concat()
        .concat();
        let input_arr = Array2::from_shape_vec((8, 2), input_mat).unwrap();
        let _ = adjacency_matrix(&input_arr, 8);
    }
}
