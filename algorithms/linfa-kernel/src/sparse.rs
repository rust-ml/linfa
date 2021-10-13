use linfa::Float;
use linfa_nn::{distance::L2Dist, NearestNeighbour};
use ndarray::{ArrayBase, Axis, Data, Ix2};
use sprs::{CsMat, CsMatBase};

/// Create sparse adjacency matrix from dense dataset
pub fn adjacency_matrix<F: Float, DT: Data<Elem = F>, N: NearestNeighbour>(
    dataset: &ArrayBase<DT, Ix2>,
    k: usize,
    nn_algo: &N,
) -> CsMat<F> {
    let n_points = dataset.len_of(Axis(0));

    // ensure that the number of neighbours is at least one and less than the total number of
    // points
    assert!(k < n_points);
    assert!(k > 0);

    let nn = nn_algo
        .from_batch(dataset, L2Dist)
        .expect("Unexpected nearest neighbour error");

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
    for (m, feature) in dataset.rows().into_iter().enumerate() {
        let mut neighbours = nn.k_nearest(feature, k + 1).unwrap();

        //dbg!(&neighbours);

        // sort by indices
        neighbours.sort_unstable_by_key(|(_, i)| *i);

        indices.push(m);
        data.push(F::one());
        added += 1;

        // push each index into the indices array
        for &(_, i) in &neighbours {
            if m != i {
                indices.push(i);
                data.push(F::one());
                added += 1;
            }
        }

        indptr.push(added);
    }

    // create CSR matrix from data, indptr and indices
    let mat = CsMatBase::new_from_unsorted((n_points, n_points), indptr, indices, data).unwrap();
    let transpose = mat.transpose_view().to_other_storage();
    let mut mat = sprs::binop::csmat_binop(mat.view(), transpose.view(), |x, y| x.add(*y));

    // ensure that all values are one
    let val: F = F::one();
    mat.map_inplace(|_| val);

    mat
}

#[cfg(test)]
mod tests {
    use super::*;
    use linfa_nn::{BallTree, KdTree};
    use ndarray::Array2;

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
        let adj_mat = adjacency_matrix(&input_arr, 1, &KdTree);
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
                } else if j % 2 == 0 && i == j + 1 {
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
        let adj_mat = adjacency_matrix(&input_arr, 2, &KdTree);
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
    fn adjacency_matrix_test_2() {
        // pts 0 & 1    pts 2 & 3    pts 4 & 5     pts 6 & 7
        // |0.| |3.1| _ |1.| |2.1| _ |2.| |1.1| _  |3.| |0.1|
        // |0.| |3.1|   |1.| |2.1|   |2.| |1.1|    |3.| |0.1|
        let input_mat = vec![
            0., 0., 3.1, 3.1, 1., 1., 2.1, 1.1, 2., 2., 1.1, 1.1, 3., 3., 0.1, 0.1,
        ];

        let input_arr = Array2::from_shape_vec((8, 2), input_mat).unwrap();
        let adj_mat = adjacency_matrix(&input_arr, 1, &BallTree);
        assert_eq!(adj_mat.nnz(), 16);

        // I expext non-zeros in the diagonal and then:
        // - point 0 to be neighbour of point 7 & vice versa
        // - point 1 to be neighbour of point 6 & vice versa
        // - point 2 to be neighbour of point 5 & vice versa
        // - point 3 to be neighbour of point 4 & vice versa

        for i in 0..8 {
            assert_eq!(*adj_mat.get(i, i).unwrap() as usize, 1);
            if i <= 3 {
                assert_eq!(*adj_mat.get(i, 7 - i).unwrap() as usize, 1);
                assert_eq!(*adj_mat.get(7 - i, i).unwrap() as usize, 1);
            }
        }
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
        let _ = adjacency_matrix(&input_arr, 0, &KdTree);
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
        let _ = adjacency_matrix(&input_arr, 8, &BallTree);
    }
}
