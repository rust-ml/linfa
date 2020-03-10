use ndarray::prelude::*;
use ndarray::OwnedRepr;
use ndarray_linalg::cholesky::*;
use ndarray_linalg::eigh::*;
use ndarray_linalg::norm::*;
use sprs::CsMat;

pub enum Order {
    Largest,
    Smallest
}

fn sorted_eig(input: ArrayView<f32, Ix2>, size: usize, order: Order) -> (Array1<f32>, Array2<f32>) {
    assert_close_l2!(&input, &input.t(), 1e-12);
    let (vals, vecs) = input.eigh(UPLO::Upper).unwrap();
    let n = input.len_of(Axis(0));

    match order {
        Order::Largest => (vals.slice_move(s![n-size..; -1]), vecs.slice_move(s![.., n-size..; -1])),
        Order::Smallest => (vals.slice_move(s![..size]), vecs.slice_move(s![..size, ..]))
    }
}

fn apply_constraints(
    mut V: ArrayViewMut<f32, Ix2>,
    fact_YY: CholeskyFactorized<OwnedRepr<f32>>,
    Y: Array2<f32>
) {
    let gram_YV = Y.t().dot(&V);

    let U = gram_YV.genrows().into_iter()
        .map(|x| fact_YY.solvec(&x).unwrap().to_vec())
        .flatten()
        .collect::<Vec<f32>>();

    let U = Array2::from_shape_vec((5, 5), U).unwrap();

    V -= &(Y.dot(&U));
}

fn orthonormalize(
    V: Array2<f32>
) -> (Array2<f32>, Array2<f32>) {
    let gram_VV = V.t().dot(&V);
    let gram_VV_fac = gram_VV.factorizec(UPLO::Upper).unwrap();
    let inv_gram_VV = gram_VV_fac.invc().unwrap();

    let V = V.dot(&inv_gram_VV);
    (V, inv_gram_VV)
}

pub fn lobpcg(
    A: CsMat<f32>,
    mut X: Array2<f32>,
    M: Option<CsMat<f32>>,
    Y: Option<Array2<f32>>,
    tol: f32, maxiter: usize,
    order: Order
) -> (Array1<f32>, Array2<f32>) {
    // the target matrix should be symmetric
    assert!(sprs::is_symmetric(&A));
    assert_eq!(A.cols(), A.rows());

    // the initital approximation should be maximal square
    // n is the dimensionality of the problem
    let (n, sizeX) = (X.rows(), X.cols());
    assert!(sizeX <= n);
    assert_eq!(n, A.cols());

    let sizeY = match Y {
        Some(ref x) => x.cols(),
        _ => 0
    };

    if (n - sizeY) < 5 * sizeX {
        panic!("Please use a different approach, the LOBPCG method only supports the calculation of a couple of eigenvectors!");
    }

    let maxiter = usize::min(n, maxiter);

    if let Some(Y) = Y {
        let fact_YY = (&Y.t() * &Y).factorizec(UPLO::Upper).unwrap();
        apply_constraints(X.view_mut(), fact_YY, Y);
    }

    let (X, _) = orthonormalize(X);
    let AX = &A * &X;
    let gram_XAX = &X.t() * &AX;

    let (lambda, eig_block) = sorted_eig(gram_XAX.view(), sizeX, Order::Largest);

    let X = &X * &eig_block;
    let AX = &AX * &eig_block;

    let mut residual_norms = Vec::new();

    for i in 0..maxiter {
        let R = &AX - &(&X * &lambda);
        let tmp = R.genrows().into_iter().map(|x| x.norm()).collect::<Vec<f32>>();
        residual_norms.push(tmp);

    }
    

    (Array1::zeros(10), Array2::eye(10))
}

mod tests {
    use super::sorted_eig;
    use super::orthonormalize;
    use super::Order;
    use ndarray::prelude::*;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;

    #[test]
    fn test_sorted_eigen() {
        let matrix = Array2::random((10, 10), Uniform::new(0., 10.));
        let matrix = matrix.t().dot(&matrix);

        // return all eigenvectors with largest first
        let (vals, vecs) = sorted_eig(matrix.view(), 10, Order::Largest);

        // calculate V * A * V' and compare to original matrix
        let diag = Array2::from_diag(&vals);
        let rec = (vecs.dot(&diag)).dot(&vecs.t());

        assert_close_l2!(&matrix, &rec, 1e-5);
    }

    #[test]
    fn test_orthonormalize() {
        let matrix = Array2::random((10, 10), Uniform::new(0., 10.));
        let matrix = matrix.t().dot(&matrix);
        dbg!(&matrix);

        let (normalized, _) = orthonormalize(matrix.clone());

        assert_close_l2!(&normalized.dot(&normalized.t()), &Array2::eye(10), 1e-5);

    }

}
