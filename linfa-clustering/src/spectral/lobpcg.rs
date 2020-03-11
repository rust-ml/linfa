use ndarray::prelude::*;
use ndarray::OwnedRepr;
use ndarray_linalg::cholesky::*;
use ndarray_linalg::triangular::*;
use ndarray_linalg::qr::*;
use ndarray_linalg::eigh::*;
use ndarray_linalg::norm::*;
use crate::ndarray::linalg::Dot;
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

fn ndarray_mask(matrix: ArrayView<f32, Ix2>, mask: &[bool]) -> Array2<f32> {
    let (rows, cols) = (matrix.rows(), matrix.cols());

    assert!(mask.len() == cols);

    let n_positive = mask.iter().filter(|x| **x).count();

    let matrix = matrix.gencolumns().into_iter().zip(mask.iter())
        .filter(|(_,x)| **x)
        .map(|(x,_)| x.to_vec())
        .flatten()
        .collect::<Vec<f32>>();

    Array2::from_shape_vec((n_positive, rows), matrix).unwrap().reversed_axes()
}

fn apply_constraints(
    mut V: ArrayViewMut<f32, Ix2>,
    fact_YY: &CholeskyFactorized<OwnedRepr<f32>>,
    Y: ArrayView<f32, Ix2>
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
    let gram_VV_fac = gram_VV_fac.into_lower();

    assert_close_l2!(&gram_VV, &gram_VV_fac.dot(&gram_VV_fac.t()), 1e-5);

    let V_t = V.reversed_axes();
    let U = gram_VV_fac.solve_triangular(UPLO::Lower, Diag::NonUnit, &V_t).unwrap();

    (U, gram_VV_fac)
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

    let mut fact_YY = None;
    if let Some(ref Y) = &Y {
        let fact_YY_tmp = Y.t().dot(Y).factorizec(UPLO::Upper).unwrap();
        apply_constraints(X.view_mut(), &fact_YY_tmp, Y.view());
        fact_YY = Some(fact_YY_tmp);
    }

    // orthonormalize the initial guess and calculate matrices AX and XAX
    let (X, _) = orthonormalize(X);
    let AX = A.dot(&X);
    let gram_XAX = X.t().dot(&AX);

    // perform eigenvalue decomposition on XAX
    let (lambda, eig_block) = sorted_eig(gram_XAX.view(), sizeX, order);

    // rotate X and AX with eigenvectors
    let X = X.dot(&eig_block);
    let AX = AX.dot(&eig_block);

    let mut activemask = vec![true; sizeX];
    let mut residual_norms = Vec::new();
    let mut previous_block_size = sizeX;

    let mut ident: Array2<f32> = Array2::eye(sizeX);
    let ident0: Array2<f32> = Array2::eye(sizeX);

    for i in 0..maxiter {
        // calculate residual
        let R = &AX - &X.dot(&lambda);

        // calculate L2 norm for every eigenvector
        let tmp = R.genrows().into_iter().map(|x| x.norm()).collect::<Vec<f32>>();
        activemask = tmp.iter().zip(activemask.iter()).map(|(x, a)| *x > tol && *a).collect();
        residual_norms.push(tmp);

        let current_block_size = activemask.iter().filter(|x| **x).count();
        if current_block_size != previous_block_size {
            previous_block_size = current_block_size;
            ident = Array2::eye(current_block_size);
        }

        if current_block_size == 0 {
            break;
        }

        let mut active_block_R = ndarray_mask(R.view(), &activemask);
        if let Some(ref M) = M {
            active_block_R = M.dot(&active_block_R);
        }
        if let (Some(ref Y), Some(ref YY)) = (&Y, &fact_YY) {
            apply_constraints(active_block_R.view_mut(), YY, Y.view());
        }

        let (R,_) = orthonormalize(R);
        let AR = A.dot(&R);
        
        // perform the Rayleigh Ritz procedure
        // compute symmetric gram matrices
        let xaw = X.t().dot(&AR);
        let waw = R.t().dot(&AR);
        let xbw = X.t().dot(&R);

        let (gramA, gramB) = if i > 0 {
            (CsMat::eye(5), CsMat::eye(5))
        } else {
            (
                /*sprs::bmat(&[[Some(CsMat::diag(&lambda)), Some(xaw)],
                              [Some(xaw.t()), Some(waw)]]),
                sprs::bmat(&[[Some(ident0), Some(xbw)],
                             [Some(xbw.t()), Some(ident)]])*/
                CsMat::eye(5),
                sprs::bmat(&[[Some(xbw.view())]])
            )
        };

        //let active_block_R = R.slice(s![:, 

    }
    

    (Array1::zeros(10), Array2::eye(10))
}

mod tests {
    use super::sorted_eig;
    use super::orthonormalize;
    use super::ndarray_mask;
    use super::Order;
    use ndarray::prelude::*;
    use ndarray_linalg::qr::*;
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
    fn test_masking() {
        let matrix = Array2::random((10, 5), Uniform::new(0., 10.));
        let masked_matrix = ndarray_mask(matrix.view(), &[true, true, false, true, false]);
        assert_close_l2!(&masked_matrix.slice(s![.., 2]), &matrix.slice(s![.., 3]), 1e-12);
    }

    #[test]
    fn test_orthonormalize() {
        let matrix = Array2::random((10, 10), Uniform::new(-10., 10.));

        let (n, l) = orthonormalize(matrix.clone());

        // check for orthogonality
        let identity = n.dot(&n.t());
        assert_close_l2!(&identity, &Array2::eye(10), 1e-4);

        // compare returned factorization with QR decomposition
        let (q, mut r) = matrix.qr().unwrap();
        assert_close_l2!(&r.mapv(|x| x.abs()) , &l.t().mapv(|x| x.abs()), 1e-5);


    }

}
