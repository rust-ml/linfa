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

fn sorted_eig(input: ArrayView<f32, Ix2>, stiff: Option<ArrayView<f32, Ix2>>, size: usize, order: &Order) -> (Array1<f32>, Array2<f32>) {
    assert_close_l2!(&input, &input.t(), 1e-12);
    let (vals, vecs) = match stiff {
        Some(x) => (input, x).eigh(UPLO::Upper).map(|x| (x.0, (x.1).0)).unwrap(),
        _ => input.eigh(UPLO::Upper).unwrap()
    };

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

    let mut iter = usize::min(n, maxiter);

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
    let (mut lambda, mut eig_block) = sorted_eig(gram_XAX.view(), None, sizeX, &order);

    // rotate X and AX with eigenvectors
    let mut X = X.dot(&eig_block);
    let mut AX = AX.dot(&eig_block);

    let mut activemask = vec![true; sizeX];
    let mut residual_norms = Vec::new();
    let mut previous_block_size = sizeX;

    let mut ident: Array2<f32> = Array2::eye(sizeX);
    let ident0: Array2<f32> = Array2::eye(sizeX);

    let mut ap: Option<(Array2<f32>, Array2<f32>)> = None;

    let final_norm = loop {
        // calculate residual
        let R = &AX - &X.dot(&lambda);

        // calculate L2 norm for every eigenvector
        let tmp = R.genrows().into_iter().map(|x| x.norm()).collect::<Vec<f32>>();
        activemask = tmp.iter().zip(activemask.iter()).map(|(x, a)| *x > tol && *a).collect();
        residual_norms.push(tmp.clone());

        let current_block_size = activemask.iter().filter(|x| **x).count();
        if current_block_size != previous_block_size {
            previous_block_size = current_block_size;
            ident = Array2::eye(current_block_size);
        }

        if current_block_size == 0 || iter == 0 {
            break tmp;
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
        let xw = X.t().dot(&R);

        let (gramA, gramB): (Array2<f32>, Array2<f32>) = if let Some((ref P, ref AP)) = ap {
            let active_P = ndarray_mask(P.view(), &activemask);
            let active_AP = ndarray_mask(AP.view(), &activemask);
            let (active_P, R) = orthonormalize(active_P);
            let active_AP = R.solve_triangular(UPLO::Lower, Diag::NonUnit, &active_AP).unwrap();

            let xap = X.t().dot(&active_AP);
            let wap = R.t().dot(&active_AP);
            let pap = active_P.t().dot(&active_AP);
            let xp = X.t().dot(&active_P);
            let wp = R.t().dot(&active_P);

            (
                stack![Axis(0),
                    stack![Axis(1), Array2::from_diag(&lambda), xaw, xap],
                    stack![Axis(1), xaw.t(), waw, wap],
                    stack![Axis(1), xap.t(), wap.t(), pap]
                ],

                stack![Axis(0),
                    stack![Axis(1), ident0, xw, xp],
                    stack![Axis(1), xw.t(), ident, wp],
                    stack![Axis(1), xp.t(), wp.t(), ident]
                ]
            )
        } else {
            (
                stack![Axis(0), 
                    stack![Axis(1), Array2::from_diag(&lambda), xaw],
                    stack![Axis(1), xaw.t(), waw]
                ],
                stack![Axis(0),
                    stack![Axis(1), ident0, xw],
                    stack![Axis(1), xw.t(), ident]
                ]
            )
        };

        let (new_lambda, eig_vecs) = sorted_eig(gramA.view(), Some(gramB.view()), sizeX, &order);
        lambda = new_lambda;

        let (pp, app, eig_X) = if let Some((ref P, ref AP)) = ap {
            let active_P = ndarray_mask(P.view(), &activemask);
            let active_AP = ndarray_mask(AP.view(), &activemask);

            let eig_X = eig_vecs.slice(s![..sizeX, ..]);
            let eig_R = eig_vecs.slice(s![sizeX..sizeX+current_block_size, ..]);
            let eig_P = eig_vecs.slice(s![sizeX+current_block_size.., ..]);

            let pp = R.dot(&eig_R) + P.dot(&eig_P);
            let app = AR.dot(&eig_R) + AP.dot(&eig_P);

            (pp, app, eig_X)
        } else {
            let eig_X = eig_vecs.slice(s![..sizeX, ..]);
            let eig_R = eig_vecs.slice(s![sizeX.., ..]);

            let pp = R.dot(&eig_R);
            let app = AR.dot(&eig_R);

            (pp, app, eig_X)
        };

        X = X.dot(&eig_X) + &pp;
        AX = AX.dot(&eig_X) + &app;

        ap = Some((pp, app));

        iter -= 1;
    };

    dbg!(&final_norm);
    
    (lambda, X)
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
        let (vals, vecs) = sorted_eig(matrix.view(), None, 10, &Order::Largest);

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
