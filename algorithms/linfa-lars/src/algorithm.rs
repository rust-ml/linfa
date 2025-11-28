use std::mem::swap;

use linfa::{
    Float, 
    dataset::{AsSingleTargets, WithLapack, WithoutLapack}, 
    prelude::Records, 
    traits::Fit
};

use ndarray::{
    Array, Array1, Array2, ArrayBase, ArrayView, ArrayView1, ArrayView2, Axis, CowArray, Data, Dimension, Ix2, NewAxis, RemoveAxis, s
};
#[cfg(not(feature="blas"))]
use linfa_linalg::triangular::{
    SolveTriangularInplace,
    UPLO
};
#[cfg(feature="blas")]
use ndarray_linalg::{
    SolveTriangularInplace,
    UPLO, 
    Diag, 
    layout::MatrixLayout,
    Lapack
};
use ndarray_stats::QuantileExt;

use crate::{
    Lars,
    LarsValidParams,
    error::LarsError,
};

impl<F, D, T> Fit<ArrayBase<D, Ix2>, T, LarsError> for LarsValidParams<F>
where
    T: AsSingleTargets<Elem = F>,
    D: Data<Elem = F>,
    F: Float
{

    type Object = Lars<F>;
    /// Fit an LARS model given a feature matrix `x` and a target variable `y`.
    ///
    /// The feature matrix `x` must have shape `(n_samples, n_features)`
    /// The target variable `y` must have shape `(n_samples)`
    ///
    /// Returns a `FittedLARS` object which contains the fitted
    /// parameters and can be used to `predict` values of the target variable
    /// for new feature values.
    fn fit(&self, dataset: &linfa::DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object, LarsError> {
        let targets = dataset.as_single_targets();
        let (intercept, y) = compute_intercept(self.fit_intercept(), targets);

        let (alphas, active, coef_path, n_iter) = lars_path(
            &dataset.records().view(),
            &y.view(),
            self.n_nonzero_coefs(),
            self.eps(),
            self.verbose(),
            F::zero()
        );

        let intercept = intercept.into_scalar();

        let hyperplane = coef_path.slice(s![.., -1]).to_owned();
        
        Ok(Lars { 
            hyperplane, 
            intercept, 
            alphas,
            n_iter,
            active,
            coef_path
        })

    }
}

/// Compute the intercept as the mean of `y` along each column and center `y` if an intercept
/// should be used, use 0 as intercept and leave `y` unchanged otherwise.
/// If `y` is 2D, mean is 1D and center is 2D. If `y` is 1D, mean is a number and center is 1D.
fn compute_intercept<F: Float, I: RemoveAxis>(
    with_intercept: bool,
    y: ArrayView<F, I>,
) -> (Array<F, I::Smaller>, CowArray<F, I>)
where
    I::Smaller: Dimension<Larger = I>,
{
    if with_intercept {
        let y_mean = y
            // Take the mean of each column (1D array counts as 1 column)
            .mean_axis(Axis(0))
            .expect("Axis 0 length of 0");
        // Subtract y_mean from each "row" of y
        let y_centered = &y - &y_mean.view().insert_axis(Axis(0));
        (y_mean, y_centered.into())
    } else {
        (Array::zeros(y.raw_dim().remove_axis(Axis(0))), y.into())
    }
}

/// Compute Least Angle Regression using LARS algorithm
/// 
/// References
/// * ["Least Angle Regression", Efron et al.](http://statweb.stanford.edu/~tibs/ftp/lars.pdf)
/// * [Wikipedia entry on the Least-angle regression](https://en.wikipedia.org/wiki/Least-angle_regression)
/// * [Wikipedia entry on the Lasso](https://en.wikipedia.org/wiki/Lasso_(statistics))
/// 
/// returns alphas, active, coef_path, n_iter
fn lars_path<F: Float>(
    x: &ArrayView2<F>, 
    y: &ArrayView1<F>,
    max_iter: usize,
    eps: F,
    verbose: usize,
    alpha_min: F
) -> (Array1<F>, Vec<usize>, Array2<F>, usize) {
    
    let n_samples = F::from(y.len()).unwrap();
    let n_features = x.nfeatures();

    let max_features = max_iter.min(n_features);
    
    let mut coefs = Array2::<F>::zeros((max_features+1, n_features));
    let mut alphas = Array1::<F>::zeros(max_features+1);

    let mut prev_coef = Array1::<F>::zeros(n_features);
    let mut prev_alpha = Array1::<F>::from_elem(1, F::zero());

    let mut gram = x.t().dot(x);
    let mut cov = x.t().dot(y);
    let mut l = Array2::<F>::default((max_features, max_features));

    let mut n_iter = 0;
    let mut n_active  = 0;

    let mut sign_active = Array1::<F>::zeros(max_features);
    let mut indices = Array1::<F>::from_iter((0..n_features).map(|i| F::from(i).unwrap()));

    let tiny_32 = F::min_positive_value();
    let equality_tolerance = F::epsilon();

    let mut active:Vec<usize> = vec![];

    // let alpha_min = F::zero();

    let mut drop = false;

    if verbose > 1 {
        println!("Step\t\tAdded\t\tDropped\t\tActive set size\t\tC");
    }

    loop{
        let c;
        let mut c_idx = 0;
        let mut c_ = F::zero();
        if !cov.is_empty() {
            
            c_idx = cov.abs().argmax().unwrap();
            
            c_ = cov[c_idx];
            
            c = c_.abs();
            
        }
        else {
            c = F::zero();
        }
        
        let mut alpha = alphas.slice(s![n_iter, NewAxis]).to_owned();
        let mut coef = coefs.row(n_iter).to_owned();
        if n_iter > 0 {
            prev_alpha = alphas.slice(s![n_iter - 1, NewAxis]).to_owned();
            prev_coef = coefs.row(n_iter-1).to_owned();
        }
        
        alpha[0] = c / n_samples;
        alphas[n_iter] = c / n_samples;

        if alpha[0] <= alpha_min + equality_tolerance {
            if (alpha[0] - alpha_min).abs() > equality_tolerance {
                if n_iter > 0 {
                    let ss = (prev_alpha[0] - alpha_min) / (prev_alpha[0] - alpha[0]);
                    coef.assign(&(&prev_coef + &( &coef - &prev_coef) * ss));
                }
                alpha[0] = alpha_min;
            }
            coefs.row_mut(n_iter).assign(&coef);
            break
        }
        
        if n_iter >= max_iter || n_active >= n_features {
            break
        }

        if !drop{
            
            sign_active[n_active] = if c_ > F::zero() {
                F::one()
            } else if c_ < F::zero() {
                -F::one()
            } else {
                F::zero()
            };
            
            let m = n_active;
            let n = c_idx + n_active;

            cov.swap(c_idx,0);
            indices.swap(m,n);

            let cov_not_shortened = cov.clone();
            cov.remove_index(Axis(0), 0);

            if m!=n {
                let n_cols = gram.ncols();
                let (mut row_m, mut row_n) = gram.multi_slice_mut((s![m, ..], s![n, ..]));
                for j in 0..n_cols {
                    swap(&mut row_m[j], &mut row_n[j]);
                }
                let n_rows = gram.nrows();
                let (mut col_m, mut col_n) = gram.multi_slice_mut((s![.., m], s![.., n]));
                for j in 0..n_rows {
                    swap(&mut col_m[j], &mut col_n[j]);
                }
            }

            let c_diff = gram[[n_active,n_active]];

            l.slice_mut(s![n_active, 0..n_active]).assign(&gram.slice(s![n_active, 0..n_active]));

            if n_active != 0 {
                let mut l_sub = l.slice(s![..n_active,..n_active]).to_owned().with_lapack();
                let mut b = l.slice(s![n_active,..n_active]).insert_axis(Axis(1)).to_owned().with_lapack();

                if l_sub.strides() == [0,0] {
                    let dimen = l_sub.clone().raw_dim();
                    let l_sub_c = l_sub.clone();
                    let (data, _offset) = l_sub_c.into_raw_vec_and_offset();
                    l_sub = Array2::from_shape_vec(dimen, data).unwrap();
 
                    let dimen_b = l_sub.clone().raw_dim();
                    let b_c = b.clone();
                    let (data, _offset) = b_c.into_raw_vec_and_offset();
                    b = Array2::from_shape_vec(dimen_b, data).unwrap();
                }
                
                #[cfg(not(feature="blas"))]
                l_sub.solve_triangular_inplace(&mut b, UPLO::Lower).unwrap();

                #[cfg(feature="blas")]
                l_sub.solve_triangular_inplace(UPLO::Lower, Diag::NonUnit, &mut b).unwrap();

                l.slice_mut(s![n_active,..n_active]).assign(&(b.without_lapack().remove_axis(Axis(1))));
            }

            let row_slice = l.slice(s![n_active, 0..n_active]);
            let v = F::from(row_slice.dot(&row_slice)).unwrap();

            let diag = (c_diff-v).abs().sqrt().max(eps);
            l[[n_active,n_active]] = diag;

            if diag < F::cast(1e-7) {
                cov.assign(&cov_not_shortened);
                cov[0] = F::zero();
                cov.swap(0, c_idx);
                continue;
            }
            active.push(indices[n_active].to_usize().unwrap());
            n_active += 1;

            if verbose > 1 {
                println!(
                    "{}\t\t{}\t\t{}\t\t{}\t\t{}",
                    n_iter,
                    active[active.len() - 1],
                    "",
                    n_active,
                    c
                );

            }
        }

        let mut a = l.slice(s![..n_active,..n_active]).to_owned();
        let mut b = sign_active.slice(s![..n_active]).to_owned();
        let mut least_squares = cholesky_solve(&mut a,&mut b);

        let mut aa;
        if least_squares.len() == 1 && least_squares[0] == F::zero() {
            least_squares.fill(F::one());
            aa = F::one();
        }
        else {
            aa = F::one()/(&least_squares * &sign_active.slice(s![..n_active])).sum().sqrt();
            if !aa.is_finite() {
                let mut i=0;
                let mut l_ = l.slice_mut(s![..n_active, ..n_active]);

                while !aa.is_finite() {
                    for j in 0..n_active {
                        l_[[j, j]] += (F::from(2).unwrap().powi(i)) * eps;
                    }

                    let mut p = l_.slice(s![..,..]).to_owned();
                    let mut q = sign_active.slice(s![..n_active]).to_owned();
                    least_squares = cholesky_solve(&mut p, &mut q);

                    let tmp = (&least_squares * &sign_active.slice(s![..n_active])).sum().max(eps);
                    aa = F::one()/tmp.sqrt();

                    i+=1;

                }
            }
            least_squares *= aa;

        }


        let corr_eq_dir = gram.slice(s![..n_active,n_active..]).t().dot(&least_squares);

        let diff = &cov.mapv(|x| c - x);
        let denom = &corr_eq_dir.mapv(|x| aa - x + tiny_32);
        let ratio = diff / denom;

        let g1 = match ratio.iter().cloned().filter(|&x| x > F::zero()).reduce(F::min){
            Some(n) => n,
            None => F::infinity()
        };


        let gamma_;
        
        let sum_n = &cov.mapv(|x| c + x);
        let denom_n = &corr_eq_dir.mapv(|x| aa + x + tiny_32);
        let ratio_n = sum_n / denom_n;

        let g2 = match ratio_n.iter().cloned().filter(|&x| x > F::zero()).reduce(F::min){
            Some(n) => n,
            None => F::infinity()
        };

        gamma_ = g2.min(g1).min(c/aa);

        drop = false;

        let z = &coef.select(Axis(0), &active) * -F::one() / (&least_squares + tiny_32);
        let z_pos_t = z.iter().cloned().filter(|&x| x > F::zero()).reduce(F::min);

        let z_pos = match z_pos_t {
            Some(n) => n,
            None => F::infinity()
        };

        if z_pos < gamma_ {
            let mut idx: Vec<usize> = z
            .iter()
            .enumerate()
            .filter_map(|(i, &value)| if value == z_pos { Some(i) } else { None } )
            .collect();

            idx.reverse();

            for &i in &idx {
                sign_active[i] = -sign_active[i];
            }

            drop = true;
        }

        n_iter += 1;

        if n_iter >= coefs.shape()[0] {
            let add_features = 2 * (max_features - n_active).max(1);

            let mut new_coefs = Array2::<F>::zeros((n_iter + add_features, n_features));
            let old_shape = coefs.shape()[0];
            new_coefs
                .slice_mut(s![..old_shape, ..])
                .assign(&coefs);

            coefs = new_coefs;
            
            let mut new_alphas = Array1::<F>::zeros(n_iter + add_features);
            new_alphas.slice_mut(s![..alphas.len()]).assign(&alphas);
            alphas = new_alphas;
        }
        
        coef.assign(&(coefs.row(n_iter)));
        prev_coef.assign(&(coefs.row(n_iter - 1)));


        
        for (i, &idx) in active.iter().enumerate() {
            coef[idx] = prev_coef[idx] + gamma_ * least_squares[i];
            
        }

        coefs.row_mut(n_iter).assign(&coef);
            
        
        cov -= &(corr_eq_dir.mapv(|v| gamma_ * v));
        
    }

    let alphas_trimmed = alphas.slice(s![..n_iter + 1]).to_owned();
    let coefs_trimmed = coefs.slice(s![..n_iter + 1, ..]).to_owned();
    let coefs_t = coefs_trimmed.t().to_owned();
    (alphas_trimmed, active, coefs_t, n_iter)


}
/// Solves a linear system `A * x = b` using a Cholesky factorization.
/// 
/// - When compiled with the `blas` feature:
///   - Uses LAPACK's Cholesky solver (`potrs`) via `ndarray-linalg`.
///   - Performs an in-place factorization and solve on the provided arrays.
/// - Without `blas`:
///   - Performs a manual two-step triangular solve:  
///     `L * z = b` then `Lᵀ * x = z`.
fn cholesky_solve<F: Float>(
    x:&mut Array2<F>,
    y:&mut Array1<F>,
) -> Array1<F> {
    
    #[cfg(feature="blas")]{
        let mut p_l = x.clone().with_lapack();
        let p_slice = p_l.as_slice_mut().unwrap();
        let mut q_l = y.view_mut().with_lapack();
        let q_slice = q_l.as_slice_mut().unwrap();
        let shape = x.raw_dim();
        let layout_m = MatrixLayout::F {
            col: shape[1] as i32,
            lda: shape[0] as i32,
        };
        F::Lapack::solve_cholesky(layout_m, UPLO::Upper, p_slice, q_slice).unwrap();
        return y.clone();
    }
    #[cfg(not(feature="blas"))]
    {
        let mut y_ia = y.view_mut().insert_axis(Axis(1));
        x.solve_triangular_inplace(&mut y_ia, UPLO::Lower).unwrap();
        x.t().solve_triangular_inplace(&mut y_ia, UPLO::Upper).unwrap();
        return y.clone();
    }
}

#[cfg(test)]
mod tests {
    use core::f64;
    use ndarray::{Array, Array1, Array2, array, s};
    use ndarray_stats::QuantileExt;
    use ndarray_rand::rand::SeedableRng;
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;
    use rand_xoshiro::Xoshiro256Plus;
    use crate::{Lars, LarsError, LarsParams, LarsValidParams};

    use super::lars_path;

    use linfa::{
        Dataset, traits::Fit
    };
    use approx::assert_abs_diff_eq;

    #[test]
    fn autotraits() {
        fn has_autotraits<T:Send + Sync + Sized + Unpin> () {}
        has_autotraits::<Lars<f64>>();
        has_autotraits::<LarsParams<f64>>();
        has_autotraits::<LarsValidParams<f64>>();
        has_autotraits::<LarsError>();
    }
    
    // sklearn result obtained using the following code:
    // x = array([[1.0, 0.0], 
    //            [0.0, 1.0]])
    // y = array([3.0, 2.0])
    // model = Lars(fit_intercept=False)
    // model.fit(x,y)
    #[test]
    fn lars_toy_example_works() {
        let dataset = Dataset::new(array![[1.0, 0.0], [0.0, 1.0]], array![3.0, 2.0]);

        let model = Lars::params().fit_intercept(false).fit(&dataset).unwrap();
        assert_abs_diff_eq!(model.hyperplane(), &array![3.0, 2.0], epsilon = 0.001);
    }
    
    #[test]
    fn lars_diabetes_1_works_like_sklearn() {
        // test that lars implementation gives very similar results to
        // sklearn implementation for the first 20 lines taken from the diabetes
        // dataset in linfa/datasets/diabetes_(data|target).csv.gz
        #[rustfmt::skip]
        let x = array![
            [3.807590643342410180e-02, 5.068011873981870252e-02, 6.169620651868849837e-02, 2.187235499495579841e-02, -4.422349842444640161e-02, -3.482076283769860309e-02, -4.340084565202689815e-02, -2.592261998182820038e-03, 1.990842087631829876e-02, -1.764612515980519894e-02],
            [-1.882016527791040067e-03, -4.464163650698899782e-02, -5.147406123880610140e-02, -2.632783471735180084e-02, -8.448724111216979540e-03, -1.916333974822199970e-02, 7.441156407875940126e-02, -3.949338287409189657e-02, -6.832974362442149896e-02, -9.220404962683000083e-02],
            [8.529890629667830071e-02, 5.068011873981870252e-02, 4.445121333659410312e-02, -5.670610554934250001e-03, -4.559945128264750180e-02, -3.419446591411950259e-02, -3.235593223976569732e-02, -2.592261998182820038e-03, 2.863770518940129874e-03, -2.593033898947460017e-02],
            [-8.906293935226029801e-02, -4.464163650698899782e-02, -1.159501450521270051e-02, -3.665644679856060184e-02, 1.219056876180000040e-02, 2.499059336410210108e-02, -3.603757004385269719e-02, 3.430885887772629900e-02, 2.269202256674450122e-02, -9.361911330135799444e-03],
            [5.383060374248070309e-03, -4.464163650698899782e-02, -3.638469220447349689e-02, 2.187235499495579841e-02, 3.934851612593179802e-03, 1.559613951041610019e-02, 8.142083605192099172e-03, -2.592261998182820038e-03, -3.199144494135589684e-02, -4.664087356364819692e-02],
            [-9.269547780327989928e-02, -4.464163650698899782e-02, -4.069594049999709917e-02, -1.944209332987930153e-02, -6.899064987206669775e-02, -7.928784441181220555e-02, 4.127682384197570165e-02, -7.639450375000099436e-02, -4.118038518800790082e-02, -9.634615654166470144e-02],
            [-4.547247794002570037e-02, 5.068011873981870252e-02, -4.716281294328249912e-02, -1.599922263614299983e-02, -4.009563984984299695e-02, -2.480001206043359885e-02, 7.788079970179680352e-04, -3.949338287409189657e-02, -6.291294991625119570e-02, -3.835665973397880263e-02],
            [6.350367559056099842e-02, 5.068011873981870252e-02, -1.894705840284650021e-03, 6.662967401352719310e-02, 9.061988167926439408e-02, 1.089143811236970016e-01, 2.286863482154040048e-02, 1.770335448356720118e-02, -3.581672810154919867e-02, 3.064409414368320182e-03],
            [4.170844488444359899e-02, 5.068011873981870252e-02, 6.169620651868849837e-02, -4.009931749229690007e-02, -1.395253554402150001e-02, 6.201685656730160021e-03, -2.867429443567860031e-02, -2.592261998182820038e-03, -1.495647502491130078e-02, 1.134862324403770016e-02],
            [-7.090024709716259699e-02, -4.464163650698899782e-02, 3.906215296718960200e-02, -3.321357610482440076e-02, -1.257658268582039982e-02, -3.450761437590899733e-02, -2.499265663159149983e-02, -2.592261998182820038e-03, 6.773632611028609918e-02, -1.350401824497050006e-02],
            [-9.632801625429950054e-02, -4.464163650698899782e-02, -8.380842345523309422e-02, 8.100872220010799790e-03, -1.033894713270950005e-01, -9.056118903623530669e-02, -1.394774321933030074e-02, -7.639450375000099436e-02, -6.291294991625119570e-02, -3.421455281914410201e-02],
            [2.717829108036539862e-02, 5.068011873981870252e-02, 1.750591148957160101e-02, -3.321357610482440076e-02, -7.072771253015849857e-03, 4.597154030400080194e-02, -6.549067247654929980e-02, 7.120997975363539678e-02, -9.643322289178400675e-02, -5.906719430815229877e-02],
            [1.628067572730669890e-02, -4.464163650698899782e-02, -2.884000768730720157e-02, -9.113481248670509197e-03, -4.320865536613589623e-03, -9.768885894535990141e-03, 4.495846164606279866e-02, -3.949338287409189657e-02, -3.075120986455629965e-02, -4.249876664881350324e-02],
            [5.383060374248070309e-03, 5.068011873981870252e-02, -1.894705840284650021e-03, 8.100872220010799790e-03, -4.320865536613589623e-03, -1.571870666853709964e-02, -2.902829807069099918e-03, -2.592261998182820038e-03, 3.839324821169769891e-02, -1.350401824497050006e-02],
            [4.534098333546320025e-02, -4.464163650698899782e-02, -2.560657146566450160e-02, -1.255635194240680048e-02, 1.769438019460449832e-02, -6.128357906048329537e-05, 8.177483968693349814e-02, -3.949338287409189657e-02, -3.199144494135589684e-02, -7.563562196749110123e-02],
            [-5.273755484206479882e-02, 5.068011873981870252e-02, -1.806188694849819934e-02, 8.040115678847230274e-02, 8.924392882106320368e-02, 1.076617872765389949e-01, -3.971920784793980114e-02, 1.081111006295440019e-01, 3.605579008983190309e-02, -4.249876664881350324e-02],
            [-5.514554978810590376e-03, -4.464163650698899782e-02, 4.229558918883229851e-02, 4.941532054484590319e-02, 2.457414448561009990e-02, -2.386056667506489953e-02, 7.441156407875940126e-02, -3.949338287409189657e-02, 5.227999979678119719e-02, 2.791705090337660150e-02],
            [7.076875249260000666e-02, 5.068011873981870252e-02, 1.211685112016709989e-02, 5.630106193231849965e-02, 3.420581449301800248e-02, 4.941617338368559792e-02, -3.971920784793980114e-02, 3.430885887772629900e-02, 2.736770754260900093e-02, -1.077697500466389974e-03],
            [-3.820740103798660192e-02, -4.464163650698899782e-02, -1.051720243133190055e-02, -3.665644679856060184e-02, -3.734373413344069942e-02, -1.947648821001150138e-02, -2.867429443567860031e-02, -2.592261998182820038e-03, -1.811826730789670159e-02, -1.764612515980519894e-02],
            [-2.730978568492789874e-02, -4.464163650698899782e-02, -1.806188694849819934e-02, -4.009931749229690007e-02, -2.944912678412469915e-03, -1.133462820348369975e-02, 3.759518603788870178e-02, -3.949338287409189657e-02, -8.944018957797799166e-03, -5.492508739331759815e-02]
        ];
        #[rustfmt::skip]
        let y = array![1.51e+02, 7.5e+01, 1.41e+02, 2.06e+02, 1.35e+02, 9.7e+01, 1.38e+02, 6.3e+01, 1.1e+02, 3.1e+02, 1.01e+02, 6.9e+01, 1.79e+02, 1.85e+02, 1.18e+02, 1.71e+02, 1.66e+02, 1.44e+02, 9.7e+01, 1.68e+02];
        let model = Lars::params()
            .fit_intercept(false)
            .verbose(2)
            .fit(&Dataset::new(x, y))
            .unwrap();

        assert_abs_diff_eq!(
            model.hyperplane(),
            &array![ -389.60441998,
            -461.1009104,   
            1600.86085833,   
            327.18441323,
            5041.91097989,  
            -964.42457319,
            -4957.76873687, 
            -5179.43823859,
            648.79523699,
            -3820.0368172 
        ],
            epsilon = 0.01
        );

    }

    #[test]
    #[allow(clippy::excessive_precision)]
    fn lars_diabetes_2_works_like_sklearn() {
        // test that lars implementation gives very similar results to
        // sklearn implementation for the last 20 lines taken from the diabetes
        // dataset in linfa/datasets/diabetes_(data|target).csv.gz
        #[rustfmt::skip]
        let x = array![
            [-7.816532399920170238e-02,5.068011873981870252e-02,7.786338762690199478e-02,5.285819123858220142e-02,7.823630595545419397e-02,6.444729954958319795e-02,2.655027262562750096e-02,-2.592261998182820038e-03,4.067226371449769728e-02,-9.361911330135799444e-03],
            [9.015598825267629943e-03,5.068011873981870252e-02,-3.961812842611620034e-02,2.875809638242839833e-02,3.833367306762140020e-02,7.352860494147960002e-02,-7.285394808472339667e-02,1.081111006295440019e-01,1.556684454070180086e-02,-4.664087356364819692e-02],
            [1.750521923228520000e-03,5.068011873981870252e-02,1.103903904628619932e-02,-1.944209332987930153e-02,-1.670444126042380101e-02,-3.819065120534880214e-03,-4.708248345611389801e-02,3.430885887772629900e-02,2.405258322689299982e-02,2.377494398854190089e-02],
            [-7.816532399920170238e-02,-4.464163650698899782e-02,-4.069594049999709917e-02,-8.141376581713200000e-02,-1.006375656106929944e-01,-1.127947298232920004e-01,2.286863482154040048e-02,-7.639450375000099436e-02,-2.028874775162960165e-02,-5.078298047848289754e-02],
            [3.081082953138499989e-02,5.068011873981870252e-02,-3.422906805671169922e-02,4.367720260718979675e-02,5.759701308243719842e-02,6.883137801463659611e-02,-3.235593223976569732e-02,5.755656502954899917e-02,3.546193866076970125e-02,8.590654771106250032e-02],
            [-3.457486258696700065e-02,5.068011873981870252e-02,5.649978676881649634e-03,-5.670610554934250001e-03,-7.311850844667000526e-02,-6.269097593696699999e-02,-6.584467611156170040e-03,-3.949338287409189657e-02,-4.542095777704099890e-02,3.205915781821130212e-02],
            [4.897352178648269744e-02,5.068011873981870252e-02,8.864150836571099701e-02,8.728689817594480205e-02,3.558176735121919981e-02,2.154596028441720101e-02,-2.499265663159149983e-02,3.430885887772629900e-02,6.604820616309839409e-02,1.314697237742440128e-01],
            [-4.183993948900609910e-02,-4.464163650698899782e-02,-3.315125598283080038e-02,-2.288496402361559975e-02,4.658939021682820258e-02,4.158746183894729970e-02,5.600337505832399948e-02,-2.473293452372829840e-02,-2.595242443518940012e-02,-3.835665973397880263e-02],
            [-9.147093429830140468e-03,-4.464163650698899782e-02,-5.686312160821060252e-02,-5.042792957350569760e-02,2.182223876920789951e-02,4.534524338042170144e-02,-2.867429443567860031e-02,3.430885887772629900e-02,-9.918957363154769225e-03,-1.764612515980519894e-02],
            [7.076875249260000666e-02,5.068011873981870252e-02,-3.099563183506899924e-02,2.187235499495579841e-02,-3.734373413344069942e-02,-4.703355284749029946e-02,3.391354823380159783e-02,-3.949338287409189657e-02,-1.495647502491130078e-02,-1.077697500466389974e-03],
            [9.015598825267629943e-03,-4.464163650698899782e-02,5.522933407540309841e-02,-5.670610554934250001e-03,5.759701308243719842e-02,4.471894645684260094e-02,-2.902829807069099918e-03,2.323852261495349888e-02,5.568354770267369691e-02,1.066170822852360034e-01],
            [-2.730978568492789874e-02,-4.464163650698899782e-02,-6.009655782985329903e-02,-2.977070541108809906e-02,4.658939021682820258e-02,1.998021797546959896e-02,1.222728555318910032e-01,-3.949338287409189657e-02,-5.140053526058249722e-02,-9.361911330135799444e-03],
            [1.628067572730669890e-02,-4.464163650698899782e-02,1.338730381358059929e-03,8.100872220010799790e-03,5.310804470794310353e-03,1.089891258357309975e-02,3.023191042971450082e-02,-3.949338287409189657e-02,-4.542095777704099890e-02,3.205915781821130212e-02],
            [-1.277963188084970010e-02,-4.464163650698899782e-02,-2.345094731790270046e-02,-4.009931749229690007e-02,-1.670444126042380101e-02,4.635943347782499856e-03,-1.762938102341739949e-02,-2.592261998182820038e-03,-3.845911230135379971e-02,-3.835665973397880263e-02],
            [-5.637009329308430294e-02,-4.464163650698899782e-02,-7.410811479030500470e-02,-5.042792957350569760e-02,-2.496015840963049931e-02,-4.703355284749029946e-02,9.281975309919469896e-02,-7.639450375000099436e-02,-6.117659509433449883e-02,-4.664087356364819692e-02],
            [4.170844488444359899e-02,5.068011873981870252e-02,1.966153563733339868e-02,5.974393262605470073e-02,-5.696818394814720174e-03,-2.566471273376759888e-03,-2.867429443567860031e-02,-2.592261998182820038e-03,3.119299070280229930e-02,7.206516329203029904e-03],
            [-5.514554978810590376e-03,5.068011873981870252e-02,-1.590626280073640167e-02,-6.764228304218700139e-02,4.934129593323050011e-02,7.916527725369119917e-02,-2.867429443567860031e-02,3.430885887772629900e-02,-1.811826730789670159e-02,4.448547856271539702e-02],
            [4.170844488444359899e-02,5.068011873981870252e-02,-1.590626280073640167e-02,1.728186074811709910e-02,-3.734373413344069942e-02,-1.383981589779990050e-02,-2.499265663159149983e-02,-1.107951979964190078e-02,-4.687948284421659950e-02,1.549073015887240078e-02],
            [-4.547247794002570037e-02,-4.464163650698899782e-02,3.906215296718960200e-02,1.215130832538269907e-03,1.631842733640340160e-02,1.528299104862660025e-02,-2.867429443567860031e-02,2.655962349378539894e-02,4.452837402140529671e-02,-2.593033898947460017e-02],
            [-4.547247794002570037e-02,-4.464163650698899782e-02,-7.303030271642410587e-02,-8.141376581713200000e-02,8.374011738825870577e-02,2.780892952020790065e-02,1.738157847891100005e-01,-3.949338287409189657e-02,-4.219859706946029777e-03,3.064409414368320182e-03]
        ];
        #[rustfmt::skip]
        let y = array![2.33e+02, 9.1e+01, 1.11e+02, 1.52e+02, 1.2e+02, 6.70e+01, 3.1e+02, 9.4e+01, 1.83e+02, 6.6e+01, 1.73e+02, 7.2e+01, 4.9e+01, 6.4e+01, 4.8e+01, 1.78e+02, 1.04e+02, 1.32e+02, 2.20e+02, 5.7e+01];
        let model = Lars::params()
            .fit_intercept(false)
            .verbose(2)
            .fit(&Dataset::new(x, y))
            .unwrap();

        assert_abs_diff_eq!(
            model.hyperplane(),
            &array![  
                -2956.37127509,  
                -27631.50761139,   
                40411.01462424,   
                25327.03442023,
                -830564.37531279,
                692446.96439378,  
                336945.69344161,   
                59728.58703503,
                224688.9698237,    
                13827.30774872
            ],
            epsilon = 0.01
        );
    }

    #[test]
    fn test_covariance() {

        let dataset = linfa_datasets::diabetes();
        let x = dataset.records().view();
        let y = dataset.targets().view();
        let (_, _, coefs, _) = lars_path(&x, &y, 500, f64::EPSILON, 0, 0.0);

        for (index, coef_) in coefs.t().axis_iter(ndarray::Axis(0)).enumerate() {
            let res = &y - &x.dot(&coef_);
            let cov = x.t().dot(&res);
            let ab_cov = cov.abs();
            let cap_c = ab_cov.max().unwrap();
            let eps = 1e-3;
            let threshold = cap_c - eps;
            let ocur = cov
                .iter()
                .filter(|v| v.abs() >= threshold)
                .count();
            if index < x.shape()[1] {
                assert!(ocur == index+1);
            }
            else {
                assert!(ocur == x.shape()[1]);
            }
        }
    }

    #[test]
    fn test_singular_matrix() {
        // Test when input is a singular matrix
        let x = array![[1.0, 1.0], [1.0, 1.0]];
        let y = array![1.0, 1.0];

        let (_, _, coef_path, _) = lars_path(&x.view(), &y.view(), 500, f64::EPSILON, 0, 0.01);

        assert_abs_diff_eq!(coef_path.t(), array![[0.0, 0.0], [1.0, 0.0]], epsilon = 0.1);
    }

    #[test]
    fn test_collinearity() {
        // Check that lars_path is robust to collinearity in input
        let mut rng = Xoshiro256Plus::seed_from_u64(0);

        let x = Array::random_using((10, 5), Uniform::new(1., 2.), &mut rng);
        let y = Array1::zeros(10);

        let (_, _, coef_path, _) = lars_path(&x.view(), &y.view(), 500, f64::EPSILON, 0, 0.0);

        assert_abs_diff_eq!(coef_path, Array2::zeros(coef_path.raw_dim()));
        
        
        let residual: Array1<f64> = x.dot(&coef_path.slice(s![.., -1]))-&y;

        assert!((residual.mapv(|x| x.powi(2))).sum() < 1.0); // just make sure it's bounded

    }

    #[test]
    fn test_lars_n_nonzero_coefs() {
        let dataset = linfa_datasets::diabetes();
        let model = Lars::params().n_nonzero_coefs(6).fit(&dataset).unwrap();
        // The path should be of length 6 + 1 in a Lars going down to 6
        // non-zero coefs
        assert!(model.hyperplane().iter().filter(|&&x| x!=0.0).count() == 6);
        assert!(model.alphas().len() == 7)
    }


}

