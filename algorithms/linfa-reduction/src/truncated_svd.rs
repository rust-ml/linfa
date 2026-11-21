//! Truncated Singular Value Decomposition
//!
//! Truncated SVD reduces the dimensionality of a dataset by factorizing the data matrix
//! `X ≈ U Σ Vᵀ` and retaining only the `k` largest singular values. The data is projected onto
//! the right singular vectors, which span the subspace capturing most of the squared magnitude of
//! the data.
//!
//! In contrast to [`Pca`](crate::Pca), this estimator does **not** center the features before
//! decomposing. It therefore operates on the raw data matrix, which makes it the appropriate
//! choice for non-negative, high-dimensional data such as term-count or TF-IDF matrices, where
//! subtracting the mean would destroy the structure of the data. This use of truncated SVD is
//! known as latent semantic analysis (LSA). When the input happens to already be centered, the
//! two estimators are equivalent. Note that this equivalence covers the components and singular
//! values, but not [`explained_variance`](TruncatedSvd::explained_variance), which is defined
//! differently here, nor `inverse_transform`, which adds no mean back.
//!
//! # Numerical caveats
//!
//! The decomposition is computed by an iterative solver (LOBPCG), which is not guaranteed to have
//! converged when it returns. For intermediate values of `embedding_size` the solver may report a
//! subspace that is not the true dominant one, in which case the singular values are inaccurate.
//! This affects [`Pca`](crate::Pca) identically, as both estimators share the same backend. The
//! effect is most pronounced when `embedding_size` is neither small nor close to the number of
//! features.
//!
//! The number of components in the fitted model may be *smaller* than the requested
//! `embedding_size`, because the backend discards singular values that are negligible relative to
//! the largest one. Always size downstream code from
//! [`n_components`](TruncatedSvd::n_components) rather than from the requested value.
//!
//! With the `blas` feature enabled the backend seeds its starting block from the thread-local
//! random number generator, so results are not reproducible between runs of the same program; in
//! particular the sign of each component is arbitrary. Without `blas` the seed is fixed and
//! results are deterministic for a given input and library version.
//!
//! # Example
//!
//! ```
//! use linfa::traits::{Fit, Predict};
//! use linfa_reduction::TruncatedSvd;
//!
//! let dataset = linfa_datasets::iris();
//!
//! // decompose the data matrix directly, without centering it first
//! let embedding = TruncatedSvd::params(2)
//!     .fit(&dataset).unwrap();
//!
//! // reduce dimensionality of the dataset
//! let dataset = embedding.predict(dataset);
//! ```
use crate::error::{ReductionError, Result};
// NOTE: the backend solver is also called `TruncatedSvd`, so it is aliased here to leave the name
// free for this crate's estimator.
#[cfg(not(feature = "blas"))]
use linfa_linalg::{lobpcg::TruncatedSvd as TruncatedSvdSolver, Order};
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix2};
#[cfg(feature = "blas")]
use ndarray_linalg::{TruncatedOrder, TruncatedSvd as TruncatedSvdSolver};
#[cfg(not(feature = "blas"))]
use rand::{prelude::SmallRng, SeedableRng};
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

use linfa::{
    dataset::Records,
    traits::{Fit, PredictInplace, Transformer},
    DatasetBase, Float, ParamGuard,
};

/// Truncated SVD hyperparameters, checked
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TruncatedSvdValidParams {
    embedding_size: usize,
}

impl TruncatedSvdValidParams {
    /// Return the target dimensionality of the embedding
    pub fn embedding_size(&self) -> usize {
        self.embedding_size
    }
}

/// Truncated SVD hyperparameters, unchecked
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TruncatedSvdParams(TruncatedSvdValidParams);

impl TruncatedSvdParams {
    /// Set the target dimensionality of the embedding
    ///
    /// This corresponds to `n_components` in scikit-learn's `TruncatedSVD`.
    pub fn embedding_size(mut self, embedding_size: usize) -> Self {
        self.0.embedding_size = embedding_size;

        self
    }
}

impl ParamGuard for TruncatedSvdParams {
    type Checked = TruncatedSvdValidParams;
    type Error = ReductionError;

    fn check_ref(&self) -> std::result::Result<&Self::Checked, Self::Error> {
        if self.0.embedding_size == 0 {
            Err(ReductionError::NonPositiveEmbeddingSize)
        } else {
            Ok(&self.0)
        }
    }

    fn check(self) -> std::result::Result<Self::Checked, Self::Error> {
        self.check_ref()?;
        Ok(self.0)
    }
}

/// Fit a truncated SVD model given a dataset
///
/// The decomposition is applied to the records of the dataset as they are, without centering the
/// features first. This is the defining difference to [`Pca`](crate::Pca).
///
/// # Parameters
///
/// * `dataset`: A dataset with records in N dimensions
///
/// # Returns
///
/// A fitted truncated SVD model holding the right singular vectors and singular values
impl<T, D: Data<Elem = f64>> Fit<ArrayBase<D, Ix2>, T, ReductionError> for TruncatedSvdValidParams {
    type Object = TruncatedSvd<f64>;

    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<TruncatedSvd<f64>> {
        if dataset.nsamples() == 0 {
            return Err(ReductionError::NotEnoughSamples);
        } else if dataset.nfeatures() < self.embedding_size {
            return Err(ReductionError::DimensionIncrease(
                self.embedding_size,
                dataset.nfeatures(),
            ));
        } else if dataset.nsamples() < self.embedding_size {
            // The backend solves an eigenproblem on the smaller of the two Gram matrices and
            // rejects a block wider than that matrix with an opaque error, so guard it here.
            return Err(ReductionError::EmbeddingLargerThanSamples(
                self.embedding_size,
                dataset.nsamples(),
            ));
        }

        let records = dataset.records();
        // NOTE: unlike `Pca`, the feature means are *not* subtracted here. The solver takes the
        // problem matrix by value, so an owned copy is required regardless.
        let x = records.to_owned();

        // estimate the truncated Singular Value Decomposition
        #[cfg(feature = "blas")]
        let result =
            TruncatedSvdSolver::new(x, TruncatedOrder::Largest).decompose(self.embedding_size)?;
        #[cfg(not(feature = "blas"))]
        let result =
            TruncatedSvdSolver::new_with_rng(x, Order::Largest, SmallRng::seed_from_u64(42))
                .decompose(self.embedding_size)?;

        // the left singular vectors are not needed: the embedding is defined by `V` alone, and
        // projecting the records is both cheaper and valid for unseen data
        let (_, singular_values, components) = result.values_vectors();

        // The explained variance of an uncentered decomposition is *not* `σ²/(n - 1)`: that
        // identity only holds when the projected data has zero mean, which centering guarantees
        // and truncated SVD does not. It is therefore measured on the projected training data.
        let transformed = records.dot(&components.t());
        let explained_variance = transformed.var_axis(Axis(0), 0.);
        let total_variance = records.var_axis(Axis(0), 0.).sum();

        Ok(TruncatedSvd {
            components,
            singular_values,
            explained_variance,
            total_variance,
        })
    }
}

/// Fitted truncated SVD model
///
/// The model contains the right singular vectors of the training data, which form the projection
/// hyperplane. Because the training data was not centered, no mean is stored and none is applied
/// when projecting.
///
/// # Example
///
/// ```
/// use linfa::traits::{Fit, Predict};
/// use linfa_reduction::TruncatedSvd;
///
/// let dataset = linfa_datasets::iris();
///
/// let embedding = TruncatedSvd::params(2)
///     .fit(&dataset).unwrap();
///
/// let dataset = embedding.predict(dataset);
/// ```
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone, PartialEq)]
pub struct TruncatedSvd<F> {
    components: Array2<F>,
    singular_values: Array1<F>,
    explained_variance: Array1<F>,
    total_variance: F,
}

impl TruncatedSvd<f64> {
    /// Create default parameter set
    ///
    /// # Parameters
    ///
    ///  * `embedding_size`: the target dimensionality
    pub fn params(embedding_size: usize) -> TruncatedSvdParams {
        TruncatedSvdParams(TruncatedSvdValidParams { embedding_size })
    }

    /// Return the components, with shape `(n_components, n_features)`
    ///
    /// The rows are the right singular vectors of the training data, in order of decreasing
    /// singular value. Note that fewer components than requested may be returned if the training
    /// data was rank deficient, and that the iterative backend may not have converged to the true
    /// dominant subspace; see the module documentation.
    pub fn components(&self) -> &Array2<f64> {
        &self.components
    }

    /// Return the singular values, in decreasing order
    ///
    /// These are only as accurate as the iterative backend that produced them, which may return a
    /// non-converged subspace for intermediate embedding sizes; see the module documentation.
    pub fn singular_values(&self) -> &Array1<f64> {
        &self.singular_values
    }

    /// Return the amount of explained variance per component
    ///
    /// This is the variance of the training data after projection onto each component. Unlike
    /// [`Pca::explained_variance`](crate::Pca::explained_variance), it cannot be derived from the
    /// singular values alone, because the projected data is not centered.
    pub fn explained_variance(&self) -> &Array1<f64> {
        &self.explained_variance
    }

    /// Return the fraction of the total variance explained by each component
    ///
    /// The denominator is the total variance of the *original* training data, so these values do
    /// not sum to one, even when as many components as features are retained. The remainder is the
    /// variance lost by truncation. Returns `NaN` if the training data had zero total variance.
    pub fn explained_variance_ratio(&self) -> Array1<f64> {
        &self.explained_variance / self.total_variance
    }

    /// Return the number of components retained by the fitted model
    ///
    /// This may be smaller than the requested embedding size if the training data was rank
    /// deficient, so downstream shapes should be taken from here rather than from the parameter.
    pub fn n_components(&self) -> usize {
        self.components.nrows()
    }

    /// Transform data back to its original space
    ///
    /// Because the model never centered the data, no mean is added back here.
    pub fn inverse_transform(&self, prediction: &Array2<f64>) -> Array2<f64> {
        prediction.dot(&self.components)
    }
}

impl<F: Float, D: Data<Elem = F>> PredictInplace<ArrayBase<D, Ix2>, Array2<F>> for TruncatedSvd<F> {
    fn predict_inplace(&self, records: &ArrayBase<D, Ix2>, targets: &mut Array2<F>) {
        assert_eq!(
            targets.shape(),
            &[records.nrows(), self.components.nrows()],
            "The number of data points must match the number of output targets."
        );
        // no mean is subtracted here, in contrast to `Pca::predict_inplace`
        *targets = records.dot(&self.components.t());
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array2<F> {
        Array2::zeros((x.nrows(), self.components.nrows()))
    }
}

impl<F: Float, D: Data<Elem = F>, T>
    Transformer<DatasetBase<ArrayBase<D, Ix2>, T>, DatasetBase<Array2<F>, T>> for TruncatedSvd<F>
{
    fn transform(&self, ds: DatasetBase<ArrayBase<D, Ix2>, T>) -> DatasetBase<Array2<F>, T> {
        let DatasetBase {
            records,
            targets,
            weights,
            ..
        } = ds;

        let mut new_records = self.default_target(&records);
        self.predict_inplace(&records, &mut new_records);

        DatasetBase::new(new_records, targets).with_weights(weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Pca;
    use approx::assert_abs_diff_eq;
    use linfa::{traits::Predict, Dataset};
    use ndarray::{array, Array1, Array2, Axis};

    /// A deterministic 8x4 matrix whose full SVD is known in closed form.
    ///
    /// `X = U diag(4, 3, 2, 1) V^T`, where `U` holds the first four normalized Sylvester-Hadamard
    /// columns of order 8 and `V` is the normalized Hadamard matrix of order 4. Both have
    /// orthonormal columns, so the singular values are exactly `[4, 3, 2, 1]` and the right
    /// singular vectors are exactly the columns of `V`.
    ///
    /// Two consequences are used repeatedly below:
    /// * `v_0` is the constant direction `[1, 1, 1, 1] / 2`;
    /// * the feature mean of `X` is exactly `sqrt(2) * v_0`, so `X - 1 mean^T` is precisely the
    ///   part of `X` spanned by `v_1, v_2, v_3`. The *leading* component therefore explains no
    ///   variance at all, and every explained variance is known analytically:
    ///   `[0, 9/8, 4/8, 1/8]`, summing to the total variance `14/8`.
    fn hadamard_fixture() -> Array2<f64> {
        // Sylvester construction: `H[i][j] = (-1)^popcount(i & j)`
        let sign = |i: usize, j: usize| {
            if (i & j).count_ones() & 1 == 0 {
                1.0
            } else {
                -1.0
            }
        };
        let u = Array2::from_shape_fn((8, 4), |(i, k)| sign(i, k) / 8f64.sqrt());
        let v = Array2::from_shape_fn((4, 4), |(f, k)| sign(f, k) / 2.);
        let s = [4., 3., 2., 1.];

        Array2::from_shape_fn((8, 4), |(i, f)| {
            (0..4).map(|k| s[k] * u[[i, k]] * v[[f, k]]).sum()
        })
    }

    /// A 6x4 matrix of full column rank without any particular structure.
    fn messy_matrix() -> Array2<f64> {
        array![
            [1.0, 2.0, 3.0, 0.5],
            [4.0, 1.0, 0.5, 2.5],
            [2.0, 7.0, 1.0, 1.5],
            [3.0, 0.5, 5.0, 4.0],
            [8.0, 2.0, 2.0, 0.25],
            [0.5, 3.0, 4.0, 6.0]
        ]
    }

    /// Deterministic full-rank matrix used to compare against scikit-learn.
    ///
    /// Every entry is a small integer, so the fixture is bit-for-bit reproducible in any language
    /// and the reference values can be regenerated from
    /// `[[float((i * 7 + j * 13) % 11) for j in range(6)] for i in range(30)]`.
    ///
    /// It is deliberately larger than the other fixtures here: the solver defaults to
    /// `2 * n_samples` iterations, so it does not converge well enough on very small inputs to
    /// reproduce reference values.
    fn reference_matrix() -> Array2<f64> {
        Array2::from_shape_fn((30, 6), |(i, j)| ((i * 7 + j * 13) % 11) as f64)
    }

    /// Frobenius norm of the difference between `X` and its rank-`k` reconstruction.
    fn reconstruction_error(x: &Array2<f64>, k: usize) -> f64 {
        let dataset = Dataset::from(x.clone());
        let model = TruncatedSvd::params(k).fit(&dataset).unwrap();
        let projected: Array2<f64> = model.predict(x);
        let reconstructed = model.inverse_transform(&projected);

        (&reconstructed - x).mapv(|v| v * v).sum().sqrt()
    }

    #[test]
    fn autotraits() {
        fn has_autotraits<T: Send + Sync + Sized + Unpin>() {}
        has_autotraits::<TruncatedSvd<f64>>();
        has_autotraits::<TruncatedSvdParams>();
        has_autotraits::<TruncatedSvdValidParams>();
    }

    /// Regression test for the defining property: truncated SVD does **not** center the features.
    ///
    /// The input has a large constant first column. Centering annihilates that column, so `Pca`
    /// puts exactly zero weight on it; truncated SVD must instead put almost all of its weight
    /// there, because the constant column dominates the squared magnitude of the raw data.
    ///
    /// If an `x - &mean` ever creeps back into `fit`, this test fails.
    #[test]
    fn test_does_not_center_features() {
        let x = array![[10., 0., 0.], [10., 1., 0.], [10., 0., 1.], [10., 1., 1.]];
        let dataset = Dataset::from(x);

        let svd = TruncatedSvd::params(1).fit(&dataset).unwrap();
        let pca = Pca::params(1).fit(&dataset).unwrap();

        let svd_leading = svd.components().row(0).to_owned();
        let pca_leading = pca.components().row(0).to_owned();

        // the constant direction carries essentially the whole signal for truncated SVD
        assert!(
            svd_leading[0].abs() > 0.9,
            "leading truncated-SVD component has no weight on the constant feature: {}",
            svd_leading
        );
        // ... and none at all for PCA, which removed it by centering
        assert_abs_diff_eq!(pca_leading[0].abs(), 0., epsilon = 1e-6);

        // the two leading directions are close to orthogonal, i.e. materially different (compared
        // on the absolute value, because the sign of a singular vector is arbitrary)
        assert!(
            svd_leading.dot(&pca_leading).abs() < 0.2,
            "truncated SVD reproduced the PCA direction: {} vs {}",
            svd_leading,
            pca_leading
        );

        // the leading singular value reflects the uncentered magnitude, sqrt(402.005) = 20.05...,
        // instead of the centered spread, 1.0
        assert_abs_diff_eq!(svd.singular_values()[0], 20.0500, epsilon = 1e-3);
        assert_abs_diff_eq!(pca.singular_values()[0], 1.0, epsilon = 1e-6);
    }

    /// On data that is already centered, truncated SVD and PCA must agree: centering is the *only*
    /// difference between the two estimators.
    #[test]
    fn test_equivalent_to_pca_on_centered_data() {
        let x = messy_matrix();
        let centered = &x - &x.mean_axis(Axis(0)).unwrap();
        let dataset = Dataset::from(centered);

        let svd = TruncatedSvd::params(2).fit(&dataset).unwrap();
        let pca = Pca::params(2).fit(&dataset).unwrap();

        assert_eq!(svd.components().nrows(), 2);
        assert_abs_diff_eq!(svd.singular_values(), pca.singular_values(), epsilon = 1e-7);

        // the components span the same directions, up to the arbitrary sign of each singular
        // vector: `C_svd * C_pca^T` must be a signed identity matrix
        let overlap = svd.components().dot(&pca.components().t());
        for (i, row) in overlap.rows().into_iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                if i == j {
                    assert_abs_diff_eq!(value.abs(), 1.0, epsilon = 1e-6);
                } else {
                    assert_abs_diff_eq!(*value, 0.0, epsilon = 1e-6);
                }
            }
        }

        // the projections agree as well, again up to a per-component sign
        let svd_proj: Array2<f64> = svd.predict(&dataset);
        let pca_proj: Array2<f64> = pca.predict(&dataset);
        assert_abs_diff_eq!(
            svd_proj.mapv(f64::abs),
            pca_proj.mapv(f64::abs),
            epsilon = 1e-6
        );
    }

    /// `components` is `(k', n_features)` and the projection is `(n_samples, k')`.
    #[test]
    fn test_output_shapes() {
        let x = hadamard_fixture();
        let dataset = Dataset::from(x.clone());
        let model = TruncatedSvd::params(2).fit(&dataset).unwrap();

        let k = model.components().nrows();
        assert_eq!(k, 2);
        assert_eq!(model.n_components(), k);
        assert_eq!(model.components().shape(), &[k, 4]);
        assert_eq!(model.singular_values().len(), k);
        assert_eq!(model.explained_variance().len(), k);
        assert_eq!(model.explained_variance_ratio().len(), k);

        let projected: Array2<f64> = model.predict(&dataset);
        assert_eq!(projected.shape(), &[8, k]);

        // the `Transformer` route must agree with `Predict`
        let transformed = model.transform(Dataset::from(x));
        assert_eq!(transformed.records().shape(), &[8, k]);
        assert_abs_diff_eq!(transformed.records(), &projected, epsilon = 1e-12);

        // ... and the reconstruction goes back to the original feature space
        assert_eq!(model.inverse_transform(&projected).shape(), &[8, 4]);
    }

    /// Singular values come back in strictly decreasing order, and match the values the fixture
    /// was built from.
    #[test]
    fn test_singular_values_descending() {
        let model = TruncatedSvd::params(4)
            .fit(&Dataset::from(hadamard_fixture()))
            .unwrap();

        // `k == n_features` is one of the block sizes the iterative backend resolves reliably, but
        // the tolerance stays loose enough to survive an unseeded starting block under `blas`.
        assert_abs_diff_eq!(
            model.singular_values(),
            &array![4., 3., 2., 1.],
            epsilon = 1e-6
        );

        let sv = model.singular_values();
        for w in sv.windows(2) {
            assert!(w[0] > w[1], "singular values are not descending: {}", sv);
        }

        // the components are the right singular vectors of the fixture, i.e. the columns of the
        // normalized Hadamard matrix, again up to sign
        // same tolerance as the singular values above, for the same reason
        assert_abs_diff_eq!(
            model.components().mapv(f64::abs),
            Array2::from_elem((4, 4), 0.5),
            epsilon = 1e-6
        );
    }

    /// Adding components can never make the rank-`k` reconstruction worse.
    #[test]
    fn test_reconstruction_error_decreases_with_k() {
        let x = messy_matrix();

        // Monotonicity holds for the *exact* rank-k reconstruction, because the dominant subspaces
        // are nested. The backend is iterative and may return a non-converged subspace for
        // intermediate block sizes (see the module documentation), which breaks that nestedness.
        // The comparison is therefore restricted to the block sizes where the solver converges
        // reliably for this fixture; under the `blas` feature the starting block is unseeded, so
        // an assertion over every `k` would be genuinely flaky rather than merely strict.
        let ks = [1usize, 2, 4];
        let errors: Vec<f64> = ks.iter().map(|&k| reconstruction_error(&x, k)).collect();
        for w in errors.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-6,
                "reconstruction error increased with k: {:?}",
                errors
            );
        }
        // With as many components as features the retained basis is complete, so the round trip is
        // exact whichever orthonormal basis the solver happened to converge to.
        assert_abs_diff_eq!(errors[2], 0.0, epsilon = 1e-6);
        // ... and a single component is genuinely lossy, so the chain is not trivially satisfied
        assert!(errors[0] > 1.0, "errors: {:?}", errors);
    }

    /// With `k` equal to the rank of the input, `inverse_transform(transform(X))` returns `X`.
    #[test]
    fn test_full_rank_reconstruction_is_exact() {
        let x = hadamard_fixture();
        let dataset = Dataset::from(x.clone());
        let model = TruncatedSvd::params(4).fit(&dataset).unwrap();
        assert_eq!(model.components().nrows(), 4);

        let projected: Array2<f64> = model.predict(&x);
        let reconstructed = model.inverse_transform(&projected);

        assert_abs_diff_eq!(reconstructed, x, epsilon = 1e-6);

        // the components are orthonormal, so `C^T C` is the projector onto the retained subspace,
        // which for a complete basis is the identity. This is sign-agnostic by construction.
        let projector = model.components().t().dot(model.components());
        assert_abs_diff_eq!(projector, Array2::eye(4), epsilon = 1e-6);
    }

    /// Rank-deficient input yields *fewer* components than requested; that is correct behavior.
    #[test]
    fn test_rank_deficient_input_returns_fewer_components() {
        // 12x10 matrix built from two outer products, hence of rank exactly 2
        let a: Vec<f64> = (0..12).map(|i| (i as f64).sin()).collect();
        let b: Vec<f64> = (0..10).map(|j| (j as f64).cos()).collect();
        let c: Vec<f64> = (0..12).map(|i| (2. * i as f64).cos()).collect();
        let d: Vec<f64> = (0..10).map(|j| (3. * j as f64).sin()).collect();
        let x = Array2::from_shape_fn((12, 10), |(i, j)| 3. * a[i] * b[j] + c[i] * d[j]);

        // ten components are requested, which is a legal request (10 <= min(12, 10))
        let model = TruncatedSvd::params(10).fit(&Dataset::from(x)).unwrap();

        // Fewer components than requested is the property under test. The exact count is *not* a
        // stable assertion: when the iterative backend stops short, the trailing eigenvalues come
        // back small but non-zero and clear its relative cutoff, so a third or fourth component can
        // survive. What must hold regardless is that the rank-2 structure is intact, i.e. anything
        // past the second singular value is negligible against the leading one.
        let n = model.n_components();
        assert!(n < 10, "fewer components than requested must be returned");
        assert_eq!(model.singular_values().len(), n);
        assert_eq!(model.components().nrows(), n);

        let sv = model.singular_values();
        for i in 2..sv.len() {
            assert!(
                sv[i] < 1e-2 * sv[0],
                "a rank-2 input yielded a significant third singular value: {}",
                sv
            );
        }
    }

    /// Hyperparameter validation reports the correct error variant for each failure mode.
    #[test]
    fn test_parameter_validation() {
        let dataset = Dataset::from(messy_matrix()); // 6 samples, 4 features

        let zero = TruncatedSvd::params(0).fit(&dataset);
        assert!(matches!(
            zero,
            Err(ReductionError::NonPositiveEmbeddingSize)
        ));

        let too_wide = TruncatedSvd::params(5).fit(&dataset);
        assert!(matches!(
            too_wide,
            Err(ReductionError::DimensionIncrease(5, 4))
        ));

        // 2 samples, 5 features: passes the feature check but not the sample check
        let few_samples = Dataset::from(Array2::<f64>::from_shape_fn((2, 5), |(i, j)| {
            (i * 5 + j) as f64
        }));
        let too_tall = TruncatedSvd::params(3).fit(&few_samples);
        assert!(matches!(
            too_tall,
            Err(ReductionError::EmbeddingLargerThanSamples(3, 2))
        ));

        let empty = Dataset::from(Array2::<f64>::zeros((0, 4)));
        assert!(matches!(
            TruncatedSvd::params(2).fit(&empty),
            Err(ReductionError::NotEnoughSamples)
        ));

        // the boundary cases are accepted
        assert!(TruncatedSvd::params(1).fit(&dataset).is_ok());
        assert!(TruncatedSvd::params(4).fit(&dataset).is_ok());
    }

    /// Cross-check `explained_variance` against the closed form it must satisfy.
    ///
    /// The implementation measures the variance of the projected training data. Because
    /// `X v_i = sigma_i u_i` with `||u_i|| = 1`, that quantity must equal
    /// `sigma_i^2 / n - (mean . v_i)^2`. Note that this identity holds for whichever unit vectors
    /// the solver returned, converged or not, so it pins the mean-corrected `ddof = 0` formula
    /// rather than validating the decomposition: it is what fails if PCA's `sigma^2 / (n - 1)` is
    /// substituted here.
    #[test]
    fn test_explained_variance_matches_closed_form() {
        for x in [messy_matrix(), hadamard_fixture()] {
            let n = x.nrows() as f64;
            let mean = x.mean_axis(Axis(0)).unwrap();
            for k in 1..=4 {
                let model = TruncatedSvd::params(k)
                    .fit(&Dataset::from(x.clone()))
                    .unwrap();

                let closed_form: Array1<f64> = model
                    .singular_values()
                    .iter()
                    .zip(model.components().rows())
                    .map(|(sigma, v)| sigma * sigma / n - mean.dot(&v).powi(2))
                    .collect();

                assert_abs_diff_eq!(model.explained_variance(), &closed_form, epsilon = 1e-7);
                // a variance can never be negative
                assert!(model.explained_variance().iter().all(|v| *v >= -1e-9));
            }
        }

        // the analytically known case: for `hadamard_fixture` the feature mean is exactly
        // `sqrt(2) * v_0`, so the *leading* component explains no variance whatsoever
        let model = TruncatedSvd::params(4)
            .fit(&Dataset::from(hadamard_fixture()))
            .unwrap();
        assert_abs_diff_eq!(
            model.explained_variance(),
            &array![0.0, 9. / 8., 4. / 8., 1. / 8.],
            epsilon = 1e-7
        );
    }

    /// The explained variance ratios do not sum to one once the decomposition is truncated on
    /// uncentered data. This mirrors scikit-learn's `TruncatedSVD` and is pinned here so that
    /// nobody "fixes" it by renormalizing.
    #[test]
    fn test_explained_variance_ratio_does_not_sum_to_one() {
        // The total variance of the fixture is 14/8 and the components contribute
        // `[0, 9/8, 4/8, 1/8]`, so the ratios of a truncated fit sum to `0` for `k = 1` and to
        // `9/14` for `k = 2`.
        //
        // `k = 3` is deliberately skipped: the iterative backend does not converge to the true
        // subspace for a block of 3 out of a 4x4 Gram matrix, which is a property of the solver
        // and not of this estimator: `Pca` is affected identically, returning bit-identical values
        // on the same input without the `blas` feature, where both estimators share a fixed seed.
        // Only assertions on exact values are affected; the property tests below cover every `k`.
        for (k, expected) in [(1, 0.0), (2, 9. / 14.)] {
            let model = TruncatedSvd::params(k)
                .fit(&Dataset::from(hadamard_fixture()))
                .unwrap();
            let sum = model.explained_variance_ratio().sum();

            assert_abs_diff_eq!(sum, expected, epsilon = 1e-7);
            assert!(
                sum < 0.99,
                "ratios were renormalized to sum to one: {}",
                sum
            );
        }
    }

    /// When the input *is* centered and every feature direction is retained, the basis is complete
    /// and the ratios do sum to one.
    #[test]
    fn test_explained_variance_ratio_sums_to_one_when_centered_and_complete() {
        let x = messy_matrix();
        let centered = &x - &x.mean_axis(Axis(0)).unwrap();
        let model = TruncatedSvd::params(4)
            .fit(&Dataset::from(centered))
            .unwrap();

        assert_eq!(model.components().nrows(), 4);
        assert_abs_diff_eq!(model.explained_variance_ratio().sum(), 1.0, epsilon = 1e-7);
    }

    /// Compare every learned quantity against scikit-learn's `TruncatedSVD`.
    ///
    /// The reference values were produced by scikit-learn 1.5.2 with
    /// `TruncatedSVD(n_components=3, algorithm="arpack")` on [`reference_matrix`], and
    /// independently reproduced from `numpy.linalg.svd` plus the documented variance formulas.
    #[test]
    fn test_matches_scikit_learn() {
        let model = TruncatedSvd::params(3)
            .fit(&Dataset::from(reference_matrix()))
            .unwrap();

        assert_abs_diff_eq!(
            model.singular_values(),
            &array![67.409442111200, 25.301101305598, 23.976151993122],
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            model.explained_variance(),
            &array![1.781289762736, 21.282358946708, 19.160436185523],
            epsilon = 1e-6
        );
        assert_abs_diff_eq!(
            model.explained_variance_ratio(),
            array![0.029710720852, 0.354975500881, 0.319583249633],
            epsilon = 1e-6
        );
        // the sign of a singular vector is arbitrary, so the components are compared by magnitude
        assert_abs_diff_eq!(
            model.components().mapv(f64::abs),
            array![
                [
                    0.423969864353,
                    0.392194236319,
                    0.402632149197,
                    0.386062226296,
                    0.409987986001,
                    0.432650432005
                ],
                [
                    0.417160863970,
                    0.256828707224,
                    0.594604739140,
                    0.337609404353,
                    0.176780937659,
                    0.511106078111
                ],
                [
                    0.334215776150,
                    0.550226558016,
                    0.147228776588,
                    0.525070260688,
                    0.536643283659,
                    0.013763888089
                ]
            ],
            epsilon = 1e-6
        );

        // The leading singular direction tracks how far the data sits from the origin, so it
        // explains almost none of the variance: the explained variance is *not* ordered like the
        // singular values. Any implementation that derives it from the spectrum alone can only
        // produce a decreasing sequence, so this single comparison rules that route out.
        let ev = model.explained_variance();
        assert!(
            ev[0] < ev[1],
            "explained variance must not inherit the singular-value ordering: {}",
            ev
        );
    }

    /// Unseen observations are projected with the learned components and nothing else.
    #[test]
    fn test_transform_of_unseen_observation() {
        let model = TruncatedSvd::params(2)
            .fit(&Dataset::from(hadamard_fixture()))
            .unwrap();

        let unseen = array![[1., -2., 3., 0.5], [-7., 0., 0., 1.]];
        let projected: Array2<f64> = model.predict(&unseen);
        let expected = unseen.dot(&model.components().t());

        assert_abs_diff_eq!(projected, expected, epsilon = 1e-12);

        // and a single held-out row projects to the corresponding row of that product
        let held_out = array![[1., -2., 3., 0.5]];
        let single: Array2<f64> = model.predict(&held_out);
        assert_abs_diff_eq!(single.row(0), projected.row(0), epsilon = 1e-12);
    }

    /// A degenerate all-zero input must not panic and must not produce NaNs.
    #[test]
    fn test_zero_matrix_is_finite() {
        let dataset = Dataset::from(Array2::<f64>::zeros((5, 3)));
        let model = TruncatedSvd::params(2)
            .fit(&dataset)
            .expect("fitting an all-zero matrix must not fail");

        assert!(
            model.singular_values().iter().all(|v| v.is_finite()),
            "singular values are not finite: {}",
            model.singular_values()
        );
        assert!(
            model.components().iter().all(|v| v.is_finite()),
            "components are not finite: {}",
            model.components()
        );
        assert_eq!(model.components().nrows(), model.singular_values().len());

        // pinned behavior: every singular value is below the solver's relative cutoff, so no
        // component at all is retained and the embedding collapses to width zero
        assert_eq!(model.n_components(), 0);

        let projected: Array2<f64> = model.predict(&dataset);
        assert_eq!(projected.shape(), &[5, 0]);

        // consequently there is nothing left to report a variance for
        assert_eq!(model.explained_variance().len(), 0);
        assert_eq!(model.explained_variance_ratio().len(), 0);
    }

    /// `inverse_transform` must not add a mean back, in contrast to `Pca`.
    #[test]
    fn test_inverse_transform_does_not_add_a_mean() {
        let x = hadamard_fixture();
        let dataset = Dataset::from(x.clone());
        let model = TruncatedSvd::params(4).fit(&dataset).unwrap();

        // the zero embedding maps back to the origin, not to the data mean
        let origin = model.inverse_transform(&Array2::zeros((1, 4)));
        assert_abs_diff_eq!(origin, Array2::<f64>::zeros((1, 4)), epsilon = 1e-12);

        // ... whereas PCA maps it to the mean, which is far from the origin here. Note that PCA
        // only retains three components on this fixture: centering removes the `v_0` direction,
        // which is exactly the direction the mean lives in.
        // `Pca` may itself retain fewer components than requested, so the probe is sized from the
        // fitted model rather than from the parameter.
        let pca = Pca::params(3).fit(&dataset).unwrap();
        let pca_k = pca.components().nrows();
        assert!(pca_k >= 1, "PCA retained no components");
        let pca_origin = pca.inverse_transform(Array2::zeros((1, pca_k)));
        let mean = x.mean_axis(Axis(0)).unwrap();
        assert_abs_diff_eq!(pca_origin.row(0).to_owned(), mean, epsilon = 1e-8);
        assert!(mean.dot(&mean) > 1.0, "the fixture mean is not far from 0");

        // `inverse_transform` is exactly the adjoint of the projection: no offset is involved
        assert_abs_diff_eq!(
            model.inverse_transform(&Array2::eye(4)),
            model.components().to_owned(),
            epsilon = 1e-12
        );
    }
}
