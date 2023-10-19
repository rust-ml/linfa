//! Linear Scaling methods

use crate::error::{PreprocessingError, Result};
use approx::abs_diff_eq;
use linfa::dataset::{AsTargets, DatasetBase, Float, WithLapack};
use linfa::traits::{Fit, Transformer};
#[cfg(not(feature = "blas"))]
use linfa_linalg::norm::Norm;
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix2, Zip};
#[cfg(feature = "blas")]
use ndarray_linalg::norm::Norm;

#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Debug, PartialEq, Eq)]
/// Possible scaling methods for [LinearScaler]
///
/// * Standard (with mean, with std): subtracts the mean to each feature and scales it by the inverse of its standard deviation
/// * MinMax (min, max): scales each feature to fit in the range `min..=max`, default values are
/// `0..=1`
/// * MaxAbs: scales each feature by the inverse of its maximum absolute value, so that it fits the range `-1..=1`
pub enum ScalingMethod<F: Float> {
    Standard(bool, bool),
    MinMax(F, F),
    MaxAbs,
}

impl<F: Float> ScalingMethod<F> {
    pub(crate) fn fit<D: Data<Elem = F>>(
        &self,
        records: &ArrayBase<D, Ix2>,
    ) -> Result<LinearScaler<F>> {
        match self {
            ScalingMethod::Standard(a, b) => Self::standardize(records, *a, *b),
            ScalingMethod::MinMax(a, b) => Self::min_max(records, *a, *b),
            ScalingMethod::MaxAbs => Self::max_abs(records),
        }
    }

    fn standardize<D: Data<Elem = F>>(
        records: &ArrayBase<D, Ix2>,
        with_mean: bool,
        with_std: bool,
    ) -> Result<LinearScaler<F>> {
        if records.dim().0 == 0 {
            return Err(PreprocessingError::NotEnoughSamples);
        }
        // safe unwrap because of above zero records check
        let means = records.mean_axis(Axis(0)).unwrap();
        let std_devs = if with_std {
            records.std_axis(Axis(0), F::zero()).mapv(|s| {
                if abs_diff_eq!(s, F::zero()) {
                    // if feature is constant then don't scale
                    F::one()
                } else {
                    F::one() / s
                }
            })
        } else {
            Array1::ones(records.dim().1)
        };
        Ok(LinearScaler {
            offsets: means,
            scales: std_devs,
            method: ScalingMethod::Standard(with_mean, with_std),
        })
    }

    fn min_max<D: Data<Elem = F>>(
        records: &ArrayBase<D, Ix2>,
        min: F,
        max: F,
    ) -> Result<LinearScaler<F>> {
        if records.dim().0 == 0 {
            return Err(PreprocessingError::NotEnoughSamples);
        } else if min > max {
            return Err(PreprocessingError::FlippedMinMaxRange);
        }

        let mins = records.fold_axis(
            Axis(0),
            F::infinity(),
            |&x, &prev| if x < prev { x } else { prev },
        );
        let mut scales =
            records.fold_axis(
                Axis(0),
                F::neg_infinity(),
                |&x, &prev| if x > prev { x } else { prev },
            );
        Zip::from(&mut scales).and(&mins).for_each(|max, min| {
            if abs_diff_eq!(*max - *min, F::zero()) {
                // if feature is constant then don't scale
                *max = F::one();
            } else {
                *max = F::one() / (*max - *min);
            }
        });
        Ok(LinearScaler {
            offsets: mins,
            scales,
            method: ScalingMethod::MinMax(min, max),
        })
    }

    fn max_abs<D: Data<Elem = F>>(records: &ArrayBase<D, Ix2>) -> Result<LinearScaler<F>> {
        if records.dim().0 == 0 {
            return Err(PreprocessingError::NotEnoughSamples);
        }
        let scales: Array1<F> = records.map_axis(Axis(0), |col| {
            let norm_max = F::cast(col.with_lapack().norm_max());
            if abs_diff_eq!(norm_max, F::zero()) {
                // if feature is constant at zero then don't scale
                F::one()
            } else {
                F::one() / norm_max
            }
        });

        let offsets = Array1::zeros(records.dim().1);
        Ok(LinearScaler {
            offsets,
            scales,
            method: ScalingMethod::MaxAbs,
        })
    }
}

impl<F: Float> std::fmt::Display for ScalingMethod<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScalingMethod::Standard(with_mean, with_std) => write!(
                f,
                "Standard scaler (with_mean = {}, with_std = {})",
                with_mean, with_std
            ),
            ScalingMethod::MinMax(min, max) => {
                write!(f, "Min-Max scaler (min = {}, max = {})", min, max)
            }
            ScalingMethod::MaxAbs => write!(f, "MaxAbs scaler"),
        }
    }
}

/// Linear Scaler: learns scaling parameters, according to the specified [method](ScalingMethod), from a dataset, producing a [fitted linear scaler](LinearScaler)
/// that can be used to scale different datasets using the same parameters.
///
///
/// ### Example
///
/// ```rust
/// use linfa::traits::{Fit, Transformer};
/// use linfa_preprocessing::linear_scaling::LinearScaler;
///
/// // Load dataset
/// let dataset = linfa_datasets::diabetes();
/// // Learn scaling parameters
/// let scaler = LinearScaler::standard().fit(&dataset).unwrap();
/// // scale dataset according to parameters
/// let dataset = scaler.transform(dataset);
/// ```
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinearScalerParams<F: Float> {
    method: ScalingMethod<F>,
}

impl<F: Float> LinearScalerParams<F> {
    /// Initializes the scaler with the specified method.
    pub fn new(method: ScalingMethod<F>) -> Self {
        Self { method }
    }

    /// Setter for the scaler method
    pub fn method(mut self, method: ScalingMethod<F>) -> Self {
        self.method = method;
        self
    }
}

impl<F: Float> LinearScaler<F> {
    /// Initializes a Standard scaler
    pub fn standard() -> LinearScalerParams<F> {
        LinearScalerParams {
            method: ScalingMethod::Standard(true, true),
        }
    }

    /// Initializes a Standard scaler that does not subract the mean to the features
    pub fn standard_no_mean() -> LinearScalerParams<F> {
        LinearScalerParams {
            method: ScalingMethod::Standard(false, true),
        }
    }

    /// Initializes a Stadard scaler that does not scale the features by the inverse of the standard deviation
    pub fn standard_no_std() -> LinearScalerParams<F> {
        LinearScalerParams {
            method: ScalingMethod::Standard(true, false),
        }
    }

    /// Initializes a MinMax scaler with range `0..=1`
    pub fn min_max() -> LinearScalerParams<F> {
        LinearScalerParams {
            method: ScalingMethod::MinMax(F::zero(), F::one()),
        }
    }

    /// Initializes a MinMax scaler with the specified minimum and maximum values for the range.
    ///
    /// If `min` is bigger than `max` then fitting will return an error on any input.
    pub fn min_max_range(min: F, max: F) -> LinearScalerParams<F> {
        LinearScalerParams {
            method: ScalingMethod::MinMax(min, max),
        }
    }

    /// Initializes a MaxAbs scaler
    pub fn max_abs() -> LinearScalerParams<F> {
        LinearScalerParams {
            method: ScalingMethod::MaxAbs,
        }
    }
}

impl<F: Float, D: Data<Elem = F>, T: AsTargets> Fit<ArrayBase<D, Ix2>, T, PreprocessingError>
    for LinearScalerParams<F>
{
    type Object = LinearScaler<F>;

    /// Fits the input dataset accordng to the scaler [method](ScalingMethod). Will return an error
    /// if the dataset does not contain any samples or (in the case of MinMax scaling) if the specified range is not valid.
    fn fit(&self, x: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {
        self.method.fit(x.records())
    }
}

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone, PartialEq, Eq)]
/// The result of fitting a [linear scaler](LinearScalerParams).
/// Scales datasets with the parameters learned during fitting.
pub struct LinearScaler<F: Float> {
    offsets: Array1<F>,
    scales: Array1<F>,
    method: ScalingMethod<F>,
}

impl<F: Float> LinearScaler<F> {
    /// Array of size `n_features` that contains the offset that will be subtracted to each feature
    pub fn offsets(&self) -> &Array1<F> {
        &self.offsets
    }

    /// Array of size `n_features` that contains the scale that will be applied to each feature
    pub fn scales(&self) -> &Array1<F> {
        &self.scales
    }

    /// Returns the method used for fitting. Useful for printing, since [ScalingMethod] implements `Display`
    pub fn method(&self) -> &ScalingMethod<F> {
        &self.method
    }
}

impl<F: Float> Transformer<Array2<F>, Array2<F>> for LinearScaler<F> {
    /// Scales an array of size (nsamples, nfeatures) according to the scaler's `offsets` and `scales`.
    /// Panics if the shape of the input array is not compatible with the shape of the dataset used for fitting.
    fn transform(&self, x: Array2<F>) -> Array2<F> {
        if x.is_empty() {
            return x;
        }
        let mut x = x;
        Zip::from(x.columns_mut())
            .and(self.offsets())
            .and(self.scales())
            .for_each(|mut col, &offset, &scale| {
                if let ScalingMethod::Standard(false, _) = self.method {
                    col.mapv_inplace(|el| (el - offset) * scale + offset);
                } else {
                    col.mapv_inplace(|el| (el - offset) * scale);
                }
            });
        match &self.method {
            ScalingMethod::MinMax(min, max) => x * (*max - *min) + *min,
            _ => x,
        }
    }
}

impl<F: Float, D: Data<Elem = F>, T: AsTargets>
    Transformer<DatasetBase<ArrayBase<D, Ix2>, T>, DatasetBase<Array2<F>, T>> for LinearScaler<F>
{
    /// Substitutes the records of the dataset with their scaled version.
    /// Panics if the shape of the records is not compatible with the shape of the dataset used for fitting.
    fn transform(&self, x: DatasetBase<ArrayBase<D, Ix2>, T>) -> DatasetBase<Array2<F>, T> {
        let feature_names = x.feature_names();
        let (records, targets, weights) = (x.records, x.targets, x.weights);
        let records = self.transform(records.to_owned());
        DatasetBase::new(records, targets)
            .with_weights(weights)
            .with_feature_names(feature_names)
    }
}

#[cfg(test)]
mod tests {
    use crate::linear_scaling::{LinearScaler, LinearScalerParams};
    use approx::assert_abs_diff_eq;
    use linfa::dataset::DatasetBase;
    use linfa::traits::{Fit, Transformer};
    use ndarray::{array, Array2, Axis};

    #[test]
    fn autotraits() {
        fn has_autotraits<T: Send + Sync + Sized + Unpin>() {}
        has_autotraits::<LinearScaler<f64>>();
        has_autotraits::<LinearScalerParams<f64>>();
        has_autotraits::<ScalingMethod<f64>>();
    }

    #[test]
    fn test_max_abs() {
        let dataset = array![[1., -1.], [2., -2.], [3., -3.], [4., -5.]].into();
        let scaler = LinearScaler::max_abs().fit(&dataset).unwrap();
        let scaled = scaler.transform(dataset);
        let col0 = scaled.records().column(0);
        let col1 = scaled.records().column(1);
        assert_abs_diff_eq!(col0, array![1. / 4., 2. / 4., 3. / 4., 1.]);
        assert_abs_diff_eq!(col1, array![-1. / 5., -2. / 5., -3. / 5., -1.]);
    }

    #[test]
    fn test_standard_scaler() {
        let dataset = array![[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]].into();
        let scaler = LinearScaler::standard().fit(&dataset).unwrap();
        assert_abs_diff_eq!(*scaler.offsets(), array![1., 0., 1. / 3.]);
        assert_abs_diff_eq!(
            *scaler.scales(),
            array![1. / 0.81, 1. / 0.81, 1. / 1.24],
            epsilon = 1e-2
        );
        let transformed = scaler.transform(dataset);
        let means = transformed.records().mean_axis(Axis(0)).unwrap();
        let std_devs = transformed.records().std_axis(Axis(0), 0.);
        assert_abs_diff_eq!(means, array![0., 0., 0.]);
        assert_abs_diff_eq!(std_devs, array![1., 1., 1.]);
    }

    #[test]
    fn test_standard_scaler_no_mean() {
        let dataset = array![[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]].into();
        let scaler = LinearScaler::standard_no_mean().fit(&dataset).unwrap();
        assert_abs_diff_eq!(*scaler.offsets(), array![1., 0., 1. / 3.]);
        assert_abs_diff_eq!(
            *scaler.scales(),
            array![1. / 0.81, 1. / 0.81, 1. / 1.24],
            epsilon = 1e-2
        );
        let transformed = scaler.transform(dataset);
        let means = transformed.records().mean_axis(Axis(0)).unwrap();
        let std_devs = transformed.records().std_axis(Axis(0), 0.);
        assert_abs_diff_eq!(means, array![1., 0., (1. / 3.)], epsilon = 1e-2);
        assert_abs_diff_eq!(std_devs, array![1., 1., 1.]);
    }

    #[test]
    fn test_standard_scaler_no_std() {
        let dataset = array![[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]].into();
        let scaler = LinearScaler::standard_no_std().fit(&dataset).unwrap();
        assert_abs_diff_eq!(*scaler.offsets(), array![1., 0., 1. / 3.]);
        assert_abs_diff_eq!(*scaler.scales(), array![1., 1., 1.],);
        let transformed = scaler.transform(dataset);
        let means = transformed.records().mean_axis(Axis(0)).unwrap();
        let std_devs = transformed.records().std_axis(Axis(0), 0.);
        assert_abs_diff_eq!(means, array![0., 0., 0.]);
        assert_abs_diff_eq!(std_devs, array![0.81, 0.81, 1.24], epsilon = 1e-2);
    }

    use super::ScalingMethod;

    #[test]
    fn test_standard_scaler_no_both() {
        let dataset = array![[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]].into();
        let scaler = LinearScalerParams::new(ScalingMethod::Standard(false, false))
            .fit(&dataset)
            .unwrap();

        let original_means = dataset.records().mean_axis(Axis(0)).unwrap();
        let original_stds = dataset.records().std_axis(Axis(0), 0.);

        assert_abs_diff_eq!(*scaler.offsets(), original_means);
        assert_abs_diff_eq!(*scaler.scales(), array![1., 1., 1.],);

        let transformed = scaler.transform(dataset);

        let means = transformed.records().mean_axis(Axis(0)).unwrap();
        let std_devs = transformed.records().std_axis(Axis(0), 0.);

        assert_abs_diff_eq!(means, original_means);
        assert_abs_diff_eq!(std_devs, original_stds, epsilon = 1e-2);
    }

    #[test]
    fn test_min_max_scaler() {
        let dataset = array![[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]].into();
        let scaler = LinearScaler::min_max().fit(&dataset).unwrap();
        assert_abs_diff_eq!(*scaler.offsets(), array![0., -1., -1.]);
        assert_abs_diff_eq!(*scaler.scales(), array![1. / 2., 1. / 2., 1. / 3.]);
        let transformed = scaler.transform(dataset);
        let mins = transformed
            .records()
            .fold_axis(
                Axis(0),
                f64::INFINITY,
                |&x, &prev| if x < prev { x } else { prev },
            );
        let maxes = transformed
            .records()
            .fold_axis(
                Axis(0),
                f64::NEG_INFINITY,
                |&x, &prev| if x > prev { x } else { prev },
            );
        assert_abs_diff_eq!(maxes, array![1., 1., 1.]);
        assert_abs_diff_eq!(mins, array![0., 0., 0.]);
    }

    #[test]
    fn test_min_max_scaler_range() {
        let dataset = array![[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]].into();
        let scaler = LinearScaler::min_max_range(5., 10.).fit(&dataset).unwrap();
        assert_abs_diff_eq!(*scaler.offsets(), array![0., -1., -1.]);
        assert_abs_diff_eq!(*scaler.scales(), array![1. / 2., 1. / 2., 1. / 3.]);
        let transformed = scaler.transform(dataset);
        let mins = transformed
            .records()
            .fold_axis(
                Axis(0),
                f64::INFINITY,
                |&x, &prev| if x < prev { x } else { prev },
            );
        let maxes = transformed
            .records()
            .fold_axis(
                Axis(0),
                f64::NEG_INFINITY,
                |&x, &prev| if x > prev { x } else { prev },
            );
        assert_abs_diff_eq!(mins, array![5., 5., 5.]);
        assert_abs_diff_eq!(maxes, array![10., 10., 10.]);
    }

    #[test]
    fn test_standard_const_feature() {
        let dataset = array![[1., 2., 2.], [2., 2., 0.], [0., 2., -1.]].into();
        let scaler = LinearScaler::standard().fit(&dataset).unwrap();
        assert_abs_diff_eq!(*scaler.offsets(), array![1., 2., 1. / 3.]);
        assert_abs_diff_eq!(
            *scaler.scales(),
            array![1. / 0.81, 1., 1. / 1.24],
            epsilon = 1e-2
        );
        let transformed = scaler.transform(dataset);
        let means = transformed.records().mean_axis(Axis(0)).unwrap();
        let std_devs = transformed.records().std_axis(Axis(0), 0.);
        assert_abs_diff_eq!(means, array![0., 0., 0.]);
        // 0 std dev on constant feature
        assert_abs_diff_eq!(std_devs, array![1., 0., 1.]);
    }

    #[test]
    fn test_max_abs_const_null_feature() {
        let dataset = array![[1., 0.], [2., 0.], [3., 0.], [4., 0.]].into();
        let scaler = LinearScaler::max_abs().fit(&dataset).unwrap();
        let scaled = scaler.transform(dataset);
        let col0 = scaled.records().column(0);
        let col1 = scaled.records().column(1);
        assert_abs_diff_eq!(col0, array![1. / 4., 2. / 4., 3. / 4., 1.]);
        // const 0 feature stays zero
        assert_abs_diff_eq!(col1, array![0., 0., 0., 0.]);
    }

    #[test]
    fn test_min_max_scaler_const_feature() {
        let dataset = array![[1., -1., 2.], [2., 0., 2.], [0., 1., 2.]].into();
        let scaler = LinearScaler::min_max().fit(&dataset).unwrap();
        assert_abs_diff_eq!(*scaler.offsets(), array![0., -1., 2.]);
        assert_abs_diff_eq!(*scaler.scales(), array![1. / 2., 1. / 2., 1.]);
        let transformed = scaler.transform(dataset);
        let mins = transformed
            .records()
            .fold_axis(
                Axis(0),
                f64::INFINITY,
                |&x, &prev| if x < prev { x } else { prev },
            );
        let maxes = transformed
            .records()
            .fold_axis(
                Axis(0),
                f64::NEG_INFINITY,
                |&x, &prev| if x > prev { x } else { prev },
            );
        // 0 max for constant feature
        assert_abs_diff_eq!(maxes, array![1., 1., 0.]);
        assert_abs_diff_eq!(mins, array![0., 0., 0.]);
    }

    #[test]
    fn test_empty_input() {
        let dataset: DatasetBase<Array2<f64>, _> =
            Array2::from_shape_vec((0, 0), vec![]).unwrap().into();
        let scaler = LinearScaler::standard().fit(&dataset);
        assert_eq!(
            scaler.err().unwrap().to_string(),
            "not enough samples".to_string()
        );
        let scaler = LinearScaler::standard_no_mean().fit(&dataset);
        assert_eq!(
            scaler.err().unwrap().to_string(),
            "not enough samples".to_string()
        );
        let scaler = LinearScaler::standard_no_std().fit(&dataset);
        assert_eq!(
            scaler.err().unwrap().to_string(),
            "not enough samples".to_string()
        );
        let scaler = LinearScaler::min_max().fit(&dataset);
        assert_eq!(
            scaler.err().unwrap().to_string(),
            "not enough samples".to_string()
        );
        let scaler = LinearScaler::max_abs().fit(&dataset);
        assert_eq!(
            scaler.err().unwrap().to_string(),
            "not enough samples".to_string()
        );
    }

    #[test]
    fn test_transform_empty_array() {
        let empty: Array2<f64> = Array2::from_shape_vec((0, 0), vec![]).unwrap();
        let dataset = array![[1., -1., 2.], [2., 0., 2.], [0., 1., 2.]].into();
        let scaler = LinearScaler::standard().fit(&dataset).unwrap();
        let transformed = scaler.transform(empty.clone());
        assert!(transformed.is_empty());
        let scaler = LinearScaler::standard_no_mean().fit(&dataset).unwrap();
        let transformed = scaler.transform(empty.clone());
        assert!(transformed.is_empty());
        let scaler = LinearScaler::standard_no_std().fit(&dataset).unwrap();
        let transformed = scaler.transform(empty.clone());
        assert!(transformed.is_empty());
        let scaler = LinearScaler::min_max().fit(&dataset).unwrap();
        let transformed = scaler.transform(empty.clone());
        assert!(transformed.is_empty());
        let scaler = LinearScaler::max_abs().fit(&dataset).unwrap();
        let transformed = scaler.transform(empty);
        assert!(transformed.is_empty());
    }

    #[test]
    fn test_retain_feature_names() {
        let dataset = linfa_datasets::diabetes();
        let original_feature_names = dataset.feature_names();
        let transformed = LinearScaler::standard()
            .fit(&dataset)
            .unwrap()
            .transform(dataset);
        assert_eq!(original_feature_names, transformed.feature_names())
    }

    #[test]
    #[should_panic]
    fn test_transform_wrong_size_array_standard() {
        let wrong_size = Array2::from_shape_vec((1, 2), vec![0., 0.]).unwrap();
        let dataset = array![[1., -1., 2.], [2., 0., 2.], [0., 1., 2.]].into();
        let scaler = LinearScaler::standard().fit(&dataset).unwrap();
        let _transformed = scaler.transform(wrong_size);
    }
    #[test]
    #[should_panic]
    fn test_transform_wrong_size_array_standard_no_mean() {
        let wrong_size = Array2::from_shape_vec((1, 2), vec![0., 0.]).unwrap();
        let dataset = array![[1., -1., 2.], [2., 0., 2.], [0., 1., 2.]].into();
        let scaler = LinearScaler::standard_no_mean().fit(&dataset).unwrap();
        let _transformed = scaler.transform(wrong_size);
    }
    #[test]
    #[should_panic]
    fn test_transform_wrong_size_array_standard_no_std() {
        let wrong_size = Array2::from_shape_vec((1, 2), vec![0., 0.]).unwrap();
        let dataset = array![[1., -1., 2.], [2., 0., 2.], [0., 1., 2.]].into();
        let scaler = LinearScaler::standard_no_std().fit(&dataset).unwrap();
        let _transformed = scaler.transform(wrong_size);
    }
    #[test]
    #[should_panic]
    fn test_transform_wrong_size_array_min_max() {
        let wrong_size = Array2::from_shape_vec((1, 2), vec![0., 0.]).unwrap();
        let dataset = array![[1., -1., 2.], [2., 0., 2.], [0., 1., 2.]].into();
        let scaler = LinearScaler::min_max().fit(&dataset).unwrap();
        let _transformed = scaler.transform(wrong_size);
    }
    #[test]
    #[should_panic]
    fn test_transform_wrong_size_array_max_abs() {
        let wrong_size = Array2::from_shape_vec((1, 2), vec![0., 0.]).unwrap();
        let dataset = array![[1., -1., 2.], [2., 0., 2.], [0., 1., 2.]].into();
        let scaler = LinearScaler::max_abs().fit(&dataset).unwrap();
        let _transformed = scaler.transform(wrong_size);
    }

    #[test]
    #[should_panic]
    fn test_min_max_wrong_range() {
        let dataset = array![[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]].into();
        let _scaler = LinearScaler::min_max_range(10., 5.).fit(&dataset).unwrap();
    }
}
