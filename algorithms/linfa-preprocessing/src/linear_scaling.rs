//! Linear Scaling methods

use crate::error::{Error, Result};
use crate::Float;
use approx::abs_diff_eq;
use linfa::dataset::{AsTargets, DatasetBase};
use linfa::traits::{Fit, Transformer};
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix2, Zip};
use ndarray_linalg::norm::Norm;

#[derive(Clone, Debug)]
/// Possible scaling methods for [LinearScaler](struct.LinearScaler.html)
///
/// * Standard (with mean, with std): subtracts the mean to each feature and scales it by the inverse of its standard deviation
/// * MinMax (min, max): scales each feature to fit in the range [min,max], default values are [0,1]
/// * MaxAbs: scales each feature by the inverse of its maximum absolute value, so that it fits the range [-1,1]
pub enum ScalingMethod<F: Float> {
    Standard(bool, bool),
    MinMax(F, F),
    MaxAbs,
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

/// Linear Scaler: learns scaling parameters, according to the specified [method](enum.ScalingMethod.html), from a dataset, producing a [fitted linear scaler](struct.FittedLinearScaler.html)
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
pub struct LinearScaler<F: Float> {
    method: ScalingMethod<F>,
}

impl<F: Float> LinearScaler<F> {
    /// Initializes the scaler with the specified method.
    pub fn new(method: ScalingMethod<F>) -> Self {
        Self { method }
    }

    /// Setter for the scaler method
    pub fn method(mut self, method: ScalingMethod<F>) -> Self {
        self.method = method;
        self
    }

    /// Initializes a Standard scaler
    pub fn standard() -> Self {
        Self {
            method: ScalingMethod::Standard(true, true),
        }
    }

    /// Initializes a Standard scaler that does not subract the mean to the features
    pub fn standard_no_mean() -> Self {
        Self {
            method: ScalingMethod::Standard(false, true),
        }
    }

    /// Initializes a Stadard scaler that does not scale the features by the inverse of the standard deviation
    pub fn standard_no_std() -> Self {
        Self {
            method: ScalingMethod::Standard(true, false),
        }
    }

    /// Initializes a MinMax scaler with range [0,1]
    pub fn min_max() -> Self {
        Self {
            method: ScalingMethod::MinMax(F::zero(), F::one()),
        }
    }

    /// Initializes a MinMax scaler with the specified minimum and maximum values for the range.
    ///
    /// If `min` is bigger than `max` then fitting will return an error on any input.
    pub fn min_max_range(min: F, max: F) -> Self {
        Self {
            method: ScalingMethod::MinMax(min, max),
        }
    }

    /// Initializes a MaxAbs scaler
    pub fn max_abs() -> Self {
        Self {
            method: ScalingMethod::MaxAbs,
        }
    }
}

impl<'a, F: Float, D: Data<Elem = F>, T: AsTargets> Fit<'a, ArrayBase<D, Ix2>, T>
    for LinearScaler<F>
{
    type Object = Result<FittedLinearScaler<F>>;

    /// Fits the input dataset accordng to the scaler [method](enum.ScalingMethod.html). Will return an error
    /// if the dataset does not contain any samples or (in the case of MinMax scaling) if the specified range is not valid.
    fn fit(&self, x: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Self::Object {
        match &self.method {
            ScalingMethod::Standard(with_mean, with_std) => {
                FittedLinearScaler::standard(x.records(), *with_mean, *with_std)
            }
            ScalingMethod::MinMax(min, max) => FittedLinearScaler::min_max(x.records(), *min, *max),
            ScalingMethod::MaxAbs => FittedLinearScaler::max_abs(x.records()),
        }
    }
}

#[derive(Debug)]
/// The result of fitting a [linear scaler](struct.LinearScaler.html).
/// Scales datasets with the parameters learned during fitting.
pub struct FittedLinearScaler<F: Float> {
    offsets: Array1<F>,
    scales: Array1<F>,
    method: ScalingMethod<F>,
}

impl<F: Float> FittedLinearScaler<F> {
    pub(crate) fn standard<D: Data<Elem = F>>(
        records: &ArrayBase<D, Ix2>,
        with_mean: bool,
        with_std: bool,
    ) -> Result<Self> {
        if records.dim().0 == 0 {
            return Err(Error::NotEnoughSamples);
        }
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
        Ok(Self {
            offsets: means,
            scales: std_devs,
            method: ScalingMethod::Standard(with_mean, with_std),
        })
    }

    pub(crate) fn min_max<D: Data<Elem = F>>(
        records: &ArrayBase<D, Ix2>,
        min: F,
        max: F,
    ) -> Result<Self> {
        if records.dim().0 == 0 {
            return Err(Error::NotEnoughSamples);
        }
        if min > max {
            return Err(Error::FlippedMinMaxRange);
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
        Zip::from(&mut scales).and(&mins).apply(|max, min| {
            if abs_diff_eq!(*max - *min, F::zero()) {
                // if feature is constant then don't scale
                *max = F::one();
            } else {
                *max = F::one() / (*max - *min);
            }
        });
        Ok(Self {
            offsets: mins,
            scales,
            method: ScalingMethod::MinMax(min, max),
        })
    }

    pub(crate) fn max_abs<D: Data<Elem = F>>(records: &ArrayBase<D, Ix2>) -> Result<Self> {
        if records.dim().0 == 0 {
            return Err(Error::NotEnoughSamples);
        }
        let scales = records.map_axis(Axis(0), |col| {
            let norm_max = F::from(col.norm_max()).unwrap();
            if abs_diff_eq!(norm_max, F::zero()) {
                // if feature is constant at zero then don't scale
                F::one()
            } else {
                F::one() / norm_max
            }
        });
        let offsets = Array1::zeros(records.dim().1);
        Ok(Self {
            offsets,
            scales,
            method: ScalingMethod::MaxAbs,
        })
    }

    /// Array of size `n_features` that contains the offset that will be subtracted to each feature
    pub fn offsets(&self) -> &Array1<F> {
        &self.offsets
    }

    /// Array of size `n_features` that contains the scale that will be applied to each feature
    pub fn scales(&self) -> &Array1<F> {
        &self.scales
    }

    /// Returns the method used for fitting. Useful for printing, since [ScalingMethod](enum.ScalingMethod.html) implements `Display`
    pub fn method(&self) -> &ScalingMethod<F> {
        &self.method
    }
}

impl<F: Float> Transformer<Array2<F>, Array2<F>> for FittedLinearScaler<F> {
    /// Scales an array of size (nsamples, nfeatures) according to the scaler's `offsets` and `scales`.
    /// Panics if the shape of the input array is not compatible with the shape of the dataset used for fitting.
    fn transform(&self, x: Array2<F>) -> Array2<F> {
        if x.is_empty() {
            return x;
        }
        let mut x = x;
        Zip::from(x.gencolumns_mut())
            .and(self.offsets())
            .and(self.scales())
            .apply(|mut col, &offset, &scale| {
                if let ScalingMethod::Standard(false, true) = self.method {
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
    Transformer<DatasetBase<ArrayBase<D, Ix2>, T>, DatasetBase<Array2<F>, T>>
    for FittedLinearScaler<F>
{
    /// Substitutes the records of the dataset with their scaled version.
    /// Panics if the shape of the records is not compatible with the shape of the dataset used for fitting.
    fn transform(&self, x: DatasetBase<ArrayBase<D, Ix2>, T>) -> DatasetBase<Array2<F>, T> {
        let transformed_records = self.transform(x.records.to_owned());
        x.with_records(transformed_records)
    }
}

#[cfg(test)]
mod tests {

    use crate::linear_scaling::LinearScaler;
    use approx::assert_abs_diff_eq;
    use linfa::dataset::DatasetBase;
    use linfa::traits::{Fit, Transformer};
    use ndarray::{array, Array2, Axis};

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
