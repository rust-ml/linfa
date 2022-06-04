//! Sample normalization methods
use linfa::dataset::{AsTargets, DatasetBase, Float, WithLapack, WithoutLapack};
use linfa::traits::Transformer;
#[cfg(not(feature = "blas"))]
use linfa_linalg::norm::Norm;
use ndarray::{Array2, ArrayBase, Axis, Data, Ix2, Zip};
#[cfg(feature = "blas")]
use ndarray_linalg::norm::Norm;

enum Norms {
    L1,
    L2,
    Max,
}

/// Norm scaler: scales all samples in a dataset to have unit norm, according to the specified norm
/// measure
///
/// ### Example
///
/// ```rust
/// use linfa::traits::Transformer;
/// use linfa_preprocessing::norm_scaling::NormScaler;
///
/// // Load dataset
/// let dataset = linfa_datasets::diabetes();
/// // Initialize scaler
/// let scaler = NormScaler::l2();
/// // Scale dataset
/// let dataset = scaler.transform(dataset);
/// ```
pub struct NormScaler {
    norm: Norms,
}

impl NormScaler {
    /// Initializes a norm scaler that uses l2 norm
    pub fn l2() -> Self {
        Self { norm: Norms::L2 }
    }

    /// Initializes a norm scaler that uses l1 norm
    pub fn l1() -> Self {
        Self { norm: Norms::L1 }
    }

    /// Initializes a norm scaler that uses max norm
    pub fn max() -> Self {
        Self { norm: Norms::Max }
    }
}

impl<F: Float> Transformer<Array2<F>, Array2<F>> for NormScaler {
    /// Scales all samples in the array of shape (nsamples, nfeatures) to have unit norm.
    fn transform(&self, x: Array2<F>) -> Array2<F> {
        // add Lapack trait bound
        let x = x.with_lapack();

        let norms = match &self.norm {
            Norms::L1 => x.map_axis(Axis(1), |row| F::cast(row.norm_l1())),
            Norms::L2 => x.map_axis(Axis(1), |row| F::cast(row.norm_l2())),
            Norms::Max => x.map_axis(Axis(1), |row| F::cast(row.norm_max())),
        };

        // remove Lapack trait bound
        let mut x = x.without_lapack();

        Zip::from(x.rows_mut())
            .and(&norms)
            .for_each(|mut row, &norm| {
                row.mapv_inplace(|el| el / norm);
            });
        x
    }
}

impl<F: Float, D: Data<Elem = F>, T: AsTargets>
    Transformer<DatasetBase<ArrayBase<D, Ix2>, T>, DatasetBase<Array2<F>, T>> for NormScaler
{
    /// Substitutes the records of the dataset with their scaled versions with unit norm.
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

    use crate::norm_scaling::NormScaler;
    use approx::assert_abs_diff_eq;
    use linfa::dataset::DatasetBase;
    use linfa::traits::Transformer;
    use ndarray::{array, Array2};

    #[test]
    fn test_norm_l2() {
        let dataset = DatasetBase::from(array![[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]]);
        let scaler = NormScaler::l2();
        let normalized_data = scaler.transform(dataset);
        let ground_truth = array![[0.4, -0.4, 0.81], [1., 0., 0.], [0., 0.7, -0.7]];
        assert_abs_diff_eq!(*normalized_data.records(), ground_truth, epsilon = 1e-2);
    }

    #[test]
    fn test_norm_l1() {
        let dataset = DatasetBase::from(array![[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]]);
        let scaler = NormScaler::l1();
        let normalized_data = scaler.transform(dataset);
        let ground_truth = array![[0.25, -0.25, 0.5], [1., 0., 0.], [0., 0.5, -0.5]];
        assert_abs_diff_eq!(*normalized_data.records(), ground_truth, epsilon = 1e-2);
    }

    #[test]
    fn test_norm_max() {
        let dataset = DatasetBase::from(array![[1., -1., 2.], [2., 0., 0.], [0., 1., -1.]]);
        let scaler = NormScaler::max();
        let normalized_data = scaler.transform(dataset);
        let ground_truth = array![[0.5, -0.5, 1.], [1., 0., 0.], [0., 1., -1.]];
        assert_abs_diff_eq!(*normalized_data.records(), ground_truth, epsilon = 1e-2);
    }

    #[test]
    fn test_no_input() {
        let input: Array2<f64> = Array2::from_shape_vec((0, 0), vec![]).unwrap();
        let ground_truth: Array2<f64> = Array2::from_shape_vec((0, 0), vec![]).unwrap();
        let scaler = NormScaler::max();
        assert_abs_diff_eq!(scaler.transform(input.clone()), ground_truth);
        let scaler = NormScaler::l1();
        assert_abs_diff_eq!(scaler.transform(input.clone()), ground_truth);
        let scaler = NormScaler::l2();
        assert_abs_diff_eq!(scaler.transform(input), ground_truth);
    }

    #[test]
    fn test_retain_feature_names() {
        let dataset = linfa_datasets::diabetes();
        let original_feature_names = dataset.feature_names();
        let transformed = NormScaler::l2().transform(dataset);
        assert_eq!(original_feature_names, transformed.feature_names())
    }
}
