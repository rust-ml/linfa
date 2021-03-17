//! Sample normalization methods
use crate::Float;
use linfa::dataset::{AsTargets, DatasetBase};
use linfa::traits::Transformer;
use ndarray::{Array2, ArrayBase, Axis, Data, Ix2, Zip};
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
/// use linfa_preprocessing::norm_scaler::NormScaler;
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
        let mut x = x;
        let norms = match &self.norm {
            Norms::L1 => x.map_axis(Axis(1), |row| F::from(row.norm_l1()).unwrap()),
            Norms::L2 => x.map_axis(Axis(1), |row| F::from(row.norm_l2()).unwrap()),
            Norms::Max => x.map_axis(Axis(1), |row| F::from(row.norm_max()).unwrap()),
        };
        Zip::from(x.genrows_mut())
            .and(&norms)
            .apply(|mut row, &norm| {
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
        let transformed_records = self.transform(x.records.to_owned());
        x.with_records(transformed_records)
    }
}

#[cfg(test)]
mod tests {

    use crate::norm_scaler::NormScaler;
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
        assert_abs_diff_eq!(scaler.transform(input.clone()), ground_truth);
    }
}
