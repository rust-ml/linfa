use crate::error::{Error, Result};
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

pub struct NormScaler {
    norm: Norms,
}

impl NormScaler {
    pub fn l2() -> Self {
        Self { norm: Norms::L2 }
    }

    pub fn l1() -> Self {
        Self { norm: Norms::L1 }
    }

    pub fn max() -> Self {
        Self { norm: Norms::Max }
    }
}

impl<F: Float> Transformer<Array2<F>, Array2<F>> for NormScaler {
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
    fn transform(&self, x: DatasetBase<ArrayBase<D, Ix2>, T>) -> DatasetBase<Array2<F>, T> {
        let transformed_records = self.transform(x.records.to_owned());
        x.with_records(transformed_records)
    }
}

#[cfg(test)]
mod tests {

    use crate::NormScaler;
    use approx::assert_abs_diff_eq;
    use linfa::dataset::DatasetBase;
    use linfa::traits::Transformer;
    use ndarray::array;

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
}
