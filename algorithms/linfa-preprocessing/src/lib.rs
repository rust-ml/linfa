use linfa::dataset::{AsTargets, DatasetBase};
use linfa::traits::{Fit, Transformer};
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix2};
use ndarray_linalg::norm::normalize;

pub mod error;

use crate::error::{Error, Result};

pub trait Float: linfa::Float + ndarray_linalg::Lapack {}

impl Float for f32 {}
impl Float for f64 {}

#[derive(Clone, Debug)]
pub enum ScalingMethod<F: Float> {
    Standard,
    MinMax(F, F),
    MaxAbs,
    Norm,
}

impl<F: Float> std::fmt::Display for ScalingMethod<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScalingMethod::Standard => write!(f, "Standard scaler"),
            ScalingMethod::MinMax(min, max) => {
                write!(f, "Min-Max scaler (min = {}, max = {})", min, max)
            }
            ScalingMethod::MaxAbs => write!(f, "MaxAbs scaler"),
            ScalingMethod::Norm => write!(f, "Normalizer"),
        }
    }
}

pub struct Scaler<F: Float> {
    method: ScalingMethod<F>,
}

impl<F: Float> Scaler<F> {
    pub fn new(method: ScalingMethod<F>) -> Self {
        Self { method }
    }

    pub fn method(mut self, method: ScalingMethod<F>) -> Self {
        self.method = method;
        self
    }
}

impl<'a, F: Float, D: Data<Elem = F>, T: AsTargets> Fit<'a, ArrayBase<D, Ix2>, T> for Scaler<F> {
    type Object = FittedScaler<F>;

    fn fit(&self, x: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Self::Object {
        match &self.method {
            ScalingMethod::Standard => FittedScaler::standard(x.records()),
            ScalingMethod::MinMax(min, max) => FittedScaler::min_max(x.records(), *min, *max),
            ScalingMethod::MaxAbs => FittedScaler::max_abs(x.records()),
            ScalingMethod::Norm => FittedScaler::norm(),
        }
    }
}

#[derive(Debug)]
pub struct FittedScaler<F: Float> {
    means: Option<Array1<F>>,
    std_devs: Option<Array1<F>>,
    maxes: Option<Array1<F>>,
    mins: Option<Array1<F>>,
    method: ScalingMethod<F>,
}

impl<F: Float> FittedScaler<F> {
    pub(crate) fn new() -> Self {
        Self {
            means: None,
            std_devs: None,
            maxes: None,
            mins: None,
            method: ScalingMethod::Standard,
        }
    }

    pub(crate) fn standard<D: Data<Elem = F>>(records: &ArrayBase<D, Ix2>) -> Self {
        let means = records.mean_axis(Axis(1));
        let std_devs = Some(records.std_axis(Axis(1), F::zero()));
        Self {
            means,
            std_devs,
            maxes: None,
            mins: None,
            method: ScalingMethod::Standard,
        }
    }

    pub(crate) fn min_max<D: Data<Elem = F>>(records: &ArrayBase<D, Ix2>, min: F, max: F) -> Self {
        let mins = Some(records.fold_axis(
            Axis(0),
            F::infinity(),
            |&x, &prev| if x < prev { x } else { prev },
        ));
        let maxes =
            Some(records.fold_axis(
                Axis(0),
                F::neg_infinity(),
                |&x, &prev| if x > prev { x } else { prev },
            ));
        Self {
            means: None,
            std_devs: None,
            maxes: maxes,
            mins: mins,
            method: ScalingMethod::MinMax(min, max),
        }
    }

    pub(crate) fn max_abs<D: Data<Elem = F>>(records: &ArrayBase<D, Ix2>) -> Self {
        let max = Some(records.fold_axis(Axis(0), F::zero(), |&x, &prev| {
            if x.abs() > prev.abs() {
                x.abs()
            } else {
                prev.abs()
            }
        }));
        Self {
            means: None,
            std_devs: None,
            maxes: max,
            mins: None,
            method: ScalingMethod::MaxAbs,
        }
    }

    pub(crate) fn norm() -> Self {
        let mut sc = Self::new();
        sc.method = ScalingMethod::Norm;
        sc
    }

    pub fn means(&self) -> Result<&Array1<F>> {
        match &self.means {
            Some(ms) => Ok(ms),
            None => Err(Error::WrongMeasureForScaler(
                "means".to_string(),
                self.method.to_string(),
            )),
        }
    }

    pub fn std_devs(&self) -> Result<&Array1<F>> {
        match &self.std_devs {
            Some(std_d) => Ok(std_d),
            None => Err(Error::WrongMeasureForScaler(
                "standard deviations".to_string(),
                self.method.to_string(),
            )),
        }
    }

    pub fn maxes(&self) -> Result<&Array1<F>> {
        match &self.maxes {
            Some(maxes) => Ok(maxes),
            None => Err(Error::WrongMeasureForScaler(
                "max".to_string(),
                self.method.to_string(),
            )),
        }
    }
    pub fn mins(&self) -> Result<&Array1<F>> {
        match &self.mins {
            Some(mins) => Ok(mins),
            None => Err(Error::WrongMeasureForScaler(
                "min".to_string(),
                self.method.to_string(),
            )),
        }
    }

    pub fn method(&self) -> ScalingMethod<F> {
        self.method.clone()
    }
}

impl<F: Float> Transformer<Array2<F>, Array2<F>> for FittedScaler<F> {
    fn transform(&self, x: Array2<F>) -> Array2<F> {
        let mut x = x;
        match &self.method {
            ScalingMethod::Norm => normalize(x, ndarray_linalg::norm::NormalizeAxis::Column).0,
            ScalingMethod::MinMax(scaler_min, scaler_max) => {
                for ((mut col, &min), &max) in x
                    .axis_iter_mut(Axis(1))
                    .zip(self.mins().unwrap().into_iter())
                    .zip(self.maxes().unwrap().into_iter())
                {
                    col.mapv_inplace(|el| (el - min) / (max - min));
                }
                x = x * (*scaler_max - *scaler_min) + *scaler_min;
                x
            }
            ScalingMethod::MaxAbs => {
                for (mut col, &max) in x
                    .axis_iter_mut(Axis(1))
                    .zip(self.maxes().unwrap().into_iter())
                {
                    col.mapv_inplace(|el| el / max);
                }
                x
            }
            ScalingMethod::Standard => {
                for ((mut col, &mean), &std) in x
                    .axis_iter_mut(Axis(1))
                    .zip(self.means().unwrap().into_iter())
                    .zip(self.std_devs().unwrap().into_iter())
                {
                    col.mapv_inplace(|el| (el - mean) / std);
                }
                x
            }
        }
    }
}

impl<F: Float, D: Data<Elem = F>, T: AsTargets>
    Transformer<DatasetBase<ArrayBase<D, Ix2>, T>, DatasetBase<Array2<F>, T>> for FittedScaler<F>
{
    fn transform(&self, x: DatasetBase<ArrayBase<D, Ix2>, T>) -> DatasetBase<Array2<F>, T> {
        let transformed_records = self.transform(x.records.to_owned());
        x.with_records(transformed_records)
    }
}

#[cfg(test)]
mod tests {

    use crate::{Scaler, ScalingMethod};
    use approx::assert_abs_diff_eq;
    use linfa::traits::{Fit, Transformer};
    use ndarray::array;

    #[test]
    fn test_max_abs() {
        let dataset = array![[1f32, -1f32], [2f32, -2f32], [3f32, -3f32], [4f32, -5f32]].into();
        let scaler = Scaler::new(ScalingMethod::MaxAbs).fit(&dataset);
        println!("{:?}", scaler);
        let scaled = scaler.transform(dataset);
        let col0 = scaled.records().column(0);
        let col1 = scaled.records().column(1);
        assert_abs_diff_eq!(col0, array![1f32 / 4f32, 2f32 / 4f32, 3f32 / 4f32, 1f32]);
        assert_abs_diff_eq!(
            col1,
            array![-1f32 / 5f32, -2f32 / 5f32, -3f32 / 5f32, -1f32]
        );
    }
}
