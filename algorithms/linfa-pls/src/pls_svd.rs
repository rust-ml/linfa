use crate::errors::{PlsError, Result};
use crate::{utils, Float};
use linfa::{dataset::Records, traits::Fit, traits::Transformer, DatasetBase};
use ndarray::{s, Array1, Array2, ArrayBase, Data, Ix2};
use ndarray_linalg::svd::*;

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone)]
pub struct PlsSvdParams {
    n_components: usize,
    scale: bool,
}

impl PlsSvdParams {
    pub fn new(n_components: usize) -> PlsSvdParams {
        PlsSvdParams {
            n_components,
            scale: true,
        }
    }

    pub fn scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        self
    }
}

impl Default for PlsSvdParams {
    fn default() -> Self {
        Self::new(2)
    }
}

#[allow(clippy::many_single_char_names)]
impl<F: Float, D: Data<Elem = F>> Fit<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>, PlsError>
    for PlsSvdParams
{
    type Object = PlsSvd<F>;

    fn fit(
        &self,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>>,
    ) -> Result<Self::Object> {
        if dataset.nsamples() < 2 {
            return Err(PlsError::NotEnoughSamplesError(format!(
                "should be greater than 1, got {}",
                dataset.records().nsamples()
            )));
        }
        // we'll compute the SVD of the cross-covariance matrix = X.T.dot(Y)
        // This matrix rank is at most min(n_samples, n_features, n_targets) so
        // n_components cannot be bigger than that.

        let rank_upper_bound = dataset
            .nsamples()
            .min(dataset.nfeatures())
            .min(dataset.targets().ncols());
        if 1 > self.n_components || self.n_components > rank_upper_bound {
            return Err(PlsError::BadComponentNumberError(format!(
                "n_components should be in [1, {}], got {}",
                rank_upper_bound, self.n_components
            )));
        }
        let (x, y, x_mean, y_mean, x_std, y_std) =
            utils::center_scale_dataset(&dataset, self.scale);

        // Compute SVD of cross-covariance matrix
        let c = x.t().dot(&y);
        let (u, _, vt) = c.svd(true, true)?;
        // safe unwraps because both parameters are set to true in above call
        let u = u.unwrap().slice_move(s![.., ..self.n_components]);
        let vt = vt.unwrap().slice_move(s![..self.n_components, ..]);
        let (u, vt) = utils::svd_flip(u, vt);
        let v = vt.reversed_axes();

        let x_weights = u;
        let y_weights = v;

        Ok(PlsSvd {
            x_mean,
            x_std,
            y_mean,
            y_std,
            x_weights,
            y_weights,
        })
    }
}

pub struct PlsSvd<F: Float> {
    x_mean: Array1<F>,
    x_std: Array1<F>,
    y_mean: Array1<F>,
    y_std: Array1<F>,
    x_weights: Array2<F>,
    y_weights: Array2<F>,
}

impl<F: Float> PlsSvd<F> {
    pub fn params(n_components: usize) -> PlsSvdParams {
        PlsSvdParams {
            n_components,
            scale: true,
        }
    }

    pub(crate) fn means(&self) -> (&Array1<F>, &Array1<F>) {
        (&self.x_mean, &self.y_mean)
    }

    pub(crate) fn stds(&self) -> (&Array1<F>, &Array1<F>) {
        (&self.x_std, &self.y_std)
    }

    pub fn weights(&self) -> (&Array2<F>, &Array2<F>) {
        (&self.x_weights, &self.y_weights)
    }
}

impl<F: Float, D: Data<Elem = F>>
    Transformer<
        DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>>,
        DatasetBase<Array2<F>, Array2<F>>,
    > for PlsSvd<F>
{
    fn transform(
        &self,
        dataset: DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>>,
    ) -> DatasetBase<Array2<F>, Array2<F>> {
        let (x_mean, y_mean) = &self.means();
        let (x_std, y_std) = &self.stds();
        let (x_weights, y_weights) = &self.weights();
        let xr = (dataset.records() - *x_mean) / *x_std;
        let x_scores = xr.dot(*x_weights);
        let yr = (dataset.targets() - *y_mean) / *y_std;
        let y_scores = yr.dot(*y_weights);
        DatasetBase::new(x_scores, y_scores)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::assert_abs_diff_eq;
    use linfa_datasets::linnerud;
    use ndarray::array;

    #[test]
    fn test_svd() -> Result<()> {
        // values checked against scikit-learn 0.24.1 PlsSVD
        let ds = linnerud();
        let pls = PlsSvd::<f64>::params(3).fit(&ds)?;
        let ds = pls.transform(ds);
        let expected_x = array![
            [-0.37144954, -0.0544441, -0.82290137],
            [-1.34032497, 0.19638169, -0.71715313],
            [-0.08234873, 0.58492788, 0.86557407],
            [-0.35496515, -0.62863268, 0.74383396],
            [0.46311708, -0.39856773, 0.39748814],
            [-1.30584148, -0.20072641, -0.3591439],
            [-0.86178968, -0.43791399, 0.2111225],
            [-0.79728366, -0.3790222, -0.32195725],
            [1.14229739, -0.93000533, 0.19761764],
            [3.03443501, 2.81149299, 0.22224139],
            [0.40921689, -0.84959246, 1.30923934],
            [1.40508381, 0.53658054, -0.09910248],
            [1.53073864, 0.29558804, -0.01949986],
            [-2.2227316, 0.19806308, -0.2536748],
            [-1.49897159, -0.4114628, 0.23494514],
            [1.3140941, 0.67110308, -0.2366431],
            [-1.88043225, -0.41844445, 0.04307104],
            [1.23661961, -0.09041449, -0.63734812],
            [1.60595982, -0.37158339, -0.01919568],
            [-1.42542371, -0.12332727, -0.73851355]
        ];
        assert_abs_diff_eq!(expected_x, ds.records(), epsilon = 1e-6);
        Ok(())
    }
}
