use crate::errors::{PlsError, Result};
use crate::utils;
use linfa::{dataset::Records, traits::Fit, traits::Transformer, DatasetBase, Float};
use ndarray::{s, Array1, Array2, ArrayBase, Data, Ix2};
use ndarray_linalg::{svd::*, Lapack, Scalar};

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

impl<F: Float + Scalar + Lapack, D: Data<Elem = F>> Fit<'_, ArrayBase<D, Ix2>, ArrayBase<D, Ix2>>
    for PlsSvdParams
{
    type Object = Result<PlsSvd<F>>;

    fn fit(
        &self,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>>,
    ) -> Result<PlsSvd<F>> {
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
        let (u, _, vt) = c.svd(true, true).unwrap();
        let mut u = u.unwrap().slice(s![.., ..self.n_components]).to_owned();
        let mut vt = vt.unwrap().slice(s![..self.n_components, ..]).to_owned();
        utils::svd_flip(&mut u, &mut vt);
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
