use ndarray::{s, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Axis, Data, Ix2};
use ndarray_stats::QuantileExt;
use std::collections::HashMap;

use crate::error::{NaiveBayesError, Result};
use crate::hyperparams::{NbValidParams, NbModel};
use linfa::dataset::{AsTargets, DatasetBase, Labels};
use linfa::traits::{Fit, FitWith, PredictInplace};
use linfa::{Float, Label};
use crate::multinomial_nb::MultinomialNb;
use crate::gaussian_nb::GaussianNb;


/// Fitted Base Naive Bayes classifier
/// 
/// 
/// Container for all fitted Naive Bayes models. Functionality common for all kinds
/// of Naive Bayes is implemented here, whereas functionality specific to individual
/// kinds is defined in corresponding variants of this enum: 
/// [`GaussianNb`](GaussianNb) and [`MultinomialNb`](MultinomialNb).
/// 
/// See [`NbParams`](crate::hyperparams::NbParams) for more information on the 
/// hyper-parameters.
///  
///
/// # Model assumptions
///
/// The family of Naive Bayes classifiers assume independence between variables. They 
/// do not model moments between variables and lack therefore in modelling capability.
/// The advantage is a linear fitting time with maximum-likelihood training in a 
/// closed form. 
/// 
/// Currently two types of Naive Bayes are supported: Gaussian Naive Bayes and
/// Multinomial Naive Bayes
///
/// # Model estimation
///
/// You can fit a single model on an entire dataset. 
/// For example, for Gaussian Naive Bayes
///
/// ```rust, ignore
/// use linfa::traits::Fit;
/// let model = GaussianNb::params().fit(&ds)?;
/// ```
///
/// You can also incrementally update a model
///
/// ```rust, ignore
/// use linfa::traits::FitWith;
/// let clf = GaussianNb::params();
/// let model = datasets.iter()
///     .try_fold(None, |prev_model, &ds| clf.fit_with(prev_model, ds))?
///     .unwrap();
/// ```
///
/// After fitting the model, you can use the [`Predict`](linfa::traits::Predict) variants
/// to predict new targets.
///
#[derive(Debug, Clone)]
pub enum BaseNb<F, L> {
    MultinomialNb(MultinomialNb<F, L>),
    GaussianNb(GaussianNb<F, L>),
}


impl<F, L, D, T> Fit<ArrayBase<D, Ix2>, T, NaiveBayesError> for NbValidParams<F, L>
where
    F: Float,
    L: Label + Ord,
    D: Data<Elem = F>,
    T: AsTargets<Elem = L> + Labels<Elem = L>
{
    type Object = BaseNb<F, L>;

    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {
        // We extract the unique classes in sorted order
        let mut unique_classes = dataset.targets.labels();
        unique_classes.sort_unstable();

        let mut model: Option<Self::Object> = None;

        // We train the model
        model = self.fit_with(model, dataset)?;

        Ok(model.unwrap())
    }
}


impl<'a, F, L, D, T> FitWith<'a, ArrayBase<D, Ix2>, T, NaiveBayesError> for NbValidParams<F, L> where 
    F: Float,
    L: Label + 'a,
    D: Data<Elem = F>,
    T: AsTargets<Elem = L> + Labels<Elem = L>, {

    type ObjectIn = Option<BaseNb<F, L>>;
    type ObjectOut = Option<BaseNb<F, L>>;

    // Implementation depends on the type of Naive Bayes used, so it is delegated 
    // to GaussianNb::fit_with or MultinomialNb::fit_with
    fn fit_with(
        &self,
        model_in: Self::ObjectIn,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
    ) -> Result<Self::ObjectOut>{
            match &self.model {
                NbModel::MultinomialNb(multinomial_nb_model) => multinomial_nb_model.fit_with(model_in, dataset),
                NbModel::GaussianNb(gaussian_nb_model) => gaussian_nb_model.fit_with(model_in, dataset),
            }
    }

}
    

impl<F, L> NbValidParams<F, L> 
where   
    F: Float,
    L: Label + Ord
    {
    // Returns a subset of x corresponding to the class specified by `ycondition`
    pub fn filter(x: ArrayView2<F>, y: ArrayView1<L>, ycondition: &L) -> Array2<F> {
        // We identify the row numbers corresponding to the class we are interested in
        let index = y
            .into_iter()
            .enumerate()
            .filter_map(|(i, y)| match *ycondition == *y {
                true => Some(i),
                false => None,
            })
            .collect::<Vec<_>>();

        // We subset x to only records corresponding to the class represented in `ycondition`
        let mut xsubset = Array2::zeros((index.len(), x.ncols()));
        index
            .into_iter()
            .enumerate()
            .for_each(|(i, r)| xsubset.row_mut(i).assign(&x.slice(s![r, ..])));

        xsubset
    }

}



impl<F: Float, L: Label, D> PredictInplace<ArrayBase<D, Ix2>, Array1<L>> for BaseNb<F, L>
where
    D: Data<Elem = F>,
{
    fn predict_inplace(&self, x: &ArrayBase<D, Ix2>, y: &mut Array1<L>) {

        assert_eq!(
            x.nrows(),
            y.len(),
            "The number of data points must match the number of output targets."
        );

        let joint_log_likelihood = self.joint_log_likelihood(x.view());

        // We store the classes and likelihood info in an vec and matrix
        // respectively for easier identification of the dominant class for
        // each input
        let nclasses = joint_log_likelihood.keys().len();
        let n = x.nrows();
        let mut classes = Vec::with_capacity(nclasses);
        let mut likelihood = Array2::zeros((nclasses, n));
        joint_log_likelihood
            .iter()
            .enumerate()
            .for_each(|(i, (&key, value))| {
                classes.push(key.clone());
                likelihood.row_mut(i).assign(value);
            });

        // Identify the class with the maximum log likelihood
        *y = likelihood.map_axis(Axis(0), |x| {
            let i = x.argmax().unwrap();
            classes[i].clone()
        });
    }

    fn default_target(&self, x: &ArrayBase<D, Ix2>) -> Array1<L> {
        Array1::default(x.nrows())
    }
}



impl<F: Float, L: Label> BaseNb<F, L> where
{
    // Compute unnormalized posterior log probability. 
    // Computation depends on the type of Naive Bayes used, so it is delegated 
    // to GaussianNb::joint_log_likelihood or MultinomialNb::joint_log_likelihood 
    pub fn joint_log_likelihood(&self, x: ArrayView2<F>) -> HashMap<&L, Array1<F>> {
        match self {
            BaseNb::MultinomialNb(model) => model.joint_log_likelihood(x),
            BaseNb::GaussianNb(model) => model.joint_log_likelihood(x)
        }
        
    }
}