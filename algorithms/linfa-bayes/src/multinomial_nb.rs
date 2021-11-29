use ndarray::{s, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Axis, Data, Ix2};
use ndarray_stats::QuantileExt;
use std::collections::HashMap;

use crate::error::{NaiveBayesError, Result};
use crate::hyperparams::{MultinomialNbParams, MultinomialNbValidParams};
use linfa::dataset::{AsTargets, DatasetBase, Labels};
use linfa::traits::{Fit, FitWith, PredictInplace};
use linfa::{Float, Label};



impl<F, L, D, T> Fit<ArrayBase<D, Ix2>, T, NaiveBayesError> for MultinomialNbValidParams<F, L>
where
    F: Float,
    L: Label + Ord,
    D: Data<Elem = F>,
    T: AsTargets<Elem = L> + Labels<Elem = L>,
{
    type Object = MultinomialNb<F, L>;

    fn fit(&self, dataset: &DatasetBase<ArrayBase<D, Ix2>, T>) -> Result<Self::Object> {
        // We extract the unique classes in sorted order
        let mut unique_classes = dataset.targets.labels();
        unique_classes.sort_unstable();

        let mut model: Option<MultinomialNb<_, _>> = None;

        // We train the model
        model = self.fit_with(model, dataset)?;

        Ok(model.unwrap())
    }
}

impl<'a, F, L, D, T> FitWith<'a, ArrayBase<D, Ix2>, T, NaiveBayesError>
    for MultinomialNbValidParams<F, L>
where
    F: Float,
    L: Label + 'a,
    D: Data<Elem = F>,
    T: AsTargets<Elem = L> + Labels<Elem = L>,
{
    type ObjectIn = Option<MultinomialNb<F, L>>;
    type ObjectOut = Option<MultinomialNb<F, L>>;

    fn fit_with(
        &self,
        model_in: Self::ObjectIn,
        dataset: &DatasetBase<ArrayBase<D, Ix2>, T>,
    ) -> Result<Self::ObjectOut> {
        let x = dataset.records();
        let y = dataset.try_single_target()?;

        let mut model = match model_in {
            Some(temp) => {
                temp
            }
            None => MultinomialNb {
                class_info: HashMap::new(),
            },
        };

        let yunique = dataset.labels();

        for class in yunique {
            // We filter for records that correspond to the current class
            let xclass = Self::filter(x.view(), y.view(), &class);
            // We count the number of occurences of the class
            let nclass = xclass.nrows();

            // We compute the update of the gaussian mean and variance
            let mut class_info = model
                .class_info
                .entry(class)
                .or_insert_with(ClassInfo::default);
            let (feature_log_prob, feature_count) = Self::update_feature_log_prob(class_info, xclass.view(), self.alpha());
            // We now update the total counts of each feature, logarithm of feature counts, and class count
            class_info.feature_log_prob = feature_log_prob;
            class_info.feature_count = feature_count;
            class_info.class_count += nclass;
        }

        // We update the priors
        let class_count_sum = model
            .class_info
            .values()
            .map(|x| x.class_count)
            .sum::<usize>();
        for info in model.class_info.values_mut() {
            info.prior = F::cast(info.class_count) / F::cast(class_count_sum);
        }

        Ok(Some(model))
    }
}



impl<F: Float, L: Label> MultinomialNbValidParams<F, L> {
    // Compute online update of feature counts and its logarithm
    fn update_feature_log_prob(
        info_old: &ClassInfo<F>,
        x_new: ArrayView2<F>,
        alpha: F,
    ) -> (Array1<F>, Array1<F>) {
        // deconstruct old state
        let (count_old, feature_log_prob_old, feature_count_old) = (&info_old.class_count, &info_old.feature_log_prob,
             &info_old.feature_count);

        // If incoming data is empty no updates required
        if x_new.nrows() == 0 {
            return (feature_log_prob_old.to_owned(), feature_count_old.to_owned())
        }

        let feature_count_new = x_new.sum_axis(Axis(0));
        
        // If previous batch was empty, we send the new feature count calculated
        let feature_count = if count_old > &0 {
            feature_count_old + feature_count_new
        }
        else {
            feature_count_new
        };

        let feature_count_smoothed = feature_count.clone() + alpha;
        let count = feature_count_smoothed.sum();

        let feature_log_prob = feature_count_smoothed.mapv(|x| x.ln() - F::cast(count).ln());
        (feature_log_prob.to_owned(), feature_count.to_owned())
    }

    // Returns a subset of x corresponding to the class specified by `ycondition`
    fn filter(x: ArrayView2<F>, y: ArrayView1<L>, ycondition: &L) -> Array2<F> {
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


#[derive(Debug, Clone)]
pub struct MultinomialNb<F, L> {
    class_info: HashMap<L, ClassInfo<F>>,
}

#[derive(Debug, Default, Clone)]
struct ClassInfo<F> {
    class_count: usize,
    prior: F,
    feature_count: Array1<F>,
    feature_log_prob: Array1<F>
}


impl<F: Float, L: Label, D> PredictInplace<ArrayBase<D, Ix2>, Array1<L>> for MultinomialNb<F, L>
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

impl<F: Float, L: Label> MultinomialNb<F, L> {
    /// Construct a new set of hyperparameters
    pub fn params() -> MultinomialNbParams<F, L> {
        MultinomialNbParams::new()
    }

    // Compute unnormalized posterior log probability
    fn joint_log_likelihood(&self, x: ArrayView2<F>) -> HashMap<&L, Array1<F>> {
        let mut joint_log_likelihood = HashMap::new();

        for (class, info) in self.class_info.iter() {
            let jointi = info.prior.ln();
            
            let nij = x.dot(&info.feature_log_prob);
            
           joint_log_likelihood.insert(class, nij + jointi);
        }

        joint_log_likelihood
    }
}


#[cfg(test)]
mod tests {
    use super::{MultinomialNb, Result};
    use linfa::{
        traits::{Fit, Predict},
        DatasetView,
    };

    use approx::assert_abs_diff_eq;
    use ndarray::{array};
    use std::collections::HashMap;

    #[test]
    fn test_multinomial_nb() -> Result<()> {
        let x = array![
            [1., 0.],
            [2., 0.],
            [3., 0.],
            [0., 1.],
            [0., 2.],
            [0., 3.]
        ];
        let y = array![1, 1, 1, 2, 2, 2];

        let data = DatasetView::new(x.view(), y.view());
        let fitted_clf = MultinomialNb::params().fit(&data)?;
        let pred = fitted_clf.predict(&x);

        assert_abs_diff_eq!(pred, y);

        let jll = fitted_clf.joint_log_likelihood(x.view());
        let mut expected = HashMap::new();
        // Computed with sklearn.naive_bayes.MultinomialNB
        expected.insert(
            &1usize,
            array![
                -0.82667857, 
                -0.96020997, 
                -1.09374136, 
                -2.77258872, 
                -4.85203026,
                -6.93147181
            ],
        );

        expected.insert(
            &2usize,
            array![
                -2.77258872, 
                -4.85203026, 
                -6.93147181, 
                -0.82667857, 
                -0.96020997, 
                -1.09374136
            ],
        );

        for (key, value) in jll.iter() {
            assert_abs_diff_eq!(value, expected.get(key).unwrap(), epsilon = 1e-6);
        }

        Ok(())
    }

    
}

