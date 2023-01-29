//! Common metrics for classification
//!
//! Scoring is essential for classification and regression tasks. This module implements
//! common scoring functions like precision, accuracy, recall, f1-score, ROC and ROC
//! Aread-Under-Curve.
use std::collections::HashMap;
use std::fmt;

use ndarray::prelude::*;
use ndarray::Data;

use crate::dataset::AsSingleTargets;
use crate::dataset::{AsTargets, DatasetBase, Label, Labels, Pr, Records};
use crate::error::{Error, Result};

/// Return tuple of class index for each element of prediction and ground_truth
fn map_prediction_to_idx<L: Label>(
    prediction: &[L],
    ground_truth: &[L],
    classes: &[L],
) -> Vec<Option<(usize, usize)>> {
    // create a map from class label to index
    let set = classes
        .iter()
        .enumerate()
        .map(|(a, b)| (b, a))
        .collect::<HashMap<_, usize>>();

    // indices for every prediction
    prediction
        .iter()
        .zip(ground_truth.iter())
        .map(|(a, b)| set.get(&a).and_then(|x| set.get(&b).map(|y| (*x, *y))))
        .collect::<Vec<Option<_>>>()
}

/// Confusion matrix for multi-label evaluation
///
/// A confusion matrix shows predictions in a matrix, where rows correspond to target and columns
/// to predicted. Diagonal entries are correct predictions, and everything off the
/// diagonal is a miss-classification.
#[derive(Clone, PartialEq)]
pub struct ConfusionMatrix<A> {
    matrix: Array2<f32>,
    members: Array1<A>,
}

impl<A> ConfusionMatrix<A> {
    fn is_binary(&self) -> bool {
        self.matrix.shape() == [2, 2]
    }

    /// Precision score, the number of correct classifications for the first class divided by total
    /// number of items in the first class
    ///
    /// ## Binary confusion matrix
    /// For binary confusion matrices (2x2 size) the precision score is calculated for the first
    /// label and corresponds to
    ///
    /// ```ignore
    /// true-label-1 / (true-label-1 + false-label-1)
    /// ```
    ///
    /// ## Multilabel confusion matrix
    /// For multilabel confusion matrices, the precision score is averaged over all classes
    /// (also known as `macro` averaging) A more precise controlled evaluation can be done by first splitting the confusion matrix with `split_one_vs_all` and then applying a different averaging scheme.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use linfa::prelude::*;
    /// use ndarray::array;
    ///
    /// // create dummy classes 0 and 1
    /// let prediction = array![0, 1, 1, 1, 0, 0, 1];
    /// let ground_truth = array![0, 0, 1, 0, 1, 0, 1];
    ///
    /// // create confusion matrix
    /// let cm = prediction.confusion_matrix(&ground_truth).unwrap();
    ///
    /// // print precision for label 0
    /// println!("{:?}", cm.precision());
    /// ```
    pub fn precision(&self) -> f32 {
        if self.is_binary() {
            self.matrix[(0, 0)] / (self.matrix[(0, 0)] + self.matrix[(1, 0)])
        } else {
            self.split_one_vs_all()
                .into_iter()
                .map(|x| x.precision())
                .sum::<f32>()
                / self.members.len() as f32
        }
    }

    /// Recall score, the number of correct classifications in the first class divided by the
    /// number of classifications in the first class
    ///
    ///
    /// ## Binary confusion matrix
    /// For binary confusion matrices (2x2 size) the recall score is calculated for the first label
    /// and corresponds to
    ///
    /// ```ignore
    /// true-label-1 / (true-label-1 + false-label-2)
    /// ```
    ///
    /// ## Multilabel confusion matrix
    /// For multilabel confusion matrices the recall score is averaged over all classes (also known
    /// as `macro` averaging). A more precise evaluation can be achieved by first splitting the
    /// confusion matrix with `split_one_vs_all` and then applying a different averaging scheme.
    ///
    /// # Example
    ///
    /// ```rust
    /// use linfa::prelude::*;
    /// use ndarray::array;
    ///
    /// // create dummy classes 0 and 1
    /// let prediction = array![0, 1, 1, 1, 0, 0, 1];
    /// let ground_truth = array![0, 0, 1, 0, 1, 0, 1];
    ///
    /// // create confusion matrix
    /// let cm = prediction.confusion_matrix(&ground_truth).unwrap();
    ///
    /// // print recall for label 0
    /// println!("{:?}", cm.recall());
    /// ```
    pub fn recall(&self) -> f32 {
        if self.is_binary() {
            self.matrix[(0, 0)] / (self.matrix[(0, 0)] + self.matrix[(0, 1)])
        } else {
            self.split_one_vs_all()
                .into_iter()
                .map(|x| x.recall())
                .sum::<f32>()
                / self.members.len() as f32
        }
    }

    /// Accuracy score
    ///
    /// The accuracy score is the ratio of correct classifications to all classifications. For
    /// multi-label confusion matrices this is the sum of diagonal entries to the sum of all
    /// entries.
    pub fn accuracy(&self) -> f32 {
        self.matrix.diag().sum() / self.matrix.sum()
    }

    /// F-beta-score
    ///
    /// The F-beta-score averages between precision and recall. It is defined as
    /// ```ignore
    /// (1.0 + b*b) * (precision * recall) / (b * b * precision + recall)
    /// ```
    pub fn f_score(&self, beta: f32) -> f32 {
        let sb = beta * beta;
        let p = self.precision();
        let r = self.recall();

        (1. + sb) * (p * r) / (sb * p + r)
    }

    /// F1-score, this is the F-beta-score for beta=1
    pub fn f1_score(&self) -> f32 {
        self.f_score(1.0)
    }

    /// Matthew Correlation Coefficients
    ///
    /// Estimates the normalized cross-correlation between target and predicted variable. The MCC
    /// is more significant than precision or recall, because all four quadrants are included in
    /// the evaluation. A generalized evaluation for multiple labels is also included.
    pub fn mcc(&self) -> f32 {
        let mut cov_xy = 0.0;
        for k in 0..self.members.len() {
            for l in 0..self.members.len() {
                for m in 0..self.members.len() {
                    cov_xy += self.matrix[(k, k)] * self.matrix[(l, m)];
                    cov_xy -= self.matrix[(k, l)] * self.matrix[(m, k)];
                }
            }
        }

        let sum = self.matrix.sum();
        let sum_over_cols = self.matrix.sum_axis(Axis(0));
        let sum_over_rows = self.matrix.sum_axis(Axis(1));

        let mut cov_xx: f32 = 0.0;
        let mut cov_yy: f32 = 0.0;
        for k in 0..self.members.len() {
            cov_xx += sum_over_rows[k] * (sum - sum_over_rows[k]);
            cov_yy += sum_over_cols[k] * (sum - sum_over_cols[k]);
        }

        cov_xy / cov_xx.sqrt() / cov_yy.sqrt()
    }

    /// Split confusion matrix in N one-vs-all binary confusion matrices
    pub fn split_one_vs_all(&self) -> Vec<ConfusionMatrix<bool>> {
        let sum = self.matrix.sum();

        (0..self.members.len())
            .map(|i| {
                let tp = self.matrix[(i, i)];
                let fp = self.matrix.row(i).sum() - tp;
                let _fn = self.matrix.column(i).sum() - tp;
                let tn = sum - tp - fp - _fn;

                ConfusionMatrix {
                    matrix: array![[tp, fp], [_fn, tn]],
                    members: Array1::from(vec![true, false]),
                }
            })
            .collect()
    }

    /// Split confusion matrix in N*(N-1)/2 one-vs-one binary confusion matrices
    pub fn split_one_vs_one(&self) -> Vec<ConfusionMatrix<bool>> {
        let n = self.members.len();
        let mut cms = Vec::with_capacity(n * (n - 1) / 2);

        for i in 0..n {
            for j in i..n {
                let tp = self.matrix[(i, i)];
                let fp = self.matrix[(i, j)];
                let _fn = self.matrix[(j, i)];
                let tn = self.matrix[(j, j)];

                cms.push(ConfusionMatrix {
                    matrix: array![[tp, fp], [_fn, tn]],
                    members: Array1::from(vec![true, false]),
                });
            }
        }

        cms
    }
}

/// Print a confusion matrix
impl<A: fmt::Display> fmt::Debug for ConfusionMatrix<A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let len = self.matrix.len_of(Axis(0));
        writeln!(f)?;
        write!(f, "{: <10}", "classes")?;
        for i in 0..len {
            write!(f, " | {: <10}", self.members[i])?;
        }
        writeln!(f)?;

        for i in 0..len {
            write!(f, "{: <10}", self.members[i])?;

            for j in 0..len {
                write!(f, " | {: <10}", self.matrix[(i, j)])?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}

/// Classification for multi-label evaluation
///
/// Contains a routine to calculate the confusion matrix, all other scores are derived form it.
pub trait ToConfusionMatrix<A, T> {
    fn confusion_matrix(&self, ground_truth: T) -> Result<ConfusionMatrix<A>>;
}

impl<L: Label, S, T> ToConfusionMatrix<L, ArrayBase<S, Ix1>> for T
where
    S: Data<Elem = L>,
    T: AsSingleTargets<Elem = L> + Labels<Elem = L>,
{
    fn confusion_matrix(&self, ground_truth: ArrayBase<S, Ix1>) -> Result<ConfusionMatrix<L>> {
        self.confusion_matrix(&ground_truth)
    }
}

impl<L: Label, S, T> ToConfusionMatrix<L, &ArrayBase<S, Ix1>> for T
where
    S: Data<Elem = L>,
    T: AsSingleTargets<Elem = L> + Labels<Elem = L>,
{
    fn confusion_matrix(&self, ground_truth: &ArrayBase<S, Ix1>) -> Result<ConfusionMatrix<L>> {
        let targets = self.as_single_targets();
        if targets.len() != ground_truth.len() {
            return Err(Error::MismatchedShapes(targets.len(), ground_truth.len()));
        }

        let classes = self.labels();

        let indices = map_prediction_to_idx(
            targets.as_slice().unwrap(),
            ground_truth.as_slice().unwrap(),
            &classes,
        );

        // count each index tuple in the confusion matrix
        let mut confusion_matrix = Array2::zeros((classes.len(), classes.len()));
        for (i1, i2) in indices.into_iter().flatten() {
            confusion_matrix[(i1, i2)] += 1.0;
        }

        Ok(ConfusionMatrix {
            matrix: confusion_matrix,
            members: Array1::from(classes),
        })
    }
}

impl<L: Label, R, R2, T, T2> ToConfusionMatrix<L, &DatasetBase<R, T>> for DatasetBase<R2, T2>
where
    R: Records,
    R2: Records,
    T: AsSingleTargets<Elem = L>,
    T2: AsSingleTargets<Elem = L> + Labels<Elem = L>,
{
    fn confusion_matrix(&self, ground_truth: &DatasetBase<R, T>) -> Result<ConfusionMatrix<L>> {
        self.targets().confusion_matrix(ground_truth.as_targets())
    }
}

impl<L: Label, S: Data<Elem = L>, T: AsSingleTargets<Elem = L> + Labels<Elem = L>, R: Records>
    ToConfusionMatrix<L, &DatasetBase<R, T>> for ArrayBase<S, Ix1>
{
    fn confusion_matrix(&self, ground_truth: &DatasetBase<R, T>) -> Result<ConfusionMatrix<L>> {
        ground_truth.confusion_matrix(self.view())
    }
}

/*
impl<A: Clone + Ord + Hash, D: Data<Elem = A>> IntoConfusionMatrix<A> for ArrayBase<D, Ix1> {
    fn into_confusion_matrix<'a, T>(self, ground_truth: T) -> ConfusionMatrix<A>
    where
        A: 'a,
        T: IntoNdProducer<Item = &'a A, Dim = Ix1, Output = ArrayView1<'a, A>>,
    {
        let tmp = ModifiedPrediction {
            prediction: self,
            classes: Vec::new(),
            weights: Vec::new(),
        };

        tmp.into_confusion_matrix(ground_truth)
    }
}

impl<A: Clone + Ord + Hash> IntoConfusionMatrix<A> for Vec<A> {
    fn into_confusion_matrix<'a, T>(self, ground_truth: T) -> ConfusionMatrix<A>
    where
        A: 'a,
        T: IntoNdProducer<Item = &'a A, Dim = Ix1, Output = ArrayView1<'a, A>>,
    {
        let tmp = ModifiedPrediction {
            prediction: Array1::from(self),
            classes: Vec::new(),
            weights: Vec::new(),
        };

        tmp.into_confusion_matrix(ground_truth)
    }
}*/

/*
 * TODO: specialization requires unstable Rust
impl IntoConfusionMatrix<bool, OwnedRepr<bool>> for Array1<bool> {
    fn confusion_matrix(self, ground_truth: &Array1<bool>) -> ConfusionMatrix<bool> {
        let mut confusion_matrix = Array2::zeros((2, 2));
        for result in self.iter().zip(ground_truth.iter()) {
            match result {
                (true, true) => confusion_matrix[(0, 0)] += 1,
                (true, false) => confusion_matrix[(1, 0)] += 1,
                (false, true) => confusion_matrix[(0, 1)] += 1,
                (false, false) => confusion_matrix[(1, 1)] += 1
            }
        }

        ConfusionMatrix {
            matrix: confusion_matrix,
            members: Array::from(vec![true, false])
        }
    }
}*/

/// Integration using the trapezoidal rule.
fn trapezoidal<A: NdFloat>(vals: &[(A, A)]) -> A {
    let mut prev_x = vals[0].0;
    let mut prev_y = vals[0].1;
    let mut integral = A::zero();

    for (x, y) in vals.iter().skip(1) {
        integral += (*x - prev_x) * (prev_y + *y) / A::from(2.0).unwrap();
        prev_x = *x;
        prev_y = *y;
    }
    integral
}

/// A Receiver Operating Characteristic for binary-label classification
///
/// The ROC curve gives insight about the seperability of a binary classification task.
#[derive(Debug, Clone, PartialEq)]
pub struct ReceiverOperatingCharacteristic {
    curve: Vec<(f32, f32)>,
    thresholds: Vec<f32>,
}

impl ReceiverOperatingCharacteristic {
    /// Returns the true-positive, false-positive curve
    pub fn get_curve(&self) -> Vec<(f32, f32)> {
        self.curve.clone()
    }

    /// Returns the threshold corresponding to each point
    pub fn get_thresholds(&self) -> Vec<f32> {
        self.thresholds.clone()
    }

    /// Returns the Area-Under-Curve metric
    pub fn area_under_curve(&self) -> f32 {
        trapezoidal(&self.curve)
    }
}

/// Classification for binary-labels
///
/// This contains Receiver-Operating-Characterstics curves and log loss as those only work for binary
/// classification tasks.
pub trait BinaryClassification<T> {
    fn roc(&self, y: T) -> Result<ReceiverOperatingCharacteristic>;
    fn log_loss(&self, y: T) -> Result<f32>;
}

impl BinaryClassification<&[bool]> for &[Pr] {
    fn roc(&self, y: &[bool]) -> Result<ReceiverOperatingCharacteristic> {
        let mut tuples = self
            .iter()
            .zip(y.iter())
            .filter_map(|(a, b)| if **a >= 0.0 { Some((*a, *b)) } else { None })
            .collect::<Vec<(Pr, bool)>>();

        tuples.sort_unstable_by(&|a: &(Pr, _), b: &(Pr, _)| match a.0.partial_cmp(&b.0) {
            Some(ord) => ord,
            None => unreachable!(),
        });

        let (mut tp, mut fp) = (0.0, 0.0);
        let mut tps_fps = Vec::new();
        let mut thresholds = Vec::new();
        let mut s0 = 0.0;

        for (s, t) in tuples {
            if (*s - s0).abs() > 1e-10 {
                tps_fps.push((tp, fp));
                thresholds.push(s);
                s0 = *s;
            }

            if t {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
        }
        tps_fps.push((tp, fp));

        let (max_tp, max_fp) = (tp, fp);
        for (tp, fp) in &mut tps_fps {
            *tp /= max_tp;
            *fp /= max_fp;
        }

        Ok(ReceiverOperatingCharacteristic {
            curve: tps_fps,
            thresholds: thresholds.into_iter().map(|x| *x).collect(),
        })
    }

    fn log_loss(&self, y: &[bool]) -> Result<f32> {
        let probabilities = aview1(self);
        probabilities.log_loss(y)
    }
}

impl<D: Data<Elem = Pr>> BinaryClassification<&[bool]> for ArrayBase<D, Ix1> {
    fn roc(&self, y: &[bool]) -> Result<ReceiverOperatingCharacteristic> {
        self.as_slice().unwrap().roc(y)
    }

    fn log_loss(&self, y: &[bool]) -> Result<f32> {
        assert_eq!(
            self.len(),
            y.len(),
            "The number of predicted points must match the length of target."
        );
        let len = self.len();
        if len == 0 {
            Err(Error::NotEnoughSamples)
        } else {
            let sum: f32 = self
                .iter()
                .map(|v| (*v).clamp(f32::EPSILON, 1. - f32::EPSILON))
                .zip(y.iter())
                .map(|(a, b)| if *b { -a.ln() } else { -(1. - a).ln() })
                .sum();
            Ok(sum / len as f32)
        }
    }
}

impl<R: Records, R2: Records, T: AsSingleTargets<Elem = bool>, T2: AsSingleTargets<Elem = Pr>>
    BinaryClassification<&DatasetBase<R, T>> for DatasetBase<R2, T2>
{
    fn roc(&self, y: &DatasetBase<R, T>) -> Result<ReceiverOperatingCharacteristic> {
        let targets = self.as_targets();
        let targets = targets.as_slice().unwrap();
        let y_targets = y.as_targets();
        let y_targets = y_targets.as_slice().unwrap();

        targets.roc(y_targets)
    }

    /// Log loss of the probabilities of the binary target
    fn log_loss(&self, y: &DatasetBase<R, T>) -> Result<f32> {
        let probabilities = self.as_single_targets();
        let y_targets = y.as_targets();
        let y_targets = y_targets.as_slice().unwrap();

        probabilities.log_loss(y_targets)
    }
}

#[cfg(test)]
mod tests {
    use super::{BinaryClassification, ConfusionMatrix, ToConfusionMatrix};
    use super::{Label, Pr};
    use approx::assert_abs_diff_eq;
    use ndarray::{array, Array1, Array2, ArrayView1};
    use rand::{distributions::Uniform, rngs::SmallRng, Rng, SeedableRng};
    use std::collections::HashMap;

    fn get_labels_map<L: Label>(cm: &ConfusionMatrix<L>) -> HashMap<L, usize> {
        cm.members
            .iter()
            .enumerate()
            .map(|(index, label)| (label.clone(), index))
            .collect()
    }

    // confusion matrices use hash sets for the labels to pair so
    // the order of the rows of the matrices is not constant.
    // we can transform the index->member mapping in `cm.members`
    // into a member->index mapping to check each element independently
    fn assert_cm_eq<L: Label>(cm: &ConfusionMatrix<L>, expected: &Array2<f32>, labels: &Array1<L>) {
        let map = get_labels_map(cm);
        for ((row, column), value) in expected.indexed_iter().map(|((r, c), v)| {
            (
                (*map.get(&labels[r]).unwrap(), *map.get(&labels[c]).unwrap()),
                v,
            )
        }) {
            let cm_value = *cm.matrix.get((row, column)).unwrap();
            assert_abs_diff_eq!(cm_value, value);
        }
    }

    fn assert_split_eq<L: Label, C: Fn(&ConfusionMatrix<bool>) -> f32>(
        cm: &ConfusionMatrix<L>,
        eval: C,
        expected: &Array1<f32>,
        labels: &Array1<L>,
    ) {
        let map = get_labels_map(cm);
        let evals = cm
            .split_one_vs_all()
            .into_iter()
            .map(|x| eval(&x))
            .collect::<Vec<_>>();
        for (index, value) in expected
            .indexed_iter()
            .map(|(i, v)| (*map.get(&labels[i]).unwrap(), v))
        {
            let evals_value = *evals.get(index).unwrap();
            assert_abs_diff_eq!(evals_value, value);
        }
    }

    #[test]
    fn test_confusion_matrix() {
        let ground_truth = ArrayView1::from(&[1, 1, 0, 1, 0, 1]);
        let predicted = ArrayView1::from(&[0, 1, 0, 1, 0, 1]);

        let cm = predicted.confusion_matrix(ground_truth).unwrap();

        let labels = array![0, 1];
        let expected = array![[2., 1.], [0., 3.]];

        assert_cm_eq(&cm, &expected, &labels);
    }

    #[test]
    fn test_cm_metrices() {
        let ground_truth = Array1::from(vec![1, 1, 0, 1, 0, 1]);
        let predicted = Array1::from(vec![0, 1, 0, 1, 0, 1]);

        let x = predicted.confusion_matrix(ground_truth).unwrap();

        let labels = array![0, 1];

        assert_abs_diff_eq!(x.accuracy(), 5.0 / 6.0_f32);
        assert_abs_diff_eq!(
            x.mcc(),
            (2. * 3. - 1. * 0.) / (2.0f32 * 3. * 3. * 4.).sqrt()
        );

        assert_split_eq(
            &x,
            |cm| ConfusionMatrix::precision(cm),
            &array![1.0, 3. / 4.],
            &labels,
        );
        assert_split_eq(
            &x,
            |cm| ConfusionMatrix::recall(cm),
            &array![2.0 / 3.0, 1.0],
            &labels,
        );
        assert_split_eq(
            &x,
            |cm| ConfusionMatrix::f1_score(cm),
            &array![4.0 / 5.0, 6.0 / 7.0],
            &labels,
        );
    }

    #[test]
    fn test_roc_curve() {
        let predicted = ArrayView1::from(&[0.1, 0.3, 0.5, 0.7, 0.8, 0.9]).mapv(Pr::new);

        let groundtruth = vec![false, true, false, true, true, true];

        let result = &[
            (0.0, 0.0),  // start
            (0.0, 0.5),  // first item is target=false
            (0.25, 0.5), // second item is target=true, but obviously false
            (0.25, 1.0), // third item is target=false, we reach max false-positive, because all other labels are positive
            (0.5, 1.0),  // the remaining three are target=true
            (0.75, 1.0),
            (1., 1.),
        ];

        let roc = predicted.roc(&groundtruth).unwrap();
        assert_eq!(roc.get_curve(), result);
    }

    #[test]
    fn test_roc_auc() {
        let mut rng = SmallRng::seed_from_u64(42);
        let predicted = Array1::linspace(0.0, 1.0, 1000).mapv(Pr::new);

        let range = Uniform::new(0, 2);

        // randomly sample ground truth
        let ground_truth = (0..1000)
            .map(|_| rng.sample(range) == 1)
            .collect::<Vec<_>>();

        // ROC Area-Under-Curve should be approximately 0.5
        let roc = predicted.roc(&ground_truth).unwrap();
        assert!((roc.area_under_curve() - 0.5) < 0.04);
    }

    #[test]
    fn split_one_vs_all() {
        let ground_truth = array![0, 2, 3, 0, 1, 2, 1, 2, 3, 2];
        let predicted = array![0, 3, 2, 0, 1, 1, 1, 3, 2, 3];

        // create a confusion matrix
        let cm = predicted.confusion_matrix(ground_truth).unwrap();

        let labels = array![0, 1, 2, 3];
        let bin_labels = array![true, false];
        let map = get_labels_map(&cm);

        // split four class confusion matrix into 4 binary confusion matrix
        let n_cm = cm.split_one_vs_all();

        let result = &[
            array![[2., 0.], [0., 8.]], // no misclassification for label=0
            array![[2., 1.], [0., 7.]], // one false-positive for label=1
            array![[0., 2.], [4., 4.]], // two false-positive and four false-negative for label=2
            array![[0., 3.], [2., 5.]], // three false-positive and two false-negative for label=3
        ];

        for (r, x) in result
            .iter()
            .zip(labels.iter())
            .map(|(r, l)| (r, n_cm.get(*map.get(l).unwrap()).unwrap()))
        {
            assert_cm_eq(x, r, &bin_labels);
        }
    }

    #[test]
    fn log_loss() {
        let ground_truth = &[false, false, false, false, true, true, true, true, true];
        let predicted =
            ArrayView1::from(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).mapv(Pr::new);

        let logloss = predicted.log_loss(ground_truth).unwrap();
        assert_abs_diff_eq!(logloss, 0.34279516);
    }

    #[test]
    #[should_panic]
    fn log_loss_empty() {
        let ground_truth = &[];
        let predicted = ArrayView1::from(&[]).mapv(Pr::new);
        predicted.log_loss(ground_truth).unwrap();
    }

    #[test]
    #[should_panic]
    fn log_loss_with_different_lengths() {
        let ground_truth = &[false, false, false, false, true, true, true, true];
        let predicted =
            ArrayView1::from(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).mapv(Pr::new);
        predicted.log_loss(ground_truth).unwrap();
    }
}
