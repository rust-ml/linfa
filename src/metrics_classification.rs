//! Common metrics for performance evaluation of classifier
//!
//! Scoring is essential for classification and regression tasks. This module implements
//! common scoring functions like precision, accuracy, recall, f1-score, ROC and ROC
//! Aread-Under-Curve.
use std::collections::HashMap;
use std::hash::Hash;
use std::iter::FromIterator;
use std::fmt;

use ndarray::prelude::*;
use ndarray::Data;

/// Return tuple of class index for each element of prediction and ground_truth 
fn map_prediction_to_idx<A: Eq + Hash, C: Data<Elem = A>, D: Data<Elem = A>>(prediction: &ArrayBase<C, Ix1>, ground_truth: &ArrayBase<D, Ix1>, classes: &[A]) -> Vec<Option<(usize, usize)>> {
    // create a map from class label to index
    let set = classes.iter().enumerate()
        .map(|(a, b)| (b, a))
        .collect::<HashMap<_, usize>>();

    // indices for every prediction
    ground_truth.iter().zip(prediction.iter()).map(|(a,b)| {
        set.get(&a)
            .and_then(|x| set.get(&b).map(|y| (*x, *y)))
    }).collect::<Vec<Option<_>>>()
}

/// A modified prediction
///
/// It can happen that only a subset of classes are of interest or the samples have different
/// weights in the resulting evaluations. For this a `ModifiedPrediction` struct offers the
/// possibility to modify a prediction before evaluation.
pub struct ModifiedPrediction<A, D: Data<Elem = A>> {
    prediction: ArrayBase<D, Ix1>,
    weights: Vec<usize>,
    classes: Vec<A>
}

/// Modify dataset weights or classes
pub trait Modify<A: PartialOrd + Eq + Hash, D: Data<Elem = A>> {
    /// Add weights to the samples
    fn with_weights(self, weights: &[usize]) -> ModifiedPrediction<A, D>;
    /// Select subset of classes
    fn reduce_classes(self, classes: &[A]) -> ModifiedPrediction<A, D>;
}

/// Implementation for prediction stored in `ndarray`
impl<A: PartialOrd + Eq + Hash + Clone, D: Data<Elem = A>> Modify<A, D> for ArrayBase<D, Ix1> {
    fn with_weights(self, weights: &[usize]) -> ModifiedPrediction<A, D> {
        ModifiedPrediction {
            prediction: self, 
            weights: weights.to_vec(),
            classes: Vec::new()
        }
    }

    fn reduce_classes(self, classes: &[A]) -> ModifiedPrediction<A, D> {
        ModifiedPrediction {
            prediction: self,
            weights: Vec::new(),
            classes: classes.to_vec()
        }
    }
}

/// Implementation for already modified prediction
impl<A: PartialOrd + Eq + Hash + Clone, D: Data<Elem = A>> Modify<A, D> for ModifiedPrediction<A, D> {
    fn with_weights(self, weights: &[usize]) -> ModifiedPrediction<A, D> {
        ModifiedPrediction {
            prediction: self.prediction,
            weights: weights.to_vec(),
            classes: self.classes
        }
    }

    fn reduce_classes(self, classes: &[A]) -> ModifiedPrediction<A, D> {
        ModifiedPrediction {
            prediction: self.prediction,
            weights: self.weights,
            classes: classes.to_vec()
        }
    }
}

/// Confusion matrix for multi-label evaluation
///
/// A confusion matrix shows predictions in a matrix, where rows correspond to target and columns
/// to predicted. The diagonal entries are correct predictions.
pub struct ConfusionMatrix<A> {
    matrix: Array2<usize>,
    members: Array1<A>
}

impl<A> ConfusionMatrix<A> {
    /// Calculate precision for every class
    pub fn precision(&self) -> Array1<f32> {
        let sum = self.matrix.sum_axis(Axis(1));

        Array1::from_iter(
            self.matrix.diag().iter()
                .zip(sum.iter())
                .map(|(a, b)| *a as f32 / *b as f32)
        )
    }

    /// Calculate recall for every class
    pub fn recall(&self) -> Array1<f32> {
        let sum = self.matrix.sum_axis(Axis(0));

        Array1::from_iter(
            self.matrix.diag().iter()
                .zip(sum.iter())
                .map(|(a, b)| *a as f32 / *b as f32)
        )
    }

    /// Return mean accuracy
    pub fn accuracy(&self) -> f32 {
        self.matrix.diag().sum() as f32 / self.matrix.sum() as f32
    }

    /// Return mean beta score
    pub fn f_score(&self, beta: f32) -> Array1<f32> {
        let sb = beta * beta;
        let precision = self.precision();
        let recall = self.recall();

        Array::from_iter(
            precision.iter().zip(recall.iter())
                .map(|(p, r)| (1.0 + sb) * (p * r) / (sb * p + r))
        )
    }

    /// Return mean beta=1 score
    pub fn f1_score(&self) -> Array1<f32> {
        self.f_score(1.0)
    }

    /// Return the Matthew Correlation Coefficients
    ///
    /// Estimates the normalized cross-correlation between target and predicted variable
    pub fn mcc(&self) -> f32 {
        let mut cov_xy = 0.0;
        for k in 0..self.members.len() {
            for l in 0..self.members.len() {
                for m in 0..self.members.len() {
                    cov_xy += self.matrix[(k, k)] as f32 * self.matrix[(l, m)] as f32;
                    cov_xy -= self.matrix[(k, l)] as f32 * self.matrix[(m, k)] as f32;
                }
            }
        }

        let sum = self.matrix.sum();
        let sum_over_cols = self.matrix.sum_axis(Axis(0));
        let sum_over_rows = self.matrix.sum_axis(Axis(1));

        let mut cov_xx: f32 = 0.0;
        let mut cov_yy: f32 = 0.0;
        for k in 0..self.members.len() {
            cov_xx += (sum_over_rows[k] * (sum - sum_over_rows[k])) as f32;
            cov_yy += (sum_over_cols[k] * (sum - sum_over_cols[k])) as f32;
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
                    members: Array1::from(vec![true, false])
                }
            })
            .collect()
    }
}

/// Print a confusion matrix
impl<A: fmt::Display> fmt::Debug for ConfusionMatrix<A> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let len = self.matrix.len_of(Axis(0));
        for _ in 0..len*4+1 {
            write!(f, "-")?;
        }
        write!(f, "\n")?;

        for i in 0..len {
            write!(f, "| ")?;

            for j in 0..len {
                write!(f, "{} | ", self.matrix[(i, j)])?;
            }
            write!(f, "\n")?;
        }

        for _ in 0..len*4+1 {
            write!(f, "-")?;
        }

        Ok(())
    }
}

/// Classification functions
///
/// Contains only routine for Confusion Matrix, as all other current metrices can be derived from
/// the entries in the matrix.
pub trait Classification<A: PartialEq + Ord, D: Data<Elem = A>> {
    fn confusion_matrix(self, ground_truth: &ArrayBase<D, Ix1>) -> ConfusionMatrix<A>;
}

impl<A: Eq + Hash + Copy + Ord, C: Data<Elem = A>, D: Data<Elem = A>> Classification<A, D> for ModifiedPrediction<A, C> {
    fn confusion_matrix(self, ground_truth: &ArrayBase<D, Ix1>) -> ConfusionMatrix<A> {
        // if we don't have any classes, create a set of predicted labels
        let classes = if self.classes.len() == 0 {
            let mut classes = ground_truth.iter().chain(self.prediction.iter()).map(|x| *x).collect::<Vec<_>>();
            // create a set
            classes.sort();
            classes.dedup();
            classes
        } else {
            self.classes
        };

        // find indices to labels
        let indices = map_prediction_to_idx(&self.prediction, ground_truth, &classes);

        // count each index tuple in the confusion matrix
        let mut confusion_matrix = Array2::zeros((classes.len(), classes.len()));
        for (i1, i2) in indices.into_iter().filter_map(|x| x) {
            confusion_matrix[(i1, i2)] += 1;
        }

        ConfusionMatrix {
            matrix: confusion_matrix, 
            members: Array1::from(classes)
        }
    }
}

impl<A: Eq + std::hash::Hash + Copy + Ord, C: Data<Elem = A>, D: Data<Elem = A>>  Classification<A, D> for ArrayBase<C, Ix1> {
    fn confusion_matrix(self, ground_truth: &ArrayBase<D, Ix1>) -> ConfusionMatrix<A> {
        let tmp = ModifiedPrediction {
            prediction: self,
            classes: Vec::new(),
            weights: Vec::new()
        };

        tmp.confusion_matrix(ground_truth)
    }
}

/*
 * TODO: specialization requires unstable Rust
impl Classification<bool, OwnedRepr<bool>> for Array1<bool> {
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
        integral = integral + (*x - prev_x) * (prev_y + *y) / A::from(2.0).unwrap();
        prev_x = *x;
        prev_y = *y;
    }
    integral
}

pub struct ReceiverOperatingCharacteristic<A> {
    curve: Vec<(A, A)>,
    thresholds: Vec<A>
}

impl<A: NdFloat> ReceiverOperatingCharacteristic<A> {
    pub fn get_curve(&self) -> Vec<(A, A)> {
        self.curve.clone()
    }

    pub fn get_thresholds(&self) -> Vec<A> {
        self.thresholds.clone()
    }

    pub fn area_under_curve(&self) -> A {
        trapezoidal(&self.curve)
    }
}

pub trait BinaryClassification<A> {
    fn roc(&self, y: &[bool]) -> ReceiverOperatingCharacteristic<A>;
}

/// The ROC curve gives insight about the seperability of a binary classification task. This
/// functions returns the ROC curve and threshold belonging to each position on the curve.
impl<A: NdFloat, D: Data<Elem = A>> BinaryClassification<A> for ArrayBase<D, Ix1> {
    fn roc(&self, y: &[bool]) -> ReceiverOperatingCharacteristic<A> {
        let mut tuples = self
            .iter()
            .zip(y.iter())
            .filter_map(|(a, b)| if *a >= A::zero() { Some((*a, *b)) } else { None })
            .collect::<Vec<(A, bool)>>();
        
        tuples.sort_unstable_by(&|a: &(A, _), b: &(A, _)| match a.0.partial_cmp(&b.0) {
            Some(ord) => ord,
            None => unreachable!(),
        });
        
        let (mut tp, mut fp) = (A::zero(), A::zero());
        let mut tps_fps = Vec::new();
        let mut thresholds = Vec::new();
        let mut s0 = A::zero();
        
        for (s, t) in tuples {
            if s != s0 {
                tps_fps.push((tp, fp));
                thresholds.push(s);
                s0 = s;
            }
        
            if t {
                tp += A::one();
            } else {
                fp += A::one();
            }
        }
        tps_fps.push((tp, fp));
        
        let (max_tp, max_fp) = (tp, fp);
        for (tp, fp) in &mut tps_fps {
            *tp /= max_tp;
            *fp /= max_fp;
        }
        
        ReceiverOperatingCharacteristic {
            curve: tps_fps,
            thresholds
        }
    }
}
#[cfg(test)]
mod tests {
    use ndarray::{ArrayView1, Array1, Data, ArrayBase, Dimension, array};
    use super::{Modify, Classification, BinaryClassification};
    use rand::{distributions::Uniform, Rng};

    fn assert_eq_slice<A: std::fmt::Debug + PartialEq + Clone, S: Data<Elem = A>, D: Dimension>(a: ArrayBase<S, D>, b: &[A]) {
        let a = a.iter().cloned().collect::<Vec<_>>();
        assert_eq!(a, b);
    }

    #[test]
    fn test_confusion_matrix() {
        let predicted = ArrayView1::from(&[0, 1, 0, 1, 0, 1]);
        let ground_truth = ArrayView1::from(&[1, 1, 0, 1, 0, 1]);

        let cm = predicted.confusion_matrix(&ground_truth);

        assert_eq_slice(cm.matrix, &[2, 0, 1, 3]);
    }

    #[test]
    fn test_cm_metrices() {
        let predicted = Array1::from(vec![0, 1, 0, 1, 0, 1]);
        let ground_truth = Array1::from(vec![1, 1, 0, 1, 0, 1]);

        let x = predicted
            .confusion_matrix(&ground_truth);

        assert_eq!(x.accuracy(), 5.0 / 6.0);
        assert_eq!(x.mcc(), (2.*3. - 1.*0.) / (2.0f32*3.*3.*4.).sqrt());
        assert_eq_slice(x.precision(), &[1.0, 3./4.]);
        assert_eq_slice(x.recall(), &[2.0 / 3.0, 1.0]);
        assert_eq_slice(x.f1_score(), &[4.0 / 5.0, 6.0 / 7.0]);
    }

    #[test]
    fn test_roc_curve() {
        let predicted = ArrayView1::from(&[0.1, 0.3, 0.5, 0.7, 0.8, 0.9]);
        let groundtruth = vec![false, true, false, true, true, true];

        let result = &[
            (0.0, 0.0),  // start 
            (0.0, 0.5),  // first item is target=false
            (0.25, 0.5), // second item is target=true, but obviously false
            (0.25, 1.0), // third item is target=false, we reach max false-positive, because all other labels are positive
            (0.5, 1.0),  // the remaining three are target=true
            (0.75, 1.0),
            (1., 1.)
        ];

        let roc = predicted.roc(&groundtruth);
        assert_eq!(roc.get_curve(), result);
    }

    #[test]
    fn test_roc_auc(){
        let predicted = Array1::linspace(0.0, 1.0, 1000);

        let mut rng = rand::thread_rng();
        let range = Uniform::new(0, 2);

        // randomly sample ground truth
        let ground_truth = (0..1000).map(|_| rng.sample(&range) == 1)
            .collect::<Vec<_>>();

        // ROC Area-Under-Curve should be approximately 0.5
        let roc = predicted.roc(&ground_truth);
        assert!((roc.area_under_curve() - 0.5) < 0.04);
    }

    #[test]
    fn split_one_vs_all() {
        let predicted =    array![0, 3, 2, 0, 1, 1, 1, 3, 2, 3];
        let ground_truth = array![0, 2, 3, 0, 1, 2, 1, 2, 3, 2];

        let cm = predicted.confusion_matrix(&ground_truth);
        let n_cm = cm.split_one_vs_all();

        println!("{:?}", n_cm);
    }
}
