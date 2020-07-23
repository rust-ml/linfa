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
fn map_prediction_to_idx<A: Eq + Hash, D: Data<Elem = A>>(prediction: &ArrayBase<D, Ix1>, ground_truth: &ArrayBase<D, Ix1>, classes: &[A]) -> Vec<Option<(usize, usize)>> {
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
    /// Estimates the correlation between target and predicted variable
    pub fn mcc(&self) -> f32 {
        let mut upper = 0.0;
        for k in 0..self.members.len() {
            for l in 0..self.members.len() {
                for m in 0..self.members.len() {
                    upper += self.matrix[(k, k)] as f32 * self.matrix[(l, m)] as f32;
                    upper -= self.matrix[(k, l)] as f32 * self.matrix[(m, k)] as f32;
                }
            }
        }

        upper
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

impl<A: Eq + Hash + Copy + Ord, D: Data<Elem = A>> Classification<A, D> for ModifiedPrediction<A, D> {
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

impl<A: Eq + std::hash::Hash + Copy + Ord, D: Data<Elem = A>>  Classification<A, D> for ArrayBase<D, Ix1> {
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


/// The ROC curve gives insight about the seperability of a binary classification task. This
/// functions returns the ROC curve and threshold belonging to each position on the curve.
pub fn roc_curve<A: NdFloat, D>(x: &ArrayBase<D, Ix1>, y: &[bool]) -> (Vec<(A, A)>, Vec<A>)
where
    D: Data<Elem = A>,
{
    let mut tuples = x
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

    (tps_fps, thresholds)
}

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

/// Return the Area Under Curve (AUC)
///
/// This function takes a prediction and ground truth and returns the ROC curve, threshold and AUC
/// value
pub fn roc_auc<A: NdFloat + PartialOrd, D>(x: &ArrayBase<D, Ix1>, y: &[bool]) -> (Vec<(A, A)>, Vec<A>, A)
where
    D: Data<Elem = A>,
{
    let (roc_curve, thresholds) = roc_curve(x, y);
    let roc_auc = trapezoidal(&roc_curve);

    (roc_curve, thresholds, roc_auc)
}

#[cfg(test)]
mod tests {
    use super::roc_auc;
    use ndarray::Array1;
    use std::iter::FromIterator;
    use super::{Modify, Classification};

    #[test]
    fn test_confusion_matrix() {
        let predicted = Array1::from(vec![0, 1, 0, 1, 0, 1]);
        let ground_truth = Array1::from(vec![1, 1, 0, 1, 0, 1]);

        let cm = predicted.confusion_matrix(&ground_truth);

        assert_eq!(cm.matrix.as_slice().unwrap(), &[2, 0, 1, 3]);
    }

    #[test]
    fn test_cm_metrices() {
        let predicted = Array1::from(vec![0, 1, 0, 1, 0, 1]);
        let ground_truth = Array1::from(vec![1, 1, 0, 1, 0, 1]);

        let x = predicted
            .confusion_matrix(&ground_truth);

        assert_eq!(x.accuracy(), 5.0 / 6.0);
        assert_eq!(x.precision().as_slice().unwrap(), &[1.0, 3./4.]);
        assert_eq!(x.recall().as_slice().unwrap(), &[2.0 / 3.0, 1.0]);
        assert_eq!(x.f1_score().as_slice().unwrap(), &[4.0 / 5.0, 6.0 / 7.0]);
    }

    #[test]
    fn test_auc() {
        let x = Array1::from_iter((0..100).map(|x| (x % 2) as f64));
        let y: Vec<_> = (0..100).map(|x| x % 2 == 1).collect();

        assert_eq!(roc_auc(&x, &y).2, 1.0);
    }
}
