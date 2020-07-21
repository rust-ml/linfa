//! Common metrics for performance evaluation of classifier
//!
//! Scoring is essential for classification and regression tasks. This module implements
//! common scoring functions like precision, accuracy, recall, f1-score, ROC and ROC
//! Aread-Under-Curve.
use ndarray::prelude::*;
use ndarray::{Data, OwnedRepr};
use std::collections::HashMap;
use std::hash::Hash;
use std::iter::FromIterator;
use std::fmt;

fn map_prediction_to_idx<A: Eq + Hash, D: Data<Elem = A>>(prediction: &ArrayBase<D, Ix1>, ground_truth: &ArrayBase<D, Ix1>, classes: &[A]) -> Vec<Option<(usize, usize)>> {
    let set = classes.iter().enumerate()
        .map(|(a, b)| (b, a))
        .collect::<HashMap<_, usize>>();

    prediction.iter().zip(ground_truth.iter()).map(|(a,b)| {
        set.get(&a)
            .and_then(|x| set.get(&b).map(|y| (*x, *y)))
    }).collect::<Vec<Option<_>>>()
}

pub struct ModifiedDataset<A, D: Data<Elem = A>> {
    prediction: ArrayBase<D, Ix1>,
    weights: Vec<usize>,
    classes: Vec<A>
}

pub trait Prepare<A: PartialOrd + Eq + Hash, D: Data<Elem = A>> {
    fn with_weights(self, weights: &[usize]) -> ModifiedDataset<A, D>;
    fn reduce_classes(self, classes: &[A]) -> ModifiedDataset<A, D>;
}

impl<A: PartialOrd + Eq + Hash + Clone, D: Data<Elem = A>> Prepare<A, D> for ArrayBase<D, Ix1> {
    fn with_weights(self, weights: &[usize]) -> ModifiedDataset<A, D> {
        ModifiedDataset {
            prediction: self, 
            weights: weights.to_vec(),
            classes: Vec::new()
        }
    }

    fn reduce_classes(self, classes: &[A]) -> ModifiedDataset<A, D> {
        ModifiedDataset {
            prediction: self,
            weights: Vec::new(),
            classes: classes.to_vec()
        }
    }
}

impl<A: PartialOrd + Eq + Hash + Clone, D: Data<Elem = A>> Prepare<A, D> for ModifiedDataset<A, D> {
    fn with_weights(self, weights: &[usize]) -> ModifiedDataset<A, D> {
        ModifiedDataset {
            prediction: self.prediction,
            weights: weights.to_vec(),
            classes: self.classes
        }
    }

    fn reduce_classes(self, classes: &[A]) -> ModifiedDataset<A, D> {
        ModifiedDataset {
            prediction: self.prediction,
            weights: self.weights,
            classes: classes.to_vec()
        }
    }
}

pub struct ConfusionMatrix<A> {
    matrix: Array2<usize>,
    members: Array1<A>
}

impl<A> ConfusionMatrix<A> {
    pub fn precision_individual(&self) -> Array1<f32> {
        let sum = self.matrix.sum_axis(Axis(0));

        Array1::from_iter(
            self.matrix.diag().iter()
                .zip(sum.iter())
                .map(|(a, b)| *a as f32 / *b as f32)
        )
    }

    pub fn recall_individual(&self) -> Array1<f32> {
        let sum = self.matrix.sum_axis(Axis(1));

        Array1::from_iter(
            self.matrix.diag().iter()
                .zip(sum.iter())
                .map(|(a, b)| *a as f32 / *b as f32)
        )
    }

    pub fn accuracy_individual(&self) -> Array1<f32> {
        let sum = self.matrix.sum();

        self.matrix.diag().mapv(|x| x as f32) / sum as f32
    }

    pub fn precision(&self) -> f32 {
        self.precision_individual().mean().unwrap()
    }

    pub fn recall(&self) -> f32 {
        self.recall_individual().mean().unwrap()
    }

    pub fn accuracy(&self) -> f32 {
        self.accuracy_individual().mean().unwrap()
    }

    pub fn f_score(&self, beta: f32) -> f32 {
        let sb = beta * beta;
        let precision = self.precision();
        let recall = self.recall();

        (1.0 + sb)* (precision * recall) / (sb * precision + recall)
    }

    pub fn f1_score(&self) -> f32 {
        self.f_score(1.0)
    }

    pub fn matthew_correlation(&self) -> f32 {
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

pub trait Classification<A: PartialEq + Ord, D: Data<Elem = A>> {
    fn accuracy(&self, ground_truth: &ArrayBase<D, Ix1>) -> f32;
    fn confusion_matrix(self, ground_truth: &ArrayBase<D, Ix1>) -> ConfusionMatrix<A>;
}

impl<A: Eq + Hash + Copy + Ord, D: Data<Elem = A>> Classification<A, D> for ModifiedDataset<A, D> {
    fn accuracy(&self, ground_truth: &ArrayBase<D, Ix1>) -> f32{
        if self.weights.len() != self.prediction.len() {
            self.prediction.iter().zip(ground_truth.iter())
                .filter(|(x, y)| x == y)
                .count() as f32 / ground_truth.len() as f32
        } else {
            let total_size: usize = self.weights.iter().map(|x| *x).sum();
            self.prediction.iter().zip(ground_truth.iter()).zip(self.weights.iter())
                .filter(|((x, y), _)| x == y)
                .map(|(_, weight)| *weight as f32)
                .sum::<f32>() / total_size as f32
        }
    }

    fn confusion_matrix(self, ground_truth: &ArrayBase<D, Ix1>) -> ConfusionMatrix<A> {
        let classes = if self.classes.len() == 0 {
            let mut classes = ground_truth.iter().chain(self.prediction.iter()).map(|x| *x).collect::<Vec<_>>();
            classes.sort();
            classes.dedup();
            classes
        } else {
            self.classes
        };

        let indices = map_prediction_to_idx(&self.prediction, ground_truth, &classes);
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
    fn accuracy(&self, ground_truth: &ArrayBase<D, Ix1>) -> f32 {
        self.iter().zip(ground_truth.iter())
            .filter(|(x, y)| x == y)
            .count() as f32 / ground_truth.len() as f32
    }

    default fn confusion_matrix(self, ground_truth: &ArrayBase<D, Ix1>) -> ConfusionMatrix<A> {
        let tmp = ModifiedDataset {
            prediction: self,
            classes: Vec::new(),
            weights: Vec::new()
        };

        tmp.confusion_matrix(ground_truth)
    }
}

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
}


/// The ROC curve gives insight about the seperability of a binary classification task. This
/// functions returns the ROC curve and threshold belonging to each position on the curve.
pub fn roc_curve<D>(x: &ArrayBase<D, Ix1>, y: &[bool]) -> (Vec<(f64, f64)>, Vec<f64>)
where
    D: Data<Elem = f64>,
{
    let mut tuples = x
        .iter()
        .zip(y.iter())
        .filter_map(|(a, b)| if *a >= 0.0 { Some((*a, *b)) } else { None })
        .collect::<Vec<(f64, bool)>>();

    tuples.sort_unstable_by(&|a: &(f64, _), b: &(f64, _)| match a.0.partial_cmp(&b.0) {
        Some(ord) => ord,
        None => unreachable!(),
    });

    let (mut tp, mut fp) = (0.0, 0.0);
    let mut tps_fps = Vec::new();
    let mut thresholds = Vec::new();
    let mut s0 = 0.0;

    for (s, t) in tuples {
        if s != s0 {
            tps_fps.push((tp, fp));
            thresholds.push(s);
            s0 = s;
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

    (tps_fps, thresholds)
}

/// Integration using the trapezoidal rule.
fn trapezoidal(vals: &[(f64, f64)]) -> f64 {
    let mut prev_x = vals[0].0;
    let mut prev_y = vals[0].1;
    let mut integral = 0.0;

    for (x, y) in vals.iter().skip(1) {
        integral = integral + (x - prev_x) * (prev_y + y) / 2.0;
        prev_x = *x;
        prev_y = *y;
    }
    integral
}

/// Return the Area Under Curve (AUC)
///
/// This function takes a prediction and ground truth and returns the ROC curve, threshold and AUC
/// value
pub fn roc_auc<D>(x: &ArrayBase<D, Ix1>, y: &[bool]) -> (Vec<(f64, f64)>, Vec<f64>, f64)
where
    D: Data<Elem = f64>,
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
    use super::{Prepare, Classification};

    #[test]
    fn test_accuracy() {
        let predicted = Array1::from(vec![1, 1, 2, 2, 3, 4]);
        let ground_truth = Array1::from(vec![1, 1, 2, 3, 3, 3]);

        let x = predicted
            .with_weights(&[0])
            .confusion_matrix(&ground_truth);

        println!("{:?}", x);
        println!("{}", x.recall());
    }

    #[test]
    fn test_auc() {
        let x = Array1::from_iter((0..100).map(|x| (x % 2) as f64));
        let y: Vec<_> = (0..100).map(|x| x % 2 == 1).collect();

        assert_eq!(roc_auc(&x, &y).2, 1.0);
    }
}
