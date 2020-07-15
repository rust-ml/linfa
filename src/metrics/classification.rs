//! Common metrics for performance evaluation of classifier
//!
//! Scoring is essential for classification and regression tasks. This module implements
//! common scoring functions like precision, accuracy, recall, f1-score, ROC and ROC
//! Aread-Under-Curve.
use ndarray::prelude::*;
use ndarray::Data;

/// Precision is the number of true positives divided by the number of prediction positives
pub fn precision<D>(x: &ArrayBase<D, Ix1>, y: &[bool], threshold: f64) -> f64
where
    D: Data<Elem = f64>,
{
    let num_positive = x.iter().filter(|a| **a > threshold).count() as f64;
    let num_true_positives = x
        .into_iter()
        .zip(y.into_iter())
        .filter(|(a, b)| **a > threshold && **b)
        .count() as f64;

    num_true_positives / num_positive
}

/// Accuracy is the number of correct classified divided by total number
pub fn accuracy<D>(x: &ArrayBase<D, Ix1>, y: &[bool], threshold: f64) -> f64
where
    D: Data<Elem = f64>,
{
    let num_correctly_classified = x
        .into_iter()
        .zip(y.into_iter())
        .filter(|(a, b)| (**a > threshold && **b) || (**a <= threshold && !**b))
        .count() as f64;

    let total_number = y.len() as f64;

    num_correctly_classified / total_number
}

/// Recall is the number of true positives, divided by ground truth positives
pub fn recall<D>(x: &ArrayBase<D, Ix1>, y: &[bool], threshold: f64) -> f64
where
    D: Data<Elem = f64>,
{
    let num_true_positives = x
        .into_iter()
        .zip(y.into_iter())
        .filter(|(a, b)| **a > threshold && **b)
        .count() as f64;

    let total_number_positives = y.iter().filter(|x| **x).count() as f64;

    num_true_positives / total_number_positives
}

/// F1 score is defined as a compromise between recall and precision
pub fn f1_score<D>(x: &ArrayBase<D, Ix1>, y: &[bool], threshold: f64) -> f64
where
    D: Data<Elem = f64>,
{
    let recall = recall(x, y, threshold);
    let precision = precision(x, y, threshold);

    2.0 * (recall * precision) / (recall + precision)
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

    #[test]
    fn test_auc() {
        let x = Array1::from_iter((0..100).map(|x| (x % 2) as f64));
        let y: Vec<_> = (0..100).map(|x| x % 2 == 1).collect();

        assert_eq!(roc_auc(&x, &y).2, 1.0);
    }
}
