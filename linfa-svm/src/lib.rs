//! # Support Vector Machines
//!
//! Support Vector Machines are a major branch of machine learning models and offer classification or
//! regression analysis of labeled datasets. They seek a discriminant, which seperates the data in
//! an optimal way, e.g. have the fewest numbers of miss-classifications and maximizes the margin
//! between positive and negative classes. A support vector
//! contributes to the discriminant and is therefore important for the classification/regression
//! task. The balance between the number of support vectors and model performance can be controlled
//! with hyperparameters.
//!
//! More details can be found [here](https://en.wikipedia.org/wiki/Support_vector_machine)
//!
//! ## Available parameters in Classification and Regression
//!
//! For supervised classification tasks the C or Nu values are used to control this balance. In
//! [fit_c](SVClassify/fn.fit_c) the
//! C value controls the penalty given to missclassification and should be in the interval (0, inf). In
//! [fit_nu](SVClassify/fn.fit_nu.html) the Nu value controls the number of support vectors and should be in the interval (0, 1].
//!
//! For supervised classification with just one class of data a special classifier is available in
//! [fit_one_class](SVClassify/fn.fit_one_class.html). It also accepts a Nu value.
//!
//! For support vector regression two flavors are available. With
//! [fit_epsilon](SVRegress/fn.fit_epsilon.html) a regression task is learned while minimizing deviation
//! larger than epsilon. In [fit_nu](SVRegress/fn.fit_nu.html) the parameter epsilon is replaced with Nu
//! again and should be in the interval (0, 1]
//!
//! ## Kernel Methods
//! Normally the resulting discriminant is linear, but with [Kernel Methods](https://en.wikipedia.org/wiki/Kernel_method) non-linear relations between the input features
//! can be learned in order improve the performance of the model.
//!  
//! For example to transform a dataset into a sparse RBF kernel with 10 non-zero distances you can
//! use `linfa_kernel`:
//! ```rust, ignore
//! use linfa_kernel::Kernel;
//! let train_kernel = Kernel::params()
//!     .method(KernelMethod::Gaussian(30.0))
//!     .transform(&train);
//! ```
//!
//! # The solver
//! This implementation uses Sequential Minimal Optimization, a widely used optimization tool for
//! convex problems. It selects in each optimization step two variables and updates the variables.
//! In each step it performs:
//!
//! 1. Find a variable, which violates the KKT conditions for the optimization problem
//! 2. Pick a second variables and crate a pair (a1, a2)
//! 3. Optimize the pair (a1, a2)
//!
//! After a couple of iterations the solution may be optimal.
//!
//! # Example
//! The wine quality data consists of 11 features, like "acid", "sugar", "sulfur dioxide", and
//! groups the quality into worst 3 to best 8. These are unified to good 8-7 and bad 3-6 to get a
//! binary classification task.
//!
//! With an RBF kernel and C-Support Vector Classification an
//! accuracy of 88.7% is reached within 79535 iterations and 316 support vectors. You can find the
//! example [here](https://github.com/rust-ml/linfa/blob/master/linfa-svm/examples/winequality.rs).
//! ```ignore
//! Fit SVM classifier with #1440 training points
//! Exited after 79535 iterations with obj = -46317.55802870996 and 316 support vectors
//!
//! classes    | bad        | good
//! bad        | 133        | 9
//! good       | 9          | 8
//!
//! accuracy 0.8867925, MCC 0.40720797
//! ```
use linfa::{dataset::Pr, Float};
use ndarray::Array1;

use std::fmt;
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};

mod classification;
mod permutable_kernel;
mod regression;
pub mod solver_smo;

use linfa_kernel::{Kernel, KernelMethod, KernelOwned, KernelParams};
pub use solver_smo::SolverParams;

/// SVM Hyperparameters
///
/// The SVM fitting process can be controlled in different ways. For classification the C and Nu
/// parameters control the ratio of support vectors and accuracy, eps controls the required
/// precision. After setting the desired parameters a model can be fitted by calling `fit`.
///
/// ## Example
///
/// ```ignore
/// let model = Svm::params()
///     .eps(0.1f64)
///     .shrinking(true)
///     .nu_weight(0.1)
///     .fit(&dataset);
/// ```
///
pub struct SvmParams<F: Float, T> {
    c: Option<(F, F)>,
    nu: Option<(F, F)>,
    solver_params: SolverParams<F>,
    phantom: PhantomData<T>,
    kernel: KernelParams<F>,
}

impl<F: Float, T> SvmParams<F, T> {
    /// Set stopping condition
    ///
    /// This parameter controls the stopping condition. It checks whether the sum of gradients of
    /// the max violating pair is below this threshold and then stops the optimization proces.
    pub fn eps(mut self, new_eps: F) -> Self {
        self.solver_params.eps = new_eps;
        self
    }

    /// Shrink active variable set
    ///
    /// This parameter controls whether the active variable set is shrinked or not. This can speed
    /// up the optimization process, but may degredade the solution performance.
    pub fn shrinking(mut self, shrinking: bool) -> Self {
        self.solver_params.shrinking = shrinking;

        self
    }

    /// Set the kernel to use for training
    ///
    /// This parameter specifies a mapping of input records to a new feature space by means
    /// of the distance function between any couple of points mapped to such new space.
    /// The SVM then applies a linear separation in the new feature space that may result in
    /// a non linear partitioning of the original input space, thus increasing the expressiveness of
    /// this model. To use the "base" SVM model it suffices to choose a `Linear` kernel.
    pub fn kernel(mut self, kernel: KernelParams<F>) -> Self {
        self.kernel = kernel;

        self
    }
}

impl<F: Float> SvmParams<F, Pr> {
    /// Set the C value for positive and negative samples.
    pub fn pos_neg_weights(mut self, c_pos: F, c_neg: F) -> Self {
        self.c = Some((c_pos, c_neg));
        self.nu = None;

        self
    }

    /// Set the Nu value for classification
    ///
    /// The Nu value should lie in range [0, 1] and sets the relation between support vectors and
    /// solution performance.
    pub fn nu_weight(mut self, nu: F) -> Self {
        self.nu = Some((nu, nu));
        self.c = None;

        self
    }
}

impl<F: Float> SvmParams<F, F> {
    /// Set the C value for regression
    pub fn c_eps(mut self, c: F, eps: F) -> Self {
        self.c = Some((c, eps));
        self.nu = None;

        self
    }

    /// Set the Nu-Eps value for regression
    pub fn nu_eps(mut self, nu: F, eps: F) -> Self {
        self.nu = Some((nu, eps));
        self.c = None;

        self
    }
}

/// Reason for stopping
///
/// SMO can either exit because a threshold is reached or the iterations are maxed out. To
/// differentiate between both this flag is passed with the solution.
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug)]
pub enum ExitReason {
    ReachedThreshold,
    ReachedIterations,
}

/// Fitted Support Vector Machines model
///
/// This is the result of the SMO optimizer and contains the support vectors, quality of solution
/// and optionally the linear hyperplane.
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Svm<F: Float, T> {
    pub alpha: Vec<F>,
    pub rho: F,
    r: Option<F>,
    exit_reason: ExitReason,
    iterations: usize,
    obj: F,
    #[cfg_attr(
        feature = "serde",
        serde(bound(
            serialize = "&'a Kernel<'a, F>: Serialize",
            deserialize = "&'a Kernel<'a, F>: Deserialize<'de>"
        ))
    )]
    kernel: KernelOwned<F>,
    linear_decision: Option<Array1<F>>,
    phantom: PhantomData<T>,
}

/// Create hyper parameter set
///
/// This creates a `SvmParams` and sets it to the default values:
///  * C values of (1, 1)
///  * Eps of 1e-7
///  * No shrinking
///  * Linear kernel
impl<F: Float, T> Svm<F, T> {
    pub fn params() -> SvmParams<F, T> {
        SvmParams {
            c: Some((F::one(), F::one())),
            nu: None,
            solver_params: SolverParams {
                eps: F::from(1e-7).unwrap(),
                shrinking: false,
            },
            phantom: PhantomData,
            kernel: Kernel::params().method(KernelMethod::Linear),
        }
    }

    /// Returns the number of support vectors
    ///
    /// This function returns the number of support vectors which have an influence on the decision
    /// outcome greater than zero.
    pub fn nsupport(&self) -> usize {
        self.alpha
            .iter()
            .filter(|x| x.abs() > F::from(1e-5).unwrap())
            .count()
    }
    pub(crate) fn with_phantom<S>(self) -> Svm<F, S> {
        Svm {
            alpha: self.alpha,
            rho: self.rho,
            r: self.r,
            exit_reason: self.exit_reason,
            obj: self.obj,
            iterations: self.iterations,
            kernel: self.kernel,
            linear_decision: self.linear_decision,
            phantom: PhantomData,
        }
    }
}

/// Display solution
///
/// In order to understand the solution of the SMO solver the objective, number of iterations and
/// required support vectors are printed here.
impl<'a, F: Float, T> fmt::Display for Svm<F, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.exit_reason {
            ExitReason::ReachedThreshold => write!(
                f,
                "Exited after {} iterations with obj = {} and {} support vectors",
                self.iterations,
                self.obj,
                self.nsupport()
            ),
            ExitReason::ReachedIterations => write!(
                f,
                "Reached maximal iterations {} with obj = {} and {} support vectors",
                self.iterations,
                self.obj,
                self.nsupport()
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::Svm;
    use linfa::dataset::Dataset;
    use linfa::prelude::*;
    use linfa_kernel::{Kernel, KernelMethod};
    use ndarray::Array1;
    #[test]
    fn test_iter_folding_for_classification() {
        let mut dataset = linfa_datasets::winequality().map_targets(|x| *x > 6);
        let kernel = Kernel::params().method(KernelMethod::Gaussian(80.0));
        let params = Svm::params().pos_neg_weights(7., 0.6).kernel(kernel);

        let avg_acc = dataset
            .iter_fold(4, |training_set| params.fit(&training_set))
            .map(|(model, valid)| {
                model
                    .predict(&valid)
                    .map_targets(|x| **x > 0.0)
                    .confusion_matrix(&valid)
                    .accuracy()
            })
            .sum::<f32>()
            / 4_f32;
        assert!(avg_acc >= 0.5)
    }

    #[test]
    fn test_iter_folding_for_regression() {
        let mut dataset: Dataset<f64, f64> = linfa_datasets::diabetes();
        let kernel = Kernel::params().method(KernelMethod::Linear);
        let params = Svm::params().kernel(kernel).c_eps(10., 0.01);

        let _avg_acc = dataset
            .iter_fold(4, |training_set| params.fit(&training_set))
            .map(|(model, valid)| Array1::from(model.predict(valid.records())).r2(valid.targets()))
            .sum::<f64>()
            / 4_f64;
    }
}
