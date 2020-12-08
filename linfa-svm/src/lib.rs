//! Support Vector Machines
//!
//! Support Vector Machines are one major branch of machine learning models and offer classification or
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
//! let dataset = ...;
//! let kernel = Kernel::gaussian_sparse(&dataset, 10);
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
//! accuracy of 0.988% is reached within 2911 iterations and 1248 support vectors. You can find the
//! example [here](https://github.com/rust-ml/linfa/blob/master/linfa-svm/examples/winequality.rs).
//! ```ignore
//! Fit SVM classifier with #1439 training points
//! Exited after 2911 iterations with obj = -248.51510322468084 and 1248 support vectors
//!
//! classes    | bad        | good
//! bad        | 1228       | 17
//! good       | 0          | 194
//!
//! accuracy 0.98818624, MCC 0.9523008
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

use permutable_kernel::Kernel;
pub use solver_smo::SolverParams;

pub struct SvmParams<F: Float, T> {
    c: Option<(F, F)>,
    nu: Option<(F, F)>,
    solver_params: SolverParams<F>,
    phantom: PhantomData<T>,
}

impl<F: Float, T> SvmParams<F, T> {
    pub fn eps(mut self, new_eps: F) -> Self {
        self.solver_params.eps = new_eps;
        self
    }

    pub fn shrinking(mut self, shrinking: bool) -> Self {
        self.solver_params.shrinking = shrinking;

        self
    }
}

impl<F: Float> SvmParams<F, Pr> {
    pub fn pos_neg_weights(mut self, c_pos: F, c_neg: F) -> Self {
        self.c = Some((c_pos, c_neg));
        self.nu = None;

        self
    }

    pub fn nu_weight(mut self, nu: F) -> Self {
        self.nu = Some((nu, nu));
        self.c = None;

        self
    }
}

impl<F: Float> SvmParams<F, F> {
    pub fn c_eps(mut self, c: F, eps: F) -> Self {
        self.c = Some((c, eps));
        self.nu = None;

        self
    }

    pub fn nu_eps(mut self, nu: F, eps: F) -> Self {
        self.nu = Some((nu, eps));
        self.c = None;

        self
    }
}

/// Support Vector Classification
#[allow(non_snake_case)]
pub mod SVClassify {
    pub use crate::classification::{fit_c, fit_nu, fit_one_class};
}

/// Support Vector Regression
#[allow(non_snake_case)]
pub mod SVRegress {
    pub use crate::regression::{fit_epsilon, fit_nu};
}

/// SMO can either exit because a threshold is reached or the iterations are maxed out
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

/// The result of the SMO solver
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Svm<'a, A: Float, T> {
    pub alpha: Vec<A>,
    pub rho: A,
    r: Option<A>,
    exit_reason: ExitReason,
    iterations: usize,
    obj: A,
    #[cfg_attr(
        feature = "serde",
        serde(bound(
            serialize = "&'a Kernel<'a, A>: Serialize",
            deserialize = "&'a Kernel<'a, A>: Deserialize<'de>"
        ))
    )]
    kernel: &'a Kernel<'a, A>,
    linear_decision: Option<Array1<A>>,
    phantom: PhantomData<T>,
}

impl<'a, A: Float, T> Svm<'a, A, T> {
    pub fn params() -> SvmParams<A, T> {
        SvmParams {
            c: Some((A::one(), A::one())),
            nu: None,
            solver_params: SolverParams {
                eps: A::from(1e-7).unwrap(),
                shrinking: false,
            },
            phantom: PhantomData,
        }
    }

    /// Returns the number of support vectors
    pub fn nsupport(&self) -> usize {
        self.alpha
            .iter()
            .filter(|x| x.abs() > A::from(1e-5).unwrap())
            .count()
    }
    pub fn with_phantom<S>(self) -> Svm<'a, A, S> {
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

impl<'a, A: Float, T> fmt::Display for Svm<'a, A, T> {
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
