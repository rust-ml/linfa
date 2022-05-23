use crate::{KernelMethod, SolverParams, Svm, SvmError};
use linfa::{platt_scaling::PlattParams, Float, ParamGuard, Platt};
use linfa_kernel::{Kernel, KernelParams};
use std::marker::PhantomData;

/// SVM Hyperparameters
///
/// The SVM fitting process can be controlled in different ways. For classification the C and Nu
/// parameters control the ratio of support vectors and accuracy, eps controls the required
/// precision. After setting the desired parameters a model can be fitted by calling `fit`.
///
/// You can specify the expected return type with the turbofish syntax. If you want to enable
/// Platt-Scaling for proper probability values, then use:
/// ```no_run
/// use linfa_svm::Svm;
/// use linfa::dataset::Pr;
/// let model = Svm::<f64, Pr>::params();
/// ```
/// or `bool` if you only wants to know the binary decision:
/// ```no_run
/// use linfa_svm::Svm;
/// let model = Svm::<f64, bool>::params();
/// ```
///
/// ## Example
///
/// ```ignore
/// use linfa_svm::Svm;
/// let model = Svm::<_, bool>::params()
///     .eps(0.1f64)
///     .shrinking(true)
///     .nu_weight(0.1)
///     .fit(&dataset);
/// ```
///
#[derive(Debug, Clone, PartialOrd, PartialEq)]
pub struct SvmValidParams<F: Float, T> {
    c: Option<(F, F)>,
    nu: Option<(F, F)>,
    solver_params: SolverParams<F>,
    phantom: PhantomData<T>,
    kernel: KernelParams<F>,
    platt: PlattParams<F, ()>,
}

impl<F: Float, T> SvmValidParams<F, T> {
    pub fn c(&self) -> Option<(F, F)> {
        self.c
    }

    pub fn nu(&self) -> Option<(F, F)> {
        self.nu
    }

    pub fn solver_params(&self) -> &SolverParams<F> {
        &self.solver_params
    }

    pub fn kernel_params(&self) -> &KernelParams<F> {
        &self.kernel
    }

    pub fn platt_params(&self) -> &PlattParams<F, ()> {
        &self.platt
    }
}

#[derive(Debug, Clone, PartialOrd, PartialEq)]
pub struct SvmParams<F: Float, T>(SvmValidParams<F, T>);

impl<F: Float, T> SvmParams<F, T> {
    /// Create hyper parameter set
    ///
    /// This creates a `SvmParams` and sets it to the default values:
    ///  * C values of (1, 1)
    ///  * Eps of 1e-7
    ///  * No shrinking
    ///  * Linear kernel
    pub fn new() -> Self {
        Self(SvmValidParams {
            c: Some((F::one(), F::one())),
            nu: None,
            solver_params: SolverParams {
                eps: F::cast(1e-7),
                shrinking: false,
            },
            phantom: PhantomData,
            kernel: Kernel::params().method(KernelMethod::Linear),
            platt: Platt::params(),
        })
    }

    /// Set stopping condition
    ///
    /// This parameter controls the stopping condition. It checks whether the sum of gradients of
    /// the max violating pair is below this threshold and then stops the optimization proces.
    pub fn eps(mut self, new_eps: F) -> Self {
        self.0.solver_params.eps = new_eps;
        self
    }

    /// Shrink active variable set
    ///
    /// This parameter controls whether the active variable set is shrinked or not. This can speed
    /// up the optimization process, but may degredade the solution performance.
    pub fn shrinking(mut self, shrinking: bool) -> Self {
        self.0.solver_params.shrinking = shrinking;
        self
    }

    /// Set the kernel to use for training
    ///
    /// This parameter specifies a mapping of input records to a new feature space by means
    /// of the distance function between any couple of points mapped to such new space.
    /// The SVM then applies a linear separation in the new feature space that may result in
    /// a non linear partitioning of the original input space, thus increasing the expressiveness of
    /// this model. To use the "base" SVM model it suffices to choose a `Linear` kernel.
    pub fn with_kernel_params(mut self, kernel: KernelParams<F>) -> Self {
        self.0.kernel = kernel;
        self
    }

    /// Set the platt params for probability calibration
    pub fn with_platt_params(mut self, platt: PlattParams<F, ()>) -> Self {
        self.0.platt = platt;
        self
    }

    /// Sets the model to use the Gaussian kernel. For this kernel the
    /// distance between two points is computed as: `d(x, x') = exp(-norm(x - x')/eps)`
    pub fn gaussian_kernel(mut self, eps: F) -> Self {
        self.0.kernel = Kernel::params().method(KernelMethod::Gaussian(eps));
        self
    }

    /// Sets the model to use the Polynomial kernel. For this kernel the
    /// distance between two points is computed as: `d(x, x') = (<x, x'> + costant)^(degree)`
    pub fn polynomial_kernel(mut self, constant: F, degree: F) -> Self {
        self.0.kernel = Kernel::params().method(KernelMethod::Polynomial(constant, degree));
        self
    }

    /// Sets the model to use the Linear kernel. For this kernel the
    /// distance between two points is computed as : `d(x, x') = <x, x'>`
    pub fn linear_kernel(mut self) -> Self {
        self.0.kernel = Kernel::params().method(KernelMethod::Linear);
        self
    }
}

impl<F: Float, T> SvmParams<F, T> {
    /// Set the C value for positive and negative samples.
    pub fn pos_neg_weights(mut self, c_pos: F, c_neg: F) -> Self {
        self.0.c = Some((c_pos, c_neg));
        self.0.nu = None;
        self
    }

    /// Set the Nu value for classification
    ///
    /// The Nu value should lie in range [0, 1] and sets the relation between support vectors and
    /// solution performance.
    pub fn nu_weight(mut self, nu: F) -> Self {
        self.0.nu = Some((nu, nu));
        self.0.c = None;
        self
    }
}

impl<F: Float> SvmParams<F, F> {
    /// Set the C value for regression
    pub fn c_eps(mut self, c: F, eps: F) -> Self {
        self.0.c = Some((c, eps));
        self.0.nu = None;
        self
    }

    /// Set the Nu-Eps value for regression
    pub fn nu_eps(mut self, nu: F, eps: F) -> Self {
        self.0.nu = Some((nu, eps));
        self.0.c = None;
        self
    }
}

impl<F: Float, L> Default for SvmParams<F, L> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: Float, L> Svm<F, L> {
    pub fn params() -> SvmParams<F, L> {
        SvmParams::new()
    }
}

impl<F: Float, L> ParamGuard for SvmParams<F, L> {
    type Checked = SvmValidParams<F, L>;
    type Error = SvmError;

    fn check_ref(&self) -> Result<&Self::Checked, Self::Error> {
        self.0.platt_params().check_ref()?;

        if self.0.solver_params.eps.is_negative()
            || self.0.solver_params.eps.is_nan()
            || self.0.solver_params.eps.is_infinite()
        {
            return Err(SvmError::InvalidEps(
                self.0.solver_params.eps.to_f32().unwrap(),
            ));
        }
        if let Some((c1, c2)) = self.0.c {
            if c1 <= F::zero() || c2 <= F::zero() {
                return Err(SvmError::InvalidC((
                    c1.to_f32().unwrap(),
                    c2.to_f32().unwrap(),
                )));
            }
        }
        if let Some((nu, _)) = self.0.nu {
            if nu <= F::zero() {
                return Err(SvmError::InvalidNu(nu.to_f32().unwrap()));
            }
        }

        Ok(&self.0)
    }

    fn check(self) -> Result<Self::Checked, Self::Error> {
        self.check_ref()?;
        Ok(self.0)
    }
}
