use friedrich::gaussian_process::GaussianProcess;
use friedrich::kernel::Kernel;
use friedrich::prior::Prior;
use friedrich::Input;
use ndarray::{ArrayBase, Ix1, Ix2, Data};

/// `GaussianProcessHyperParams` holds the hyperparameters for `GaussianProcessRegressor`.
///
/// In a regression problem you need to identify a function which captures the relationship
/// between the input and output variable(s) based on a set of labeled training data.
///
/// A [Gaussian Process](https://en.wikipedia.org/wiki/Gaussian_process) provides us with
/// a probability distribution over a family of functions.
/// In other words, it defines a solution space for our regression problem: what functions we are
/// considering when looking for the **optimal** solution.
///
/// To control our solution space, we have two handles:
/// - a mean function `m(x)`;
/// - a covariance kernel `k(x,y)`.
/// Names match behaviour: if we draw `n` points (`X`) from our input domain,
/// then the corresponding set of random variables from our gaussian process will be
/// distributed according to a `n`-dimensional Gaussian distribution, with mean `m(X)` and covariance
/// matrix `K(X, X)`, where `m_i(X) = m(x_i)` and `K_{i,j}(X, X) = k(x_i, x_j)` [`0 <= i,j < n`].
///
/// `m(x)` and `k(x,y)` act as a **prior** - they codify our assumptions on the solution of
/// our regression problem **before** we see any training data.
///
/// Given a set of labeled training data, we can refine our assumptions:
/// we compute the posterior distribution combining the evidence of the training data
/// with our prior distribution.
///
/// We can then:
/// - use [MAP](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation) to provide a
///   solution to the regression problem (i.e. use the mean of the posterior distribution as prediction
///   for unlabeled data points);
/// - sample a solution from the function family described by our posterior distribution.
///
/// A brief introduction to Gaussian Processes for machine learning purposes can be found
/// [here](http://cs229.stanford.edu/section/cs229-gaussian_processes.pdf).
/// A more detailed reference (a whole book on the subject!) is [Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/chapters/),
/// by Rasmussen and Williams.
pub struct GaussianProcessHyperParams<KernelType: Kernel, MeanType: Prior> {
    /// Mean function of our gaussian prior - `m(x)`.
    ///
    /// The mean of our gaussian prior might itself have a set of hyperparameters.
    /// If `should_fit_mean` is equal to `true`, those hyperparameters will be optimised
    /// based on the training data.
    mean: MeanType,
    /// Kernel function of our gaussian prior - `k(x, y)`.
    ///
    /// The kernel function of our gaussian prior might itself have a set of hyperparameters.
    /// If `should_fit_kernel` is equal to `true`, those hyperparameters will be optimised
    /// based on the training data.
    kernel: KernelType,
    /// Noise-level in the data.
    noise: f64,
    /// Determine if the hyperparameters of the covariance kernel of our prior should be fitted on the training data.
    should_fit_kernel: bool,
    /// Determine if the hyperparameters of the mean of our prior should be fitted on the training data.
    should_fit_mean: bool,
    /// Maximum number of iterations when optimising hyperparameters.
    max_n_iterations: usize,
    /// ???
    convergence_fraction: f64,
}

/// An helper struct used to construct a set of [valid hyperparameters](struct.GaussianProcessHyperParams.html) for
/// the [Gaussian Process regression algorithm](struct.GaussianProcess.html) (using the builder pattern).
pub struct GaussianProcessHyperParamsBuilder<KernelType: Kernel, MeanType: Prior> {
    mean: MeanType,
    kernel: KernelType,
    noise: f64,
    should_fit_kernel: bool,
    should_fit_mean: bool,
    max_n_iterations: usize,
    convergence_fraction: f64,
}

impl<KernelType: Kernel, MeanType: Prior> GaussianProcessHyperParams<KernelType, MeanType> {
    /// `new` lets us configure our training algorithm parameters.
    ///
    /// Defaults are provided if optional parameters are not specified:
    /// * `noise = 0.1`;
    /// * `should_fit_kernel = false`;
    /// * `should_fit_mean = false`;
    /// * `max_n_iterations = 100`;
    /// * `convergence_fraction = 0.05`;
    pub fn new() -> GaussianProcessHyperParamsBuilder<KernelType, MeanType> {
        let mean = MeanType::default(todo!());
        let kernel = KernelType::default();
        let noise = 0.1; // 10% of output std by default
        let should_fit_kernel = false;
        let should_fit_mean = false;
        let max_n_iterations = 100;
        let convergence_fraction = 0.05;
        GaussianProcessHyperParamsBuilder {
            mean,
            kernel,
            noise,
            should_fit_kernel,
            should_fit_mean,
            max_n_iterations,
            convergence_fraction,
        }
    }
}

impl<KernelType: Kernel, MeanType: Prior> GaussianProcessHyperParamsBuilder<KernelType, MeanType> {
    /// Set the mean of the gaussian process, `m(x)`.
    ///
    /// See the documentation on `Prior`s for more information.
    pub fn mean<NewMeanType: Prior>(
        self,
        mean: NewMeanType,
    ) -> GaussianProcessHyperParamsBuilder<KernelType, NewMeanType> {
        GaussianProcessHyperParamsBuilder {
            mean,
            kernel: self.kernel,
            noise: self.noise,
            should_fit_kernel: self.should_fit_kernel,
            should_fit_mean: self.should_fit_mean,
            max_n_iterations: self.max_n_iterations,
            convergence_fraction: self.convergence_fraction,
        }
    }

    /// Sets the noise parameter.
    ///
    /// It correspond to the standard deviation of the noise in the outputs of the training set.
    pub fn noise(self, noise: f64) -> Self {
        assert!(noise > 0., "The noise parameter should be strictly over 0.");
        Self { noise, ..self }
    }

    /// Set the kernel `k(x,y)` of the gaussian process.
    ///
    /// See the documentation on `Kernel`s for more information.
    pub fn kernel<NewKernelType: Kernel>(
        self,
        kernel: NewKernelType,
    ) -> GaussianProcessHyperParamsBuilder<NewKernelType, MeanType> {
        GaussianProcessHyperParamsBuilder {
            mean: self.mean,
            kernel,
            noise: self.noise,
            should_fit_kernel: self.should_fit_kernel,
            should_fit_mean: self.should_fit_mean,
            max_n_iterations: self.max_n_iterations,
            convergence_fraction: self.convergence_fraction,
        }
    }

    /// Modifies the stopping criteria of the gradient descent.
    ///
    /// The optimizer runs for a maximum of `max_n_iterations`.
    pub fn max_n_iterations(self, max_n_iterations: usize) -> Self {
        Self {
            max_n_iterations,
            ..self
        }
    }

    /// Modifies the stopping criteria of the gradient descent.
    ///
    /// The optimizer stops prematurely if all gradients are below
    /// `convergence_fraction` time their associated parameter.
    pub fn convergence_fraction(self, convergence_fraction: f64) -> Self {
        Self {
            convergence_fraction,
            ..self
        }
    }

    /// The kernel function of our gaussian prior might itself have a set of hyperparameters.
    ///
    /// If `should_fit_kernel` is equal to `true`, those hyperparameters will be optimised
    /// based on the training data.
    pub fn should_fit_kernel(self) -> Self {
        Self {
            should_fit_kernel: true,
            ..self
        }
    }

    /// The mean of our gaussian prior might itself have a set of hyperparameters.
    ///
    /// If `should_fit_mean` is equal to `true`, those hyperparameters will be optimised
    /// based on the training data.
    pub fn should_fit_mean(self) -> Self {
        Self {
            should_fit_mean: true,
            ..self
        }
    }

    /// Finalise the builder pattern and a set of [valid hyperparameters](struct.GaussianProcessHyperParams.html)
    /// for the [Gaussian Process regression algorithm](struct.GaussianProcess.html).
    pub fn build(self) -> GaussianProcessHyperParams<KernelType, MeanType> {
        GaussianProcessHyperParams {
            mean: self.mean,
            kernel: self.kernel,
            should_fit_mean: self.should_fit_mean,
            should_fit_kernel: self.should_fit_kernel,
            noise: self.noise,
            max_n_iterations: self.max_n_iterations,
            convergence_fraction: self.convergence_fraction,
        }
    }
}

impl<KernelType: Kernel, MeanType: Prior> GaussianProcessHyperParams<KernelType, MeanType> {
    pub fn train<S, T>(
        mut self,
        training_inputs: ArrayBase<S, Ix2>,
        training_outputs: ArrayBase<T, Ix1>,
    ) -> GaussianProcess<KernelType, MeanType>
    where
        S: Data<Elem = f64>,
        T: Data<Elem = f64>,
    {
        let training_inputs = Input::to_dmatrix(&training_inputs);
        let training_outputs = <ArrayBase<T, Ix2> as Input>::to_dvector(&training_outputs);
        if self.should_fit_kernel {
            self.kernel
                .heuristic_fit(&training_inputs, &training_outputs);
        }

        let mut gp = GaussianProcess::<KernelType, MeanType>::new(
            self.mean,
            self.kernel,
            self.noise,
            training_inputs,
            training_outputs,
        );

        gp.fit_parameters(
            self.should_fit_mean,
            self.should_fit_kernel,
            self.max_n_iterations,
            self.convergence_fraction,
        );
        gp
    }
}
