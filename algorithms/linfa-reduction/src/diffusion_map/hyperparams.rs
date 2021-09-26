use crate::{DiffusionMap, ReductionError};
use linfa::{param_guard::TransformGuard, ParamGuard};

/// Diffusion map hyperparameters
///
/// The diffusion map algorithms has only two explicit hyperparameter. The first is the stepsize.
/// As the algorithm calculates the closeness of points after a number of steps taken in the
/// diffusion graph, a larger step size introduces a more global behaviour of the projection while
/// a smaller one (especially one) just projects close obserations closely together.
/// The second parameter is the embedding size and defines the target dimensionality.
pub struct DiffusionMapValidParams {
    steps: usize,
    embedding_size: usize,
}

impl DiffusionMapValidParams {
    pub fn steps(&self) -> usize {
        self.steps
    }

    pub fn embedding_size(&self) -> usize {
        self.embedding_size
    }
}

/// Diffusion map hyperparameters
///
/// The diffusion map algorithms has only two explicit hyperparameter. The first is the stepsize.
/// As the algorithm calculates the closeness of points after a number of steps taken in the
/// diffusion graph, a larger step size introduces a more global behaviour of the projection while
/// a smaller one (especially one) just projects close obserations closely together.
/// The second parameter is the embedding size and defines the target dimensionality.
pub struct DiffusionMapParams(DiffusionMapValidParams);

impl DiffusionMapParams {
    /// Set the number of steps in the diffusion operator
    ///
    /// The diffusion map algorithm expresses the transition probability with a kernel matrix and
    /// then takes multiple steps along the diffusion operator. In practice scales the
    /// eigenvalues of the decomposition exponentially with the number of steps.
    pub fn steps(mut self, steps: usize) -> Self {
        self.0.steps = steps;

        self
    }

    pub fn embedding_size(mut self, embedding_size: usize) -> Self {
        self.0.embedding_size = embedding_size;

        self
    }

    /// Creates the set of default parameters
    ///
    /// # Parameters
    ///
    /// * `embedding_size`: the number of dimensions in the projection
    ///
    /// # Returns
    ///
    /// Parameter set with number of steps = 1
    pub fn new(embedding_size: usize) -> DiffusionMapParams {
        Self(DiffusionMapValidParams {
            steps: 1,
            embedding_size,
        })
    }
}

impl Default for DiffusionMapParams {
    fn default() -> Self {
        Self::new(2)
    }
}

impl<F> DiffusionMap<F> {
    pub fn params(embedding_size: usize) -> DiffusionMapParams {
        DiffusionMapParams::new(embedding_size)
    }
}

impl ParamGuard for DiffusionMapParams {
    type Checked = DiffusionMapValidParams;
    type Error = ReductionError;

    fn check_ref(&self) -> Result<&Self::Checked, Self::Error> {
        if self.0.steps == 0 {
            Err(ReductionError::StepsZero)
        } else if self.0.embedding_size == 0 {
            Err(ReductionError::EmbeddingTooSmall(self.0.embedding_size))
        } else {
            Ok(&self.0)
        }
    }

    fn check(self) -> Result<Self::Checked, Self::Error> {
        self.check_ref()?;
        Ok(self.0)
    }
}
impl TransformGuard for DiffusionMapParams {}
