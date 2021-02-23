use linfa::error::{Error, Result};

/// Diffusion map hyperparameters
///
/// The diffusion map algorithms has only two explicit hyperparameter. The first is the stepsize.
/// As the algorithm calculates the closeness of points after a number of steps taken in the
/// diffusion graph, a larger step size introduces a more global behaviour of the projection while
/// a smaller one (especially one) just projects close obserations closely together.
/// The second parameter is the embedding size and defines the target dimensionality.
pub struct DiffusionMapParams {
    pub steps: usize,
    pub embedding_size: usize,
}

impl DiffusionMapParams {
    /// Set the number of steps in the diffusion operator
    ///
    /// The diffusion map algorithm expresses the transition probability with a kernel matrix and
    /// then takes multiple steps along the diffusion operator. This scales in practice the
    /// eigenvalues of the decomposition exponentially with the number of steps.
    pub fn steps(mut self, steps: usize) -> Self {
        self.steps = steps;

        self
    }

    /// Validates the parameter
    pub fn validate(&self) -> Result<()> {
        if self.steps == 0 {
            return Err(Error::Parameters(
                "number of steps should be larger than zero".into(),
            ));
        }

        if self.embedding_size == 0 {
            return Err(Error::Parameters(
                "embedding size should be larger than zero".into(),
            ));
        }

        Ok(())
    }
}
