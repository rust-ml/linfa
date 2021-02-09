use linfa::error::{Error, Result};

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
