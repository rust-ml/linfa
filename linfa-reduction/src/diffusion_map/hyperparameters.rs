pub struct DiffusionMapHyperParams {
    steps: usize,
    embedding_size: usize,
}

pub struct DiffusionMapHyperParamsBuilder {
    steps: usize,
    embedding_size: usize,
}

impl DiffusionMapHyperParamsBuilder {
    pub fn steps(mut self, steps: usize) -> Self {
        self.steps = steps;

        self
    }

    pub fn build(self) -> DiffusionMapHyperParams {
        DiffusionMapHyperParams::build(self.steps, self.embedding_size)
    }
}

impl DiffusionMapHyperParams {
    // Violates the convention that new should return a value of type `Self`
    #[allow(clippy::new_ret_no_self)]
    pub fn new(embedding_size: usize) -> DiffusionMapHyperParamsBuilder {
        DiffusionMapHyperParamsBuilder {
            steps: 10,
            embedding_size,
        }
    }

    pub fn steps(&self) -> usize {
        self.steps
    }

    pub fn embedding_size(&self) -> usize {
        self.embedding_size
    }

    pub fn build(steps: usize, embedding_size: usize) -> Self {
        DiffusionMapHyperParams {
            steps,
            embedding_size,
        }
    }
}
