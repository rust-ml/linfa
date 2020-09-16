use crate::dataset::{Data, Targets, Dataset};

pub trait Transformer<D: Data, T> {
    //fn default(&self) -> T;
    fn transform(&self, x: D) -> T;
}

pub trait Fit<D: Data, T> {
    type Object: Predict<D, T>;

    fn fit(&self, dataset: Dataset<D, T>) -> Self::Object;
}

pub trait IncrementalFit<D: Data, T> {
    fn fit_update(self, dataset: Dataset<D, T>) -> Self;
    fn reset_state(&self);
}

pub trait Predict<D: Data, T> {
    fn predict(&self, x: D) -> T;
}
