use crate::Float;

pub enum Kind {
    CSvc,
    NuSvc,
    OneClassSvc,
    EpsSvr,
    NuSvr,
}

/// Parameters of the solver routine
#[derive(Clone)]
pub struct SolverParams<A: Float> {
    /// Stopping condition
    pub eps: A,
    /// Should we shrink, e.g. ignore bounded alphas
    pub shrinking: bool,
}
