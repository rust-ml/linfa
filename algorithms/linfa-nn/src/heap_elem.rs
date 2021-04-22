use std::cmp::{Ordering, Reverse};

use linfa::Float;
use ordered_float::NotNan;

pub(crate) struct HeapElem<F: Float, T> {
    pub(crate) dist: Reverse<NotNan<F>>,
    pub(crate) elem: T,
}

impl<F: Float, T> HeapElem<F, T> {
    pub(crate) fn new(dist: F, elem: T) -> Self {
        Self {
            dist: Reverse(NotNan::new(dist).unwrap()),
            elem,
        }
    }
}

impl<F: Float, T> PartialEq for HeapElem<F, T> {
    fn eq(&self, other: &Self) -> bool {
        self.dist.eq(&other.dist)
    }
}
impl<F: Float, T> Eq for HeapElem<F, T> {}

impl<F: Float, T> PartialOrd for HeapElem<F, T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}

impl<F: Float, T> Ord for HeapElem<F, T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.cmp(&other.dist)
    }
}
