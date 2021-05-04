use std::cmp::{Ordering, Reverse};

use linfa::Float;
use noisy_float::{checkers::FiniteChecker, NoisyFloat};

pub(crate) struct HeapElem<D: Ord, T> {
    pub(crate) dist: D,
    pub(crate) elem: T,
}

impl<D: Ord, T> PartialEq for HeapElem<D, T> {
    fn eq(&self, other: &Self) -> bool {
        self.dist.eq(&other.dist)
    }
}
impl<D: Ord, T> Eq for HeapElem<D, T> {}

impl<D: Ord, T> PartialOrd for HeapElem<D, T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}

impl<D: Ord, T> Ord for HeapElem<D, T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.cmp(&other.dist)
    }
}

pub(crate) type MinHeapElem<F, T> = HeapElem<Reverse<NoisyFloat<F, FiniteChecker>>, T>;

impl<F: Float, T> MinHeapElem<F, T> {
    pub(crate) fn new(dist: F, elem: T) -> Self {
        Self {
            dist: Reverse(NoisyFloat::new(dist)),
            elem,
        }
    }
}
