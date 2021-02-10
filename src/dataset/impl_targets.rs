use std::collections::HashSet;

use super::{DatasetBase, Label, Records, Targets, Result, super::error::Error, TargetsWithLabels, AsProbabilites, Pr};
use ndarray::{ArrayBase, Data, Dimension, ViewRepr, Ix0, Ix1, Ix2, Ix3, Array1, CowArray, Axis, Array2};

impl<L, S: Data<Elem = L>> Targets for ArrayBase<S, Ix1> {
    type Elem = L;

    fn try_single_target(&self) -> Result<CowArray<L, Ix1>> {
        Ok(CowArray::from(self.view()))
    }

    fn multi_targets(&self) -> CowArray<L, Ix2> {
        Ok(self.view().insert_axis(Axis(1)))
    }
}

impl<L, S: Data<Elem = L>> Targets for ArrayBase<S, Ix2> {
    type Elem = L;

    fn try_single_target(&self) -> Result<CowArray<L, Ix1>> {
        if self.len_of(Axis(1)) > 1 {
            return Err(Error::MultipleTargets);
        }

        Ok(CowArray::from(self.row(0)))
    }

    fn multi_targets(&self) -> CowArray<L, Ix2> {
        Ok(CowArray::from(self.view()))
    }
}

impl<L: Label, T: Targets + AsProbabilites> Targets for TargetsWithLabels<L, T> {
    type Elem = L;

    fn try_single_target(&self) -> Result<CowArray<L, Ix1>> {
        self.targets.try_single_target()
    }

    fn multi_targets(&self) -> CowArray<L, Ix2> {
        self.targets.multi_targets()
    }
}

impl<L: Label, S: Data<Elem = Pr>> Targets for TargetsWithLabels<L, ArrayBase<S, Ix3>> {
    type Elem = L;

    fn try_single_target(&self) -> Result<CowArray<L, Ix1>> {
        let res = self.multi_targets();

        if res.len_of(Axis(1)) > 1 {
            return Err(Error::MultipleTargets);
        }

        Ok(res.remove_axis(Axis(1)))
    }

    fn multi_targets(&self) -> CowArray<L, Ix2> {
        let init_vals = (..self.labels.len()).map(|i| (i, f32::INFINITY)).collect();
        let res = self.targets.fold_axis(Axis(2), init_vals, |a, b| {
            if a.1 > b.1 {
                return b;
            } else {
                return a;
            }
        });

        //let labels = self.labels.into_iter().collect::<Vec<_>>();
        //res.map_axis(Axis(1), |a| {});
        panic!("")
    }
}

impl<S: Data<Elem = Pr>> AsProbabilites for ArrayBase<S, Ix3> {
    fn multi_target_probabilities<'a>(&'a self) -> CowArray<'a, Pr, Ix3> {
        CowArray::from(self.view())
    }
}

impl Targets for () {
    type Elem = ();

    fn try_single_target(&self) -> Result<CowArray<(), Ix1>> {
        Ok(CowArray::from(Array1::zeros(0)))
    }

    fn multi_targets(&self) -> CowArray<(), Ix2> {
        CowArray::from(Array2::zeros(0))
    }

}

/// A NdArray with discrete labels can act as labels
impl<L: Label, R: Records, S: Data<Elem = L>> DatasetBase<R, ArrayBase<S, Ix1>> {
    pub fn labels(&self) -> Array1<L> {
        self.iter()
            .cloned()
            .collect::<HashSet<L>>()
            .into_iter()
            .collect()
    }
}

/// A NdArray with discrete labels can act as labels
impl<L: Label, R: Records, S: Data<Elem = L>> DatasetBase<R, ArrayBase<S, Ix2>> {
    pub fn labels(&self) -> Array1<L> {
        self.iter()
            .cloned()
            .collect::<HashSet<L>>()
            .into_iter()
            .collect()
    }
}

impl<R: Records, L: Label, T: Targets<Elem = L> + AsProbabilites> DatasetBase<R, T> {
    pub fn with_labels(self, labels: &[L]) -> DatasetBase<R, TargetsWithLabels<L, T>> {
        let targets = TargetsWithLabels {
            targets: self.targets,
            labels: labels.iter().cloned().collect(),
        };

        DatasetBase {
            records: self.records,
            weights: self.weights,
            targets,
            feature_names: self.feature_names,
        }
    }
}
/*
/// A NdArray can act as targets
impl<L, S: Data<Elem = L>, D: Dimension> Targets for ArrayBase<S, D> {
    type Elem = L;
    type Dim = D;

    fn view<'a>(&'a self) -> ArrayBase<ViewRepr<&'a L>, D> {
        self.view()
    }
}

/// A NdArray with discrete labels can act as labels
impl<L: Label, R, S: Data<Elem = L>> DatasetBase<R, ArrayBase<S, Ix1>> {
    pub fn labels(&self) -> Array1<L> {
        self.iter()
            .cloned()
            .collect::<HashSet<L>>()
            .into_iter()
            .collect()
    }
}

/// A NdArray with discrete labels can act as labels
impl<L: Label, T: Targets<Elem = L, Dim = Ix2>> T {
    pub fn labels(&self) -> Array1<L> {
        self.iter()
            .cloned()
            .collect::<HashSet<L>>()
            .into_iter()
            .collect()
    }
}

/// Empty targets for datasets with just observations
impl Targets for () {
    type Elem = ();
    type Dim = Ix0;

    fn view<'a>(&'a self) -> ArrayBase<ViewRepr<&'a ()>, Ix0> {
        &[()]
    }
}

impl<'a, T: Targets> Targets for &'a T {
    type Elem = T::Elem;

    fn view<'b: 'a>(&'b self) -> ArrayBase<ViewRepr<&'b T::Elem>, T::Dim> {
        (*self).view()
    }
}


impl<L: Label, T: Targets<Elem = L>> Targets for TargetsWithLabels<L, T> {
    type Elem = T::Elem;
    type Dim = T::Dim;

    fn view<'a>(&'a self) -> ArrayBase<ViewRepr<&'a T::Elem>, T::Dim> {
        self.targets.view()
    }
}

*/
