use std::collections::HashSet;

use super::{DatasetBase, Label, Records, ToTargets, Result, super::error::Error, TargetsWithLabels, ToProbabilities, Pr, Labels};
use ndarray::{ArrayBase, Data, Dimension, Ix1, Ix2, Ix3, Array1, CowArray, Axis, ArrayView2};

impl<L, S: Data<Elem = L>> ToTargets for ArrayBase<S, Ix1> {
    type Elem = L;

    fn to_multi_targets(&self) -> ArrayView2<L> {
        self.view().insert_axis(Axis(1))
    }
}

impl<L, S: Data<Elem = L>> ToTargets for ArrayBase<S, Ix2> {
    type Elem = L;

    fn to_multi_targets(&self) -> ArrayView2<L> {
        self.view()
    }
}

impl<L: Label, T: ToTargets<Elem = L> + ToProbabilities> ToTargets for TargetsWithLabels<L, T> {
    type Elem = L;

    fn to_multi_targets(&self) -> ArrayView2<L> {
        self.targets.to_multi_targets()
    }
}

/*
impl<L: Label, S: Data<Elem = Pr>> ToTargets for TargetsWithLabels<L, ArrayBase<S, Ix3>> {
    type Elem = L;

    fn to_multi_targets(&self) -> CowArray<L, Ix2> {
        /*let init_vals = (..self.labels.len()).map(|i| (i, f32::INFINITY)).collect();
        let res = self.targets.fold_axis(Axis(2), init_vals, |a, b| {
            if a.1 > b.1 {
                return b;
            } else {
                return a;
            }
        });*/

        //let labels = self.labels.into_iter().collect::<Vec<_>>();
        //res.map_axis(Axis(1), |a| {});
        panic!("")
    }
}*/

impl<S: Data<Elem = Pr>> ToProbabilities for ArrayBase<S, Ix3> {
    fn to_multi_target_probabilities<'a>(&'a self) -> CowArray<'a, Pr, Ix3> {
        CowArray::from(self.view())
    }
}

/*
impl ToTargets for () {
    type Elem = ();

    fn try_single_target(&self) -> Result<CowArray<(), Ix1>> {
        Ok(CowArray::from(Array1::zeros(0)))
    }

    fn multi_targets(&self) -> CowArray<(), Ix2> {
        CowArray::from(Array2::zeros(0))
    }

}*/

/// A NdArray with discrete labels can act as labels
impl<L: Label, R: Records, D: Data<Elem = L>, I: Dimension> Labels for DatasetBase<R, ArrayBase<D, I>> {
    type Elem = L;

    fn label_set(&self) -> HashSet<L> {
        self.targets.iter()
            .cloned()
            .collect()
    }
}

/// A NdArray with discrete labels can act as labels
impl<L: Label, R: Records, T: ToTargets<Elem = L>> Labels for DatasetBase<R, TargetsWithLabels<L, T>> {
    type Elem = L;

    fn label_set(&self) -> HashSet<L> {
        self.targets.labels.clone()
    }
}

impl<R: Records, L: Label, T: ToTargets<Elem = L> + ToProbabilities> DatasetBase<R, T> {
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
