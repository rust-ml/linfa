use std::collections::HashSet;

use super::{DatasetBase, Label, Records, AsTargets, AsTargetsMut, TargetsWithLabels, AsProbabilities, Pr, Labels, FromTargetArray};
use ndarray::{ArrayBase, Data, Dimension, Ix1, Ix2, Ix3, CowArray, Axis, ArrayView2, ArrayViewMut2, DataMut, Array2, Array1, ArrayView1, OwnedRepr, ViewRepr};

impl<'a, L, S: Data<Elem = L>> AsTargets for ArrayBase<S, Ix1> {
    type Elem = L;

    fn as_multi_targets(&self) -> ArrayView2<L> {
        self.view().insert_axis(Axis(1))
    }
}

impl<'a, L: Clone + 'a, S: Data<Elem = L>> FromTargetArray<'a, L> for ArrayBase<S, Ix2> {
    type Owned = ArrayBase<OwnedRepr<L>, Ix2>;
    type View = ArrayBase<ViewRepr<&'a L>, Ix2>;

    fn new_targets(targets: Array2<L>) -> Self::Owned {
        targets
    }

    fn new_targets_view(targets: ArrayView2<'a, L>) -> Self::View {
        targets
    }
}

/*
impl<'a, L: 'a, S: Data<Elem = L> , I: Dimension> FromTargetArray<&'a ArrayBase<S, I>, ArrayBase<ViewRepr<&'a L>, I>> for ArrayBase<S, I> {
    fn from_array(array: &'a ArrayBase<S, I>) -> ArrayBase<ViewRepr<&'a L>, I> {
        array.view()
    }
}*/

impl<L, S: DataMut<Elem = L>> AsTargetsMut for ArrayBase<S, Ix1> {
    type Elem = L;

    fn as_multi_targets_mut<'a>(&'a mut self) -> ArrayViewMut2<'a, Self::Elem> {
        self.view_mut().insert_axis(Axis(1))
    }
}

impl<L, S: Data<Elem = L>> AsTargets for ArrayBase<S, Ix2> {
    type Elem = L;

    fn as_multi_targets(&self) -> ArrayView2<L> {
        self.view()
    }
}

impl<L, S: DataMut<Elem = L>> AsTargetsMut for ArrayBase<S, Ix2> {
    type Elem = L;

    fn as_multi_targets_mut<'a>(&'a mut self) -> ArrayViewMut2<'a, Self::Elem> {
        self.view_mut()
    }
}

impl<T: AsTargets> AsTargets for &T {
    type Elem = T::Elem;

    fn as_multi_targets(&self) -> ArrayView2<Self::Elem> {
        (*self).as_multi_targets()
    }
}

impl<L: Label, T: AsTargets<Elem = L> + AsProbabilities> AsTargets for TargetsWithLabels<L, T> {
    type Elem = L;

    fn as_multi_targets(&self) -> ArrayView2<L> {
        self.targets.as_multi_targets()
    }
}

impl<L: Label, T: AsTargetsMut<Elem = L>> AsTargetsMut for TargetsWithLabels<L, T> {
    type Elem = L;

    fn as_multi_targets_mut<'a>(&'a mut self) -> ArrayViewMut2<'a, Self::Elem> {
        self.targets.as_multi_targets_mut()
    }
}

impl<'a, L: Label + 'a, T> FromTargetArray<'a, L> for TargetsWithLabels<L, T>
where
    T: AsTargets<Elem = L> + FromTargetArray<'a, L>,
    T::Owned: Labels<Elem = L>,
    T::View: Labels<Elem = L>,
{
    type Owned = TargetsWithLabels<L, T::Owned>;
    type View = TargetsWithLabels<L, T::View>;

    fn new_targets(targets: Array2<L>) -> Self::Owned {
        let targets = T::new_targets(targets);

        TargetsWithLabels {
            labels: targets.label_set(),
            targets,
        }
    }

    fn new_targets_view(targets: ArrayView2<'a, L>) -> Self::View {
        let targets = T::new_targets_view(targets);

        TargetsWithLabels {
            labels: targets.label_set(),
            targets,
        }
    }
}
/*
impl<L: Label, S: Data<Elem = Pr>> AsTargets for TargetsWithLabels<L, ArrayBase<S, Ix3>> {
    type Elem = L;

    fn as_multi_targets(&self) -> CowArray<L, Ix2> {
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

impl<S: Data<Elem = Pr>> AsProbabilities for ArrayBase<S, Ix3> {
    fn as_multi_target_probabilities<'a>(&'a self) -> CowArray<'a, Pr, Ix3> {
        CowArray::from(self.view())
    }
}

/*
impl AsTargets for () {
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
impl<L: Label, R: Records, T: AsTargets<Elem = L>> Labels for DatasetBase<R, TargetsWithLabels<L, T>> {
    type Elem = L;

    fn label_set(&self) -> HashSet<L> {
        self.targets.labels.clone()
    }
}

impl<R: Records, L: Label, T: AsTargets<Elem = L> + AsProbabilities> DatasetBase<R, T> {
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
