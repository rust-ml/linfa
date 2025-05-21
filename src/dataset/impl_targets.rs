use std::collections::HashMap;

use super::{
    AsMultiTargets, AsMultiTargetsMut, AsProbabilities, AsSingleTargets, AsSingleTargetsMut,
    AsTargets, AsTargetsMut, CountedTargets, DatasetBase, FromTargetArray, FromTargetArrayOwned,
    Label, Labels, Pr, TargetDim,
};
use ndarray::{
    Array, Array1, Array2, ArrayBase, ArrayView, ArrayViewMut, Axis, CowArray, Data, DataMut,
    Dimension, Ix1, Ix2, Ix3, OwnedRepr, ViewRepr,
};

impl TargetDim for Ix1 {}
impl TargetDim for Ix2 {}

impl<L, S: Data<Elem = L>, I: TargetDim> AsTargets for ArrayBase<S, I> {
    type Elem = L;
    type Ix = I;

    fn as_targets(&self) -> ArrayView<L, I> {
        self.view()
    }
}

impl<T: AsTargets<Ix = Ix1>> AsSingleTargets for T {}
impl<T: AsTargets<Ix = Ix2>> AsMultiTargets for T {}

impl<L: Clone, S: Data<Elem = L>, I: TargetDim> FromTargetArrayOwned for ArrayBase<S, I> {
    type Owned = ArrayBase<OwnedRepr<L>, I>;

    /// Returns an owned representation of the target array
    fn new_targets(targets: Array<L, I>) -> Self::Owned {
        targets
    }
}

impl<'a, L: Clone + 'a, S: Data<Elem = L>, I: TargetDim> FromTargetArray<'a> for ArrayBase<S, I> {
    type View = ArrayBase<ViewRepr<&'a L>, I>;

    /// Returns a reference to the target array
    fn new_targets_view(targets: ArrayView<'a, L, I>) -> Self::View {
        targets
    }
}

impl<L, S: DataMut<Elem = L>, I: TargetDim> AsTargetsMut for ArrayBase<S, I> {
    type Elem = L;
    type Ix = I;

    fn as_targets_mut(&mut self) -> ArrayViewMut<Self::Elem, I> {
        self.view_mut()
    }
}

impl<T: AsTargetsMut<Ix = Ix1>> AsSingleTargetsMut for T {}
impl<T: AsTargetsMut<Ix = Ix2>> AsMultiTargetsMut for T {}

impl<T: AsTargets> AsTargets for &T {
    type Elem = T::Elem;
    type Ix = T::Ix;

    fn as_targets(&self) -> ArrayView<Self::Elem, Self::Ix> {
        (*self).as_targets()
    }
}

impl<L: Label, T: AsTargets<Elem = L>> AsTargets for CountedTargets<L, T> {
    type Elem = L;
    type Ix = T::Ix;

    fn as_targets(&self) -> ArrayView<Self::Elem, Self::Ix> {
        self.targets.as_targets()
    }
}

impl<L: Label, T: AsTargetsMut<Elem = L>> AsTargetsMut for CountedTargets<L, T> {
    type Elem = L;
    type Ix = T::Ix;

    fn as_targets_mut(&mut self) -> ArrayViewMut<Self::Elem, Self::Ix> {
        self.targets.as_targets_mut()
    }
}

impl<L: Label, T> FromTargetArrayOwned for CountedTargets<L, T>
where
    T: FromTargetArrayOwned<Elem = L>,
    T::Owned: Labels<Elem = L>,
{
    type Owned = CountedTargets<L, T::Owned>;

    fn new_targets(targets: Array<L, T::Ix>) -> Self::Owned {
        let targets = T::new_targets(targets);
        CountedTargets {
            labels: targets.label_count(),
            targets,
        }
    }
}

impl<'a, L: Label + 'a, T> FromTargetArray<'a> for CountedTargets<L, T>
where
    T: FromTargetArray<'a, Elem = L>,
    T::View: Labels<Elem = L>,
{
    type View = CountedTargets<L, T::View>;

    fn new_targets_view(targets: ArrayView<'a, L, T::Ix>) -> Self::View {
        let targets = T::new_targets_view(targets);

        CountedTargets {
            labels: targets.label_count(),
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
    fn as_multi_target_probabilities(&self) -> CowArray<'_, Pr, Ix3> {
        CowArray::from(self.view())
    }
}

/// A NdArray with discrete labels can act as labels
impl<L: Label, S: Data<Elem = L>, I: Dimension> Labels for ArrayBase<S, I> {
    type Elem = L;

    fn label_count(&self) -> Vec<HashMap<L, usize>> {
        self.columns()
            .into_iter()
            .map(|x| {
                let mut map = HashMap::new();

                for i in x {
                    *map.entry(i.clone()).or_insert(0) += 1;
                }

                map
            })
            .collect()
    }
}

/// Counted labels can act as labels
impl<L: Label, T: AsTargets<Elem = L>> Labels for CountedTargets<L, T> {
    type Elem = L;

    fn label_count(&self) -> Vec<HashMap<L, usize>> {
        self.labels.clone()
    }
}

impl<F: Copy, L: Copy + Label, D, T> DatasetBase<ArrayBase<D, Ix2>, T>
where
    D: Data<Elem = F>,
    T: AsTargets<Elem = L>,
{
    /// Transforms the input dataset by keeping only those samples whose label appears in `labels`.
    ///
    /// In the multi-target case a sample is kept if *any* of its targets appears in `labels`.
    ///
    /// Sample weights and feature names are preserved by this transformation.
    #[allow(clippy::type_complexity)]
    pub fn with_labels(
        &self,
        labels: &[L],
    ) -> DatasetBase<Array2<F>, CountedTargets<L, Array<L, T::Ix>>> {
        let targets = self.targets.as_targets();
        let old_weights = self.weights();

        let mut records_arr = Vec::new();
        let mut targets_arr = Vec::new();
        let mut weights = Vec::new();

        let mut map = vec![HashMap::new(); self.ntargets()];

        for (i, (r, t)) in self
            .records()
            .rows()
            .into_iter()
            .zip(targets.axis_iter(Axis(0)))
            .enumerate()
        {
            let any_exists = t.iter().any(|a| labels.contains(a));

            if any_exists {
                for (map, val) in map.iter_mut().zip(t.iter()) {
                    *map.entry(*val).or_insert(0) += 1;
                }

                records_arr.push(r);
                targets_arr.push(t);

                if let Some(weight) = old_weights {
                    weights.push(weight[i]);
                }
            }
        }

        let nsamples = records_arr.len();

        let records_arr = records_arr.into_iter().flatten().copied().collect();
        let targets_arr = targets_arr.into_iter().flatten().copied().collect();

        let records =
            Array2::from_shape_vec(self.records.raw_dim().nsamples(nsamples), records_arr).unwrap();
        let targets = Array::from_shape_vec(
            self.targets.as_targets().raw_dim().nsamples(nsamples),
            targets_arr,
        )
        .unwrap();

        let targets = CountedTargets {
            targets,
            labels: map,
        };

        DatasetBase {
            records,
            weights: Array1::from(weights),
            targets,
            feature_names: self.feature_names.clone(),
            target_names: self.target_names.clone(),
        }
    }
}
