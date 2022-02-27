use std::collections::HashMap;

use super::{
    AsMultiTargets, AsMultiTargetsMut, AsProbabilities, AsSingleTargets, CountedTargets,
    DatasetBase, FromTargetArray, Label, Labels, Pr, Records,
};
use ndarray::{
    Array1, Array2, ArrayBase, ArrayView2, ArrayViewMut2, Axis, CowArray, Data, DataMut, Dimension,
    Ix1, Ix2, Ix3, OwnedRepr, ViewRepr,
};

impl<'a, L, S: Data<Elem = L>> AsMultiTargets for ArrayBase<S, Ix1> {
    type Elem = L;

    fn as_multi_targets(&self) -> ArrayView2<L> {
        self.view().insert_axis(Axis(1))
    }

    fn ntargets(&self) -> usize {
        1
    }
}

impl<'a, L, S: Data<Elem = L>> AsMultiTargets for ArrayBase<S, Ix2> {
    type Elem = L;

    fn as_multi_targets(&self) -> ArrayView2<L> {
        self.view()
    }

    fn ntargets(&self) -> usize {
        self.len_of(Axis(1))
    }
}

impl<'a, L, S: Data<Elem = L>> AsSingleTargets for ArrayBase<S, Ix1> {}

impl<'a, L: Clone + 'a, S: Data<Elem = L>> FromTargetArray<'a, L> for ArrayBase<S, Ix1> {
    type Owned = ArrayBase<OwnedRepr<L>, Ix1>;
    type View = ArrayBase<ViewRepr<&'a L>, Ix1>;

    fn new_targets(targets: Array2<L>) -> Self::Owned {
        let new_shape = &targets.nrows();
        targets.into_shape(*new_shape).unwrap()
    }

    fn new_targets_view(targets: ArrayView2<'a, L>) -> Self::View {
        let new_shape = &targets.nrows();
        targets.into_shape(*new_shape).unwrap()
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

impl<L, S: DataMut<Elem = L>> AsMultiTargetsMut for ArrayBase<S, Ix1> {
    type Elem = L;

    fn as_multi_targets_mut(&mut self) -> ArrayViewMut2<'_, Self::Elem> {
        self.view_mut().insert_axis(Axis(1))
    }

    fn ntargets(&self) -> usize {
        1
    }
}

impl<L, S: DataMut<Elem = L>> AsMultiTargetsMut for ArrayBase<S, Ix2> {
    type Elem = L;

    fn as_multi_targets_mut(&mut self) -> ArrayViewMut2<'_, Self::Elem> {
        self.view_mut()
    }

    fn ntargets(&self) -> usize {
        self.len_of(Axis(1))
    }
}

impl<T: AsMultiTargets> AsMultiTargets for &T {
    type Elem = T::Elem;

    fn as_multi_targets(&self) -> ArrayView2<Self::Elem> {
        (*self).as_multi_targets()
    }

    fn ntargets(&self) -> usize {
        (*self).ntargets()
    }
}

impl<T: AsSingleTargets> AsSingleTargets for &T {}

impl<L: Label, T: AsSingleTargets<Elem = L>> AsSingleTargets for CountedTargets<L, T> {}

impl<L: Label, T: AsMultiTargets<Elem = L>> AsMultiTargets for CountedTargets<L, T> {
    type Elem = L;

    fn as_multi_targets(&self) -> ArrayView2<Self::Elem> {
        self.targets.as_multi_targets()
    }

    fn ntargets(&self) -> usize {
        self.targets.ntargets()
    }
}

impl<L: Label, T: AsMultiTargetsMut<Elem = L>> AsMultiTargetsMut for CountedTargets<L, T> {
    type Elem = L;

    fn as_multi_targets_mut(&mut self) -> ArrayViewMut2<'_, Self::Elem> {
        self.targets.as_multi_targets_mut()
    }

    fn ntargets(&self) -> usize {
        self.targets.ntargets()
    }
}

impl<'a, L: Label + 'a, T> FromTargetArray<'a, L> for CountedTargets<L, T>
where
    T: FromTargetArray<'a, L>,
    T::Owned: Labels<Elem = L>,
    T::View: Labels<Elem = L>,
{
    type Owned = CountedTargets<L, T::Owned>;
    type View = CountedTargets<L, T::View>;

    fn new_targets(targets: Array2<L>) -> Self::Owned {
        let targets = T::new_targets(targets);

        CountedTargets {
            labels: targets.label_count(),
            targets,
        }
    }

    fn new_targets_view(targets: ArrayView2<'a, L>) -> Self::View {
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
impl<L: Label, T> Labels for CountedTargets<L, T> {
    type Elem = L;

    fn label_count(&self) -> Vec<HashMap<L, usize>> {
        self.labels.clone()
    }
}

impl<F: Copy, L: Copy + Label, D, T> DatasetBase<ArrayBase<D, Ix2>, T>
where
    D: Data<Elem = F>,
    T: AsMultiTargets<Elem = L>,
{
    /// Transforms the input dataset by keeping only those samples whose label appears in `labels`.
    ///
    /// In the multi-target case a sample is kept if *any* of its targets appears in `labels`.
    ///
    /// Sample weights and feature names are preserved by this transformation.
    pub fn with_labels(
        &self,
        labels: &[L],
    ) -> DatasetBase<Array2<F>, CountedTargets<L, Array2<L>>> {
        let targets = self.targets.as_multi_targets();
        let old_weights = self.weights();

        let mut records_arr = Vec::new();
        let mut targets_arr = Vec::new();
        let mut weights = Vec::new();

        let mut map = vec![HashMap::new(); targets.len_of(Axis(1))];

        for (i, (r, t)) in self
            .records()
            .rows()
            .into_iter()
            .zip(targets.rows().into_iter())
            .enumerate()
        {
            let any_exists = t.iter().any(|a| labels.contains(a));

            if any_exists {
                for (map, val) in map.iter_mut().zip(t.iter()) {
                    *map.entry(*val).or_insert(0) += 1;
                }

                records_arr.push(r.insert_axis(Axis(1)));
                targets_arr.push(t.insert_axis(Axis(1)));

                if let Some(weight) = old_weights {
                    weights.push(weight[i]);
                }
            }
        }

        let nsamples = records_arr.len();
        let nfeatures = self.nfeatures();
        let ntargets = self.ntargets();

        let records_arr = records_arr.into_iter().flatten().copied().collect();
        let targets_arr = targets_arr.into_iter().flatten().copied().collect();

        let records = Array2::from_shape_vec((nsamples, nfeatures), records_arr).unwrap();
        let targets = Array2::from_shape_vec((nsamples, ntargets), targets_arr).unwrap();

        let targets = CountedTargets {
            targets,
            labels: map,
        };

        DatasetBase {
            records,
            weights: Array1::from(weights),
            targets,
            feature_names: self.feature_names.clone(),
        }
    }
}
