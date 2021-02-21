use std::collections::HashMap;

use super::{
    AsProbabilities, AsTargets, AsTargetsMut, CountedTargets, DatasetBase, Float, FromTargetArray,
    Label, Labels, Pr, Records,
};
use ndarray::{
    stack, Array1, Array2, ArrayBase, ArrayView2, ArrayViewMut2, Axis, CowArray, Data, DataMut,
    Dimension, Ix1, Ix2, Ix3, OwnedRepr, ViewRepr,
};

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

    fn as_multi_targets_mut(&mut self) -> ArrayViewMut2<'_, Self::Elem> {
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

    fn as_multi_targets_mut(&mut self) -> ArrayViewMut2<'_, Self::Elem> {
        self.view_mut()
    }
}

impl<T: AsTargets> AsTargets for &T {
    type Elem = T::Elem;

    fn as_multi_targets(&self) -> ArrayView2<Self::Elem> {
        (*self).as_multi_targets()
    }
}

impl<L: Label, T: AsTargets<Elem = L>> AsTargets for CountedTargets<L, T> {
    type Elem = L;

    fn as_multi_targets(&self) -> ArrayView2<L> {
        self.targets.as_multi_targets()
    }
}

impl<L: Label, T: AsTargetsMut<Elem = L>> AsTargetsMut for CountedTargets<L, T> {
    type Elem = L;

    fn as_multi_targets_mut(&mut self) -> ArrayViewMut2<'_, Self::Elem> {
        self.targets.as_multi_targets_mut()
    }
}

impl<'a, L: Label + 'a, T> FromTargetArray<'a, L> for CountedTargets<L, T>
where
    T: AsTargets<Elem = L> + FromTargetArray<'a, L>,
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
        self.gencolumns()
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

/// A NdArray with discrete labels can act as labels
impl<L: Label, R: Records, T: AsTargets<Elem = L>> Labels for DatasetBase<R, CountedTargets<L, T>> {
    type Elem = L;

    fn label_count(&self) -> Vec<HashMap<L, usize>> {
        self.targets.labels.clone()
    }
}

impl<F: Float, L: Copy + Label, D, T> DatasetBase<ArrayBase<D, Ix2>, T>
where
    D: Data<Elem = F>,
    T: AsTargets<Elem = L>,
{
    pub fn with_labels(
        &self,
        labels: &[&[L]],
    ) -> DatasetBase<Array2<F>, CountedTargets<L, Array2<L>>> {
        let targets = self.targets.as_multi_targets();
        let old_weights = self.weights();

        let mut records_arr = Vec::new();
        let mut targets_arr = Vec::new();
        let mut weights = Vec::new();

        let mut map = vec![HashMap::new(); targets.len_of(Axis(1))];

        for (i, (r, t)) in self
            .records()
            .genrows()
            .into_iter()
            .zip(targets.genrows().into_iter())
            .enumerate()
        {
            let any_exists = t.iter().zip(labels.iter()).any(|(a, b)| b.contains(&a));

            if any_exists {
                for (map, val) in map.iter_mut().zip(t.iter()) {
                    *map.entry(val.clone()).or_insert(0) += 1;
                }

                records_arr.push(r.insert_axis(Axis(1)));
                targets_arr.push(t.insert_axis(Axis(1)));

                if let Some(weight) = old_weights {
                    weights.push(weight[i]);
                }
            }
        }

        let records: Array2<F> = stack(Axis(0), &records_arr).unwrap();
        let targets = stack(Axis(0), &targets_arr).unwrap();

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
