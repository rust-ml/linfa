use crate::dbscan::hyperparameters::DBScanHyperParams;
use ndarray::{s, Array1, ArrayBase, Axis, Data, Ix1, Ix2};
use ndarray_stats::DeviationExt;

pub fn predict(
    hyperparameters: &DBScanHyperParams,
    observations: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> Array1<Option<usize>> {
    let mut result = Array1::from_elem(observations.dim().1, None);
    let mut latest_id = 0;
    for (i, obs) in observations.axis_iter(Axis(1)).enumerate() {
        if result[i].is_some() {
            continue;
        }
        let n = find_neighbors(&obs, observations, hyperparameters.tolerance());
        if n.len() < hyperparameters.minimum_points() {
            continue;
        }
        // Now go over the neighbours adding them to the cluster
        let mut search_queue = n
            .iter()
            .filter(|x| result[[**x]].is_none())
            .copied()
            .collect::<Vec<_>>();
        while !search_queue.is_empty() {
            let cand = search_queue.remove(0);

            result[cand] = Some(latest_id);

            let mut n = find_neighbors(
                &observations.slice(s![.., cand]),
                observations,
                hyperparameters.tolerance(),
            )
            .iter()
            .filter(|x| result[[**x]].is_none() && !search_queue.contains(x))
            .copied()
            .collect::<Vec<_>>();

            search_queue.append(&mut n);
        }
        latest_id += 1;
    }
    result
}

fn find_neighbors(
    candidate: &ArrayBase<impl Data<Elem = f64>, Ix1>,
    observations: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    eps: f64,
) -> Vec<usize> {
    let mut res = vec![];
    for (i, obs) in observations.axis_iter(Axis(1)).enumerate() {
        if candidate.l2_dist(&obs).unwrap() < eps {
            res.push(i);
        }
    }
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, s, Array2};

    #[test]
    fn nested_clusters() {
        // Create a circuit of points and then a cluster in the centre
        // and ensure they are identified as two separate clusters
        let params = DBScanHyperParams::new(4).tolerance(1.0).build();

        let mut data: Array2<f64> = Array2::zeros((2, 50));
        let rising = Array1::linspace(0.0, 8.0, 10);
        data.slice_mut(s![0, 0..10]).assign(&rising);
        data.slice_mut(s![0, 10..20]).assign(&rising);
        data.slice_mut(s![1, 20..30]).assign(&rising);
        data.slice_mut(s![1, 30..40]).assign(&rising);

        data.slice_mut(s![1, 0..10]).fill(0.0);
        data.slice_mut(s![1, 10..20]).fill(8.0);
        data.slice_mut(s![0, 20..30]).fill(0.0);
        data.slice_mut(s![0, 30..40]).fill(8.0);

        data.slice_mut(s![.., 40..]).fill(5.0);

        let labels = predict(&params, &data);

        assert!(labels.slice(s![..40]).iter().all(|x| x == &Some(0)));
        assert!(labels.slice(s![40..]).iter().all(|x| x == &Some(1)));
    }

    #[test]
    fn non_cluster_points() {
        let params = DBScanHyperParams::new(4).build();

        let data: Array2<f64> = Array2::zeros((2, 3));

        let labels = predict(&params, &data);
        assert!(labels.iter().all(|x| x.is_none()));

        let mut data: Array2<f64> = Array2::zeros((2, 5));
        data.slice_mut(s![.., 0]).assign(&arr1(&[10.0, 10.0]));

        let labels = predict(&params, &data);
        let expected = arr1(&[None, Some(0), Some(0), Some(0), Some(0)]);
        assert_eq!(labels, expected);
    }
}
