use crate::dbscan::hyperparameters::DBScanHyperParams;
use ndarray::{Array1, ArrayBase, ArrayView, Axis, Data, Ix1, Ix2};
use ndarray_stats::DeviationExt;

pub fn dbscan(
    hyperparameters: &DBScanHyperParams,
    observations: &ArrayBase<impl Data<Elem = f64>, Ix2>,
) -> Array1<Option<usize>> {
    let mut cluster_memberships = Array1::from_elem(observations.dim().1, None);
    let mut current_cluster_id = 0;
    for (i, obs) in observations.axis_iter(Axis(1)).enumerate() {
        if cluster_memberships[i].is_some() {
            continue;
        }
        let (neighbor_count, mut search_queue) = find_neighbors(
            &obs,
            observations,
            hyperparameters.tolerance(),
            &cluster_memberships,
        );
        if neighbor_count < hyperparameters.minimum_points() {
            continue;
        }
        // Now go over the neighbours adding them to the cluster
        cluster_memberships[i] = Some(current_cluster_id);

        while !search_queue.is_empty() {
            let candidate = search_queue.remove(0);

            cluster_memberships[candidate.0] = Some(current_cluster_id);

            let (_, mut neighbors) = find_neighbors(
                &candidate.1,
                observations,
                hyperparameters.tolerance(),
                &cluster_memberships,
            );

            search_queue.append(&mut neighbors);
        }
        current_cluster_id += 1;
    }
    cluster_memberships
}

fn find_neighbors<'a>(
    candidate: &ArrayBase<impl Data<Elem = f64>, Ix1>,
    observations: &'a ArrayBase<impl Data<Elem = f64>, Ix2>,
    eps: f64,
    clusters: &Array1<Option<usize>>,
) -> (usize, Vec<(usize, ArrayView<'a, f64, Ix1>)>) {
    let mut res = vec![];
    let mut count = 0;
    for (i, obs) in observations.axis_iter(Axis(1)).enumerate() {
        if candidate.l2_dist(&obs).unwrap() < eps {
            count += 1;
            if clusters[[i]].is_none() {
                res.push((i, obs));
            }
        }
    }
    (count, res)
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

        let labels = dbscan(&params, &data);

        assert!(labels.slice(s![..40]).iter().all(|x| x == &Some(0)));
        assert!(labels.slice(s![40..]).iter().all(|x| x == &Some(1)));
    }

    #[test]
    fn non_cluster_points() {
        let params = DBScanHyperParams::new(4).build();
        let mut data: Array2<f64> = Array2::zeros((2, 5));
        data.slice_mut(s![.., 0]).assign(&arr1(&[10.0, 10.0]));

        let labels = dbscan(&params, &data);
        let expected = arr1(&[None, Some(0), Some(0), Some(0), Some(0)]);
        assert_eq!(labels, expected);
    }

    #[test]
    fn dataset_too_small() {
        let params = DBScanHyperParams::new(4).build();

        let data: Array2<f64> = Array2::zeros((2, 3));

        let labels = dbscan(&params, &data);
        assert!(labels.iter().all(|x| x.is_none()));
    }
}
