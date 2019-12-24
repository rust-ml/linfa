use crate::dbscan::hyperparameters::DBScanHyperParams;
use ndarray::{Array1, ArrayBase, Axis, Data, Ix1, Ix2};

pub fn predict(
    hyperparameters: DBScanHyperParams,
    observations: &ArrayBase<impl Data<Elem = f64> + Sync, Ix2>,
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

            let mut n = find_neighbors(&obs, observations, hyperparameters.tolerance())
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
    candidate: &ArrayBase<impl Data<Elem = f64> + Sync, Ix1>,
    observations: &ArrayBase<impl Data<Elem = f64> + Sync, Ix2>,
    eps: f64,
) -> Vec<usize> {
    let mut res = vec![];
    for (i, obs) in observations.axis_iter(Axis(1)).enumerate() {
        if distance(candidate, &obs) < eps {
            res.push(i);
        }
    }
    res
}

fn distance(
    lhs: &ArrayBase<impl Data<Elem = f64> + Sync, Ix1>,
    rhs: &ArrayBase<impl Data<Elem = f64> + Sync, Ix1>,
) -> f64 {
    let mut acc = 0.0;
    for (l, r) in lhs.iter().zip(rhs.iter()) {
        acc += (l - r).powi(2);
    }
    acc.sqrt()
}
