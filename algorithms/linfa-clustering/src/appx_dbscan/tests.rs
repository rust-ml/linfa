use crate::{AppxDbscan, AppxDbscanParams, AppxDbscanParamsError, AppxDbscanValidParams, Dbscan};
use linfa::traits::Transformer;
use linfa::ParamGuard;
use linfa_datasets::generate;
use linfa_nn::distance::L2Dist;
use ndarray::{arr1, arr2, concatenate, s, Array1, Array2};
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use rand_xoshiro::Xoshiro256Plus;
use std::collections::HashMap;

#[test]
fn autotraits() {
    fn has_autotraits<T: Send + Sync + Sized + Unpin>() {}
    has_autotraits::<AppxDbscan>();
    has_autotraits::<Dbscan>();
    has_autotraits::<AppxDbscanValidParams<f64, L2Dist>>();
    has_autotraits::<AppxDbscanParams<f64, L2Dist>>();
}

#[test]
fn appx_dbscan_parity() {
    let mut rng = Xoshiro256Plus::seed_from_u64(40);
    let min_points = 4;
    let tolerance = 0.8;
    let centroids = arr2(&[
        [-99.9, -88.3, 78.9],
        [-69.3, 90.1, -87.3],
        [20., 43.2, 10.2],
        [-1.3, 56.0, 98.9],
    ]);

    // Points too far from any centroid to be part of a cluster
    let outliers = arr2(&[[40.0, 55.5, 78.0], [-33.3, -1., 0.3], [-87.1, 0., 33.3]]);
    // Each cluster of 100 points is situated within a 2x2x2 cube. On average the points are 0.08
    // units apart, so they should all be in the same cluster
    let clusters =
        generate::blobs_with_distribution(100, &centroids, Uniform::new(-1., 1.), &mut rng);
    let dataset = concatenate![ndarray::Axis(0), clusters, outliers];

    let appx_res = AppxDbscan::params(min_points)
        .tolerance(tolerance)
        .slack(1e-6)
        .transform(&dataset)
        .unwrap();
    let ex_res = Dbscan::params(min_points)
        .tolerance(tolerance)
        .check()
        .unwrap()
        .transform(&dataset);

    // The order of the labels of the clusters in the two algorithms may not be the same
    // but it does not affect the result. We have to create a mapping from the exact labels
    // to the approximate labels.
    let mut ex_appx_correspondence: HashMap<i64, i64> = HashMap::new();
    // For each point in the dataset get the exact and approximate label
    for (i, (ex_label, appx_label)) in ex_res.iter().zip(appx_res.iter()).enumerate() {
        println!("{:?} = {:?} {}", ex_label, appx_label, dataset.row(i));
        // Get the value of the exact and approximate labels, defaulting to -1 if the
        // point is not in any cluster
        let ex_value = match ex_label {
            Some(value) => *value as i64,
            None => -1,
        };
        let appx_value = match appx_label {
            Some(value) => *value as i64,
            None => -1,
        };
        // assert that every exact noise point is also an approximated noise point
        if ex_value == -1 {
            assert!(appx_value == -1);
        }
        // assert that the approximated label is the one associated to the exact label
        // of the current points
        let expected_appx_val = ex_appx_correspondence.entry(ex_value).or_insert(appx_value);
        assert_eq!(*expected_appx_val, appx_value);
    }
}

#[test]
fn non_cluster_points() {
    let mut data: Array2<f64> = Array2::zeros((5, 2));
    data.row_mut(0).assign(&arr1(&[10.0, 10.0]));

    let labels = AppxDbscan::params(4).check().unwrap().transform(&data);

    let expected = arr1(&[None, Some(0), Some(0), Some(0), Some(0)]);
    assert_eq!(labels, expected);
}

#[test]
fn test_border() {
    let data: Array2<f64> = arr2(&[
        // Outlier
        [0.0, 2.0],
        // Core point
        [0.0, 0.0],
        // Border points
        [0.0, 1.0],
        [0.0, -1.0],
        [-1.0, 0.0],
        [1.0, 0.0],
    ]);

    // Run the approximate dbscan with tolerance of 1.1, 5 min points for density and
    // a negligible slack
    let labels = AppxDbscan::params(5)
        .tolerance(1.1)
        .slack(1e-5)
        .transform(&data)
        .unwrap();

    assert_eq!(labels[0], None);
    for id in labels.slice(s![1..]).iter() {
        assert_eq!(id, &Some(0));
    }
}

#[test]
fn test_outliers() {
    let mut data: Array2<f64> = Array2::zeros((100, 2));
    // 50 equally spaced values between 0 and 0.8
    let linspace_center = Array1::linspace(0.0, 0.8, 50);
    // Let the first 50 points in data be points with the same value in
    //both dimensions and such values taken from the array above
    data.column_mut(0)
        .slice_mut(s![0..50])
        .assign(&linspace_center);
    data.column_mut(1)
        .slice_mut(s![0..50])
        .assign(&linspace_center);
    // All points have values between 0 and 0.8 in each dimension.

    // Let's now create points with values between -1000 and -5
    //and others with values between 5 and 1000

    // 25 equally spaced values between 5 and 1000
    let linspace_out = Array1::linspace(5., 1000., 25);
    // 25 equally spaced values between -1000 and -5
    let linspace_out_neg = Array1::linspace(-1000., -5., 25);
    // Let the other 50 points in data be points with the same value in
    //both dimensions and such values taken from the arrays above
    data.column_mut(0)
        .slice_mut(s![50..75])
        .assign(&linspace_out);
    data.column_mut(1)
        .slice_mut(s![50..75])
        .assign(&linspace_out);
    data.column_mut(0)
        .slice_mut(s![75..100])
        .assign(&linspace_out_neg);
    data.column_mut(1)
        .slice_mut(s![75..100])
        .assign(&linspace_out_neg);

    // Now let's run the approximate dbscan with tolerance of 1, 2 min points for density and
    // a negligible slack
    let labels = AppxDbscan::params(2)
        .tolerance(1.0)
        .slack(1e-4)
        .transform(&data)
        .unwrap();
    // we should find that the first 50 points are all in the same cluster (cluster 0)
    // and that the other points are so far away from one another that they are all noise points
    for i in 0..50 {
        assert!(labels[i].is_some());
        assert_eq!(labels[i].unwrap(), 0);
        assert!(labels[i + 50].is_none());
    }
}

#[test]
// Identical to the one in the Exact DBSCAN
fn nested_clusters() {
    // Create a circuit of points and then a cluster in the centre
    // and ensure they are identified as two separate clusters
    let mut data: Array2<f64> = Array2::zeros((50, 2));
    let rising = Array1::linspace(0.0, 8.0, 10);
    data.column_mut(0).slice_mut(s![0..10]).assign(&rising);
    data.column_mut(0).slice_mut(s![10..20]).assign(&rising);
    data.column_mut(1).slice_mut(s![20..30]).assign(&rising);
    data.column_mut(1).slice_mut(s![30..40]).assign(&rising);

    data.column_mut(1).slice_mut(s![0..10]).fill(0.0);
    data.column_mut(1).slice_mut(s![10..20]).fill(8.0);
    data.column_mut(0).slice_mut(s![20..30]).fill(0.0);
    data.column_mut(0).slice_mut(s![30..40]).fill(8.0);

    data.column_mut(0).slice_mut(s![40..]).fill(5.0);
    data.column_mut(1).slice_mut(s![40..]).fill(5.0);

    let labels = AppxDbscan::params(2)
        .tolerance(1.0)
        .slack(1e-4)
        .transform(&data)
        .unwrap();

    assert!(labels.slice(s![..40]).iter().all(|x| x == &Some(0)));
    assert!(labels.slice(s![40..]).iter().all(|x| x == &Some(1)));
}

/* ONLY HERE TO CHECK AGAINST PREVIOUS IMPLEMENTATION OF APPX_DBSCAN
#[test]
fn test_exp(){
    let mock_params = params_from_file(&"./e_shop.txt");
    let mock_points = read_points_from_file::<&str,6>(&"./e_shop.txt",&mock_params);
    let points_vec : Vec<f64> = mock_points.iter().map(|x| x.to_vec()).flatten().collect();
    let dataset = Array2::from_shape_vec((mock_params.cardinality, 6), points_vec).unwrap();
    let params = AppxDbscanParams::new(15).tolerance(1.5).slack(0.0001).build();
    let appx_res = AppxDbscan::predict(&params, &dataset);
    let appx_clusters: i64 = appx_res
        .iter()
        .filter(|x| x.is_some())
        .map(|x| x.unwrap() as i64)
        .max()
        .unwrap_or(-1)
        + 1;
    let appx_noise = appx_res.iter().filter(|x| x.is_none()).count();
    assert_eq!(appx_clusters,7);
    assert_eq!(appx_noise, 98);
}*/

#[test]
fn tolerance_cannot_be_zero() {
    let res = AppxDbscan::params(2).tolerance(0.0).slack(0.1).check();
    assert!(matches!(res, Err(AppxDbscanParamsError::Tolerance)));
}

#[test]
fn slack_cannot_be_zero() {
    let res = AppxDbscan::params(2).tolerance(0.1).slack(0.0).check();
    assert!(matches!(res, Err(AppxDbscanParamsError::Slack)));
}

#[test]
fn min_points_at_least_2() {
    let res = AppxDbscan::params(1).tolerance(0.1).slack(0.1).check();
    assert!(matches!(res, Err(AppxDbscanParamsError::MinPoints)));
}

#[test]
fn tolerance_should_be_positive() {
    let res = AppxDbscan::params(2).tolerance(-1.0).slack(0.1).check();
    assert!(matches!(res, Err(AppxDbscanParamsError::Tolerance)));
}

#[test]
fn slack_should_be_positive() {
    let res = AppxDbscan::params(2).tolerance(0.1).slack(-1.0).check();
    assert!(matches!(res, Err(AppxDbscanParamsError::Slack)));
}
