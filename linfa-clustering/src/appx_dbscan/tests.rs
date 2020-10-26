use crate::{generate_blobs, AppxDbscan, AppxDbscanHyperParams, Dbscan};
use linfa::traits::Transformer;
use ndarray::Array2;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use rand_isaac::Isaac64Rng;

#[test]
fn appx_dbscan_test_100() {
    let mut rng = Isaac64Rng::seed_from_u64(40);
    let min_points = 4;
    let n_features = 3;
    let tolerance = 0.8;
    let centroids =
        Array2::random_using((min_points, n_features), Uniform::new(-30., 30.), &mut rng);
    let dataset = generate_blobs(100, &centroids, &mut rng);
    let appx_res = AppxDbscan::params(min_points)
        .tolerance(tolerance)
        .slack(1e-4)
        .build()
        .transform(&dataset);
    let ex_res = Dbscan::params(min_points)
        .tolerance(tolerance)
        .build()
        .transform(&dataset);
    //cluster indexes start at 0
    let appx_clusters: i64 = appx_res
        .iter()
        .filter(|x| x.is_some())
        .map(|x| x.unwrap() as i64)
        .max()
        .unwrap_or(-1)
        + 1;
    let ex_clusters: i64 = ex_res
        .iter()
        .filter(|x| x.is_some())
        .map(|x| x.unwrap() as i64)
        .max()
        .unwrap_or(-1)
        + 1;

    let appx_noise = appx_res.iter().filter(|x| x.is_none()).count();
    let ex_noise = ex_res.iter().filter(|x| x.is_none()).count();
    assert_eq!(appx_clusters, ex_clusters);
    assert_eq!(appx_noise, ex_noise);
}
/* ONLY HERE TO CHECK AGAINST PREVIOUS IMPLEMENTATION OF APPX_DBSCAN
#[test]
fn test_exp(){
    let mock_params = params_from_file(&"./e_shop.txt");
    let mock_points = read_points_from_file::<&str,6>(&"./e_shop.txt",&mock_params);
    let points_vec : Vec<f64> = mock_points.iter().map(|x| x.to_vec()).flatten().collect();
    let dataset = Array2::from_shape_vec((mock_params.cardinality, 6), points_vec).unwrap();
    let params = AppxDbscanHyperParams::new(15).tolerance(1.5).slack(0.0001).build();
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
#[should_panic]
fn tolerance_cannot_be_zero() {
    AppxDbscanHyperParams::new(2)
        .tolerance(0.0)
        .slack(0.1)
        .build();
}

#[test]
#[should_panic]
fn slack_cannot_be_zero() {
    AppxDbscanHyperParams::new(2)
        .tolerance(0.1)
        .slack(0.0)
        .build();
}

#[test]
#[should_panic]
fn min_points_at_least_2() {
    AppxDbscanHyperParams::new(1)
        .tolerance(0.1)
        .slack(0.1)
        .build();
}

#[test]
#[should_panic]
fn tolerance_should_be_positive() {
    AppxDbscanHyperParams::new(2)
        .tolerance(-1.0)
        .slack(0.1)
        .build();
}

#[test]
#[should_panic]
fn slack_should_be_positive() {
    AppxDbscanHyperParams::new(2)
        .tolerance(0.1)
        .slack(-1.0)
        .build();
}
