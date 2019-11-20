use linfa_clustering::KMeansHyperParams;

#[test]
#[should_panic]
fn n_clusters_cannot_be_zero() {
    KMeansHyperParams::new(0).build();
}

#[test]
#[should_panic]
fn tolerance_has_to_positive() {
    KMeansHyperParams::new(1)
        .tolerance(-1.)
        .build();
}

#[test]
#[should_panic]
fn tolerance_cannot_be_zero() {
    KMeansHyperParams::new(1)
        .tolerance(0.)
        .build();
}

#[test]
#[should_panic]
fn max_n_iterations_cannot_be_zero() {
    KMeansHyperParams::new(1)
        .tolerance(1.)
        .max_n_iterations(0)
        .build();
}
