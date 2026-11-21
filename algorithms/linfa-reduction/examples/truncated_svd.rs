use linfa::prelude::*;
use linfa_reduction::{Pca, TruncatedSvd};

// Truncated SVD on the iris dataset. Unlike PCA, the features are not centered before the
// decomposition, which is what makes this estimator applicable to sparse, non-negative data such
// as term-count or TF-IDF matrices. This example contrasts the two on the same input.
fn main() {
    let dataset = linfa_datasets::iris();

    let embedding = TruncatedSvd::params(2).fit(&dataset).unwrap();

    // The fitted model can hold fewer components than requested, so report what it actually kept
    println!("retained components:      {}", embedding.n_components());
    println!("singular values:          {}", embedding.singular_values());
    println!(
        "explained variance:       {}",
        embedding.explained_variance()
    );
    println!(
        "explained variance ratio: {}",
        embedding.explained_variance_ratio()
    );

    // The iris features have large, non-zero means. PCA subtracts them and therefore finds a
    // different leading direction than truncated SVD, which decomposes the raw matrix.
    let pca = Pca::params(2).fit(&dataset).unwrap();
    let svd_direction = embedding.components().row(0).to_owned();
    let pca_direction = pca.components().row(0).to_owned();

    println!("\nleading direction, truncated SVD: {}", svd_direction);
    println!("leading direction, PCA:           {}", pca_direction);
    // the sign of a singular vector is arbitrary, so compare the directions by absolute cosine
    println!(
        "absolute cosine between them:     {:.4}",
        svd_direction.dot(&pca_direction).abs()
    );

    let reduced = embedding.predict(&dataset);
    println!("\nprojected records: {:?}", reduced.shape());
}
