/// Compute a safe dimension for a projection with precision `eps`,
/// using the Johnson-Lindestrauss Lemma.
///
/// References:
/// - [D. Achlioptas, JCSS](https://www.sciencedirect.com/science/article/pii/S0022000003000254)
/// - [Li et al., SIGKDD'06](https://hastie.su.domains/Papers/Ping/KDD06_rp.pdf)
pub(crate) fn johnson_lindenstrauss_min_dim(n_samples: usize, eps: f64) -> usize {
    let log_samples = (n_samples as f64).log2();
    let value = 4. * log_samples * (eps.powi(2) / 2. - eps.powi(3) / 3.);
    value.ceil() as usize
}
