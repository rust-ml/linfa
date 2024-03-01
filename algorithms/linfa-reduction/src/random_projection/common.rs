/// Compute a safe dimension for a projection with precision `eps`,
/// using the Johnson-Lindestrauss Lemma.
///
/// References:
/// - [D. Achlioptas, JCSS](https://www.sciencedirect.com/science/article/pii/S0022000003000254)
/// - [Li et al., SIGKDD'06](https://hastie.su.domains/Papers/Ping/KDD06_rp.pdf)
pub(crate) fn johnson_lindenstrauss_min_dim(n_samples: usize, eps: f64) -> usize {
    let log_samples = (n_samples as f64).ln();
    let value = 4. * log_samples / (eps.powi(2) / 2. - eps.powi(3) / 3.);
    value as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    /// Test against values computed by the scikit-learn implementation
    /// of `johnson_lindenstrauss_min_dim`.
    fn test_johnson_lindenstrauss() {
        assert_eq!(johnson_lindenstrauss_min_dim(100, 0.05), 15244);
        assert_eq!(johnson_lindenstrauss_min_dim(100, 0.1), 3947);
        assert_eq!(johnson_lindenstrauss_min_dim(100, 0.2), 1062);
        assert_eq!(johnson_lindenstrauss_min_dim(100, 0.5), 221);
        assert_eq!(johnson_lindenstrauss_min_dim(1000, 0.05), 22867);
        assert_eq!(johnson_lindenstrauss_min_dim(1000, 0.1), 5920);
        assert_eq!(johnson_lindenstrauss_min_dim(1000, 0.2), 1594);
        assert_eq!(johnson_lindenstrauss_min_dim(1000, 0.5), 331);
        assert_eq!(johnson_lindenstrauss_min_dim(5000, 0.05), 28194);
        assert_eq!(johnson_lindenstrauss_min_dim(5000, 0.1), 7300);
        assert_eq!(johnson_lindenstrauss_min_dim(5000, 0.2), 1965);
        assert_eq!(johnson_lindenstrauss_min_dim(5000, 0.5), 408);
        assert_eq!(johnson_lindenstrauss_min_dim(10000, 0.05), 30489);
        assert_eq!(johnson_lindenstrauss_min_dim(10000, 0.1), 7894);
        assert_eq!(johnson_lindenstrauss_min_dim(10000, 0.2), 2125);
        assert_eq!(johnson_lindenstrauss_min_dim(10000, 0.5), 442);
    }
}
