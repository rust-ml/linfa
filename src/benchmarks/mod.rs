#[cfg(feature = "benchmarks")]
pub mod config {
    #[cfg(not(target_os = "windows"))]
    use criterion::Criterion;
    #[cfg(not(target_os = "windows"))]
    use pprof::criterion::{Output, PProfProfiler};
    use std::time::Duration;

    #[cfg(not(target_os = "windows"))]
    pub fn get_default_profiling_configs() -> Criterion {
        Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)))
    }

    pub fn get_default_benchmark_configs() -> (usize, Duration, f64, Duration, f64) {
        let sample_size: usize = 200;
        let measurement_time: Duration = Duration::new(10, 0);
        let confidence_level: f64 = 0.97;
        let warm_up_time: Duration = Duration::new(10, 0);
        let noise_threshold: f64 = 0.05;
        (
            sample_size,
            measurement_time,
            confidence_level,
            warm_up_time,
            noise_threshold,
        )
    }
}

