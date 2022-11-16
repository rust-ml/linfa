#[cfg(feature = "benchmarks")]
pub mod config {
    use criterion::{measurement::WallTime, BenchmarkGroup, Criterion};
    #[cfg(not(target_os = "windows"))]
    use pprof::criterion::{Output, PProfProfiler};
    use std::time::Duration;

    #[cfg(not(target_os = "windows"))]
    pub fn get_default_profiling_configs() -> Criterion {
        Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)))
    }

    pub fn set_default_benchmark_configs(benchmark: &mut BenchmarkGroup<WallTime>) {
        let sample_size: usize = 200;
        let measurement_time: Duration = Duration::new(10, 0);
        let confidence_level: f64 = 0.97;
        let warm_up_time: Duration = Duration::new(10, 0);
        let noise_threshold: f64 = 0.05;

        benchmark
            .sample_size(sample_size)
            .measurement_time(measurement_time)
            .confidence_level(confidence_level)
            .warm_up_time(warm_up_time)
            .noise_threshold(noise_threshold);
    }
}
