use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use linfa::{dataset::DatasetBase, traits::Fit};
use linfa_ica::fast_ica::{FastIca, GFunc};
use ndarray::{array, concatenate};
use ndarray::{Array, Array2, Axis};
use ndarray_rand::{rand::SeedableRng, rand_distr::Uniform, RandomExt};
use pprof::criterion::{Output, PProfProfiler};
use rand_xoshiro::Xoshiro256Plus;

fn perform_ica(size: usize, gfunc: GFunc) {
    let sources_mixed = create_data(size);

    let ica = FastIca::params().gfunc(gfunc).random_state(10);

    let ica = ica.fit(&DatasetBase::from(sources_mixed.view()));
}

fn create_data(nsamples: usize) -> Array2<f64> {
    // Creating a sine wave signal
    let source1 = Array::linspace(0., 8., nsamples).mapv(|x| (2f64 * x).sin());

    // Creating a sawtooth signal
    let source2 = Array::linspace(0., 8., nsamples).mapv(|x| {
        let tmp = (4f64 * x).sin();
        if tmp > 0. {
            return 1.;
        }
        -1.
    });

    // Column concatenating both the signals
    let mut sources_original = concatenate![
        Axis(1),
        source1.insert_axis(Axis(1)),
        source2.insert_axis(Axis(1))
    ];

    // Adding noise to the signals
    let mut rng = Xoshiro256Plus::seed_from_u64(42);
    sources_original +=
        &Array::random_using((nsamples, 2), Uniform::new(0.0, 1.0), &mut rng).mapv(|x| x * 0.2);

    // Mixing the two signals
    let mixing = array![[1., 1.], [0.5, 2.]];
    let sources_mixed = sources_original.dot(&mixing.t());

    sources_mixed
}

fn bench(c: &mut Criterion) {
    for (gfunc, name) in [
        (GFunc::Cube, "GFunc_Cube"),
        (GFunc::Logcosh(1.0), "GFunc_Logcosh"),
        (GFunc::Exp, "Exp"),
    ] {
        let mut group = c.benchmark_group("Fast ICA");
        let sizes: [usize; 3] = [1_000, 10_000, 100_000];
        for size in sizes {
            let input = (size, gfunc);
            group.bench_with_input(BenchmarkId::new(name, size), &input, |b, (size, gfunc)| {
                b.iter(|| perform_ica(*size, *gfunc));
            });
        }
        group.finish();
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = bench
}
criterion_main!(benches);
