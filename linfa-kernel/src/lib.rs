//! Kernel methods
//!

mod sparse;

use ndarray::prelude::*;
use ndarray::{linalg::Dot, Data};
use sprs::CsMat;

use linfa::{dataset::Dataset, dataset::Records, dataset::Targets, traits::Transformer, Float};

/// Distance function between two data points
type SimFnc<F> = Box<dyn Fn(ArrayView1<F>, ArrayView1<F>) -> F>;

/// Kernel representation, can be either dense or sparse
#[derive(Clone)]
pub enum KernelType {
    Dense,
    Sparse(usize),
}

/// Storage for the kernel matrix
#[derive(Debug)]
pub enum KernelInner<F: Float> {
    Dense(Array2<F>),
    Sparse(CsMat<F>),
}

/// A generic kernel
///
///
pub struct Kernel<R: Records>
where
    R::Elem: Float,
{
    pub inner: KernelInner<R::Elem>,
    pub fnc: SimFnc<R::Elem>,
    pub dataset: R,
    pub linear: bool,
}

impl<'a, F: Float> Kernel<ArrayView2<'a, F>> {
    pub fn new<G: Fn(ArrayView1<F>, ArrayView1<F>) -> F + 'static>(
        dataset: ArrayView2<'a, F>,
        fnc: G,
        kind: KernelType,
        linear: bool,
    ) -> Kernel<ArrayView2<'a, F>> {
        let inner = match kind {
            KernelType::Dense => KernelInner::Dense(dense_from_fn(&dataset, &fnc)),
            KernelType::Sparse(k) => KernelInner::Sparse(sparse_from_fn(&dataset, k, &fnc)),
        };

        Kernel {
            inner,
            fnc: Box::new(fnc),
            dataset,
            linear,
        }
    }

    pub fn dot(&self, rhs: &ArrayView2<F>) -> Array2<F> {
        match &self.inner {
            KernelInner::Dense(mat) => mat.dot(rhs),
            KernelInner::Sparse(mat) => mat.dot(rhs),
        }
    }

    pub fn sum(&self) -> Array1<F> {
        match &self.inner {
            KernelInner::Dense(mat) => mat.sum_axis(Axis(1)),
            KernelInner::Sparse(mat) => {
                let mut sum = Array1::zeros(mat.cols());
                for (val, i) in mat.iter() {
                    let (_, col) = i;
                    sum[col] += *val;
                }

                sum
            }
        }
    }

    pub fn size(&self) -> usize {
        match &self.inner {
            KernelInner::Dense(mat) => mat.ncols(),
            KernelInner::Sparse(mat) => mat.cols(),
        }
    }

    pub fn to_upper_triangle(&self) -> Vec<F> {
        match &self.inner {
            KernelInner::Dense(mat) => mat
                .indexed_iter()
                .filter(|((row, col), _)| col > row)
                .map(|(_, val)| *val)
                .collect(),
            KernelInner::Sparse(mat) => {
                let mat = mat.to_dense();
                mat.indexed_iter()
                    .filter(|((row, col), _)| col > row)
                    .map(|(_, val)| *val)
                    .collect()
            }
        }
    }

    pub fn diagonal(&self) -> Array1<F> {
        match &self.inner {
            KernelInner::Dense(mat) => mat.diag().to_owned(),
            KernelInner::Sparse(_) => self
                .dataset
                .outer_iter()
                .map(|x| (self.fnc)(x.view(), x.view()))
                .collect(),
        }
    }

    pub fn column(&self, i: usize) -> Vec<F> {
        match &self.inner {
            KernelInner::Dense(mat) => mat.column(i).to_vec(),
            KernelInner::Sparse(mat) => (0..self.size())
                .map(|j| *mat.get(j, i).unwrap_or(&F::neg_zero()))
                .collect::<Vec<_>>(),
        }
    }

    pub fn weighted_sum(&self, weights: &[F], sample: ArrayView1<F>) -> F {
        self.dataset
            .outer_iter()
            .zip(weights.iter())
            .map(|(x, a)| (*self.fnc)(x, sample) * *a)
            .sum()
    }

    pub fn is_linear(&self) -> bool {
        self.linear
    }

    pub fn params() -> KernelParams<F> {
        KernelParams {
            kind: KernelType::Dense,
            method: KernelMethod::Gaussian(F::from(0.5).unwrap()),
        }
    }
}

impl<'a, F: Float> Records for Kernel<ArrayView2<'a, F>> {
    type Elem = F;

    fn observations(&self) -> usize {
        self.size()
    }
}

pub enum KernelMethod<F> {
    Gaussian(F),
    Linear,
    Polynomial(F, F),
}

impl<F: Float> KernelMethod<F> {
    pub fn method(&self) -> SimFnc<F> {
        match *self {
            KernelMethod::Gaussian(eps) => Box::new(move |a: ArrayView1<F>, b: ArrayView1<F>| {
                let distance = a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| (*x - *y) * (*x - *y))
                    .sum::<F>();

                (-distance / eps).exp()
            }),
            KernelMethod::Linear => Box::new(move |a: ArrayView1<F>, b: ArrayView1<F>| a.dot(&b)),
            KernelMethod::Polynomial(c, d) => {
                Box::new(move |a: ArrayView1<F>, b: ArrayView1<F>| (a.dot(&b) + c).powf(d))
            }
        }
    }

    pub fn is_linear(&self) -> bool {
        match *self {
            KernelMethod::Linear => true,
            _ => false,
        }
    }
}

pub struct KernelParams<F> {
    kind: KernelType,
    method: KernelMethod<F>,
}

impl<F: Float> KernelParams<F> {
    pub fn method(mut self, method: KernelMethod<F>) -> KernelParams<F> {
        self.method = method;

        self
    }

    pub fn kind(mut self, kind: KernelType) -> KernelParams<F> {
        self.kind = kind;

        self
    }
}

impl<'a, F: Float> Transformer<&'a Array2<F>, Kernel<ArrayView2<'a, F>>> for KernelParams<F> {
    fn transform(&self, x: &'a Array2<F>) -> Kernel<ArrayView2<'a, F>> {
        let fnc = self.method.method();
        let is_linear = self.method.is_linear();

        Kernel::new(x.view(), fnc, self.kind.clone(), is_linear)
    }
}

impl<'a, F: Float> Transformer<ArrayView2<'a, F>, Kernel<ArrayView2<'a, F>>> for KernelParams<F> {
    fn transform(&self, x: ArrayView2<'a, F>) -> Kernel<ArrayView2<'a, F>> {
        let fnc = self.method.method();
        let is_linear = self.method.is_linear();

        Kernel::new(x, fnc, self.kind.clone(), is_linear)
    }
}

impl<'a, F: Float, T: Targets>
    Transformer<&'a Dataset<Array2<F>, T>, Dataset<Kernel<ArrayView2<'a, F>>, &'a T>>
    for KernelParams<F>
{
    fn transform(&self, x: &'a Dataset<Array2<F>, T>) -> Dataset<Kernel<ArrayView2<'a, F>>, &'a T> {
        let fnc = self.method.method();
        let is_linear = self.method.is_linear();

        let kernel = Kernel::new(x.records.view(), fnc, self.kind.clone(), is_linear);

        Dataset::new(kernel, &x.targets)
    }
}

impl<'a, F: Float, T: Targets>
    Transformer<
        &'a Dataset<ArrayView2<'a, F>, T>,
        Dataset<Kernel<ArrayView2<'a, F>>, &'a [T::Elem]>,
    > for KernelParams<F>
{
    fn transform(
        &self,
        x: &'a Dataset<ArrayView2<'a, F>, T>,
    ) -> Dataset<Kernel<ArrayView2<'a, F>>, &'a [T::Elem]> {
        let fnc = self.method.method();
        let is_linear = self.method.is_linear();

        let kernel = Kernel::new(x.records, fnc, self.kind.clone(), is_linear);

        Dataset::new(kernel, x.targets.as_slice())
    }
}

fn dense_from_fn<F: Float, D: Data<Elem = F>, T: Fn(ArrayView1<F>, ArrayView1<F>) -> F>(
    dataset: &ArrayBase<D, Ix2>,
    fnc: &T,
) -> Array2<F> {
    let n_observations = dataset.len_of(Axis(0));
    let mut similarity = Array2::eye(n_observations);

    for i in 0..n_observations {
        for j in 0..n_observations {
            let a = dataset.row(i);
            let b = dataset.row(j);

            similarity[(i, j)] = fnc(a, b);
        }
    }

    similarity
}

fn sparse_from_fn<F: Float, D: Data<Elem = F>, T: Fn(ArrayView1<F>, ArrayView1<F>) -> F>(
    dataset: &ArrayBase<D, Ix2>,
    k: usize,
    fnc: &T,
) -> CsMat<F> {
    let mut data = sparse::adjacency_matrix(dataset, k);

    for (i, mut vec) in data.outer_iterator_mut().enumerate() {
        for (j, val) in vec.iter_mut() {
            let a = dataset.row(i);
            let b = dataset.row(j);

            *val = fnc(a, b);
        }
    }

    data
}
