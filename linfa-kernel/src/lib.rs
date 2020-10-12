//! Kernel methods
//!

mod sparse;

use ndarray::prelude::*;
use ndarray::{linalg::Dot, NdFloat};
use sprs::CsMat;

use linfa::{Float, dataset::Data};

type SimFnc<F> = Box<dyn Fn(ArrayView1<F>, ArrayView1<F>) -> F>;

pub enum KernelType {
    Dense,
    Sparse(usize),
}

#[derive(Debug)]
pub enum KernelInner<F: Float> {
    Dense(Array2<F>),
    Sparse(CsMat<F>),
}

pub struct Kernel<'a, F: Float, D: Data<Elem = F>> {
    pub inner: KernelInner<F>,
    pub fnc: SimFnc<F>,
    pub dataset: &'a D,
    pub linear: bool,
}

impl<'a, F: Float, D: Data<Elem = F>> Kernel<'a, F, D> {
    pub fn new<F: Fn(ArrayView1<F>, ArrayView1<F>) -> F + 'static>(
        dataset: &'a Data,
        fnc: F,
        kind: KernelType,
        linear: bool,
    ) -> Kernel<'a, F, D> {
        let inner = match kind {
            KernelType::Dense => KernelInner::Dense(dense_from_fn(dataset, &fnc)),
            KernelType::Sparse(k) => KernelInner::Sparse(sparse_from_fn(dataset, k, &fnc)),
        };

        Kernel {
            inner,
            fnc: Box::new(fnc),
            dataset,
            linear,
        }
    }

    pub fn dot(&self, rhs: &ArrayView2<A>) -> Array2<A> {
        match &self.inner {
            KernelInner::Dense(mat) => mat.dot(rhs),
            KernelInner::Sparse(mat) => mat.dot(rhs),
        }
    }

    pub fn sum(&self) -> Array1<A> {
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

    pub fn to_upper_triangle(&self) -> Vec<A> {
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

    pub fn diagonal(&self) -> Array1<A> {
        match &self.inner {
            KernelInner::Dense(mat) => mat.diag().to_owned(),
            KernelInner::Sparse(_) => self
                .dataset
                .outer_iter()
                .map(|x| (self.fnc)(x.view(), x.view()))
                .collect(),
        }
    }

    pub fn column(&self, i: usize) -> Vec<A> {
        match &self.inner {
            KernelInner::Dense(mat) => mat.column(i).to_vec(),
            KernelInner::Sparse(mat) => (0..self.size())
                .map(|j| *mat.get(j, i).unwrap_or(&A::neg_zero()))
                .collect::<Vec<_>>(),
        }
    }

    pub fn weighted_sum(&self, weights: &[A], sample: ArrayView1<A>) -> A {
        self.dataset
            .outer_iter()
            .zip(weights.iter())
            .map(|(x, a)| (*self.fnc)(x, sample) * *a)
            .sum()
    }

    pub fn is_linear(&self) -> bool {
        self.linear
    }

    pub fn linear(dataset: &'a ArrayBase<D, Ix2>) -> Kernel<A, D> {
        let fnc = |a: ArrayView1<A>, b: ArrayView1<A>| a.dot(&b);

        Kernel::new(dataset, fnc, KernelType::Dense, true)
    }

    pub fn linear_sparse(dataset: &'a ArrayBase<D, Ix2>, nneigh: usize) -> Kernel<A, D> {
        let fnc = |a: ArrayView1<A>, b: ArrayView1<A>| a.dot(&b);

        Kernel::new(dataset, fnc, KernelType::Sparse(nneigh), true)
    }

    pub fn gaussian(dataset: &'a ArrayBase<D, Ix2>, eps: A) -> Kernel<A, D> {
        let fnc = move |a: ArrayView1<A>, b: ArrayView1<A>| {
            let distance = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (*x - *y) * (*x - *y))
                .sum::<A>();

            (-distance / eps).exp()
        };

        Kernel::new(dataset, fnc, KernelType::Dense, false)
    }

    pub fn gaussian_sparse(dataset: &'a ArrayBase<D, Ix2>, eps: A, nneigh: usize) -> Kernel<A, D> {
        let fnc = move |a: ArrayView1<A>, b: ArrayView1<A>| {
            let distance = a
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (*x - *y) * (*x - *y))
                .sum::<A>();

            (-distance / eps).exp()
        };

        Kernel::new(dataset, fnc, KernelType::Sparse(nneigh), false)
    }

    pub fn polynomial(dataset: &'a ArrayBase<D, Ix2>, c: A, d: A) -> Kernel<A, D> {
        let fnc = move |a: ArrayView1<A>, b: ArrayView1<A>| (a.dot(&b) + c).powf(d);

        Kernel::new(dataset, fnc, KernelType::Dense, false)
    }

    pub fn polynomial_sparse(
        dataset: &'a ArrayBase<D, Ix2>,
        c: A,
        d: A,
        nneigh: usize,
    ) -> Kernel<A, D> {
        let fnc = move |a: ArrayView1<A>, b: ArrayView1<A>| (a.dot(&b) + c).powf(d);

        Kernel::new(dataset, fnc, KernelType::Sparse(nneigh), false)
    }
}

fn dense_from_fn<A: NdFloat, D: Data<Elem = A>, T: Fn(ArrayView1<A>, ArrayView1<A>) -> A>(
    dataset: &ArrayBase<D, Ix2>,
    fnc: &T,
) -> Array2<A> {
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

fn sparse_from_fn<
    A: NdFloat + Default + std::iter::Sum,
    D: Data<Elem = A>,
    T: Fn(ArrayView1<A>, ArrayView1<A>) -> A,
>(
    dataset: &ArrayBase<D, Ix2>,
    k: usize,
    fnc: &T,
) -> CsMat<A> {
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
