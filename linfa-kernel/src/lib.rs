mod sparse;

use std::iter::FromIterator;
use ndarray::prelude::*;
use ndarray::{NdFloat, linalg::Dot};
use sprs::CsMat;

type SimFnc<A> = Box<dyn Fn(ArrayView1<A>, ArrayView1<A>) -> A>;

pub enum KernelType {
    Dense,
    Sparse(usize)
}

pub enum KernelInner<A: NdFloat> {
    Dense(Array2<A>),
    Sparse(CsMat<A>)
}

pub struct Kernel<A: NdFloat> {
    inner: KernelInner<A>,
    fnc: SimFnc<A>,
    pub dataset: Array2<A>
}

impl<A: NdFloat + Default + std::iter::Sum> Kernel<A> {
    pub fn new<F: Fn(ArrayView1<A>, ArrayView1<A>) -> A + 'static>(dataset: Array2<A>, fnc: F, kind: KernelType) -> Kernel<A> {
        let inner = match kind {
            KernelType::Dense => KernelInner::Dense(dense_from_fn(&dataset, &fnc)),
            KernelType::Sparse(k) => KernelInner::Sparse(sparse_from_fn(&dataset, k, &fnc))
        };

        Kernel {
            inner,
            fnc: Box::new(fnc),
            dataset
        }
    }

    pub fn from_dense(kernel: Array2<A>) -> Kernel<A> {
        Kernel {
            inner: KernelInner::Dense(kernel),
            fnc: Box::new(|_, _| { A::zero() }),
            dataset: array![[A::zero()]]
        }
    }

    pub fn mul_similarity(&self, rhs: &ArrayView2<A>) -> Array2<A> {
        match &self.inner {
            KernelInner::Dense(mat) => mat.dot(rhs),
            KernelInner::Sparse(mat) => mat.dot(rhs)
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
            KernelInner::Sparse(mat) => mat.cols()
        }
    }

    pub fn to_upper_triangle(&self) -> Vec<A> {
        match &self.inner {
            KernelInner::Dense(mat) => {
                mat.indexed_iter().filter(|((row, col), _)| col > row)
                    .map(|(_, val)| *val).collect()
            },
            KernelInner::Sparse(mat) => {
                let mat = mat.to_dense();
                mat.indexed_iter().filter(|((row, col), _)| col > row)
                    .map(|(_, val)| *val).collect()
            }
        }
    }

    pub fn diagonal(&self) -> Array1<A> {
        Array::from_iter(
            self.dataset.outer_iter().map(|x| (self.fnc)(x.view(), x.view()))
        )
    }

    pub fn column(&self, i: usize) -> Vec<A> {
        match &self.inner {
            KernelInner::Dense(mat) => mat.column(i).to_vec(),
            KernelInner::Sparse(mat) => {
                (0..self.size())
                    .map(|j| *mat.get(j, i).unwrap_or(&A::neg_zero()))
                    .collect::<Vec<_>>()
            }
        }
    }

    pub fn weighted_sum(&self, weights: &[A], sample: ArrayView1<A>) -> A {
        self.dataset.outer_iter().zip(weights.iter())
            .map(|(x, a)| (*self.fnc)(x, sample) * *a)
            .sum()
    }

    pub fn gaussian(dataset: Array2<A>, eps: A) -> Kernel<A> {
        let fnc = move |a: ArrayView1<A>, b: ArrayView1<A>| {
            let distance = a.iter().zip(b.iter()).map(|(x, y)| (*x-*y)*(*x-*y))
                .sum::<A>();

            (-distance / eps).exp()
        };

        Kernel::new(dataset, fnc, KernelType::Dense)
    }
}

fn dense_from_fn<A: NdFloat, T: Fn(ArrayView1<A>, ArrayView1<A>) -> A>(dataset: &Array2<A  >, fnc: &T) -> Array2<A> {
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

fn sparse_from_fn<A: NdFloat + Default + std::iter::Sum, T: Fn(ArrayView1<A>, ArrayView1<A>) -> A>(dataset: &Array2<  A>, k: usize, fnc: &T) -> CsMat<A> {
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

