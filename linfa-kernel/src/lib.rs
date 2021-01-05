//! Kernel methods
//!
mod sparse;

use ndarray::prelude::*;
use ndarray::Data;
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};
use sprs::CsMat;
use std::ops::Mul;

use linfa::{dataset::DatasetBase, dataset::Records, dataset::Targets, traits::Transformer, Float};

/// Kernel representation, can be either dense or sparse
#[derive(Clone)]
pub enum KernelType {
    Dense,
    Sparse(usize),
}

/// Storage for the kernel matrix
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug)]
pub enum KernelInner<F: Float> {
    Dense(Array2<F>),
    Sparse(CsMat<F>),
}

/// A generic kernel
///
///
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Kernel<R: Records>
where
    R::Elem: Float,
{
    #[cfg_attr(
        feature = "serde",
        serde(bound(
            serialize = "KernelInner<R::Elem>: Serialize",
            deserialize = "KernelInner<R::Elem>: Deserialize<'de>"
        ))
    )]
    pub inner: KernelInner<R::Elem>,
    #[cfg_attr(
        feature = "serde",
        serde(bound(
            serialize = "KernelMethod<R::Elem>: Serialize",
            deserialize = "KernelMethod<R::Elem>: Deserialize<'de>"
        ))
    )]
    pub method: KernelMethod<R::Elem>,
    pub dataset: R,
    pub linear: bool,
}

impl<'a, F: Float> Kernel<ArrayView2<'a, F>> {
    pub fn new(
        dataset: ArrayView2<'a, F>,
        method: KernelMethod<F>,
        kind: KernelType,
        linear: bool,
    ) -> Kernel<ArrayView2<'a, F>> {
        let inner = match kind {
            KernelType::Dense => KernelInner::Dense(dense_from_fn(&dataset, &method)),
            KernelType::Sparse(k) => KernelInner::Sparse(sparse_from_fn(&dataset, k, &method)),
        };

        Kernel {
            inner,
            method,
            dataset,
            linear,
        }
    }

    pub fn dot(&self, rhs: &ArrayView2<F>) -> Array2<F> {
        match &self.inner {
            KernelInner::Dense(mat) => mat.mul(rhs),
            KernelInner::Sparse(mat) => mat.mul(rhs),
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
                .map(|x| self.method.distance(x.view(), x.view()))
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
            .map(|(x, a)| self.method.distance(x, sample) * *a)
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

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone)]
pub enum KernelMethod<F> {
    Gaussian(F),
    Linear,
    Polynomial(F, F),
}

impl<F: Float> KernelMethod<F> {
    pub fn distance(&self, a: ArrayView1<F>, b: ArrayView1<F>) -> F {
        match *self {
            KernelMethod::Gaussian(eps) => {
                let distance = a
                    .iter()
                    .zip(b.iter())
                    .map(|(x, y)| (*x - *y) * (*x - *y))
                    .sum::<F>();

                (-distance / eps).exp()
            }
            KernelMethod::Linear => a.mul(&b).sum(),
            KernelMethod::Polynomial(c, d) => (a.mul(&b).sum() + c).powf(d),
        }
    }

    pub fn is_linear(&self) -> bool {
        matches!(*self, KernelMethod::Linear)
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
        let is_linear = self.method.is_linear();

        Kernel::new(x.view(), self.method.clone(), self.kind.clone(), is_linear)
    }
}

impl<'a, F: Float> Transformer<ArrayView2<'a, F>, Kernel<ArrayView2<'a, F>>> for KernelParams<F> {
    fn transform(&self, x: ArrayView2<'a, F>) -> Kernel<ArrayView2<'a, F>> {
        let is_linear = self.method.is_linear();

        Kernel::new(x, self.method.clone(), self.kind.clone(), is_linear)
    }
}

impl<'a, F: Float, T: Targets>
    Transformer<&'a DatasetBase<Array2<F>, T>, DatasetBase<Kernel<ArrayView2<'a, F>>, &'a T>>
    for KernelParams<F>
{
    fn transform(
        &self,
        x: &'a DatasetBase<Array2<F>, T>,
    ) -> DatasetBase<Kernel<ArrayView2<'a, F>>, &'a T> {
        let is_linear = self.method.is_linear();

        let kernel = Kernel::new(
            x.records.view(),
            self.method.clone(),
            self.kind.clone(),
            is_linear,
        );

        DatasetBase::new(kernel, &x.targets)
    }
}

impl<'a, F: Float, T: Targets>
    Transformer<
        &'a DatasetBase<ArrayView2<'a, F>, T>,
        DatasetBase<Kernel<ArrayView2<'a, F>>, &'a [T::Elem]>,
    > for KernelParams<F>
{
    fn transform(
        &self,
        x: &'a DatasetBase<ArrayView2<'a, F>, T>,
    ) -> DatasetBase<Kernel<ArrayView2<'a, F>>, &'a [T::Elem]> {
        let is_linear = self.method.is_linear();

        let kernel = Kernel::new(x.records, self.method.clone(), self.kind.clone(), is_linear);

        DatasetBase::new(kernel, x.targets.as_slice())
    }
}

fn dense_from_fn<F: Float, D: Data<Elem = F>>(
    dataset: &ArrayBase<D, Ix2>,
    method: &KernelMethod<F>,
) -> Array2<F> {
    let n_observations = dataset.len_of(Axis(0));
    let mut similarity = Array2::eye(n_observations);

    for i in 0..n_observations {
        for j in 0..n_observations {
            let a = dataset.row(i);
            let b = dataset.row(j);

            similarity[(i, j)] = method.distance(a, b);
        }
    }

    similarity
}

fn sparse_from_fn<F: Float, D: Data<Elem = F>>(
    dataset: &ArrayBase<D, Ix2>,
    k: usize,
    method: &KernelMethod<F>,
) -> CsMat<F> {
    // compute adjacency matrix between points in the input dataset:
    // one point for each row
    let mut data = sparse::adjacency_matrix(dataset, k);

    // iterate through each row of the adjacency matrix where each
    // row is represented by a vec containing a pair (col_index, value)
    // for each non-zero element in the row
    for (i, mut vec) in data.outer_iterator_mut().enumerate() {
        // If there is a non-zero element in row i at index j
        // then it means that points i and j in the input matrix are
        // k-neighbours and their distance is stored in position (i,j)
        for (j, val) in vec.iter_mut() {
            let a = dataset.row(i);
            let b = dataset.row(j);

            *val = method.distance(a, b);
        }
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};
    use std::f64::consts;

    #[test]
    fn sparse_from_fn_test() {
        // pts 0 & 1    pts 2 & 3    pts 4 & 5     pts 6 & 7
        // |0.| |0.1| _ |1.| |1.1| _ |2.| |2.1| _  |3.| |3.1|
        // |0.| |0.1|   |1.| |1.1|   |2.| |2.1|    |3.| |3.1|
        let input_mat = vec![
            0., 0., 0.1, 0.1, 1., 1., 1.1, 1.1, 2., 2., 2.1, 2.1, 3., 3., 3.1, 3.1,
        ];
        let input_arr = Array2::from_shape_vec((8, 2), input_mat).unwrap();
        let adj_mat = sparse_from_fn(&input_arr, 1, &KernelMethod::Linear);
        assert_eq!(adj_mat.nnz(), 16);

        // 2*0^2
        assert_eq!(*adj_mat.get(0, 0).unwrap() as usize, 0);
        // 2*0.1^2
        assert_eq!((*adj_mat.get(1, 1).unwrap() * 100.) as usize, 2);
        // 2*1^2
        assert_eq!(*adj_mat.get(2, 2).unwrap() as usize, 2);
        // 2*1.1^2
        assert_eq!((*adj_mat.get(3, 3).unwrap() * 100.) as usize, 242);
        // 2 * 2^2
        assert_eq!(*adj_mat.get(4, 4).unwrap() as usize, 8);
        // 2 * 2.1^2
        assert_eq!((*adj_mat.get(5, 5).unwrap() * 100.) as usize, 882);
        // 2 * 3^2
        assert_eq!(*adj_mat.get(6, 6).unwrap() as usize, 18);
        // 2 * 3.1^2
        assert_eq!((*adj_mat.get(7, 7).unwrap() * 100.) as usize, 1922);

        // 2*(0 * 0.1)
        assert_eq!(*adj_mat.get(0, 1).unwrap() as usize, 0);
        assert_eq!(*adj_mat.get(1, 0).unwrap() as usize, 0);

        // 2*(1 * 1.1)
        assert_eq!((*adj_mat.get(2, 3).unwrap() * 10.) as usize, 22);
        assert_eq!((*adj_mat.get(3, 2).unwrap() * 10.) as usize, 22);

        // 2*(2 * 2.1)
        assert_eq!((*adj_mat.get(4, 5).unwrap() * 10.) as usize, 84);
        assert_eq!((*adj_mat.get(5, 4).unwrap() * 10.) as usize, 84);

        // 2*(3 * 3.1)
        assert_eq!((*adj_mat.get(6, 7).unwrap() * 10.) as usize, 186);
        assert_eq!((*adj_mat.get(7, 6).unwrap() * 10.) as usize, 186);
    }

    #[test]
    fn dense_from_fn_test() {
        // pts 0 & 1    pts 2 & 3    pts 4 & 5     pts 6 & 7
        // |0.| |0.1| _ |1.| |1.1| _ |2.| |2.1| _  |3.| |3.1|
        // |0.| |0.1|   |1.| |1.1|   |2.| |2.1|    |3.| |3.1|
        let input_mat = vec![
            0., 0., 0.1, 0.1, 1., 1., 1.1, 1.1, 2., 2., 2.1, 2.1, 3., 3., 3.1, 3.1,
        ];
        let input_arr = Array2::from_shape_vec((8, 2), input_mat).unwrap();
        let method: KernelMethod<f64> = KernelMethod::Linear;

        let similarity_matrix = dense_from_fn(&input_arr, &method);

        for i in 0..8 {
            for j in 0..8 {
                assert!(
                    (similarity_matrix.row(i)[j]
                        - method.distance(input_arr.row(i), input_arr.row(j)))
                    .abs()
                        <= f64::EPSILON
                );
            }
        }
    }

    #[test]
    fn gaussian_test() {
        let gauss_1 = KernelMethod::Gaussian(1.);

        let p1 = Array1::from_shape_vec(2, vec![0., 0.]).unwrap();
        let p2 = Array1::from_shape_vec(2, vec![0., 0.]).unwrap();
        let distance = gauss_1.distance(p1.view(), p2.view());
        let expected = 1.;

        assert!(((distance - expected) as f64).abs() <= f64::EPSILON);

        let p1 = Array1::from_shape_vec(2, vec![1., 1.]).unwrap();
        let p2 = Array1::from_shape_vec(2, vec![5., 5.]).unwrap();
        let distance = gauss_1.distance(p1.view(), p2.view());
        let expected = (consts::E).powf(-32.);
        // this fails with e^-31 or e^-33 so f64::EPSILON still holds
        assert!(((distance - expected) as f64).abs() <= f64::EPSILON);

        let gauss_01 = KernelMethod::Gaussian(0.1);

        let p1 = Array1::from_shape_vec(2, vec![0., 0.]).unwrap();
        let p2 = Array1::from_shape_vec(2, vec![0., 0.]).unwrap();
        let distance = gauss_01.distance(p1.view(), p2.view());
        let expected = 1.;

        assert!(((distance - expected) as f64).abs() <= f64::EPSILON);

        let p1 = Array1::from_shape_vec(2, vec![1., 1.]).unwrap();
        let p2 = Array1::from_shape_vec(2, vec![2., 2.]).unwrap();
        let distance = gauss_01.distance(p1.view(), p2.view());
        let expected = (consts::E).powf(-20.);

        assert!(((distance - expected) as f64).abs() <= f64::EPSILON);
    }

    #[test]
    fn poly2_test() {
        let pol_0 = KernelMethod::Polynomial(0., 2.);

        let p1 = Array1::from_shape_vec(2, vec![0., 0.]).unwrap();
        let p2 = Array1::from_shape_vec(2, vec![0., 0.]).unwrap();
        let distance = pol_0.distance(p1.view(), p2.view());
        let expected = 0.;

        assert!(((distance - expected) as f64).abs() <= f64::EPSILON);

        let p1 = Array1::from_shape_vec(2, vec![1., 1.]).unwrap();
        let p2 = Array1::from_shape_vec(2, vec![5., 5.]).unwrap();
        let distance = pol_0.distance(p1.view(), p2.view());
        let expected = 100.;
        assert!(((distance - expected) as f64).abs() <= f64::EPSILON);

        let pol_2 = KernelMethod::Polynomial(2., 2.);

        let p1 = Array1::from_shape_vec(2, vec![0., 0.]).unwrap();
        let p2 = Array1::from_shape_vec(2, vec![0., 0.]).unwrap();
        let distance = pol_2.distance(p1.view(), p2.view());
        let expected = 4.;

        assert!(((distance - expected) as f64).abs() <= f64::EPSILON);

        let p1 = Array1::from_shape_vec(2, vec![1., 1.]).unwrap();
        let p2 = Array1::from_shape_vec(2, vec![2., 2.]).unwrap();
        let distance = pol_2.distance(p1.view(), p2.view());
        let expected = 36.;

        assert!(((distance - expected) as f64).abs() <= f64::EPSILON);
    }

    #[test]
    fn test_is_linear() {
        /*let input = Array2::eye(8);
        let linear_kernel : Kernel<Array2<f64>> = Kernel::new();*/
    }
}
