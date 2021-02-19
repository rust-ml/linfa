//! Kernel methods
//!
pub mod inner;
mod sparse;

pub use inner::{Inner, KernelInner};
use ndarray::prelude::*;
use ndarray::Data;
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};
use sprs::{CsMat, CsMatView};
use std::ops::Mul;

use linfa::{dataset::DatasetBase, dataset::Records, dataset::AsTargets, dataset::FromTargetArray, traits::Transformer, Float};

/// Kernel representation, can be either dense or sparse
#[derive(Clone)]
pub enum KernelType {
    Dense,
    /// A sparse kernel requires to define a number of neighbours
    /// between 1 and the total number of samples in input minus one.
    Sparse(usize),
}

/// A generic kernel
///
///
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct KernelBase<K1: Inner, K2: Inner>
where
    K1::Elem: Float,
    K2::Elem: Float,
{
    #[cfg_attr(
        feature = "serde",
        serde(bound(
            serialize = "KernelInner<K1, K2>: Serialize",
            deserialize = "KernelInner<K1, K2>: Deserialize<'de>"
        ))
    )]
    pub inner: KernelInner<K1, K2>,
    #[cfg_attr(
        feature = "serde",
        serde(bound(
            serialize = "KernelMethod<R::Elem>: Serialize",
            deserialize = "KernelMethod<R::Elem>: Deserialize<'de>"
        ))
    )]
    pub method: KernelMethod<K1::Elem>,
}

pub type Kernel<F> = KernelBase<Array2<F>, CsMat<F>>;
pub type KernelView<'a, F> = KernelBase<ArrayView2<'a, F>, CsMatView<'a, F>>;

impl<F: Float, K1: Inner<Elem = F>, K2: Inner<Elem = F>> KernelBase<K1, K2> {
    /// Wheter the kernel is a linear kernel
    ///
    /// ## Returns
    ///
    /// - `true`: if the kernel is linear
    /// - `false`: otherwise
    pub fn is_linear(&self) -> bool {
        self.method.is_linear()
    }

    /// Generates the default set of parameters for building a kernel.
    /// Use this to initialize a set of parameters to be customized using `KernelParams`'s methods
    ///
    /// ## Example
    ///
    /// ```rust
    ///
    /// use linfa_kernel::Kernel;
    /// use linfa::traits::Transformer;
    /// use ndarray::Array2;
    ///
    /// let data = Array2::from_shape_vec((3,2), vec![1., 2., 3., 4., 5., 6.,]).unwrap();
    ///
    /// // Build a kernel from `data` with the defaul parameters
    /// let params = Kernel::params();
    /// let kernel = params.transform(&data);
    ///
    /// ```
    pub fn params() -> KernelParams<F> {
        KernelParams {
            kind: KernelType::Dense,
            method: KernelMethod::Gaussian(F::from(0.5).unwrap()),
        }
    }

    /// Performs the matrix product between the kernel matrix
    /// and the input
    ///
    /// ## Parameters
    ///
    /// - `rhs`: The matrix on the right-hand side of the multiplication
    ///
    /// ## Returns
    ///
    /// A new matrix containing the matrix product between the kernel
    /// and `rhs`
    ///
    /// ## Panics
    ///
    /// If the shapes of kernel and `rhs` are not compatible for multiplication
    pub fn dot(&self, rhs: &ArrayView2<F>) -> Array2<F> {
        match &self.inner {
            KernelInner::Dense(inn) => inn.dot(rhs),
            KernelInner::Sparse(inn) => inn.dot(rhs),
        }
    }

    /// Sums all elements in the same row of the kernel matrix
    ///
    /// ## Returns
    ///
    /// A new array with the sum of all the elements in each row
    pub fn sum(&self) -> Array1<F> {
        match &self.inner {
            KernelInner::Dense(inn) => inn.sum(),
            KernelInner::Sparse(inn) => inn.sum(),
        }
    }

    /// Gives the size of the side of the square kernel matrix
    pub fn size(&self) -> usize {
        match &self.inner {
            KernelInner::Dense(inn) => inn.size(),
            KernelInner::Sparse(inn) => inn.size(),
        }
    }

    /// Getter for a column of the kernel matrix
    ///
    /// ## Params
    ///
    /// - `i`: the index of the column
    ///
    /// ## Returns
    ///
    /// The i-th column of the kernel matrix, stored as a `Vec`
    ///
    /// ## Panics
    ///
    /// If `i` is out of bounds
    pub fn column(&self, i: usize) -> Vec<F> {
        match &self.inner {
            KernelInner::Dense(inn) => inn.column(i),
            KernelInner::Sparse(inn) => inn.column(i),
        }
    }

    /// Getter for the data in the upper triangle of the kernel
    /// matrix
    ///
    /// ## Returns
    ///
    /// A copy of all elements in the upper triangle of the kernel
    /// matrix, stored in a `Vec`
    pub fn to_upper_triangle(&self) -> Vec<F> {
        match &self.inner {
            KernelInner::Dense(inn) => inn.to_upper_triangle(),
            KernelInner::Sparse(inn) => inn.to_upper_triangle(),
        }
    }

    /// Getter for the elements in the diagonal of the kernel matrix
    ///
    /// ## Returns
    ///
    /// A new array containing the copy of all elements in the diagonal fo
    /// the kernel matrix
    pub fn diagonal(&self) -> Array1<F> {
        match &self.inner {
            KernelInner::Dense(inn) => inn.diagonal(),
            KernelInner::Sparse(inn) => inn.diagonal(),
        }
    }
}

impl<'a, F: Float> Kernel<F> {
    pub fn new(dataset: ArrayView2<'a, F>, method: KernelMethod<F>, kind: KernelType) -> Kernel<F> {
        let inner = match kind {
            KernelType::Dense => KernelInner::Dense(dense_from_fn(&dataset, &method)),
            KernelType::Sparse(k) => KernelInner::Sparse(sparse_from_fn(&dataset, k, &method)),
        };

        Kernel { inner, method }
    }

    /// Gives a KernelView which has a view on the original kernel's inner matrix
    pub fn view(&'a self) -> KernelView<'a, F> {
        KernelView {
            inner: match &self.inner {
                KernelInner::Dense(inn) => KernelInner::Dense(inn.view()),
                KernelInner::Sparse(inn) => KernelInner::Sparse(inn.view()),
            },
            method: self.method.clone(),
        }
    }
}

impl<'a, F: Float> KernelView<'a, F> {
    pub fn to_owned(&self) -> Kernel<F> {
        Kernel {
            inner: match &self.inner {
                KernelInner::Dense(inn) => KernelInner::Dense(inn.to_owned()),
                KernelInner::Sparse(inn) => KernelInner::Sparse(inn.to_owned()),
            },
            method: self.method.clone(),
        }
    }
}

impl<F: Float, K1: Inner<Elem = F>, K2: Inner<Elem = F>> Records for KernelBase<K1, K2> {
    type Elem = F;

    fn nsamples(&self) -> usize {
        self.size()
    }

    fn nfeatures(&self) -> usize {
        self.size()
    }
}

/// The inner product definition used by a kernel.
///
/// There are three methods available:
///
/// - Gaussian(eps):  `d(x, x') = exp(-norm(x - x')/eps) `
/// - Linear: `d(x, x') = <x, x'>`
/// - Polynomial(constant, degree):  `d(x, x') = (<x, x'> + costant)^(degree)`
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Debug, Clone)]
pub enum KernelMethod<F> {
    /// Gaussian(eps): exp(-norm(x - x')/eps)
    Gaussian(F),
    /// Euclidean inner product
    Linear,
    /// Polynomial(constant, degree):  ` (<x, x'> + costant)^(degree)`
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

/// Defines the set of prameters needed to build a kernel
pub struct KernelParams<F> {
    /// Wether to construct a dense or sparse kernel
    kind: KernelType,
    /// The inner product used by the kernel
    method: KernelMethod<F>,
}

impl<F: Float> KernelParams<F> {
    /// Setter for `method`. Can be chained with `kind` and `transform`.
    ///
    /// ## Arguments
    ///
    /// - `method`: The inner product that will be used by the kernel
    ///
    /// ## Returns
    ///
    /// The modified set of params
    ///
    /// ## Example
    ///
    /// ```rust
    /// use linfa_kernel::{Kernel, KernelMethod};
    /// use linfa::traits::Transformer;
    /// use ndarray::Array2;
    ///
    /// let data = Array2::from_shape_vec((3,2), vec![1., 2., 3., 4., 5., 6.,]).unwrap();
    ///
    /// // Build a kernel from `data` with the defaul parameters
    /// // and then set the preferred method
    /// let params = Kernel::params().method(KernelMethod::Linear);
    /// let kernel = params.transform(&data);
    /// ```
    pub fn method(mut self, method: KernelMethod<F>) -> KernelParams<F> {
        self.method = method;

        self
    }

    /// Setter for `kind`. Can be chained with `method` and `transform`.
    ///
    /// ## Arguments
    ///
    /// - `kind`: The kind of kernel to build, either dense or sparse
    ///
    /// ## Returns
    ///
    /// The modified set of params
    ///
    /// ## Example
    ///
    /// ```rust
    /// use linfa_kernel::{Kernel, KernelType};
    /// use linfa::traits::Transformer;
    /// use ndarray::Array2;
    ///
    /// let data = Array2::from_shape_vec((3,2), vec![1., 2., 3., 4., 5., 6.,]).unwrap();
    ///
    /// // Build a kernel from `data` with the defaul parameters
    /// // and then set the preferred kind
    /// let params = Kernel::params().kind(KernelType::Dense);
    /// let kernel = params.transform(&data);
    /// ```
    pub fn kind(mut self, kind: KernelType) -> KernelParams<F> {
        self.kind = kind;
        self
    }
}

impl<F: Float> Transformer<&Array2<F>, Kernel<F>> for KernelParams<F> {
    /// Builds a kernel from a view of the input data.
    ///
    /// ## Parameters
    ///
    /// - `x`: view of a matrix of records (#records, #features)
    ///
    /// A kernel build from `x` according to the parameters on which
    /// this method is called
    ///
    /// ## Panics
    ///
    /// If the kernel type is `Sparse` and the number of neighbors specified is
    /// not between 1 and #records-1
    fn transform(&self, x: &Array2<F>) -> Kernel<F> {
        Kernel::new(x.view(), self.method.clone(), self.kind.clone())
    }
}

impl<'a, F: Float> Transformer<ArrayView2<'a, F>, Kernel<F>> for KernelParams<F> {
    /// Builds a kernel from a view of the input data.
    ///
    /// ## Parameters
    ///
    /// - `x`: view of a matrix of records (#records, #features)
    ///
    /// A kernel build from `x` according to the parameters on which
    /// this method is called
    ///
    /// ## Panics
    ///
    /// If the kernel type is `Sparse` and the number of neighbors specified is
    /// not between 1 and #records-1
    fn transform(&self, x: ArrayView2<'a, F>) -> Kernel<F> {
        Kernel::new(x, self.method.clone(), self.kind.clone())
    }
}

impl<'a, F: Float> Transformer<&ArrayView2<'a, F>, Kernel<F>> for KernelParams<F> {
    /// Builds a kernel from a view of the input data.
    ///
    /// ## Parameters
    ///
    /// - `x`: view of a matrix of records (#records, #features)
    ///
    /// A kernel build from `x` according to the parameters on which
    /// this method is called
    ///
    /// ## Panics
    ///
    /// If the kernel type is `Sparse` and the number of neighbors specified is
    /// not between 1 and #records-1
    fn transform(&self, x: &ArrayView2<'a, F>) -> Kernel<F> {
        Kernel::new(*x, self.method.clone(), self.kind.clone())
    }
}

impl<'a, F: Float, T: AsTargets> Transformer<DatasetBase<Array2<F>, T>, DatasetBase<Kernel<F>, T>>
    for KernelParams<F>
{
    /// Builds a new Dataset with the kernel as the records and the same targets as the input one.
    ///
    /// It takes ownership of the original dataset.
    ///
    /// ## Parameters
    ///
    /// - `x`: A dataset with a matrix of records (#records, #features) and any targets
    ///
    /// ## Returns
    ///
    /// A new dataset with:
    ///  - records: a kernel build from `x.records()` according to the parameters on which
    /// this method is called
    ///  - targets: same as `x.targets()`
    ///
    /// ## Panics
    ///
    /// If the kernel type is `Sparse` and the number of neighbors specified is
    /// not between 1 and #records-1
    fn transform(&self, x: DatasetBase<Array2<F>, T>) -> DatasetBase<Kernel<F>, T> {
        let kernel = Kernel::new(x.records.view(), self.method.clone(), self.kind.clone());
        DatasetBase::new(kernel, x.targets)
    }
}

impl<'a, F: Float, L: 'a, T: AsTargets<Elem = L> + FromTargetArray<'a, L>>
    Transformer<&'a DatasetBase<Array2<F>, T>, DatasetBase<Kernel<F>, T::View>>
    for KernelParams<F>
{
    /// Builds a new Dataset with the kernel as the records and the same targets as the input one.
    ///
    /// ## Parameters
    ///
    /// - `x`: A dataset with a matrix of records (#records, #features) and any targets
    ///
    /// ## Returns
    ///
    /// A new dataset with:
    ///  - records: a kernel build from `x.records()` according to the parameters on which
    /// this method is called
    ///  - targets: same as `x.targets()`
    ///
    /// ## Panics
    ///
    /// If the kernel type is `Sparse` and the number of neighbors specified is
    /// not between 1 and #records-1
    fn transform(&self, x: &'a DatasetBase<Array2<F>, T>) -> DatasetBase<Kernel<F>, T::View> {
        let kernel = Kernel::new(x.records.view(), self.method.clone(), self.kind.clone());
        DatasetBase::new(kernel, T::new_targets_view(x.as_multi_targets()))
    }
}

// lifetime 'b allows the kernel to borrow the underlying data
// for a possibly shorter time than 'a, useful in fold_fit
impl<'a, 'b, F: Float, L: 'b, T: AsTargets<Elem = L> + FromTargetArray<'b, L>> 
    Transformer<&'b DatasetBase<ArrayView2<'a, F>, T>, DatasetBase<Kernel<F>, T::View>>
    for KernelParams<F>
{
    /// Builds a new Dataset with the kernel as the records and the same targets as the input one.
    ///
    /// ## Parameters
    ///
    /// - `x`: A dataset with a matrix of records (##records, ##features) and any targets
    ///
    /// ## Returns
    ///
    /// A new dataset with:
    ///  - records: a kernel build from `x.records()` according to the parameters on which
    /// this method is called
    ///  - targets: a slice of `x.targets()`
    ///
    /// ## Panics
    ///
    /// If the kernel type is `Sparse` and the number of neighbors specified is
    /// not between 1 and ##records-1
    fn transform(
        &self,
        x: &'b DatasetBase<ArrayView2<'a, F>, T>,
    ) -> DatasetBase<Kernel<F>, T::View> {
        let kernel = Kernel::new(x.records.view(), self.method.clone(), self.kind.clone());

        DatasetBase::new(kernel, T::new_targets_view(x.as_multi_targets()))
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
    use linfa::Dataset;
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
    fn test_kernel_dot() {
        let input_vec: Vec<f64> = (0..100).map(|v| v as f64 * 0.1).collect();
        let vec_to_multiply: Vec<f64> = (0..100).map(|v| v as f64 * 0.3).collect();
        let input_arr = Array2::from_shape_vec((10, 10), input_vec).unwrap();
        let to_multiply = Array2::from_shape_vec((10, 10), vec_to_multiply).unwrap();

        // dense kernel dot
        let mul_mat = dense_from_fn(&input_arr, &KernelMethod::Linear).dot(&to_multiply);
        let kernel = KernelView::params()
            .kind(KernelType::Dense)
            .method(KernelMethod::Linear)
            .transform(input_arr.view());
        let mul_ker = kernel.dot(&to_multiply.view());
        assert!(matrices_almost_equal(mul_mat.view(), mul_ker.view()));

        // sparse kernel dot
        let mul_mat = sparse_from_fn(&input_arr, 3, &KernelMethod::Linear).mul(&to_multiply.view());
        let kernel = KernelView::params()
            .kind(KernelType::Sparse(3))
            .method(KernelMethod::Linear)
            .transform(input_arr.view());
        let mul_ker = kernel.dot(&to_multiply.view());
        assert!(matrices_almost_equal(mul_mat.view(), mul_ker.view()));
    }

    #[test]
    fn test_kernel_upper_triangle() {
        // symmetric vec, kernel matrix is a "cross" of ones
        let input_vec: Vec<f64> = (0..50).map(|v| v as f64 * 0.1).collect();
        let input_arr_1 = Array2::from_shape_vec((5, 10), input_vec.clone()).unwrap();
        let mut input_arr_2 = Array2::from_shape_vec((5, 10), input_vec).unwrap();
        input_arr_2.invert_axis(Axis(0));
        let input_arr = ndarray::stack(Axis(0), &[input_arr_1.view(), input_arr_2.view()]).unwrap();

        for kind in vec![KernelType::Dense, KernelType::Sparse(1)] {
            let kernel = KernelView::params()
                .kind(kind)
                // Such a value for eps brings to zero the inner product
                // between any two points that are not equal
                .method(KernelMethod::Gaussian(1e-5))
                .transform(input_arr.view());
            let mut kernel_upper_triang = kernel.to_upper_triangle();
            assert_eq!(kernel_upper_triang.len(), 45);
            //so that i can use pop()
            kernel_upper_triang.reverse();
            for i in 0..9 {
                for j in (i + 1)..10 {
                    if j == (9 - i) {
                        assert_eq!(kernel_upper_triang.pop().unwrap() as usize, 1);
                    } else {
                        assert_eq!(kernel_upper_triang.pop().unwrap() as usize, 0);
                    }
                }
            }
            assert!(kernel_upper_triang.is_empty());
        }
    }

    /* #[test]
    fn test_kernel_weighted_sum() {
        let input_vec: Vec<f64> = (0..100).map(|v| v as f64 * 0.1).collect();
        let input_arr = Array2::from_shape_vec((10, 10), input_vec).unwrap();
        let weights = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];
        for kind in vec![KernelType::Dense, KernelType::Sparse(1)] {
            let kernel = KernelView::params()
                .kind(kind)
                // Such a value for eps brings to zero the inner product
                // between any two points that are not equal
                .method(KernelMethod::Gaussian(1e-5))
                .transform(input_arr.view());
            for (sample, w) in input_arr.outer_iter().zip(&weights) {
                // with that kernel, only the input samples have non
                // zero inner product with the samples used to generate the matrix.
                // In particular, they have inner product equal to one only for the
                // column corresponding to themselves
                //let w_sum = kernel.weighted_sum(&weights, sample);
                //assert!(values_almost_equal(&w_sum, w));
            }
        }
    }*/

    #[test]
    fn test_kernel_sum() {
        let input_vec: Vec<f64> = (0..100).map(|v| v as f64 * 0.1).collect();
        let input_arr = Array2::from_shape_vec((10, 10), input_vec).unwrap();

        let method = KernelMethod::Linear;

        // dense kernel sum
        let cols_sum = dense_from_fn(&input_arr, &method).sum_axis(Axis(1));
        let kernel = KernelView::params()
            .kind(KernelType::Dense)
            .method(method.clone())
            .transform(input_arr.view());
        let kers_sum = kernel.sum();
        assert!(arrays_almost_equal(cols_sum.view(), kers_sum.view()));

        // sparse kernel sum
        let cols_sum = sparse_from_fn(&input_arr, 3, &method)
            .to_dense()
            .sum_axis(Axis(1));
        let kernel = KernelView::params()
            .kind(KernelType::Sparse(3))
            .method(method)
            .transform(input_arr.view());
        let kers_sum = kernel.sum();
        assert!(arrays_almost_equal(cols_sum.view(), kers_sum.view()));
    }

    #[test]
    fn test_kernel_diag() {
        let input_vec: Vec<f64> = (0..100).map(|v| v as f64 * 0.1).collect();
        let input_arr = Array2::from_shape_vec((10, 10), input_vec).unwrap();

        let method = KernelMethod::Linear;

        // dense kernel diag
        let input_diagonal = dense_from_fn(&input_arr, &method).diag().into_owned();
        let kernel = KernelView::params()
            .kind(KernelType::Dense)
            .method(method.clone())
            .transform(input_arr.view());
        let kers_diagonal = kernel.diagonal();
        assert!(arrays_almost_equal(
            input_diagonal.view(),
            kers_diagonal.view()
        ));

        // sparse kernel diag
        let input_diagonal: Vec<_> = sparse_from_fn(&input_arr, 3, &method)
            .outer_iterator()
            .enumerate()
            .map(|(i, row)| *row.get(i).unwrap())
            .collect();
        let input_diagonal = Array1::from_shape_vec(10, input_diagonal).unwrap();
        let kernel = KernelView::params()
            .kind(KernelType::Sparse(3))
            .method(method)
            .transform(input_arr.view());
        let kers_diagonal = kernel.diagonal();
        assert!(arrays_almost_equal(
            input_diagonal.view(),
            kers_diagonal.view()
        ));
    }

    // inspired from scikit learn's tests
    #[test]
    fn test_kernel_transform_from_array2() {
        let input_vec: Vec<f64> = (0..100).map(|v| v as f64 * 0.1).collect();
        let input = Array2::from_shape_vec((50, 2), input_vec).unwrap();
        // checks that the transform for Array2 builds the right kernel
        // according to its input params.
        check_kernel_from_array2_type(&input, KernelType::Dense);
        check_kernel_from_array2_type(&input, KernelType::Sparse(3));
        // checks that the transform for ArrayView2 builds the right kernel
        // according to its input params.
        check_kernel_from_array_view_2_type(input.view(), KernelType::Dense);
        check_kernel_from_array_view_2_type(input.view(), KernelType::Sparse(3));
    }

    // inspired from scikit learn's tests
    #[test]
    fn test_kernel_transform_from_dataset() {
        let input_vec: Vec<f64> = (0..100).map(|v| v as f64 * 0.1).collect();
        let input_arr = Array2::from_shape_vec((50, 2), input_vec).unwrap();
        let input = Dataset::from(input_arr);
        // checks that the transform for dataset builds the right kernel
        // according to its input params.
        check_kernel_from_dataset_type(&input, KernelType::Dense);
        check_kernel_from_dataset_type(&input, KernelType::Sparse(3));

        // checks that the transform for dataset view builds the right kernel
        // according to its input params.
        check_kernel_from_dataset_view_type(&input.view(), KernelType::Dense);
        check_kernel_from_dataset_view_type(&input.view(), KernelType::Sparse(3));
    }

    fn check_kernel_from_dataset_type<'a, L: 'a, T: AsTargets<Elem = L> + FromTargetArray<'a, L>>(
        input: &'a DatasetBase<Array2<f64>, T>,
        k_type: KernelType,
    ) {
        let methods = vec![
            KernelMethod::Linear,
            KernelMethod::Gaussian(0.1),
            KernelMethod::Polynomial(1., 2.),
        ];
        for method in methods {
            let kernel_ref = Kernel::new(input.records().view(), method.clone(), k_type.clone());
            let kernel_tr = Kernel::params()
                .kind(k_type.clone())
                .method(method.clone())
                .transform(input);
            match (&kernel_ref.inner, &kernel_tr.records().inner) {
                (KernelInner::Dense(m1), KernelInner::Dense(m2)) => {
                    assert!(kernels_almost_equal(m1, m2))
                }
                (KernelInner::Sparse(m1), KernelInner::Sparse(m2)) => {
                    assert!(kernels_almost_equal(m1, m2))
                }
                _ => panic!("Kernel inners must match!"),
            };
        }
    }

    fn check_kernel_from_dataset_view_type<'a, L: 'a, T: AsTargets<Elem = L> + FromTargetArray<'a, L>>(
        input: &'a DatasetBase<ArrayView2<'a, f64>, T>,
        k_type: KernelType,
    ) {
        let methods = vec![
            KernelMethod::Linear,
            KernelMethod::Gaussian(0.1),
            KernelMethod::Polynomial(1., 2.),
        ];
        for method in methods {
            let kernel_ref = Kernel::new(*input.records(), method.clone(), k_type.clone());
            let kernel_tr = Kernel::params()
                .kind(k_type.clone())
                .method(method.clone())
                .transform(input);
            match (&kernel_ref.inner, &kernel_tr.records().inner) {
                (KernelInner::Dense(m1), KernelInner::Dense(m2)) => {
                    assert!(kernels_almost_equal(m1, m2))
                }
                (KernelInner::Sparse(m1), KernelInner::Sparse(m2)) => {
                    assert!(kernels_almost_equal(m1, m2))
                }
                _ => panic!("Kernel inners must match!"),
            };
        }
    }

    fn check_kernel_from_array2_type(input: &Array2<f64>, k_type: KernelType) {
        let methods = vec![
            KernelMethod::Linear,
            KernelMethod::Gaussian(0.1),
            KernelMethod::Polynomial(1., 2.),
        ];
        for method in methods {
            let kernel_ref = Kernel::new(input.view(), method.clone(), k_type.clone());
            let kernel_tr = Kernel::params()
                .kind(k_type.clone())
                .method(method.clone())
                .transform(input.view());
            match (&kernel_ref.inner, &kernel_tr.inner) {
                (KernelInner::Dense(m1), KernelInner::Dense(m2)) => {
                    assert!(kernels_almost_equal(m1, m2))
                }
                (KernelInner::Sparse(m1), KernelInner::Sparse(m2)) => {
                    assert!(kernels_almost_equal(m1, m2))
                }
                _ => panic!("Kernel inners must match!"),
            };
        }
    }

    fn check_kernel_from_array_view_2_type(input: ArrayView2<f64>, k_type: KernelType) {
        let methods = vec![
            KernelMethod::Linear,
            KernelMethod::Gaussian(0.1),
            KernelMethod::Polynomial(1., 2.),
        ];
        for method in methods {
            let kernel_ref = Kernel::new(input, method.clone(), k_type.clone());
            let kernel_tr = Kernel::params()
                .kind(k_type.clone())
                .method(method.clone())
                .transform(input);
            match (&kernel_ref.inner, &kernel_tr.inner) {
                (KernelInner::Dense(m1), KernelInner::Dense(m2)) => {
                    assert!(kernels_almost_equal(m1, m2))
                }
                (KernelInner::Sparse(m1), KernelInner::Sparse(m2)) => {
                    assert!(kernels_almost_equal(m1, m2))
                }
                _ => panic!("Kernel inners must match!"),
            };
        }
    }

    fn matrices_almost_equal(reference: ArrayView2<f64>, transformed: ArrayView2<f64>) -> bool {
        for (ref_row, tr_row) in reference
            .axis_iter(Axis(0))
            .zip(transformed.axis_iter(Axis(0)))
        {
            if !arrays_almost_equal(ref_row, tr_row) {
                return false;
            }
        }
        true
    }

    fn arrays_almost_equal(reference: ArrayView1<f64>, transformed: ArrayView1<f64>) -> bool {
        for (ref_item, tr_item) in reference.iter().zip(transformed.iter()) {
            if !values_almost_equal(ref_item, tr_item) {
                return false;
            }
        }
        true
    }

    fn kernels_almost_equal<K: Inner<Elem = f64>>(reference: &K, transformed: &K) -> bool {
        for i in 0..reference.size() {
            if !vecs_almost_equal(reference.column(i), transformed.column(i)) {
                return false;
            }
        }
        true
    }

    fn vecs_almost_equal(reference: Vec<f64>, transformed: Vec<f64>) -> bool {
        for (ref_item, tr_item) in reference.iter().zip(transformed.iter()) {
            if !values_almost_equal(ref_item, tr_item) {
                return false;
            }
        }
        true
    }

    fn values_almost_equal(v1: &f64, v2: &f64) -> bool {
        (v1 - v2).abs() <= f64::EPSILON
    }
}
