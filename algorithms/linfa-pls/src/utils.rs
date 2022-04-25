use linfa::{
    dataset::{WithLapack, WithoutLapack},
    DatasetBase, Float,
};
use ndarray::{s, Array1, Array2, ArrayBase, ArrayView2, Axis, Data, DataMut, Ix1, Ix2, Zip};
#[cfg(feature = "blas")]
use ndarray_linalg::svd::*;
#[cfg(not(feature = "blas"))]
use ndarray_linalg_rs::svd::*;
use ndarray_stats::QuantileExt;

pub fn outer<F: Float>(
    a: &ArrayBase<impl Data<Elem = F>, Ix1>,
    b: &ArrayBase<impl Data<Elem = F>, Ix1>,
) -> Array2<F> {
    let mut outer = Array2::zeros((a.len(), b.len()));
    Zip::from(outer.rows_mut()).and(a).for_each(|mut out, ai| {
        out.assign(&b.mapv(|v| *ai * v));
    });
    outer
}

/// Calculates the pseudo inverse of a matrix
pub fn pinv2<F: Float>(x: ArrayView2<F>, cond: Option<F>) -> Array2<F> {
    let x = x.with_lapack();
    #[cfg(feature = "blas")]
    let (opt_u, s, opt_vh) = x.svd(true, true).unwrap();
    #[cfg(not(feature = "blas"))]
    let (opt_u, s, opt_vh) = x.svd(true, true).unwrap().sort_svd_desc();
    let u = opt_u.unwrap();
    let vh = opt_vh.unwrap();

    let cond = cond
        .unwrap_or(F::cast(*s.max().unwrap()) * F::cast(x.nrows().max(x.ncols())) * F::epsilon());

    let rank = s.fold(0, |mut acc, v| {
        if F::cast(*v) > cond {
            acc += 1
        };
        acc
    });

    let mut ucut = u.slice_move(s![.., ..rank]);
    ucut /= &s.slice(s![..rank]).mapv(F::Lapack::cast);

    vh.slice(s![..rank, ..]).t().dot(&ucut.t()).without_lapack()
}

#[allow(clippy::type_complexity)]
pub fn center_scale_dataset<F: Float, D: Data<Elem = F>>(
    dataset: &DatasetBase<ArrayBase<D, Ix2>, ArrayBase<D, Ix2>>,
    scale: bool,
) -> (
    Array2<F>,
    Array2<F>,
    Array1<F>,
    Array1<F>,
    Array1<F>,
    Array1<F>,
) {
    let (xnorm, x_mean, x_std) = center_scale(dataset.records(), scale);
    let (ynorm, y_mean, y_std) = center_scale(dataset.targets(), scale);
    (xnorm, ynorm, x_mean, y_mean, x_std, y_std)
}

fn center_scale<F: Float>(
    x: &ArrayBase<impl Data<Elem = F>, Ix2>,
    scale: bool,
) -> (Array2<F>, Array1<F>, Array1<F>) {
    let x_mean = x.mean_axis(Axis(0)).unwrap();
    let (xnorm, x_std) = if scale {
        let mut x_std = x.std_axis(Axis(0), F::one());
        x_std.mapv_inplace(|v| if v == F::zero() { F::one() } else { v });
        ((x - &x_mean) / &x_std, x_std)
    } else {
        ((x - &x_mean), Array1::ones(x.ncols()))
    };

    (xnorm, x_mean, x_std)
}

pub fn svd_flip_1d<F: Float>(
    x_weights: &mut ArrayBase<impl DataMut<Elem = F>, Ix1>,
    y_weights: &mut ArrayBase<impl DataMut<Elem = F>, Ix1>,
) {
    let biggest_abs_val_idx = x_weights.mapv(|v| v.abs()).argmax().unwrap();
    let sign: F = x_weights[biggest_abs_val_idx].signum();
    x_weights.map_inplace(|v| *v *= sign);
    y_weights.map_inplace(|v| *v *= sign);
}

pub fn svd_flip<F: Float>(
    u: ArrayBase<impl Data<Elem = F>, Ix2>,
    v: ArrayBase<impl Data<Elem = F>, Ix2>,
) -> (Array2<F>, Array2<F>) {
    // columns of u, rows of v
    let abs_u = u.mapv(|v| v.abs());
    let max_abs_val_indices = abs_u.map_axis(Axis(0), |col| col.argmax().unwrap());
    let mut signs = Array1::<F>::zeros(u.ncols());
    let range: Vec<usize> = (0..u.ncols()).collect();
    Zip::from(&mut signs)
        .and(&max_abs_val_indices)
        .and(&range)
        .for_each(|s, &i, &j| *s = u[[i, j]].signum());
    (&u * &signs, &v * &signs.insert_axis(Axis(1)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_outer() {
        let a = array![1., 2., 3.];
        let b = array![2., 3.];
        let expected = array![[2., 3.], [4., 6.], [6., 9.]];
        assert_abs_diff_eq!(expected, outer(&a, &b));
    }

    #[test]
    fn test_pinv2() {
        let a = array![[1., 2., 3.], [4., 5., 6.], [7., 8., 10.]];
        let a_pinv2 = pinv2(a.view(), None);
        assert_abs_diff_eq!(a.dot(&a_pinv2), Array2::eye(3), epsilon = 1e-6)
    }
}
