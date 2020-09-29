use crate::float::Float;
use ndarray::Array1;
use ndarray::Zip;

use crate::error::{LinearError, Result};

pub struct TweedieDistribution {
    power: f64,
    lower_bound: f64,
    inclusive: bool,
}

impl TweedieDistribution {
    pub fn new(power: f64) -> Result<Self> {
        // Based on the `power` value, the lower bound of `y` is selected
        let dist = match power {
            power if power <= 0. => Self {
                power,
                lower_bound: std::f64::NEG_INFINITY,
                inclusive: false,
            },
            power if power > 0. && power < 1. => {
                return Err(LinearError::InvalidValue(format!(
                    "Power value cannot be between 0 and 1, got: {}",
                    power
                )));
            }
            power if power >= 1. && power < 2. => Self {
                power,
                lower_bound: 0.,
                inclusive: true,
            },
            power if power >= 2. => Self {
                power,
                lower_bound: 0.,
                inclusive: false,
            },
            _ => unreachable!(),
        };

        Ok(dist)
    }

    // Returns `true` if y is in the valid range
    pub fn in_range<A: Float>(&self, y: &Array1<A>) -> bool {
        if self.inclusive {
            return y.iter().all(|&x| x >= A::from(self.lower_bound).unwrap());
        }
        y.iter().all(|&x| x > A::from(self.lower_bound).unwrap())
    }

    fn unit_variance<A: Float>(&self, ypred: &Array1<A>) -> Array1<A> {
        // ypred ^ power
        ypred.mapv(|x| x.powf(A::from(self.power).unwrap()))
    }

    fn unit_deviance<A: Float>(&self, y: &Array1<A>, ypred: &Array1<A>) -> Result<Array1<A>> {
        match self.power {
            power if power < 0. => {
                let mut left = y.mapv(|x| {
                    if x < A::from(0.).unwrap() {
                        return A::from(0.).unwrap();
                    }
                    x
                });
                left.mapv_inplace(|x| {
                    x.powf(A::from(2. - self.power).unwrap())
                        / A::from((1. - self.power) * (2. - self.power)).unwrap()
                });

                let middle = y * &ypred.mapv(|x| {
                    x.powf(A::from(1. - self.power).unwrap()) / A::from(1. - power).unwrap()
                });

                let right = ypred.mapv(|x| {
                    x.powf(A::from(2. - self.power).unwrap()) / A::from(2. - self.power).unwrap()
                });

                Ok((left - middle + right).mapv(|x| A::from(2.).unwrap() * x))
            }
            // Normal distribution
            // (y - ypred)^2
            power if power == 0. => Ok((y - ypred).mapv(|x| x.powi(2))),
            power if power < 1. => Err(LinearError::InvalidValue(format!(
                "Power value cannot be between 0 and 1, got: {}",
                power
            ))),
            // Poisson distribution
            // 2 * (y * log(y / ypred) - y + ypred)
            power if (power - 1.).abs() < 1e-6 => {
                let mut div = y / ypred;
                Zip::from(&mut div).and(y).apply(|y, &x| {
                    if x == A::from(0.).unwrap() {
                        *y = A::from(0.).unwrap();
                    } else {
                        *y = A::from(2.).unwrap() * (x * y.ln());
                    }
                });
                Ok(div - y + ypred)
            }
            // Gamma distribution
            // 2 * (log(ypred / y) + (y / ypred) - 1)
            power if (power - 2.).abs() < 1e-6 => {
                let mut temp = (ypred / y).mapv(|x| x.ln()) + (y / ypred);
                temp.mapv_inplace(|x| x - A::from(1.).unwrap());
                Ok(temp.mapv(|x| A::from(2.).unwrap() * x))
            }
            power => {
                let left = y.mapv(|x| {
                    x.powf(A::from(2. - power).unwrap())
                        / A::from((1. - power) * (2. - power)).unwrap()
                });

                let middle = y * &ypred
                    .mapv(|x| x.powf(A::from(1. - power).unwrap()) / A::from(1. - power).unwrap());

                let right = ypred
                    .mapv(|x| x.powf(A::from(2. - power).unwrap()) / A::from(2. - power).unwrap());

                Ok((left - middle + right).mapv(|x| A::from(2.).unwrap() * x))
            }
        }
    }

    fn unit_deviance_derivative<A: Float>(&self, y: &Array1<A>, ypred: &Array1<A>) -> Array1<A> {
        ((y - ypred) / &self.unit_variance(ypred)).mapv(|x| A::from(-2.).unwrap() * x)
    }

    pub fn deviance<A: Float>(&self, y: &Array1<A>, ypred: &Array1<A>) -> Result<A> {
        Ok(self.unit_deviance(y, ypred)?.sum())
    }

    pub fn deviance_derivative<A: Float>(&self, y: &Array1<A>, ypred: &Array1<A>) -> Array1<A> {
        self.unit_deviance_derivative(y, ypred)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_distribution_error() {
        let tweedie = TweedieDistribution::new(0.2);
        assert!(tweedie.is_err());
    }

    macro_rules! test_bounds {
        ($($name:ident: ($dist:expr, $input:expr, $expected:expr),)*) => {
            $(
                #[test]
                fn $name() {
                    let output = $dist.in_range(&$input);
                    assert_eq!(output, $expected);
                }
            )*
        };
    }

    test_bounds! {
        test_bounds_normal: (TweedieDistribution::new(0.).unwrap(), array![-1., 0., 1.], true),
        test_bounds_poisson1: (TweedieDistribution::new(1.).unwrap(), array![-1., 0., 1.], false),
        test_bounds_poisson2: (TweedieDistribution::new(1.).unwrap(), array![0., 1., 2.], true),
        test_bounds_tweedie1: (TweedieDistribution::new(1.5).unwrap(), array![-1., 0., 1.], false),
        test_bounds_tweedie2: (TweedieDistribution::new(1.5).unwrap(), array![0., 1., 4.], true),
        test_bounds_gamma1: (TweedieDistribution::new(2.).unwrap(), array![-1., 0., 1.], false),
        test_bounds_gamma2: (TweedieDistribution::new(2.).unwrap(), array![0., 1., 2.], false),
        test_bounds_gamma3: (TweedieDistribution::new(2.).unwrap(), array![1., 2., 3.], true),
        test_bounds_inverse_gaussian: (TweedieDistribution::new(3.).unwrap(), array![-1., 0., 1.], false),
        test_bounds_tweedie3: (TweedieDistribution::new(3.5).unwrap(), array![-1., 0., 1.], false),
    }

    macro_rules! test_deviance {
        ($($name:ident: ($dist:expr, $input:expr),)*) => {
            $(
                #[test]
                fn $name() {
                    let output = $dist.deviance(&$input, &$input).unwrap();
                    assert_abs_diff_eq!(output, 0.0, epsilon=1e-9);
                }
            )*
        }
    }

    test_deviance! {
        test_deviance_normal: (TweedieDistribution::new(0.).unwrap(), array![-1.5, -0.1, 0.1, 2.5]),
        test_deviance_poisson: (TweedieDistribution::new(1.).unwrap(), array![0.1, 1.5]),
        test_deviance_gamma: (TweedieDistribution::new(2.).unwrap(), array![0.1, 1.5]),
        test_deviance_inverse_gaussian: (TweedieDistribution::new(3.).unwrap(), array![0.1, 1.5]),
        test_deviance_tweedie1: (TweedieDistribution::new(-2.5).unwrap(), array![0.1, 1.5]),
        test_deviance_tweedie2: (TweedieDistribution::new(-1.).unwrap(), array![0.1, 1.5]),
        test_deviance_tweedie3: (TweedieDistribution::new(1.5).unwrap(), array![0.1, 1.5]),
        test_deviance_tweedie4: (TweedieDistribution::new(2.5).unwrap(), array![0.1, 1.5]),
        test_deviance_tweedie5: (TweedieDistribution::new(-4.).unwrap(), array![0.1, 1.5]),
    }

    macro_rules! test_deviance_derivative {
        ($($name:ident: {dist: $dist:expr, y: $y:expr, ypred: $ypred:expr, expected: $expected:expr,},)*) => {
            $(
                #[test]
                fn $name() {
                    let output = $dist.deviance_derivative(&$y, &$ypred);
                    println!("{:?}", $expected);
                    println!("{:?}", output);
                    assert_abs_diff_eq!(output, $expected, epsilon=1e-6);
                }
            )*
        };
    }

    test_deviance_derivative! {
        test_derivative_normal: {
            dist: TweedieDistribution::new(0.).unwrap(),
            y: array![
                0.94225502, 1.10863089, 0.99620489, 0.9383247, 0.81709632, 1.03933563, 0.83102873,
                1.28521452, 1.35710428, 0.77688304
            ],
            ypred: array![
                1.73398006, 1.6375258, 1.56424946, 1.86392134, 0.88813238, 1.12646493, 0.85124713,
                2.11783437, 2.13526103, 1.64689519
            ],
            expected: array![
                1.58345008, 1.05778984, 1.13608912, 1.85119328, 0.14207212, 0.1742586, 0.04043679,
                1.66523969, 1.5563135, 1.7400243
            ],
        },
        test_derivative_poisson: {
            dist: TweedieDistribution::new(1.).unwrap(),
            y: array![
                0.94225502, 1.10863089, 0.99620489, 0.9383247, 0.81709632, 1.03933563, 0.83102873,
                1.28521452, 1.35710428, 0.77688304
            ],
            ypred: array![
                1.73398006, 1.6375258, 1.56424946, 1.86392134, 0.88813238, 1.12646493, 0.85124713,
                2.11783437, 2.13526103, 1.64689519
            ],
            expected: array![
                0.91318817, 0.64596835, 0.72628385, 0.99317135, 0.15996728, 0.15469509, 0.047503,
                0.78629364, 0.72886335, 1.05654829
            ],
        },
        test_derivative_gamma: {
            dist: TweedieDistribution::new(2.).unwrap(),
            y: array![
                0.94225502, 1.10863089, 0.99620489, 0.9383247, 0.81709632, 1.03933563, 0.83102873,
                1.28521452, 1.35710428, 0.77688304
            ],
            ypred: array![
                1.73398006, 1.6375258, 1.56424946, 1.86392134, 0.88813238, 1.12646493, 0.85124713,
                2.11783437, 2.13526103, 1.64689519
            ],
            expected: array![
                0.52664283, 0.39447827, 0.46430181, 0.53283973, 0.18011648, 0.13732793, 0.05580401,
                0.37127249, 0.34134625, 0.64153949
            ],
        },
        test_derivative_inverse_gaussian: {
            dist: TweedieDistribution::new(3.).unwrap(),
            y: array![
                0.94225502, 1.10863089, 0.99620489, 0.9383247, 0.81709632, 1.03933563, 0.83102873,
                1.28521452, 1.35710428, 0.77688304
            ],
            ypred: array![
                1.73398006, 1.6375258, 1.56424946, 1.86392134, 0.88813238, 1.12646493, 0.85124713,
                2.11783437, 2.13526103, 1.64689519
            ],
            expected: array![
                0.30371908, 0.24089896, 0.29682082, 0.28587029, 0.20280364, 0.12191052, 0.06555559,
                0.17530761, 0.1598616, 0.38954482
            ],
        },
        test_derivative_tweedie1: {
            dist: TweedieDistribution::new(-2.5).unwrap(),
            y: array![
                0.94225502, 1.10863089, 0.99620489, 0.9383247, 0.81709632, 1.03933563, 0.83102873,
                1.28521452, 1.35710428, 0.77688304
            ],
            ypred: array![
                1.73398006, 1.6375258, 1.56424946, 1.86392134, 0.88813238, 1.12646493, 0.85124713,
                2.11783437, 2.13526103, 1.64689519
            ],
            expected: array![
                6.26923606,
                3.62969199,
                3.47678178,
                8.78052969,
                0.10560953,
                0.23468666,
                0.02703435,
                10.86942904,
                10.36870504,
                6.05647896
            ],
        },
        test_derivative_tweedie2: {
            dist: TweedieDistribution::new(-1.).unwrap(),
            y: array![
                0.94225502, 1.10863089, 0.99620489, 0.9383247, 0.81709632, 1.03933563, 0.83102873,
                1.28521452, 1.35710428, 0.77688304
            ],
            ypred: array![
                1.73398006, 1.6375258, 1.56424946, 1.86392134, 0.88813238, 1.12646493, 0.85124713,
                2.11783437, 2.13526103, 1.64689519
            ],
            expected: array![
                2.74567086, 1.73215816, 1.77712679, 3.45047865, 0.12617885, 0.1962962, 0.03442171,
                3.52670184, 3.32313557, 2.86563764
            ],
        },
        test_derivative_tweedie3: {
            dist: TweedieDistribution::new(1.5).unwrap(),
            y: array![
                0.94225502, 1.10863089, 0.99620489, 0.9383247, 0.81709632, 1.03933563, 0.83102873,
                1.28521452, 1.35710428, 0.77688304
            ],
            ypred: array![
                1.73398006, 1.6375258, 1.56424946, 1.86392134, 0.88813238, 1.12646493, 0.85124713,
                2.11783437, 2.13526103, 1.64689519
            ],
            expected: array![
                0.69348684, 0.50479746, 0.58070208, 0.72746214, 0.16974317, 0.14575307, 0.05148648,
                0.54030473, 0.49879331, 0.8232967
            ],
        },
        test_derivative_tweedie4: {
            dist: TweedieDistribution::new(2.5).unwrap(),
            y: array![
                0.94225502, 1.10863089, 0.99620489, 0.9383247, 0.81709632, 1.03933563, 0.83102873,
                1.28521452, 1.35710428, 0.77688304
            ],
            ypred: array![
                1.73398006, 1.6375258, 1.56424946, 1.86392134, 0.88813238, 1.12646493, 0.85124713,
                2.11783437, 2.13526103, 1.64689519
            ],
            expected: array![
                0.39993934, 0.3082684, 0.37123368, 0.39028586, 0.19112372, 0.1293898, 0.06048359,
                0.25512133, 0.23359829, 0.49990837
            ],
        },
        test_derivative_tweedie5: {
            dist: TweedieDistribution::new(-4.).unwrap(),
            y: array![
                0.94225502, 1.10863089, 0.99620489, 0.9383247, 0.81709632, 1.03933563, 0.83102873,
                1.28521452, 1.35710428, 0.77688304
            ],
            ypred: array![
                1.73398006, 1.6375258, 1.56424946, 1.86392134, 0.88813238, 1.12646493, 0.85124713,
                2.11783437, 2.13526103, 1.64689519
            ],
            expected: array![
                1.43146513e+01,
                7.60592435e+00,
                6.80199725e+00,
                2.23440599e+01,
                8.83933634e-02,
                2.80585306e-01,
                2.12324135e-02,
                3.34999932e+01,
                3.23519886e+01,
                1.28002707e+01
            ],
        },
    }
}
