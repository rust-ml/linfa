use linfa_supervised::LinearRegression;
use linfa_supervised::RidgeRegression;
use ndarray::array;

fn linear_regression() {
    let mut linear_regression = LinearRegression::new(false);
    let x = array![[1.0], [2.0], [3.0], [4.0]];
    let y = array![1.0, 2.0, 3.0, 4.0];
    linear_regression.fit(&x, &y);
    let x_hat = array![[6.0], [7.0]];
    println!("{:#?}", linear_regression.predict(&x_hat));

    let mut linear_regression2 = LinearRegression::new(true);
    let x2 = array![[1.0], [2.0], [3.0], [4.0]];
    let y2 = array![2.0, 3.0, 4.0, 5.0];
    linear_regression2.fit(&x2, &y2);
    let x2_hat = array![[6.0], [7.0]];
    println!("{:#?}", linear_regression2.predict(&x2_hat));
}

fn ridge_regression() {
    let mut ridge_regression = RidgeRegression::new(0.0);
    let x = array![[1.0], [2.0], [3.0], [4.0]];
    let y = array![1.0, 2.0, 3.0, 4.0];
    ridge_regression.fit(&x, &y);
    let x_hat = array![[6.0], [7.0]];
    println!("{:#?}", ridge_regression.predict(&x_hat));

    let mut ridge_regression2 = RidgeRegression::new(1.0);
    let x2 = array![[1.0], [2.0], [3.0], [4.0]];
    let y2 = array![2.0, 3.0, 4.0, 5.0];
    ridge_regression2.fit(&x2, &y2);
    let x2_hat = array![[6.0], [7.0]];
    println!("{:#?}", ridge_regression2.predict(&x2_hat));
}

fn main() {
    linear_regression();
    ridge_regression();
}
