#[cfg(any(feature = "openblas-system", feature = "netlib-system"))]
fn main() {
    println!("cargo:rustc-link-lib=lapacke");
    println!("cargo:rustc-link-lib=lapack");
    println!("cargo:rustc-link-lib=cblas");
}

#[cfg(not(any(feature = "openblas-system", feature = "netlib-system")))]
fn main() {}
