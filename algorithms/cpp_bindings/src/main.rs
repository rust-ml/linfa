#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("cpp_bindings/include/LogisticRegression.h");
        type LogisticRegression;
        fn train() -> f64;
    }
}

fn main() {
    ffi::train();
}
