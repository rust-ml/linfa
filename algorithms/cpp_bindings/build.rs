fn main() {
  // Link OpenMP library
  println!("cargo:rustc-link-arg=-fopenmp");
  /*https://stackoverflow.com/a/68824012/23471793 */
  
  cxx_build::bridge("src/main.rs")
      .file("src/LogisticRegression.cc")
      .flag("-fopenmp")
      .std("c++11")
      .opt_level(3)
      .compile("cxx-demo");

  println!("cargo:rerun-if-changed=src/main.rs");
  println!("cargo:rerun-if-changed=src/LogisticRegression.cc");
  println!("cargo:rerun-if-changed=include/LogisticRegression.h");
}
