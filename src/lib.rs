#[macro_use]
extern crate ndarray;
extern crate num_traits as libnum;

pub mod preprocessing;


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
