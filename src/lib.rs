#[macro_use]
extern crate ndarray;

use ndarray::prelude::*;

pub mod preprocessing;


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
