mod random_forest;
mod adaboost;
pub use random_forest::*;
pub use adaboost::*;

pub use linfa::error::Result;
pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
