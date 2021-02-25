use thiserror::Error;

pub type Result<T> = std::result::Result<T, SvmResult>;

#[derive(Error, Debug)]
pub enum SvmResult {
    #[error(transparent)]
    BaseCrate(#[from] linfa::Error),
}
