use std::fmt;

#[derive(Debug)]
pub enum TinyError {
    InvalidShape(String),
    ShapeMismatch {
        left: Vec<usize>,
        right: Vec<usize>,
    },
    InvalidDimension(String),
    UnsupportedDType(String),
    Metal(String),
}

impl fmt::Display for TinyError {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            TinyError::InvalidShape(message) => {
                write!(
                    f,
                    "invalid shape: {message}"
                )
            }

            TinyError::ShapeMismatch {
                left,
                right,
            } => {
                write!(
                    f,
                    "shape mismatch: left={left:?}, right={right:?}"
                )
            }

            TinyError::InvalidDimension(message) => {
                write!(
                    f,
                    "invalid dimension: {message}"
                )
            }

            TinyError::UnsupportedDType(dtype) => {
                write!(
                    f,
                    "unsupported dtype: {dtype}"
                )
            }

            TinyError::Metal(message) => {
                write!(
                    f,
                    "metal error: {message}"
                )
            }
        }
    }
}

impl std::error::Error for TinyError {}

pub type Result<T> =
    std::result::Result<T, TinyError>;
