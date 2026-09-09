use std::fmt;

#[derive(Debug)]
pub enum TinyError {
    InvalidShape(String),
    ShapeMismatch {
        left: Vec<usize>,
        right: Vec<usize>,
    },
    InvalidDimension(String),

    InvalidTokenId {
        token_id: u32,
        vocab_size: usize,
    },

    UnsupportedDType(String),
    NonContiguousTensor(String),
    PositionOutOfRange {
        start_pos: usize,
        seq_len: usize,
        max_seq_len: usize,
    },

    Sampling(String),

    Tokenizer(String),

    Io(String),

    ModelFormat(String),

    MissingWeight(String),

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

            TinyError::NonContiguousTensor(message) => {
                write!(
                    f,
                    "non-contiguous tensor: {message}"
                )
            }

            TinyError::InvalidTokenId {
                token_id,
                vocab_size,
            } => {
                write!(
                    f,
                    "invalid token id {token_id}: vocabulary size is {vocab_size}"
                )
            }

            TinyError::PositionOutOfRange {
                start_pos,
                seq_len,
                max_seq_len,
            } => {
                write!(
                    f,
                    "position range [{start_pos}, {}) exceeds max sequence length {max_seq_len}",
                    start_pos + seq_len,
                )
            }

            TinyError::Sampling(message) => {
                write!(
                    f,
                    "sampling error: {message}"
                )
            }

            TinyError::Io(message) => {
                write!(
                    f,
                    "I/O error: {message}"
                )
            }

            TinyError::ModelFormat(message) => {
                write!(
                    f,
                    "model format error: {message}"
                )
            }

            TinyError::MissingWeight(name) => {
                write!(
                    f,
                    "missing model weight: {name}"
                )
            }

            TinyError::Tokenizer(message) => {
                write!(
                    f,
                    "tokenizer error: {message}"
                )
            }
        }
    }
}

impl std::error::Error for TinyError {}

pub type Result<T> =
    std::result::Result<T, TinyError>;

impl From<std::io::Error> for TinyError {
    fn from(
        error: std::io::Error,
    ) -> Self {
        TinyError::Io(
            error.to_string(),
        )
    }
}
