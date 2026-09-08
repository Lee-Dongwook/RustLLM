mod add;
mod matmul;

pub use add::vector_add;
pub use matmul::{
    matrix_multiply_naive,
    matrix_multiply_tiled,
};
