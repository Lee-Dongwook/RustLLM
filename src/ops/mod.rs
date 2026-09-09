mod add;
mod matmul;
mod contiguous;

pub use add::vector_add;
pub use matmul::{
    matrix_multiply_naive,
    matrix_multiply_tiled_8,
    matrix_multiply_tiled_16,
    matrix_multiply_tiled_32,
};
pub use contiguous::materialize_contiguous_f32;

