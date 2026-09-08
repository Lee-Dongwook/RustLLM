mod benchmark;
mod metal;
mod ops;
mod tensor;
mod error;

use benchmark::run_matmul_benchmarks;
use metal::MetalContext;

use tensor::{
    DType,
    Shape,
};

fn main() -> error::Result<()> {
    let shape =
        Shape::new(&[
            2,
            3,
            4,
        ])?;

    println!(
        "shape = {:?}",
        shape.dims(),
    );

    println!(
        "rank = {}",
        shape.rank(),
    );

    println!(
        "numel = {}",
        shape.numel(),
    );

    println!(
        "dtype = {:?}",
        DType::F32,
    );

    println!(
        "bytes per element = {}",
        DType::F32.size_in_bytes(),
    );

    Ok(())
}
