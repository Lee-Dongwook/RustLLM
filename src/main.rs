mod metal;
mod ops;

use metal::{
    MetalBuffer,
    MetalContext
};

use crate::ops::vector_add;

fn main() {
    let context = MetalContext::new();

    println!("GPU: {}", context.device.name());

    let a = MetalBuffer::from_slice(&context, &[1.0,2.0,3.0,4.0]);
    let b = MetalBuffer::from_slice(&context, &[10.0, 20.0, 30.0, 40.0]);

    let result = vector_add(&context, &a, &b);

    println!("A      = {:?}", a.as_slice());
    println!("B      = {:?}", b.as_slice());
    println!("Result = {:?}", result.as_slice());
}
