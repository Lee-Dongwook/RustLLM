use std::ffi::c_void;
use std::mem;

use ::metal::{
    MTLResourceOptions,
    MTLSize,
};

mod metal;

use metal::MetalContext;

fn main() {
    let context = MetalContext::new();

    println!("GPU: {}", context.device.name());

    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];

    assert_eq!(a.len(), b.len());

    let count = a.len();

    let buffer_size = (count * mem::size_of::<f32>()) as u64;

    let buffer_a = context.device.new_buffer_with_data(
        a.as_ptr() as *const c_void, 
        buffer_size, 
        MTLResourceOptions::StorageModeShared
    );

    let buffer_b = context.device.new_buffer_with_data(
        b.as_ptr() as *const c_void, 
        buffer_size, 
        MTLResourceOptions::StorageModeShared
    );

    let buffer_result = context.device.new_buffer(
        buffer_size,
        MTLResourceOptions::StorageModeShared
    );

    let shader_source = include_str!("../kernels/vector_add.metal");

    let pipeline = context.create_pipeline(shader_source, "vector_add");

    let command_buffer = context.command_queue.new_command_buffer();

    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(&buffer_a), 0);
    encoder.set_buffer(1, Some(&buffer_b), 0);
    encoder.set_buffer(2, Some(&buffer_result), 0);

    let grid_size = MTLSize::new(count as u64, 1, 1);

    let thread_group_size = MTLSize::new(count as u64, 1,1);

    encoder.dispatch_threads(
        grid_size,
        thread_group_size
    );

    encoder.end_encoding();

    command_buffer.commit();

    command_buffer.wait_until_completed();

    let result_ptr = buffer_result.contents() as *const f32;

    let result = unsafe {
        std::slice::from_raw_parts(result_ptr, count)
    };

    println!("A      = {:?}", a);
    println!("B      = {:?}", b);
    println!("Result = {:?}", result);
}
