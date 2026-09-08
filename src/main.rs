use ::metal::MTLSize;

mod metal;

use metal::{
    MetalBuffer,
    MetalContext
};

fn main() {
    let context = MetalContext::new();

    println!("GPU: {}", context.device.name());

    let a = MetalBuffer::from_slice(&context, &[1.0,2.0,3.0,4.0]);
    let b = MetalBuffer::from_slice(&context, &[10.0, 20.0, 30.0, 40.0]);

    assert_eq!(a.len(), b.len(), "두 Buffer의 길이가 다릅니다.");

    let result = MetalBuffer::empty(&context, a.len());

    let shader_source = include_str!("../kernels/vector_add.metal");

    let pipeline = context.create_pipeline(shader_source, "vector_add");

    let command_buffer = context.command_queue.new_command_buffer();

    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline);

    encoder.set_buffer(0, Some(a.raw()), 0);
    encoder.set_buffer(1, Some(b.raw()), 0);
    encoder.set_buffer(2, Some(result.raw()), 0);

    let grid_size = MTLSize::new(a.len() as u64, 1, 1);

    let thread_group_size = MTLSize::new(a.len() as u64, 1,1);

    encoder.dispatch_threads(
        grid_size,
        thread_group_size
    );

    encoder.end_encoding();

    command_buffer.commit();

    command_buffer.wait_until_completed();

    println!("A      = {:?}", a.as_slice());
    println!("B      = {:?}", b.as_slice());
    println!("Result = {:?}", result.as_slice());
}
