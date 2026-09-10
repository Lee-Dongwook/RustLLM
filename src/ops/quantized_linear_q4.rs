use crate::{
    error::{Result, TinyError},
    metal::{MetalBuffer, MetalContext},
};
use ::metal::MTLSize;
use std::{ffi::c_void, mem};

pub fn quantized_linear_q4_f16(
    context: &MetalContext,
    input: &MetalBuffer,
    weight: &MetalBuffer,
    scales: &MetalBuffer,
    m: usize,
    k: usize,
    n: usize,
    block_size: usize,
) -> Result<MetalBuffer> {
    let packed_rows = k.div_ceil(2);
    let scale_rows = k.div_ceil(block_size);
    if input.len() != m * k || weight.len() != packed_rows * n || scales.len() != scale_rows * n {
        return Err(TinyError::InvalidShape(
            "Q4 quantized linear buffer shape mismatch".into(),
        ));
    }
    let output = MetalBuffer::empty_with_element_size(context, m * n, 2);
    let pipeline = context.pipeline(
        include_str!("../../kernels/quantized_linear_q4.metal"),
        "quantized_linear_q4_f16",
    );
    let command = context.command_queue.new_command_buffer();
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline.as_ref());
    encoder.set_buffer(0, Some(input.raw()), 0);
    encoder.set_buffer(1, Some(weight.raw()), 0);
    encoder.set_buffer(2, Some(scales.raw()), 0);
    encoder.set_buffer(3, Some(output.raw()), 0);
    for (index, value) in [m, k, n, block_size].iter().enumerate() {
        let value = u32::try_from(*value)
            .map_err(|_| TinyError::InvalidShape("Q4 dimension exceeds u32".into()))?;
        encoder.set_bytes(
            (index + 4) as u64,
            mem::size_of::<u32>() as u64,
            &value as *const u32 as *const c_void,
        );
    }
    encoder.dispatch_threads(MTLSize::new(n as u64, m as u64, 1), MTLSize::new(16, 16, 1));
    encoder.end_encoding();
    command.commit();
    command.wait_until_completed();
    Ok(output)
}
