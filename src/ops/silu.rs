use std::ffi::c_void;
use std::mem;

use ::metal::{CommandBufferRef, MTLSize};

use crate::error::{Result, TinyError};

use crate::metal::{MetalBuffer, MetalContext, MetalExecution};

pub fn silu_f32(context: &MetalContext, input: &MetalBuffer) -> Result<MetalBuffer> {
    let execution = MetalExecution::new(context);

    let output = silu_f32_encode(context, execution.command_buffer(), input)?;

    execution.finish();

    Ok(output)
}

pub(crate) fn silu_f32_encode(
    context: &MetalContext,
    command_buffer: &CommandBufferRef,
    input: &MetalBuffer,
) -> Result<MetalBuffer> {
    if input.is_empty() {
        return Err(TinyError::InvalidShape(
            "SiLU input cannot be empty".to_string(),
        ));
    }

    let output = MetalBuffer::empty(context, input.len());

    let shader_source = include_str!("../../kernels/silu.metal");

    let pipeline = context.pipeline(shader_source, "silu_f32");

    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline.as_ref());

    encoder.set_buffer(0, Some(input.raw()), 0);

    encoder.set_buffer(1, Some(output.raw()), 0);

    let numel = u32::try_from(input.len())
        .map_err(|_| TinyError::InvalidShape("SiLU input size exceeds u32".to_string()))?;

    encoder.set_bytes(
        2,
        mem::size_of::<u32>() as u64,
        &numel as *const u32 as *const c_void,
    );

    let grid = MTLSize::new(input.len() as u64, 1, 1);

    let threads_per_group = MTLSize::new(input.len().min(256) as u64, 1, 1);

    encoder.dispatch_threads(grid, threads_per_group);

    encoder.end_encoding();

    Ok(output)
}

pub fn silu_f16(context: &MetalContext, input: &MetalBuffer) -> Result<MetalBuffer> {
    let execution = MetalExecution::new(context);

    let output = silu_f16_encode(context, execution.command_buffer(), input)?;

    execution.finish();

    Ok(output)
}

pub(crate) fn silu_f16_encode(
    context: &MetalContext,
    command_buffer: &CommandBufferRef,
    input: &MetalBuffer,
) -> Result<MetalBuffer> {
    if input.is_empty() {
        return Err(TinyError::InvalidShape(
            "SiLU input cannot be empty".to_string(),
        ));
    }

    let output = MetalBuffer::empty_with_element_size(context, input.len(), 2);

    let shader_source = include_str!("../../kernels/silu.metal");

    let pipeline = context.pipeline(shader_source, "silu_f16");

    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline.as_ref());

    encoder.set_buffer(0, Some(input.raw()), 0);

    encoder.set_buffer(1, Some(output.raw()), 0);

    let grid = MTLSize::new(input.len() as u64, 1, 1);

    let threads_per_group = MTLSize::new(input.len().min(256) as u64, 1, 1);

    encoder.dispatch_threads(grid, threads_per_group);

    encoder.end_encoding();

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::silu_f16_encode;

    use crate::{
        metal::{MetalContext, MetalExecution},
        ops::mul_f16_encode,
        tensor::{DType, Tensor},
    };

    #[test]
    fn batches_silu_then_mul_in_one_command_buffer() {
        let Ok(context) = MetalContext::new() else {
            return;
        };

        let gate = Tensor::from_f16_slice(
            &context,
            &[half::f16::from_f32(1.0), half::f16::from_f32(-1.0)],
            &[2],
        )
        .unwrap();

        let up = Tensor::from_f16_slice(
            &context,
            &[half::f16::from_f32(2.0), half::f16::from_f32(3.0)],
            &[2],
        )
        .unwrap();

        let execution = MetalExecution::new(&context);

        let activated = silu_f16_encode(
            &context,
            execution.command_buffer(),
            gate.metal_buffer().unwrap(),
        )
        .unwrap();

        let multiplied = mul_f16_encode(
            &context,
            execution.command_buffer(),
            &activated,
            up.metal_buffer().unwrap(),
        )
        .unwrap();

        execution.finish();

        let result = Tensor::from_metal_buffer(multiplied, &[2], DType::F16).unwrap();

        let values = result.to_f32_vec().unwrap();

        assert!((values[0] - 1.4621172).abs() < 0.02);

        assert!((values[1] + 0.8068242).abs() < 0.02);
    }
}
