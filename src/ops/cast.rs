use crate::{
    error::{Result, TinyError},
    metal::MetalContext,
    tensor::{DType, Tensor},
};
use ::metal::MTLSize;

const SHADER_SOURCE: &str = include_str!("../../kernels/cast.metal");

pub fn cast(context: &MetalContext, input: &Tensor, dtype: DType) -> Result<Tensor> {
    if input.dtype() == dtype {
        return Ok(input.clone());
    }

    match (input.dtype(), dtype) {
        (DType::F32, DType::F16) => cast_f32_to_f16(context, input),

        (DType::F16, DType::F32) => cast_f16_to_f32(context, input),

        _ => Err(TinyError::UnsupportedDType(format!(
            "cannot cast {:?} to {:?}",
            input.dtype(),
            dtype,
        ))),
    }
}

fn cast_f32_to_f16(context: &MetalContext, input: &Tensor) -> Result<Tensor> {
    let input = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous(context)?
    };

    let output = Tensor::empty(context, input.shape().dims(), DType::F16)?;

    let pipeline = context.pipeline(SHADER_SOURCE, "cast_f32_to_f16");

    dispatch_cast(context, &input, &output, pipeline.as_ref())?;

    Ok(output)
}

fn cast_f16_to_f32(context: &MetalContext, input: &Tensor) -> Result<Tensor> {
    let input = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous(context)?
    };

    let output = Tensor::empty(context, input.shape().dims(), DType::F32)?;

    let pipeline = context.pipeline(SHADER_SOURCE, "cast_f16_to_f32");

    dispatch_cast(context, &input, &output, pipeline.as_ref())?;

    Ok(output)
}

fn dispatch_cast(
    context: &MetalContext,
    input: &Tensor,
    output: &Tensor,
    pipeline: &::metal::ComputePipelineState,
) -> Result<()> {
    let numel = input.numel();
    u32::try_from(numel)
        .map_err(|_| TinyError::InvalidShape("cast element count exceeds u32".to_string()))?;

    let command_buffer = context.command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_buffer(0, Some(input.metal_buffer()?.raw()), 0);
    encoder.set_buffer(1, Some(output.metal_buffer()?.raw()), 0);
    encoder.dispatch_threads(
        MTLSize::new(numel as u64, 1, 1),
        MTLSize::new(numel.min(256) as u64, 1, 1),
    );
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
    Ok(())
}
