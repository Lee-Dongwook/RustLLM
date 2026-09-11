use std::ffi::c_void;
use std::mem;

use ::metal::{MTLSize, CommandBufferRef};

use crate::error::{Result, TinyError};

use crate::metal::{MetalBuffer, MetalContext, MetalExecution};

pub fn mul_f32(context: &MetalContext, a: &MetalBuffer, b: &MetalBuffer,) -> Result<MetalBuffer> {
    let execution = MetalExecution::new(context);

    let output = mul_f32_encode(context, execution.command_buffer(), a, b,)?;

    execution.finish();

    Ok(output)
}

pub(crate) fn mul_f32_encode(context: &MetalContext, command_buffer: &CommandBufferRef, a: &MetalBuffer, b: &MetalBuffer) -> Result<MetalBuffer> {
    if a.len() != b.len() {
        return Err(TinyError::InvalidShape(format!(
            "element-wise multiply requires equal buffer lengths, got {} and {}",
            a.len(),
            b.len(),
        )));
    }

    let output = MetalBuffer::empty(context, a.len());

    let shader_source = include_str!("../../kernels/mul.metal");

    let pipeline = context.pipeline(shader_source, "mul_f32");

    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline.as_ref());

    encoder.set_buffer(0, Some(a.raw()), 0);

    encoder.set_buffer(1, Some(b.raw()), 0);

    encoder.set_buffer(2, Some(output.raw()), 0);

    let numel = u32::try_from(a.len())
        .map_err(|_| TinyError::InvalidShape("multiply input size exceeds u32".to_string()))?;

    encoder.set_bytes(
        3,
        mem::size_of::<u32>() as u64,
        &numel as *const u32 as *const c_void,
    );

    let grid = MTLSize::new(a.len() as u64, 1, 1);

    let threads_per_group = MTLSize::new(a.len().min(256) as u64, 1, 1);

    encoder.dispatch_threads(grid, threads_per_group);

    encoder.end_encoding();

    Ok(output)
}

pub fn mul_f16(context: &MetalContext, a: &MetalBuffer, b: &MetalBuffer) -> Result<MetalBuffer> {
    let execution = MetalExecution::new(context);

    let output = mul_f16_encode(context, execution.command_buffer(), a, b,)?;

    execution.finish();

    Ok(output)
}

pub(crate) fn mul_f16_encode(context: &MetalContext, command_buffer: &CommandBufferRef, a: &MetalBuffer, b: &MetalBuffer) -> Result<MetalBuffer> {
    if a.len() != b.len() {
        return Err(TinyError::InvalidShape(format!(
            "element-wise multiply requires equal buffer lengths, got {} and {}",
            a.len(),
            b.len(),
        )));
    }

    let output = MetalBuffer::empty_with_element_size(context, a.len(), 2);

    let shader_source = include_str!("../../kernels/mul.metal");

    let pipeline = context.pipeline(shader_source, "mul_f16");

    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline.as_ref());

    encoder.set_buffer(0, Some(a.raw()), 0);

    encoder.set_buffer(1, Some(b.raw()), 0);

    encoder.set_buffer(2, Some(output.raw()), 0);

    let grid = MTLSize::new(a.len() as u64, 1, 1);

    let threads_per_group = MTLSize::new(a.len().min(256) as u64, 1, 1);

    encoder.dispatch_threads(grid, threads_per_group);

    encoder.end_encoding();

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::mul_f16_encode;

    use crate::{
        metal::{
            MetalContext,
            MetalExecution,
        },
        ops::add_f16_encode,
        tensor::{
            DType,
            Tensor,
        },
    };

    #[test]
    fn batches_add_then_mul_in_one_command_buffer() {
        let Ok(context) =
            MetalContext::new()
        else {
            return;
        };

        let a =
            Tensor::from_f16_slice(
                &context,
                &[
                    half::f16::from_f32(1.0),
                    half::f16::from_f32(2.0),
                ],
                &[2],
            )
            .unwrap();

        let b =
            Tensor::from_f16_slice(
                &context,
                &[
                    half::f16::from_f32(10.0),
                    half::f16::from_f32(20.0),
                ],
                &[2],
            )
            .unwrap();

        let c =
            Tensor::from_f16_slice(
                &context,
                &[
                    half::f16::from_f32(2.0),
                    half::f16::from_f32(3.0),
                ],
                &[2],
            )
            .unwrap();

        let execution =
            MetalExecution::new(
                &context,
            );

        /*
         * 1 + 10 = 11
         * 2 + 20 = 22
         */
        let added =
            add_f16_encode(
                &context,
                execution.command_buffer(),
                a.metal_buffer().unwrap(),
                b.metal_buffer().unwrap(),
            )
            .unwrap();

        /*
         * 11 * 2 = 22
         * 22 * 3 = 66
         */
        let multiplied =
            mul_f16_encode(
                &context,
                execution.command_buffer(),
                &added,
                c.metal_buffer().unwrap(),
            )
            .unwrap();

        /*
         * 중요한 부분:
         *
         * 위 두 kernel 사이에는
         * commit도 없고
         * wait도 없다.
         *
         * 여기서 처음 한 번만 실행한다.
         */
        execution.finish();

        let result =
            Tensor::from_metal_buffer(
                multiplied,
                &[2],
                DType::F16,
            )
            .unwrap();

        let values =
            result
                .to_f32_vec()
                .unwrap();

        assert!(
            (values[0] - 22.0)
                .abs()
                < 0.01
        );

        assert!(
            (values[1] - 66.0)
                .abs()
                < 0.01
        );
    }
}
