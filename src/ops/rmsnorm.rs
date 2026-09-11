use std::ffi::c_void;
use std::mem;

use ::metal::{CommandBufferRef,MTLSize};

use crate::error::{Result, TinyError};

use crate::metal::{MetalBuffer, MetalContext, MetalExecution};
use crate::tensor::{DType, Tensor};

const THREADGROUP_SIZE: u64 = 256;

pub fn rmsnorm_f32(context: &MetalContext, input: &MetalBuffer, weight: &MetalBuffer, rows: usize, hidden_size: usize, epsilon: f32) -> Result<MetalBuffer> {
    let execution = MetalExecution::new(context);

    let output = rmsnorm_f32_encode(context, execution.command_buffer(), input, weight, rows, hidden_size, epsilon)?;

    execution.finish();

    Ok(output)
}

pub(crate) fn rmsnorm_f32_encode(
    context: &MetalContext,
    command_buffer: &CommandBufferRef,
    input: &MetalBuffer,
    weight: &MetalBuffer,
    rows: usize,
    hidden_size: usize,
    epsilon: f32,
) -> Result<MetalBuffer> {
    if hidden_size == 0 {
        return Err(TinyError::InvalidShape(
            "RMSNorm hidden size cannot be zero".to_string(),
        ));
    }

    if input.len() != rows * hidden_size {
        return Err(TinyError::InvalidShape(format!(
            "RMSNorm input has {} elements, expected {}",
            input.len(),
            rows * hidden_size,
        )));
    }

    if weight.len() != hidden_size {
        return Err(TinyError::InvalidShape(format!(
            "RMSNorm weight has {} elements, expected {}",
            weight.len(),
            hidden_size,
        )));
    }

    let output = MetalBuffer::empty(context, input.len());

    let shader_source = include_str!("../../kernels/rmsnorm.metal");

    let pipeline = context.pipeline(shader_source, "rmsnorm_f32");

    if THREADGROUP_SIZE > pipeline.max_total_threads_per_threadgroup() {
        return Err(TinyError::Metal(format!(
            "RMSNorm requires {THREADGROUP_SIZE} threads per group, but pipeline supports at most {}",
            pipeline.max_total_threads_per_threadgroup(),
        )));
    }

    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(pipeline.as_ref());

    encoder.set_buffer(0, Some(input.raw()), 0);

    encoder.set_buffer(1, Some(weight.raw()), 0);

    encoder.set_buffer(2, Some(output.raw()), 0);

    let hidden_size_u32 = u32::try_from(hidden_size)
        .map_err(|_| TinyError::InvalidShape("RMSNorm hidden size exceeds u32".to_string()))?;

    encoder.set_bytes(
        3,
        mem::size_of::<u32>() as u64,
        &hidden_size_u32 as *const u32 as *const c_void,
    );

    encoder.set_bytes(
        4,
        mem::size_of::<f32>() as u64,
        &epsilon as *const f32 as *const c_void,
    );

    let thread_groups = MTLSize::new(rows as u64, 1, 1);

    let threads_per_group = MTLSize::new(THREADGROUP_SIZE, 1, 1);

    encoder.dispatch_thread_groups(thread_groups, threads_per_group);

    encoder.end_encoding();

    Ok(output)
}

pub fn rmsnorm_f16(context: &MetalContext, input: &MetalBuffer, weight: &MetalBuffer, rows: usize, hidden_size: usize, epsilon: f32) -> Result<MetalBuffer> {
    let execution = MetalExecution::new(context);
    let output = rmsnorm_f16_encode(context, execution.command_buffer(), input, weight, rows, hidden_size, epsilon)?;

    execution.finish();

    Ok(output)
}

pub(crate) fn rmsnorm_f16_encode(
    context: &MetalContext,
    command_buffer: &CommandBufferRef,
    input: &MetalBuffer,
    weight: &MetalBuffer,
    rows: usize,
    hidden_size: usize,
    epsilon: f32,
) -> Result<MetalBuffer> {
    if hidden_size == 0 {
        return Err(TinyError::InvalidShape(
            "RMSNorm hidden size cannot be zero".to_string(),
        ));
    }
    if input.len() != rows * hidden_size {
        return Err(TinyError::InvalidShape(format!(
            "RMSNorm input has {} elements, expected {}",
            input.len(),
            rows * hidden_size,
        )));
    }
    if weight.len() != hidden_size {
        return Err(TinyError::InvalidShape(format!(
            "RMSNorm weight has {} elements, expected {}",
            weight.len(),
            hidden_size,
        )));
    }

    let output = MetalBuffer::empty_with_element_size(context, input.len(), 2);
    let shader_source = include_str!("../../kernels/rmsnorm.metal");
    let pipeline = context.pipeline(shader_source, "rmsnorm_f16");

    if THREADGROUP_SIZE > pipeline.max_total_threads_per_threadgroup() {
        return Err(TinyError::Metal(format!(
            "RMSNorm requires {THREADGROUP_SIZE} threads per group, but pipeline supports at most {}",
            pipeline.max_total_threads_per_threadgroup(),
        )));
    }

    let hidden_size_u32 = u32::try_from(hidden_size)
        .map_err(|_| TinyError::InvalidShape("RMSNorm hidden size exceeds u32".to_string()))?;
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline.as_ref());
    encoder.set_buffer(0, Some(input.raw()), 0);
    encoder.set_buffer(1, Some(weight.raw()), 0);
    encoder.set_buffer(2, Some(output.raw()), 0);
    encoder.set_bytes(
        3,
        mem::size_of::<u32>() as u64,
        &hidden_size_u32 as *const u32 as *const c_void,
    );
    encoder.set_bytes(
        4,
        mem::size_of::<f32>() as u64,
        &epsilon as *const f32 as *const c_void,
    );
    encoder.dispatch_thread_groups(
        MTLSize::new(rows as u64, 1, 1),
        MTLSize::new(THREADGROUP_SIZE, 1, 1),
    );
    encoder.end_encoding();

    Ok(output)
}

pub fn rmsnorm(
    context: &MetalContext,
    input: &Tensor,
    weight: &Tensor,
    epsilon: f32,
) -> Result<Tensor> {
    if input.dtype() != weight.dtype() {
        return Err(TinyError::UnsupportedDType(format!(
            "RMSNorm dtype mismatch: input {:?}, weight {:?}",
            input.dtype(),
            weight.dtype(),
        )));
    }
    if input.rank() == 0 {
        return Err(TinyError::InvalidDimension(
            "RMSNorm input must have at least one dimension".to_string(),
        ));
    }
    if !epsilon.is_finite() || epsilon <= 0.0 {
        return Err(TinyError::InvalidShape(format!(
            "RMSNorm epsilon must be positive and finite, got {epsilon}",
        )));
    }

    let hidden_size = input.dim(input.rank() - 1)?;
    if weight.rank() != 1 || weight.numel() != hidden_size {
        return Err(TinyError::ShapeMismatch {
            left: input.shape().dims().to_vec(),
            right: weight.shape().dims().to_vec(),
        });
    }

    // F16 stride materialization is not implemented yet, so do not silently
    // treat a strided view as contiguous input.
    if input.dtype() == DType::F16 && (!input.is_contiguous() || !weight.is_contiguous()) {
        return Err(TinyError::NonContiguousTensor(
            "F16 RMSNorm currently requires contiguous input and weight".to_string(),
        ));
    }

    let input = input.contiguous(context)?;
    let weight = weight.contiguous(context)?;
    let rows = input.numel() / hidden_size;

    let buffer = match input.dtype() {
        DType::F32 => rmsnorm_f32(
            context,
            input.metal_buffer()?,
            weight.metal_buffer()?,
            rows,
            hidden_size,
            epsilon,
        )?,
        DType::F16 => rmsnorm_f16(
            context,
            input.metal_buffer()?,
            weight.metal_buffer()?,
            rows,
            hidden_size,
            epsilon,
        )?,
    };

    Tensor::from_metal_buffer(buffer, input.shape().dims(), input.dtype())
}

#[cfg(test)]
mod tests {
    use super::{
        rmsnorm_f16,
        rmsnorm_f16_encode,
        rmsnorm_f32,
    };

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

    const EPSILON: f32 = 1e-5;

    fn cpu_rmsnorm(
        input: &[f32],
        weight: &[f32],
        rows: usize,
        hidden_size: usize,
        epsilon: f32,
    ) -> Vec<f32> {
        assert_eq!(
            input.len(),
            rows * hidden_size,
        );

        assert_eq!(
            weight.len(),
            hidden_size,
        );

        let mut output =
            vec![0.0f32; input.len()];

        for row in 0..rows {
            let row_start =
                row * hidden_size;

            let row_end =
                row_start + hidden_size;

            let row_input =
                &input[row_start..row_end];

            let mean_square =
                row_input
                    .iter()
                    .map(|value| {
                        value * value
                    })
                    .sum::<f32>()
                    / hidden_size as f32;

            let inv_rms =
                1.0
                    / (mean_square + epsilon)
                        .sqrt();

            for column in 0..hidden_size {
                let index =
                    row_start + column;

                output[index] =
                    input[index]
                        * inv_rms
                        * weight[column];
            }
        }

        output
    }

    fn assert_close(
        actual: &[f32],
        expected: &[f32],
        tolerance: f32,
    ) {
        assert_eq!(
            actual.len(),
            expected.len(),
        );

        for (
            index,
            (actual, expected),
        ) in actual
            .iter()
            .zip(expected.iter())
            .enumerate()
        {
            let error =
                (actual - expected).abs();

            assert!(
                error <= tolerance,
                "value {index} differs by {error}: \
                 actual={actual}, \
                 expected={expected}, \
                 tolerance={tolerance}",
            );
        }
    }

    fn metal_context() -> Option<MetalContext> {
        MetalContext::new().ok()
    }

    #[test]
    fn rmsnorm_f32_matches_cpu_reference() {
        let Some(context) =
            metal_context()
        else {
            return;
        };

        /*
         * 두 row를 일부러 사용한다.
         *
         * row 0 = [1, 2, 3, 4]
         * row 1 = [-1, 0.5, -0.5, 2]
         */
        let input_values = [
            1.0f32,
            2.0,
            3.0,
            4.0,
            -1.0,
            0.5,
            -0.5,
            2.0,
        ];

        /*
         * 모든 feature에 동일한 weight를 쓰지 않고
         * 서로 다른 값을 써서 weight 적용도 검증한다.
         */
        let weight_values = [
            1.0f32,
            1.5,
            0.5,
            2.0,
        ];

        let rows = 2;
        let hidden_size = 4;

        let input =
            Tensor::from_f32_slice(
                &context,
                &input_values,
                &[rows, hidden_size],
            )
            .unwrap();

        let weight =
            Tensor::from_f32_slice(
                &context,
                &weight_values,
                &[hidden_size],
            )
            .unwrap();

        let expected =
            cpu_rmsnorm(
                &input_values,
                &weight_values,
                rows,
                hidden_size,
                EPSILON,
            );

        let output_buffer =
            rmsnorm_f32(
                &context,
                input
                    .metal_buffer()
                    .unwrap(),
                weight
                    .metal_buffer()
                    .unwrap(),
                rows,
                hidden_size,
                EPSILON,
            )
            .unwrap();

        let output =
            Tensor::from_metal_buffer(
                output_buffer,
                &[rows, hidden_size],
                DType::F32,
            )
            .unwrap();

        let actual =
            output
                .to_f32_vec()
                .unwrap();

        assert_close(
            &actual,
            &expected,
            1e-4,
        );
    }

    #[test]
    fn rmsnorm_f16_matches_cpu_reference() {
        let Some(context) =
            metal_context()
        else {
            return;
        };

        let input_values = [
            1.0f32,
            2.0,
            3.0,
            4.0,
            -1.0,
            0.5,
            -0.5,
            2.0,
        ];

        let weight_values = [
            1.0f32,
            1.5,
            0.5,
            2.0,
        ];

        let rows = 2;
        let hidden_size = 4;

        let input_f16: Vec<half::f16> =
            input_values
                .iter()
                .copied()
                .map(half::f16::from_f32)
                .collect();

        let weight_f16: Vec<half::f16> =
            weight_values
                .iter()
                .copied()
                .map(half::f16::from_f32)
                .collect();

        let input =
            Tensor::from_f16_slice(
                &context,
                &input_f16,
                &[rows, hidden_size],
            )
            .unwrap();

        let weight =
            Tensor::from_f16_slice(
                &context,
                &weight_f16,
                &[hidden_size],
            )
            .unwrap();

        let expected =
            cpu_rmsnorm(
                &input_values,
                &weight_values,
                rows,
                hidden_size,
                EPSILON,
            );

        let output_buffer =
            rmsnorm_f16(
                &context,
                input
                    .metal_buffer()
                    .unwrap(),
                weight
                    .metal_buffer()
                    .unwrap(),
                rows,
                hidden_size,
                EPSILON,
            )
            .unwrap();

        let output =
            Tensor::from_metal_buffer(
                output_buffer,
                &[rows, hidden_size],
                DType::F16,
            )
            .unwrap();

        assert_eq!(
            output.dtype(),
            DType::F16,
        );

        let actual =
            output
                .to_f32_vec()
                .unwrap();

        /*
         * F16 저장 때문에 F32보다 tolerance를 넓게 잡는다.
         */
        assert_close(
            &actual,
            &expected,
            0.02,
        );
    }

    #[test]
    fn batches_rmsnorm_then_add_in_one_command_buffer() {
        let Some(context) =
            metal_context()
        else {
            return;
        };

        let input_values = [
            1.0f32,
            2.0,
            3.0,
            4.0,
            -1.0,
            0.5,
            -0.5,
            2.0,
        ];

        let weight_values = [
            1.0f32,
            1.5,
            0.5,
            2.0,
        ];

        let rows = 2;
        let hidden_size = 4;

        let input_f16: Vec<half::f16> =
            input_values
                .iter()
                .copied()
                .map(half::f16::from_f32)
                .collect();

        let weight_f16: Vec<half::f16> =
            weight_values
                .iter()
                .copied()
                .map(half::f16::from_f32)
                .collect();

        let zeros =
            vec![
                half::f16::from_f32(0.0);
                rows * hidden_size
            ];

        let input =
            Tensor::from_f16_slice(
                &context,
                &input_f16,
                &[rows, hidden_size],
            )
            .unwrap();

        let weight =
            Tensor::from_f16_slice(
                &context,
                &weight_f16,
                &[hidden_size],
            )
            .unwrap();

        let zero_tensor =
            Tensor::from_f16_slice(
                &context,
                &zeros,
                &[rows, hidden_size],
            )
            .unwrap();

        let expected =
            cpu_rmsnorm(
                &input_values,
                &weight_values,
                rows,
                hidden_size,
                EPSILON,
            );

        /*
         * 여기서 command buffer를 딱 하나 만든다.
         */
        let execution =
            MetalExecution::new(
                &context,
            );

        /*
         * Kernel #1
         *
         * RMSNorm을 encode만 한다.
         *
         * 아직 commit도,
         * wait도 하지 않는다.
         */
        let normalized =
            rmsnorm_f16_encode(
                &context,
                execution.command_buffer(),
                input
                    .metal_buffer()
                    .unwrap(),
                weight
                    .metal_buffer()
                    .unwrap(),
                rows,
                hidden_size,
                EPSILON,
            )
            .unwrap();

        /*
         * Kernel #2
         *
         * normalized는 아직 GPU 계산이 끝난 상태가 아닐 수 있다.
         *
         * 하지만 동일 command buffer에서 다음 kernel의
         * input으로 바로 사용할 수 있다.
         *
         * + 0 이므로 최종 결과는 RMSNorm과 동일해야 한다.
         */
        let added =
            add_f16_encode(
                &context,
                execution.command_buffer(),
                &normalized,
                zero_tensor
                    .metal_buffer()
                    .unwrap(),
            )
            .unwrap();

        /*
         * 여기에서 처음이자 마지막으로
         * command buffer를 commit하고 기다린다.
         */
        execution.finish();

        let output =
            Tensor::from_metal_buffer(
                added,
                &[rows, hidden_size],
                DType::F16,
            )
            .unwrap();

        let actual =
            output
                .to_f32_vec()
                .unwrap();

        assert_close(
            &actual,
            &expected,
            0.02,
        );
    }
}
